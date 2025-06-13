#!/bin/bash

# Strict error handling
set -e
set -o pipefail

# Ensure models are downloaded to the local models directory
export HF_HOME="./models"
export HUGGINGFACE_HUB_CACHE="./models" # For some huggingface libs

# --- Global Variables (will be sourced from config.sh) ---
LOG_DIR="./logs" # Default, can be overridden by config
VENV_DIR="./vllm_env" # Default, can be overridden by config
CONFIG_FILE="./config/config.sh"
# PIDs for cleanup
VLLM_PID=""
NGROK_PID=""
KEEPALIVE_PID=""
# For Ngrok URL
FINAL_NGROK_URL=""


# --- Log File Initialisation ---
# Initial setup, might be updated after config load if LOG_DIR changes
mkdir -p "$LOG_DIR" # Ensure log directory exists
DEPLOYMENT_LOG="$LOG_DIR/deployment.log"
VLLM_LOG="$LOG_DIR/vllm_server.log"
NGROK_LOG="$LOG_DIR/ngrok_tunnel.log"
KEEPALIVE_LOG="$LOG_DIR/keep_alive.log"

# Function to log messages to both stdout and the deployment log
log() {
    # POSIX compliant date formatting. Works on Linux and macOS without gdate.
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$DEPLOYMENT_LOG"
}

# --- Function Definitions ---

load_config() {
    log "INFO: Loading configuration..."
    if [ -f "$CONFIG_FILE" ]; then
        # shellcheck source=./config/config.sh
        source "$CONFIG_FILE"
        log "INFO: Configuration loaded from $CONFIG_FILE."
        # Update log paths if LOG_DIR was changed in config
        DEPLOYMENT_LOG="$LOG_DIR/deployment.log"
        VLLM_LOG="$LOG_DIR/vllm_server.log"
        NGROK_LOG="$LOG_DIR/ngrok_tunnel.log"
        KEEPALIVE_LOG="$LOG_DIR/keep_alive.log"
        mkdir -p "$LOG_DIR" # Ensure it exists
    else
        log "ERROR: Configuration file $CONFIG_FILE not found!"
        log "Please copy config/config.sh.example to config/config.sh and customize it."
        exit 1
    fi
}

check_config_vars() {
    log "INFO: Checking required configuration variables..."
    local required_vars=( \
        MODEL_ID_GPU MODEL_ID_CPU USE_TEST_MODEL \
        VLLM_EXTRA_ARGS_GPU VLLM_EXTRA_ARGS_CPU \
        NGROK_AUTHTOKEN NGROK_TUNNEL_NAME \
        VLLM_HOST VLLM_PORT \
        VENV_DIR LOG_DIR MODEL_CACHE_DIR \
        ENABLE_KEEP_ALIVE IDLE_THRESHOLD_MINUTES \
        KEEPALIVE_PROMPT_INTERVAL_MINUTES KEEPALIVE_PROMPTS \
        KEEPALIVE_TARGET_ENDPOINT \
    )
    local missing_vars=0
    for var in "${required_vars[@]}"; do
        # Check if var is unset or empty string
        if [ -z "${!var+x}" ] || [ -z "${!var}" ]; then
            # KEEPALIVE_PROMPTS is special as it's an array and might be empty if not set.
            # The check for array emptiness is done later if ENABLE_KEEP_ALIVE is true.
            if [ "$var" = "KEEPALIVE_PROMPTS" ]; then
                if [ "$ENABLE_KEEP_ALIVE" = "true" ] && [ ${#KEEPALIVE_PROMPTS[@]} -eq 0 ]; then
                    log "ERROR: ENABLE_KEEP_ALIVE is true but KEEPALIVE_PROMPTS array is empty in $CONFIG_FILE."
                    missing_vars=1
                fi
                continue # Handled specifically
            fi
            log "ERROR: Required configuration variable '$var' is not set or is empty in $CONFIG_FILE."
            missing_vars=1
        fi
    done
    if [ "$missing_vars" -eq 1 ]; then
        log "ERROR: Please ensure all required variables are set and not empty in $CONFIG_FILE."
        exit 1
    fi
    # Specific check for KEEPALIVE_PROMPTS array emptiness if keep_alive is enabled
    if [ "$ENABLE_KEEP_ALIVE" = "true" ] && [ ${#KEEPALIVE_PROMPTS[@]} -eq 0 ]; then
        # This condition might be redundant if KEEPALIVE_PROMPTS being empty makes ${!var} (i.e. ${KEEPALIVE_PROMPTS}) effectively empty.
        # However, explicit check for array count is more robust for arrays.
        log "ERROR: ENABLE_KEEP_ALIVE is true but KEEPALIVE_PROMPTS array is empty in $CONFIG_FILE."
        exit 1
    fi
    log "INFO: All required configuration variables seem present and non-empty where required."
}

pre_flight_checks() {
    log "INFO: Performing pre-flight checks..."

    # Check for CUDA and NVIDIA drivers
    if ! command -v nvidia-smi &> /dev/null; then
        log "WARNING: nvidia-smi command not found. If using GPU, please ensure NVIDIA drivers and CUDA are installed."
        # For CPU mode, this might be okay. Add more checks if GPU is explicitly selected later.
    else
        log "INFO: NVIDIA-SMI output:"
        (nvidia-smi || log "WARNING: nvidia-smi command failed to execute.") | tee -a "$DEPLOYMENT_LOG"
    fi

    # Check Python version
    if ! command -v python3 &> /dev/null; then
        log "ERROR: python3 command not found. Python 3.8 or higher is required."
        exit 1
    fi
    PYTHON_VERSION=$(python3 -c "import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")")
    # Using awk for version comparison as bc might not be available
    if [ "$(echo "$PYTHON_VERSION 3.8" | awk '{if ($1 < $2) print "true"; else print "false";}')" = "true" ]; then
        log "ERROR: Python 3.8 or higher is required. Current version: $PYTHON_VERSION"
        exit 1
    fi
    log "INFO: Python version: $PYTHON_VERSION"

    # Check for essential commands
    local essential_commands=("curl" "wget" "unzip" "git" "sudo" "awk" "pkill")
    for cmd in "${essential_commands[@]}"; do
        if ! command -v $cmd &> /dev/null; then
            log "WARNING: Essential command '$cmd' not found. The script might fail if it needs it."
            # Consider installing them or making them hard requirements if absolutely necessary
        fi
    done
    log "INFO: Pre-flight checks complete."
}

setup_venv() {
    log "INFO: Setting up Python virtual environment in $VENV_DIR..."
    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        log "INFO: Virtual environment not found. Creating..."
        python3 -m venv "$VENV_DIR"
        log "INFO: Virtual environment created."
    else
        log "INFO: Virtual environment already exists."
    fi
    # Activation is tricky in scripts. We will call pip/python directly from venv.
    # Example: $VENV_DIR/bin/python script.py
    # Example: $VENV_DIR/bin/pip install package
    log "INFO: Python virtual environment setup complete. Use $VENV_DIR/bin/python and $VENV_DIR/bin/pip."
}

install_dependencies() {
    log "INFO: Installing system dependencies (if any identified as missing and auto-install is desired)..."
    # Example: sudo apt update && sudo apt install -y some-package
    # For now, pre-flight checks just warn. If auto-install is needed, add logic here.

    log "INFO: Installing Python dependencies from venv..."
    # Activate venv for this block or call pip directly
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install vllm packaging # packaging for version comparison if needed later

    # Check if vLLM installed correctly
    if ! "$VENV_DIR/bin/python" -c "import vllm" &> /dev/null; then
        log "ERROR: vLLM installation failed. Please check the logs above."
        log "You might need to install a specific CUDA version of vLLM (e.g., pip install vllm[cu121])."
        exit 1
    fi
    log "INFO: vLLM installed successfully."
    log "INFO: Dependencies installation complete."
}

start_vllm() {
    log "INFO: Preparing to start vLLM server..."
    local model_to_use
    local vllm_args_to_use

    if [ "$USE_TEST_MODEL" = "true" ]; then
        log "INFO: Test mode enabled. Using CPU model: $MODEL_ID_CPU"
        model_to_use="$MODEL_ID_CPU"
        vllm_args_to_use="$VLLM_EXTRA_ARGS_CPU"
    else
        log "INFO: GPU mode enabled. Using GPU model: $MODEL_ID_GPU"
        model_to_use="$MODEL_ID_GPU"
        vllm_args_to_use="$VLLM_EXTRA_ARGS_GPU"
    fi

    log "INFO: Model: $model_to_use"
    log "INFO: Host: $VLLM_HOST, Port: $VLLM_PORT"
    log "INFO: Model Cache Dir: $MODEL_CACHE_DIR" # Added log for visibility

    # Construct VLLM command
    # Ensure HF_HOME and MODEL_CACHE_DIR are set for this command execution environment
    # shellcheck disable=SC2086
    VLLM_COMMAND_STR="HF_HOME='$MODEL_CACHE_DIR' HUGGINGFACE_HUB_CACHE='$MODEL_CACHE_DIR' $VENV_DIR/bin/python -m vllm.entrypoints.openai.api_server \
      --model '$model_to_use' \
      --host '$VLLM_HOST' \
      --port '$VLLM_PORT' \
      --trust-remote-code \
      $vllm_args_to_use" # $vllm_args_to_use is intentionally not quoted to allow multiple args

    log "INFO: Running vLLM command: $VLLM_COMMAND_STR"
    # Using eval to correctly interpret the command string with arguments and environment variables
    eval "$VLLM_COMMAND_STR" > "$VLLM_LOG" 2>&1 &
    VLLM_PID=$!
    log "INFO: vLLM server started with PID $VLLM_PID. Check $VLLM_LOG for output."

    log "INFO: Giving vLLM server some time to initialize (approx. 60-120 seconds)..."
    sleep 60 # Initial sleep
    # Basic check if server is up, can be improved
    if ! curl -sf "http://$VLLM_HOST:$VLLM_PORT/health"; then # vLLM uses /health for readiness
      log "WARNING: vLLM server might not be ready after 60s. Waiting a bit more..."
      sleep 60
      if ! curl -sf "http://$VLLM_HOST:$VLLM_PORT/health"; then
        log "ERROR: vLLM server failed to start or is not responding on http://$VLLM_HOST:$VLLM_PORT/health after 120s. Check $VLLM_LOG."
        # Optionally kill $VLLM_PID here if we want to be strict
        # exit 1 # Or allow script to continue for ngrok/other parts
        return 1 # Indicate failure
      fi
    fi
    log "INFO: vLLM server appears to be responding."
    return 0 # Indicate success
}

start_ngrok() {
    log "INFO: Starting ngrok tunnel..."

    # Check for ngrok and install if not found
    if ! command -v ngrok &> /dev/null; then
        log "INFO: Ngrok not found, attempting to download and install..."
        # Determine architecture for ngrok download
        local ARCH
        ARCH=$(uname -m)
        local NGROK_ZIP_FILENAME
        local NGROK_OS
        NGROK_OS=$(uname -s | tr '[:upper:]' '[:lower:]')

        case "$ARCH" in
            x86_64) ARCH="amd64" ;;
            aarch64 | arm64) ARCH="arm64" ;;
            armv7l) ARCH="arm" ;; # May need specific handling or different URL
            *) log "ERROR: Unsupported architecture: $ARCH for ngrok auto-install."
               log "INFO: Please install ngrok manually: https://ngrok.com/download"
               return 1 ;;
        esac

        # Ngrok download URL structure can change. This is a common pattern.
        # Example: https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.zip
        # The new URL is https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip (older version)
        # Let's use a more generic URL pattern if possible, or stick to a known working one.
        # The URL from original script: https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
        # This URL might be for a specific ngrok version (v2 or early v3).
        # For ngrok v3, URL is like: https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.zip
        # Let's use the one from the original script as it was specified.
        NGROK_ZIP_FILENAME="ngrok-stable-${NGROK_OS}-${ARCH}.zip"
        # The provided URL was generic for linux-amd64, let's adjust based on detected OS/ARCH or use a fixed known good one.
        # For simplicity, sticking to the provided linux-amd64 URL, assuming it's the target.
        # If more general solution is needed, this part needs expansion.
        if [ "$NGROK_OS" != "linux" ] || [ "$ARCH" != "amd64" ]; then
            log "WARNING: Auto-install for ngrok is configured for linux-amd64. Your system: ${NGROK_OS}-${ARCH}."
            log "INFO: Attempting to use generic linux-amd64 ngrok. If issues, install manually from https://ngrok.com/download"
        fi

        NGROK_ZIP="ngrok-stable-linux-amd64.zip" # Defaulting to this
        NGROK_URL="https://bin.equinox.io/c/4VmDzA7iaHb/$NGROK_ZIP"

        log "INFO: Downloading ngrok from $NGROK_URL..."
        wget "$NGROK_URL" -O "$NGROK_ZIP"
        if ! command -v unzip &> /dev/null; then
            log "ERROR: unzip command not found, cannot install ngrok. Please install unzip."
            rm "$NGROK_ZIP" # Clean up downloaded zip
            return 1
        fi
        sudo unzip -o "$NGROK_ZIP" -d "/usr/local/bin" # Ensure /usr/local/bin is in PATH and writable by sudo
        rm "$NGROK_ZIP"
        log "INFO: Ngrok installed to /usr/local/bin."
    else
        log "INFO: Ngrok already installed."
    fi

    log "INFO: Configuring Ngrok Authtoken..."
    ngrok config add-authtoken "$NGROK_AUTHTOKEN"
    log "INFO: Ngrok authtoken configured."

    log "INFO: Connecting Ngrok to http://$VLLM_HOST:$VLLM_PORT for tunnel definition starting with '$NGROK_TUNNEL_NAME'"
    # Ngrok v3 uses --domain for custom domain, for named tunnels it's usually part of the config file or a label.
    # The command `ngrok http "$VLLM_PORT" --label "$NGROK_TUNNEL_NAME"` seems to refer to labels which might be a paid feature or specific config.
    # A simpler way for a dynamic URL if NGROK_TUNNEL_NAME is just for logging:
    # nohup ngrok http "$VLLM_PORT" --log=stdout > "$NGROK_LOG" 2>&1 &
    # If NGROK_TUNNEL_NAME is a specific domain/subdomain, syntax is like:
    # nohup ngrok http "$VLLM_PORT" --domain "$NGROK_TUNNEL_NAME.ngrok.io" --log=stdout > "$NGROK_LOG" 2>&1 & (for ngrok free/paid plans)
    # Let's assume NGROK_TUNNEL_NAME is a desired subdomain if it contains dots, or just a metadata label.
    # The original script used --label. Ngrok docs state labels are for traffic routing, not subdomain naming.
    # For subdomain, it should be --subdomain for older ngrok or --domain for newer.
    # If NGROK_TUNNEL_NAME is meant to be a specific subdomain on ngrok.io:
    # NGROK_CMD="ngrok http $VLLM_PORT --subdomain $NGROK_TUNNEL_NAME" (for v2 style, if applicable for their account)
    # NGROK_CMD="ngrok http $VLLM_PORT --domain $NGROK_TUNNEL_NAME.ngrok.dev" (for v3 style, if using ngrok.dev domains)
    # Let's use a generic approach that works with free tier (dynamic subdomain) and log the provided name.
    log "INFO: Starting ngrok with dynamic subdomain. The NGROK_TUNNEL_NAME ('$NGROK_TUNNEL_NAME') will be used as a reference."

    nohup ngrok http "$VLLM_PORT" --log=stdout > "$NGROK_LOG" 2>&1 &
    NGROK_PID=$!
    log "INFO: Ngrok tunnel started with PID $NGROK_PID. Check $NGROK_LOG for output."

    log "INFO: Waiting for Ngrok tunnel URL... (up to 30 seconds)"
    sleep 30 # Give ngrok time to establish tunnel

    # Attempt to retrieve URL from ngrok API (localhost:4040)
    # This python script assumes the first tunnel is the one we want.
    # It also assumes python3 and json module are available.
    NGROK_PUBLIC_URL=$(curl -s http://localhost:4040/api/tunnels | "$VENV_DIR/bin/python" -c "import sys, json; data = json.load(sys.stdin); print(next(t['public_url'] for t in data['tunnels'] if t['proto'] == 'https' and t['config']['addr'] == 'http://$VLLM_HOST:$VLLM_PORT'), None)" || true)

    if [ -z "$NGROK_PUBLIC_URL" ] || [ "$NGROK_PUBLIC_URL" = "None" ]; then
        log "WARNING: Failed to retrieve Ngrok URL via API. Trying log parsing (less reliable)..."
        # Fallback to log parsing from the original script
        NGROK_PUBLIC_URL=$(grep -o "url=https://[^ ]*" "$NGROK_LOG" | head -n 1 | cut -d'=' -f2 || true)
    fi

    if [ -z "$NGROK_PUBLIC_URL" ]; then
        log "ERROR: Failed to retrieve Ngrok URL. Check $NGROK_LOG for errors."
        log "You might need to manually inspect '$NGROK_LOG' or run 'ngrok http $VLLM_PORT' in a new terminal."
        return 1 # Indicate failure
    else
        log "INFO: Ngrok tunnel active. External URL (OpenAI-compatible endpoint): $NGROK_PUBLIC_URL/v1"
        # Store for main function to display
        export FINAL_NGROK_URL="$NGROK_PUBLIC_URL/v1"
    fi
    return 0 # Indicate success
}

start_keep_alive_service() {
    if [ "$ENABLE_KEEP_ALIVE" != "true" ]; then
        log "INFO: Keep-alive service is disabled by configuration."
        return 0
    fi

    log "INFO: Starting keep-alive service in background."
    log "INFO: Initial delay: $IDLE_THRESHOLD_MINUTES minutes."
    log "INFO: Prompt interval: $KEEPALIVE_PROMPT_INTERVAL_MINUTES minutes."
    log "INFO: Target endpoint: $KEEPALIVE_TARGET_ENDPOINT"
    # log "INFO: Prompts: ${KEEPALIVE_PROMPTS[*]}" # Log prompts carefully if they are sensitive

    (
        # This subshell runs in the background
        # Wait for the initial idle threshold
        sleep $((IDLE_THRESHOLD_MINUTES * 60))

        log_keep_alive() {
            # Use the main log function but identify the source
            log "KEEPALIVE: $*"
        }

        log_keep_alive "INFO: Keep-alive service activated after initial delay."

        local prompt_index=0
        # Ensure KEEPALIVE_PROMPTS is treated as an array
        declare -a prompts_array=("${KEEPALIVE_PROMPTS[@]}") # Make a local copy to be safe
        local num_prompts=${#prompts_array[@]}

        if [ "$num_prompts" -eq 0 ]; then
            log_keep_alive "ERROR: No prompts defined in KEEPALIVE_PROMPTS array. Keep-alive service exiting."
            exit 1 # Exit subshell
        fi

        while true; do
            local current_prompt="${prompts_array[prompt_index]}"
            local model_for_request="$MODEL_ID_CPU" # Default to CPU model for keep-alive
            if [ "$USE_TEST_MODEL" = "false" ]; then model_for_request="$MODEL_ID_GPU"; fi
            # Allow override for keep-alive model specifically
            if [ -n "$KEEPALIVE_MODEL_ID_OVERRIDE" ] && [ -n "$KEEPALIVE_MODEL_ID_OVERRIDE" ]; then
                model_for_request="$KEEPALIVE_MODEL_ID_OVERRIDE"
            fi

            log_keep_alive "INFO: Sending keep-alive prompt (index $prompt_index): \`$current_prompt\` to model \`$model_for_request\` at \`$KEEPALIVE_TARGET_ENDPOINT\`"

            # Construct JSON payload for OpenAI compatible chat completions
            # Ensure proper escaping for the prompt within JSON
            JSON_ESCAPED_PROMPT=$(printf '%s' "$current_prompt" | python3 -c 'import json, sys; print(json.dumps(sys.stdin.read()))')

            JSON_PAYLOAD=$(cat <<END_JSON
{
    "model": "$model_for_request",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": $JSON_ESCAPED_PROMPT}
    ],
    "temperature": 0.7,
    "max_tokens": 50
}
END_JSON
)
            # Send the request using main log for curl output
            log_keep_alive "DEBUG: JSON Payload for keep-alive: $JSON_PAYLOAD"
            # Send request and log response to main deployment log, identified by KEEPALIVE prefix
            curl_output_and_code=$(curl -s -w "HTTP_STATUS_CODE:%{http_code}" -X POST "$KEEPALIVE_TARGET_ENDPOINT" \
               -H "Content-Type: application/json" \
               -d "$JSON_PAYLOAD" 2>&1) # Capture stdout and stderr

            http_status_code=$(echo "$curl_output_and_code" | grep -o 'HTTP_STATUS_CODE:[0-9]*$' | cut -d':' -f2)
            response_body=$(echo "$curl_output_and_code" | sed 's/HTTP_STATUS_CODE:[0-9]*$//')


            if [[ "$http_status_code" -eq 200 ]]; then
                log_keep_alive "INFO: Keep-alive prompt sent successfully. Status: $http_status_code. Response: $response_body"
            else
                log_keep_alive "ERROR: Keep-alive prompt failed. Status: $http_status_code. Response: $response_body"
            fi

            prompt_index=$(((prompt_index + 1) % num_prompts)) # Cycle through prompts
            sleep $((KEEPALIVE_PROMPT_INTERVAL_MINUTES * 60))
        done
    ) >> "$KEEPALIVE_LOG" 2>&1 & # Redirect stdout/stderr of subshell to keepalive log
    KEEPALIVE_PID=$!
    log "INFO: Keep-alive service started with PID $KEEPALIVE_PID. Logging to $KEEPALIVE_LOG and summarized in main log."
}

cleanup() {
    log "INFO: Shutdown signal received or script ending. Cleaning up background processes..."
    # Add children of this script to the kill list too if any (like those from nohup not captured)
    # pkill -P $$ # Kills children of this script's PID ($$) - use with care

    if [ -n "$KEEPALIVE_PID" ] && kill -0 "$KEEPALIVE_PID" 2>/dev/null; then
        log "INFO: Stopping Keep-Alive service (PID: $KEEPALIVE_PID)..."
        kill "$KEEPALIVE_PID" 2>/dev/null || log "WARN: Failed to kill keep-alive PID $KEEPALIVE_PID. It might have already exited."
    fi
    if [ -n "$NGROK_PID" ] && kill -0 "$NGROK_PID" 2>/dev/null; then
        log "INFO: Stopping Ngrok tunnel (PID: $NGROK_PID)..."
        kill "$NGROK_PID" 2>/dev/null || log "WARN: Failed to kill ngrok PID $NGROK_PID. It might have already exited."
    fi
    # Also try to kill ngrok by name/port as it can be resilient
    if command -v pkill &> /dev/null; then
      pkill -f "ngrok http $VLLM_PORT" || true
    fi

    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        log "INFO: Stopping vLLM server (PID: $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || log "WARN: Failed to kill vLLM PID $VLLM_PID. It might have already exited."
    fi

    # General cleanup of potential orphaned vLLM processes by pattern
    # This is aggressive and might kill unrelated processes if patterns are too broad.
    if command -v pkill &> /dev/null; then
      # pkill -f "vllm.entrypoints.openai.api_server.*--port $VLLM_PORT" || true
      log "INFO: Attempting to find and kill orphaned vLLM processes listening on port $VLLM_PORT..."
      # Find PID using the specific port and kill it. This is safer.
      local orphaned_vllm_pid
      orphaned_vllm_pid=$(netstat -tulnp 2>/dev/null | grep ":$VLLM_PORT" | awk '{print $7}' | cut -d'/' -f1 | grep -o '[0-9]*' || true)
      if [ -n "$orphaned_vllm_pid" ]; then
        log "INFO: Found potential orphaned vLLM process $orphaned_vllm_pid on port $VLLM_PORT. Attempting to kill..."
        kill "$orphaned_vllm_pid" 2>/dev/null || log "WARN: Failed to kill process $orphaned_vllm_pid on port $VLLM_PORT."
      fi
    fi

    log "INFO: Cleanup attempt complete. Exiting."
    # exit 0 # Exit cleanly after trap - this is handled by trap itself
}

main() {
    # Initialize log file (primary DEPLOYMENT_LOG)
    # The log function itself appends, so clear or header the log for a new run.
    echo "--- Log for deployment started at $(date '+%Y-%m-%d %H:%M:%S') --- Script PID: $$ ---" > "$DEPLOYMENT_LOG"

    # Ensure cleanup runs on script exit or interruption
    trap 'cleanup' EXIT INT TERM

    log "INFO: Script PID: $$"

    load_config # Loads config and potentially updates LOG_DIR related vars
    check_config_vars # Checks variables loaded by load_config

    # Global vars like LOG_DIR might have been updated by load_config.
    # Re-initialize log file paths here IF they are not re-set within load_config
    # Current load_config handles this.

    pre_flight_checks
    setup_venv
    install_dependencies # Installs into venv

    if ! start_vllm; then # If start_vllm returns non-zero
        log "ERROR: vLLM server failed to start properly. Aborting further steps."
        # cleanup trap will handle stopping any partial services
        exit 1
    fi

    if ! start_ngrok; then # If start_ngrok returns non-zero
        log "ERROR: Ngrok tunnel failed to start properly."
        # cleanup trap will handle stopping vLLM etc.
        # Decide if script should exit or continue without ngrok
        # For now, we exit as ngrok is critical for external access.
        exit 1
    fi

    start_keep_alive_service # Starts in background if enabled

    log "INFO: --- Setup Summary ---"
    if [ -n "$VLLM_PID" ]; then
        log "INFO: vLLM Server PID: $VLLM_PID (Logs: $VLLM_LOG)"
    fi
    if [ -n "$NGROK_PID" ]; then
        log "INFO: Ngrok Tunnel PID: $NGROK_PID (Logs: $NGROK_LOG)"
        if [ -n "$FINAL_NGROK_URL" ]; then
            log "INFO: Ngrok External URL (OpenAI-compatible): $FINAL_NGROK_URL"
            log "INFO: Example Chat Completions API call (also echoed to console):"

            local example_model_id="$MODEL_ID_CPU"
            if [ "$USE_TEST_MODEL" = "false" ]; then example_model_id="$MODEL_ID_GPU"; fi
            # This heredoc goes to the log file
            cat <<EOCURL_EXAMPLE_LOG >> "$DEPLOYMENT_LOG"

curl "$FINAL_NGROK_URL" \\
  -H "Content-Type: application/json" \\
  -d '{
        "model": "$example_model_id",
        "messages": [
          {"role": "system", "content": "You are a helpful AI assistant."},
          {"role": "user", "content": "Tell me a short story."}
        ],
        "temperature": 0.7,
        "max_tokens": 100
      }'
EOCURL_EXAMPLE_LOG
            # Echo to console for user convenience (without actual newlines in commands)
            echo -e "\nINFO: Example Chat Completions API call (details in $DEPLOYMENT_LOG):"
            echo "curl \"$FINAL_NGROK_URL\" \\"
            echo "  -H \"Content-Type: application/json\" \\"
            echo "  -d '{"
            echo "        \"model\": \"$example_model_id\","
            echo "        \"messages\": ["
            echo "          {\"role\": \"system\", \"content\": \"You are a helpful AI assistant.\"},"
            echo "          {\"role\": \"user\", \"content\": \"Tell me a short story.\"} "
            echo "        ],"
            echo "        \"temperature\": 0.7,"
            echo "        \"max_tokens\": 100"
            echo "      }'\n"

        else
            log "WARNING: Ngrok URL was not captured. Cannot provide example API call."
        fi
    else
        log "INFO: Ngrok tunnel not started or PID not captured."
    fi

    if [ "$ENABLE_KEEP_ALIVE" = "true" ] && [ -n "$KEEPALIVE_PID" ]; then
        log "INFO: Keep-Alive Service PID: $KEEPALIVE_PID (Logs: $KEEPALIVE_LOG)"
    elif [ "$ENABLE_KEEP_ALIVE" = "true" ]; then
        log "WARNING: Keep-alive service was enabled but PID not captured or service failed to start."
    else
        log "INFO: Keep-alive service is disabled."
    fi

    log "INFO: All services initiated. Monitor logs in $LOG_DIR for details."
    log "INFO: To stop services manually if needed: kill $VLLM_PID $NGROK_PID $KEEPALIVE_PID (if PIDs are set and processes are running)"
    log "INFO: Alternatively, sending Ctrl+C to this script will trigger cleanup."
    log "INFO: To reactivate the venv later for manual operations, run: source $VENV_DIR/bin/activate"
    log "INFO: Script will now wait for background processes. Press Ctrl+C to stop this script and its children..."

    # Wait for all background PIDs that were successfully started
    declare -a PIDS_TO_WAIT_FOR
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then PIDS_TO_WAIT_FOR+=("$VLLM_PID"); fi
    if [ -n "$NGROK_PID" ] && kill -0 "$NGROK_PID" 2>/dev/null; then PIDS_TO_WAIT_FOR+=("$NGROK_PID"); fi
    # KEEPALIVE_PID is a background daemon, not typically waited upon to exit by itself.
    # If KEEPALIVE_PID exits, it's usually an error or intentional stop.
    # The main script should stay alive as long as VLLM and Ngrok are meant to be running.
    # If KEEPALIVE_PID is added to wait, and it's a long-running loop, it might prevent script from exiting if other main tasks finish.
    # However, if all are daemons, `wait` without args waits for all children.
    # `wait` with specific PIDs will exit if any of *those* PIDs exit. This might be too soon.
    # A better approach for daemons is to just sleep or read indefinitely, relying on trap for cleanup.
    # Or, wait for primary service like vLLM. If vLLM dies, then script can exit.

    if [ ${#PIDS_TO_WAIT_FOR[@]} -gt 0 ]; then
        # Wait for any of the primary background jobs to exit.
        # If one exits (e.g. vLLM crashes), the script will proceed past wait and then exit (triggering cleanup).
        wait -n "${PIDS_TO_WAIT_FOR[@]}"
        log "INFO: A monitored background process has exited. Initiating cleanup..."
    else
        log "INFO: No primary background processes to wait for explicitly (e.g. vLLM or Ngrok failed to start). Script will exit."
    fi
    # Cleanup is handled by the trap on EXIT
}


# --- Script Execution ---
# Call main, passing all script arguments to it
main "$@"
