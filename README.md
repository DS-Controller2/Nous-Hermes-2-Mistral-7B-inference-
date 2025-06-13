# Local LLM Server Deployment with vLLM and Ngrok

This project provides a shell script (`deploy_llm_server.sh`) to automate the setup and deployment of a Large Language Model (LLM) serving endpoint using [vLLM](https://github.com/vllm-project/vllm) and exposing it publicly via an [Ngrok](https://ngrok.com/) tunnel. It includes features for easy configuration, selection between GPU and CPU test models, local model caching, and an optional keep-alive service to prevent the model from idling.

## Features

*   **Automated Setup**: Single script to set up the environment, install dependencies, and launch services.
*   **vLLM Integration**: Leverages vLLM for high-throughput LLM serving.
*   **Ngrok Tunneling**: Exposes your local LLM server to the internet via Ngrok for easy access and testing.
*   **Configurable Models**: Easily switch between a full GPU-powered model and a lightweight CPU model for testing via configuration.
*   **Local Model Caching**: Hugging Face models are downloaded and cached within the project's `./models` directory, controlled by setting `HF_HOME`.
*   **Keep-Alive Service**: Optional service to periodically send prompts to the LLM to keep it "warm" and prevent idling, especially useful for environments that might suspend inactive processes.
*   **Structured Logging**: Outputs and errors from different components are logged into separate files in the `./logs` directory.
*   **Modular Script**: The main deployment script is organized into functions for better readability and maintainability.

## Prerequisites

Before running the deployment script, ensure you have the following installed on your system:

*   **`git`**: For cloning the repository.
*   **`bash`**: The script is written in bash.
*   **`python3` (version 3.8 or higher)**: Required by vLLM.
*   **`pip3` and `python3-venv`**: For managing Python dependencies. Often installed via `python3-pip` and `python3-venv` packages on Debian/Ubuntu.
*   **`curl`, `wget`, `unzip`**: For downloading and installing Ngrok and potentially other dependencies.
*   **Build Tools**: `build-essential` (or equivalent for your distribution) might be needed for some Python package installations.
*   **(For GPU Mode)**:
    *   **NVIDIA GPU**: With compatible drivers installed.
    *   **CUDA Toolkit**: Version compatible with your drivers and vLLM. vLLM often requires CUDA 11.8 or 12.1. Check vLLM documentation for specific version compatibility.
*   **(Optional, for system-wide Ngrok installation by script)**: `sudo` access, if Ngrok is not already installed and you want the script to attempt a system-wide installation to `/usr/local/bin`.

## Directory Structure

The project is organized as follows:

```
.
├── deploy_llm_server.sh     # Main deployment script
├── config/
│   └── config.sh.example    # Example configuration file
├── logs/
│   ├── .gitkeep             # Ensures the directory is tracked by git
│   ├── deployment.log       # Log for the main deployment script
│   ├── vllm_server.log      # Log for the vLLM server
│   ├── ngrok_tunnel.log     # Log for the Ngrok tunnel
│   └── keep_alive.log       # Log for the keep-alive service
├── models/
│   └── .gitkeep             # Models downloaded by Hugging Face/vLLM will be stored here
└── vllm_env/                # Python virtual environment (created by the script)
    └── ...
```

## Local Model Caching (`HF_HOME`)

The `deploy_llm_server.sh` script sets the `HF_HOME` (and `HUGGINGFACE_HUB_CACHE`) environment variable to `./models` (or the path configured in `MODEL_CACHE_DIR`). This directs the Hugging Face libraries (used by vLLM) to download and cache models within this specified directory. This makes the project more portable and helps manage model storage specific to this setup.

## Configuration

Configuration is managed via a shell script located at `config/config.sh`. You must create this file by copying the example:

```bash
cp config/config.sh.example config/config.sh
```

Then, **edit `config/config.sh`** to set your desired parameters. **Do not commit your `config/config.sh` file if it contains sensitive information like your `NGROK_AUTHTOKEN`.**

Key configuration variables in `config/config.sh.example` (and thus in your `config/config.sh`):

### Model Configuration
*   `USE_TEST_MODEL`: Set to `"true"` to use `MODEL_ID_CPU` and `VLLM_EXTRA_ARGS_CPU`. Set to `"false"` (or any other string) for GPU mode using `MODEL_ID_GPU`.
*   `MODEL_ID_GPU`: Hugging Face identifier for the GPU model (e.g., `"NousResearch/Nous-Hermes-2-SOLAR-10.7B"`).
*   `MODEL_ID_CPU`: Hugging Face identifier for the smaller CPU test model (e.g., `"gpt2"`).
*   `VLLM_EXTRA_ARGS_GPU`: Additional arguments for vLLM in GPU mode (e.g., `"--tensor-parallel-size 1 --gpu-memory-utilization 0.90"`). For AWQ models, include `--quantization awq`.
*   `VLLM_EXTRA_ARGS_CPU`: Additional arguments for vLLM in CPU mode (e.g., `"--device cpu --max-model-len 1024"`).

### vLLM Server Configuration
*   `VLLM_HOST`: Host IP for the vLLM server (default: `"0.0.0.0"`).
*   `VLLM_PORT`: Port for the vLLM server (default: `"8000"`).

### Ngrok Configuration
*   `NGROK_AUTHTOKEN`: Your Ngrok authentication token. **Required.**
*   `NGROK_TUNNEL_NAME`: Descriptive name for your Ngrok tunnel (default: `"vllm-server-tunnel"`). Used for logging and reference.

### Directory Configuration
*   `VENV_DIR`: Path to the Python virtual environment (default: `"./vllm_env"`).
*   `LOG_DIR`: Path to the directory for log files (default: `"./logs"`).
*   `MODEL_CACHE_DIR`: Path for Hugging Face model cache (default: `"./models"`). Script sets `HF_HOME` and `HUGGINGFACE_HUB_CACHE` to this.

### Keep-Alive Service
*   `ENABLE_KEEP_ALIVE`: Set to `"true"` to enable, `"false"` to disable (default: `"false"`).
*   `IDLE_THRESHOLD_MINUTES`: Initial delay (minutes) before the first keep-alive prompt (default: `"60"`).
*   `KEEPALIVE_PROMPT_INTERVAL_MINUTES`: Interval (minutes) between subsequent keep-alive prompts (default: `"5"`).
*   `KEEPALIVE_PROMPTS`: Bash array of sample prompts to cycle through (e.g., `("What is AI?" "Tell me a joke.")`).
*   `KEEPALIVE_MODEL_ID_OVERRIDE`: Optional. Specify a model ID for keep-alive requests, otherwise uses the currently active model.
*   `KEEPALIVE_TARGET_ENDPOINT`: The API endpoint for sending keep-alive prompts (default uses configured `VLLM_HOST` and `VLLM_PORT`).

## How to Run

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Set up Configuration:**
    Copy the example configuration file and customize it with your settings (especially `NGROK_AUTHTOKEN` and desired models).
    ```bash
    cp config/config.sh.example config/config.sh
    nano config/config.sh  # Or use your preferred editor
    ```

3.  **Run the Deployment Script:**
    Execute the script from the project root directory:
    ```bash
    ./deploy_llm_server.sh
    ```
    The script will perform pre-flight checks, set up the Python virtual environment, install dependencies, and start the vLLM server and Ngrok tunnel.

## Expected Output

If successful, the script will output:
*   Logs of its progress to the console and to `logs/deployment.log`.
*   The public Ngrok URL for accessing your LLM server (e.g., `https://<random_string>.ngrok.io/v1`).
*   An example `curl` command to test the OpenAI-compatible chat completions endpoint.
*   PIDs of the started services (vLLM, Ngrok, Keep-Alive).

Example API call (will be shown in script output):
```bash
curl "YOUR_NGROK_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
        "model": "your_chosen_model_id",
        "messages": [
          {"role": "system", "content": "You are a helpful AI assistant."},
          {"role": "user", "content": "Tell me a short story."}
        ],
        "temperature": 0.7,
        "max_tokens": 100
      }'
```

## Monitoring Logs

All logs are stored in the `./logs/` directory:
*   `deployment.log`: Main log for the `deploy_llm_server.sh` script.
*   `vllm_server.log`: Output from the vLLM server. Check here for vLLM errors.
*   `ngrok_tunnel.log`: Output from the Ngrok tunnel service. Check here for Ngrok connection issues.
*   `keep_alive.log`: Logs from the keep-alive service, if enabled.

## Stopping Services

1.  **Press `Ctrl+C`** in the terminal where `deploy_llm_server.sh` is running. The script has a trap to attempt to shut down the background services (vLLM, Ngrok, Keep-Alive).
2.  If `Ctrl+C` doesn't stop all processes, you may need to **manually kill them** using their PIDs. The PIDs are logged in `logs/deployment.log` and also printed to the console when the services start.
    ```bash
    kill <VLLM_PID> <NGROK_PID> <KEEPALIVE_PID>
    ```
    You can find these PIDs in the `logs/deployment.log` or in the initial output of the script. The `cleanup` function in the script also attempts to find and kill processes by name or port if PIDs are not available or direct kill fails.

## Troubleshooting

*   **`nvidia-smi` command not found**: Ensure NVIDIA drivers are correctly installed and in your system's PATH. This is crucial for GPU mode.
*   **Python version errors**: Make sure `python3` points to version 3.8 or higher.
*   **vLLM installation issues**:
    *   Check `logs/deployment.log` for pip installation errors.
    *   You might need a specific CUDA version of vLLM (e.g., `pip install vllm[cu121]`). The script installs the generic `vllm`. If this fails, you might need to adjust the `install_dependencies` function in `deploy_llm_server.sh` or install vLLM manually in the activated venv (`source vllm_env/bin/activate; pip install vllm[cuXXX]`) before running the script's dependency install again.
*   **Model download issues**: Ensure you have internet connectivity. Large models can take a significant time to download. Check `logs/vllm_server.log` for download progress or errors. The script configures models to be cached in the `./models` directory (or as specified in `MODEL_CACHE_DIR`).
*   **Ngrok errors**:
    *   Invalid authtoken: Double-check `NGROK_AUTHTOKEN` in `config/config.sh`.
    *   Tunnel connection issues: Check `logs/ngrok_tunnel.log`. Ensure your firewall isn't blocking Ngrok.
*   **"Address already in use" for vLLM port**: Ensure no other service is using the `VLLM_PORT` (default 8000). The cleanup function attempts to find and kill processes on this port if they are orphaned.
*   **Permission errors**: Ensure `deploy_llm_server.sh` is executable (`chmod +x deploy_llm_server.sh`). If ngrok auto-installation is used, `sudo` access might be prompted for moving ngrok to `/usr/local/bin`.
EOF
