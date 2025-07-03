#!/bin/bash

# data_creation.sh - Script to manage CT report analysis with different LLM providers

# Default values
CSV_PATH="/data2/CT-RATE/reports/train/train_reports.csv"
LLM_PROVIDER="vllm"
VLLM_METHOD="http"
VLLM_MODEL="meta-llama/Llama-2-7b-chat-hf"
VLLM_SERVER_URL="http://localhost:8000/v1/completions"
OUTPUT_FILE="/home/compu/WHY/ct-chat/results/ct_report_analysis_results.json"

# Function to display usage information
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help                      Show this help message"
    echo "  -p, --provider PROVIDER         LLM provider (openai, huggingface, vllm) [default: vllm]"
    echo "  -c, --csv PATH                  Path to CSV file [default: $CSV_PATH]"
    echo "  -m, --model MODEL               Model name [default: $VLLM_MODEL]"
    echo "  -v, --vllm-method METHOD        vLLM method (http, direct) [default: http]"
    echo "  -u, --vllm-url URL              vLLM server URL [default: $VLLM_SERVER_URL]"
    echo "  -o, --output FILE               Output JSON file [default: $OUTPUT_FILE]"
    echo "  -n, --num-samples COUNT         Number of samples to process [default: $NUM_SAMPLES]"
    echo "  -s, --start-server              Start vLLM server before processing"
    echo
    echo "Environment variables (can be set instead of using command line options):"
    echo "  OPENAI_API_KEY                  Required for OpenAI provider"
    echo "  HF_TOKEN                        Required for Hugging Face provider"
    echo "  VLLM_MODEL                      Model name for vLLM"
    echo "  VLLM_METHOD                     vLLM method (http or direct)"
    echo "  VLLM_SERVER_URL                 URL for vLLM HTTP server"
    echo
    echo "Examples:"
    echo "  $0 --provider vllm --vllm-method direct --model meta-llama/Llama-2-7b-chat-hf"
    echo "  $0 --provider openai --num-samples 10"
    echo "  $0 --start-server --provider vllm"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        -p|--provider)
            LLM_PROVIDER="$2"
            shift 2
            ;;
        -c|--csv)
            CSV_PATH="$2"
            shift 2
            ;;
        -m|--model)
            VLLM_MODEL="$2"
            shift 2
            ;;
        -v|--vllm-method)
            VLLM_METHOD="$2"
            shift 2
            ;;
        -u|--vllm-url)
            VLLM_SERVER_URL="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -n|--num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        -s|--start-server)
            START_SERVER=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if provider is valid
if [[ "$LLM_PROVIDER" != "openai" && "$LLM_PROVIDER" != "huggingface" && "$LLM_PROVIDER" != "vllm" ]]; then
    echo "Error: Invalid provider '$LLM_PROVIDER'. Must be one of: openai, huggingface, vllm"
    exit 1
fi

# Check if vLLM method is valid
if [[ "$LLM_PROVIDER" == "vllm" && "$VLLM_METHOD" != "http" && "$VLLM_METHOD" != "direct" ]]; then
    echo "Error: Invalid vLLM method '$VLLM_METHOD'. Must be one of: http, direct"
    exit 1
fi

# Check for required environment variables
if [[ "$LLM_PROVIDER" == "openai" && -z "$OPENAI_API_KEY" ]]; then
    echo "Warning: OPENAI_API_KEY environment variable is not set, which is required for OpenAI provider"
    echo "You can set it with: export OPENAI_API_KEY=your_api_key"
fi

if [[ "$LLM_PROVIDER" == "huggingface" && -z "$HF_TOKEN" ]]; then
    echo "Warning: HF_TOKEN environment variable is not set, which is required for Hugging Face provider"
    echo "You can set it with: export HF_TOKEN=your_token"
fi

# Start vLLM server if requested
if [[ -n "$START_SERVER" && "$LLM_PROVIDER" == "vllm" && "$VLLM_METHOD" == "http" ]]; then
    echo "Starting vLLM server with model $VLLM_MODEL..."
    python -m vllm.entrypoints.openai.api_server \
        --model "$VLLM_MODEL" \
        --port 8000 &
    
    SERVER_PID=$!
    echo "vLLM server started with PID $SERVER_PID"
    echo "Waiting 30 seconds for server to initialize..."
    sleep 30
fi

# Export environment variables for the Python script
export LLM_PROVIDER="$LLM_PROVIDER"
export VLLM_MODEL="$VLLM_MODEL"
export VLLM_METHOD="$VLLM_METHOD"
export VLLM_SERVER_URL="$VLLM_SERVER_URL"

# Create a temporary Python script to override parameters
TMP_SCRIPT=$(mktemp)
cat > "$TMP_SCRIPT" <<EOF
import os
import sys
# Add the current directory to the path
sys.path.append('/home/compu/WHY')
from ct_chat.Dataset.data_creator import process_csv_and_generate_json

# Override parameters
csv_file_path = "$CSV_PATH"

# Call the main function
if __name__ == "__main__":
    process_csv_and_generate_json()
EOF

# Run the Python script
echo "Running CT report analysis with provider: $LLM_PROVIDER"
echo "CSV file: $CSV_PATH"
if [[ "$LLM_PROVIDER" == "vllm" ]]; then
    echo "vLLM method: $VLLM_METHOD"
    echo "vLLM model: $VLLM_MODEL"
    if [[ "$VLLM_METHOD" == "http" ]]; then
        echo "vLLM server URL: $VLLM_SERVER_URL"
    fi
fi
echo "Number of samples: $NUM_SAMPLES"
echo "Output file: $OUTPUT_FILE"
echo

python "$TMP_SCRIPT"
rm "$TMP_SCRIPT"

# Terminate vLLM server if we started it
if [[ -n "$START_SERVER" && -n "$SERVER_PID" ]]; then
    echo "Terminating vLLM server (PID $SERVER_PID)..."
    kill $SERVER_PID
fi

echo "Analysis complete. Results saved to $OUTPUT_FILE"