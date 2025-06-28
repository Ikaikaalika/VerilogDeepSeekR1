#!/bin/bash

# Thunder Compute Optimized Training Script for DeepSeek R1 Verilog Fine-tuning
# This script handles both supervised fine-tuning and PPO training phases

set -e

# Default configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data/processed"
OUTPUT_BASE="$PROJECT_DIR/checkpoints"
DEEPSPEED_CONFIG="$PROJECT_DIR/configs/deepspeed_config.json"
LOG_DIR="$PROJECT_DIR/logs"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_BASE"

# Parse command line arguments
PHASE="both"  # sft, ppo, or both
MODEL_NAME="deepseek-ai/deepseek-r1-distill-qwen-32b"
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
WANDB_PROJECT="deepseek-verilog-thunder"
RUN_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-base)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --deepspeed-config)
            DEEPSPEED_CONFIG="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --phase              Training phase: sft, ppo, or both (default: both)"
            echo "  --model-name         Base model name (default: deepseek-ai/deepseek-r1-distill-qwen-32b)"
            echo "  --num-gpus           Number of GPUs to use (default: auto-detect)"
            echo "  --wandb-project      Wandb project name"
            echo "  --run-name           Wandb run name"
            echo "  --data-dir           Data directory (default: data/processed)"
            echo "  --output-base        Base output directory (default: checkpoints)"
            echo "  --deepspeed-config   DeepSpeed config file"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set run name if not provided
if [[ -z "$RUN_NAME" ]]; then
    RUN_NAME="thunder-$(date +%Y%m%d-%H%M%S)"
fi

echo "=============================================="
echo "DeepSeek R1 Verilog Fine-tuning on Thunder"
echo "=============================================="
echo "Phase: $PHASE"
echo "Model: $MODEL_NAME"
echo "GPUs: $NUM_GPUS"
echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_BASE"
echo "DeepSpeed Config: $DEEPSPEED_CONFIG"
echo "Wandb Project: $WANDB_PROJECT"
echo "Run Name: $RUN_NAME"
echo "=============================================="

# Check prerequisites
echo "Checking prerequisites..."

if ! command -v accelerate &> /dev/null; then
    echo "Error: accelerate not found. Please install with: pip install accelerate"
    exit 1
fi

if ! command -v deepspeed &> /dev/null; then
    echo "Error: deepspeed not found. Please install with: pip install deepspeed"
    exit 1
fi

if [[ ! -f "$DEEPSPEED_CONFIG" ]]; then
    echo "Error: DeepSpeed config not found: $DEEPSPEED_CONFIG"
    exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
    echo "Error: Data directory not found: $DATA_DIR"
    echo "Please run: python scripts/download_and_preprocess.py"
    exit 1
fi

# Check for required data files
for file in train.json val.json test.json; do
    if [[ ! -f "$DATA_DIR/$file" ]]; then
        echo "Error: Required data file not found: $DATA_DIR/$file"
        echo "Please run the data preprocessing pipeline first"
        exit 1
    fi
done

echo "Prerequisites check passed!"

# Function to run supervised fine-tuning
run_sft() {
    echo "Starting Supervised Fine-tuning Phase..."
    
    SFT_OUTPUT="$OUTPUT_BASE/deepseek-verilog-sft"
    SFT_LOG="$LOG_DIR/sft_$(date +%Y%m%d_%H%M%S).log"
    
    SFT_RUN_NAME="${RUN_NAME}-sft"
    
    echo "SFT Output: $SFT_OUTPUT"
    echo "SFT Log: $SFT_LOG"
    
    # DeepSpeed distributed training command
    deepspeed --num_gpus=$NUM_GPUS "$PROJECT_DIR/scripts/train_supervised.py" \
        --model_name "$MODEL_NAME" \
        --use_lora \
        --lora_r 64 \
        --lora_alpha 32 \
        --lora_dropout 0.1 \
        --data_dir "$DATA_DIR" \
        --output_dir "$SFT_OUTPUT" \
        --max_sequence_length 2048 \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --learning_rate 2e-5 \
        --weight_decay 0.01 \
        --warmup_ratio 0.1 \
        --max_grad_norm 1.0 \
        --eval_strategy steps \
        --eval_steps_ratio 0.1 \
        --save_steps_ratio 0.1 \
        --load_best_model_at_end \
        --metric_for_best_model eval_loss \
        --logging_steps 50 \
        --deepspeed "$DEEPSPEED_CONFIG" \
        --fp16 \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_run_name "$SFT_RUN_NAME" \
        --seed 42 \
        2>&1 | tee "$SFT_LOG"
    
    if [[ $? -eq 0 ]]; then
        echo "Supervised fine-tuning completed successfully!"
        echo "Model saved to: $SFT_OUTPUT"
        echo "Log saved to: $SFT_LOG"
    else
        echo "Supervised fine-tuning failed!"
        exit 1
    fi
}

# Function to run PPO training
run_ppo() {
    echo "Starting PPO Training Phase..."
    
    # Check for SFT model
    SFT_OUTPUT="$OUTPUT_BASE/deepseek-verilog-sft"
    if [[ ! -d "$SFT_OUTPUT" ]]; then
        echo "Error: SFT model not found at $SFT_OUTPUT"
        echo "Please run supervised fine-tuning first or specify correct path"
        exit 1
    fi
    
    PPO_OUTPUT="$OUTPUT_BASE/deepseek-verilog-ppo"
    PPO_LOG="$LOG_DIR/ppo_$(date +%Y%m%d_%H%M%S).log"
    
    PPO_RUN_NAME="${RUN_NAME}-ppo"
    
    echo "PPO Model Path: $SFT_OUTPUT"
    echo "PPO Output: $PPO_OUTPUT"
    echo "PPO Log: $PPO_LOG"
    
    # PPO training - typically single node but can use multiple GPUs
    python "$PROJECT_DIR/scripts/train_ppo.py" \
        --model_path "$SFT_OUTPUT" \
        --base_model_name "$MODEL_NAME" \
        --data_dir "$DATA_DIR" \
        --output_dir "$PPO_OUTPUT" \
        --max_sequence_length 2048 \
        --num_ppo_steps 1000 \
        --batch_size 32 \
        --mini_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --learning_rate 1.41e-5 \
        --ppo_epochs 4 \
        --target_kl 0.1 \
        --cliprange 0.2 \
        --max_new_tokens 512 \
        --temperature 0.7 \
        --top_p 0.9 \
        --syntax_weight 0.3 \
        --functional_weight 0.4 \
        --quality_weight 0.2 \
        --adherence_weight 0.1 \
        --reward_scale 1.0 \
        --reward_clip 5.0 \
        --eval_frequency 100 \
        --eval_samples 50 \
        --save_freq 250 \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_run_name "$PPO_RUN_NAME" \
        --log_with_step 10 \
        --seed 42 \
        2>&1 | tee "$PPO_LOG"
    
    if [[ $? -eq 0 ]]; then
        echo "PPO training completed successfully!"
        echo "Model saved to: $PPO_OUTPUT"
        echo "Log saved to: $PPO_LOG"
    else
        echo "PPO training failed!"
        exit 1
    fi
}

# Function to run evaluation
run_evaluation() {
    echo "Running final evaluation..."
    
    FINAL_MODEL="$OUTPUT_BASE/deepseek-verilog-ppo"
    if [[ ! -d "$FINAL_MODEL" ]]; then
        FINAL_MODEL="$OUTPUT_BASE/deepseek-verilog-sft"
    fi
    
    if [[ ! -d "$FINAL_MODEL" ]]; then
        echo "No trained model found for evaluation"
        return 1
    fi
    
    EVAL_LOG="$LOG_DIR/evaluation_$(date +%Y%m%d_%H%M%S).log"
    
    python "$PROJECT_DIR/scripts/evaluate_model.py" \
        --model_path "$FINAL_MODEL" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_BASE/evaluation" \
        --max_sequence_length 2048 \
        --num_samples 100 \
        --temperature 0.7 \
        --top_p 0.9 \
        2>&1 | tee "$EVAL_LOG"
    
    echo "Evaluation completed! Log saved to: $EVAL_LOG"
}

# Main execution
echo "Starting training pipeline..."

case $PHASE in
    "sft")
        run_sft
        ;;
    "ppo")
        run_ppo
        ;;
    "both")
        run_sft
        echo ""
        echo "Waiting 30 seconds before starting PPO..."
        sleep 30
        run_ppo
        ;;
    *)
        echo "Invalid phase: $PHASE"
        echo "Valid phases: sft, ppo, both"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "Training pipeline completed!"
echo "=============================================="

# Run evaluation if both phases completed
if [[ "$PHASE" == "both" ]] || [[ "$PHASE" == "ppo" ]]; then
    echo ""
    run_evaluation
fi

echo "All tasks completed successfully!"
echo "Check the logs directory for detailed logs: $LOG_DIR"
echo "Check the checkpoints directory for models: $OUTPUT_BASE"