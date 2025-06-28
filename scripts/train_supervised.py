#!/usr/bin/env python3
"""
Supervised Fine-tuning Script for DeepSeek R1 Verilog Generation
Complete training pipeline with configuration management.
"""

import sys
import os
import logging
import argparse
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
import wandb
from transformers import set_seed

from models.model_setup import DeepSeekVerilogModel
from training.supervised_trainer import (
    VerilogTrainingArguments, 
    train_verilog_model, 
    load_verilog_dataset
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train DeepSeek R1 for Verilog generation")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, 
                       default="deepseek-ai/deepseek-r1-distill-qwen-32b",
                       help="Base model name or path")
    parser.add_argument("--use_lora", action="store_true", default=True,
                       help="Use LoRA for efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=64,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data/processed",
                       help="Directory containing processed datasets")
    parser.add_argument("--max_sequence_length", type=int, default=2048,
                       help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./checkpoints/deepseek-verilog-sft",
                       help="Output directory for model checkpoints")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                       help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                       help="Evaluation batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Warmup ratio")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Maximum gradient norm")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Evaluation arguments
    parser.add_argument("--eval_strategy", type=str, default="steps",
                       choices=["steps", "epoch", "no"],
                       help="Evaluation strategy")
    parser.add_argument("--eval_steps_ratio", type=float, default=0.1,
                       help="Evaluation steps as ratio of total steps")
    parser.add_argument("--save_steps_ratio", type=float, default=0.1,
                       help="Save steps as ratio of total steps")
    parser.add_argument("--load_best_model_at_end", action="store_true", default=True,
                       help="Load best model at end of training")
    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss",
                       help="Metric for best model selection")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum new tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p for generation")
    
    # Logging arguments
    parser.add_argument("--wandb_project", type=str, default="deepseek-verilog-finetune",
                       help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="Wandb run name")
    parser.add_argument("--logging_steps", type=int, default=50,
                       help="Logging steps")
    parser.add_argument("--report_to", type=str, default="wandb",
                       help="Reporting platform")
    
    # DeepSpeed arguments
    parser.add_argument("--deepspeed", type=str, default=None,
                       help="DeepSpeed config file path")
    parser.add_argument("--bf16", action="store_true", default=False,
                       help="Use bf16 precision")
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Use fp16 precision")
    
    # Resume training
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    return parser.parse_args()

def create_lora_config(args):
    """Create LoRA configuration from arguments"""
    return {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    }

def create_training_arguments(args):
    """Create training arguments from parsed args"""
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = VerilogTrainingArguments(
        # Basic training settings
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        # Optimization settings
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        
        # Evaluation settings
        eval_strategy=args.eval_strategy,
        eval_steps_ratio=args.eval_steps_ratio,
        save_steps_ratio=args.save_steps_ratio,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=False if "loss" in args.metric_for_best_model else True,
        
        # Logging settings
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        
        # Precision settings
        bf16=args.bf16,
        fp16=args.fp16,
        dataloader_pin_memory=False,
        
        # DeepSpeed
        deepspeed=args.deepspeed,
        
        # Save settings
        save_total_limit=3,
        save_strategy="steps",
        
        # Verilog-specific settings
        max_sequence_length=args.max_sequence_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        
        # Memory optimization
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    
    return training_args

def save_training_config(args, output_dir: str):
    """Save training configuration"""
    config = vars(args)
    config_path = Path(output_dir) / "training_config.json"
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
        
    logger.info(f"Training configuration saved to {config_path}")

def main():
    """Main training function"""
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    logger.info("=" * 80)
    logger.info("DEEPSEEK R1 VERILOG FINE-TUNING")
    logger.info("=" * 80)
    
    # Save configuration
    save_training_config(args, args.output_dir)
    
    # Load datasets
    logger.info("Loading datasets...")
    try:
        train_dataset, val_dataset, test_dataset = load_verilog_dataset(args.data_dir)
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        logger.error("Please run the data preprocessing pipeline first:")
        logger.error("python scripts/download_and_preprocess.py")
        sys.exit(1)
    
    # Setup model
    logger.info(f"Setting up model: {args.model_name}")
    
    lora_config = create_lora_config(args) if args.use_lora else None
    
    model_wrapper = DeepSeekVerilogModel(
        model_name=args.model_name,
        use_lora=args.use_lora,
        lora_config=lora_config
    )
    
    model, tokenizer = model_wrapper.setup_for_training()
    
    # Save model configuration
    model_wrapper.save_model_config(args.output_dir)
    
    # Print model info
    model_info = model_wrapper.get_model_info()
    logger.info("Model Information:")
    for key, value in model_info.items():
        if key != 'model_config':  # Skip large config dict
            logger.info(f"  {key}: {value}")
    
    # Create training arguments
    training_args = create_training_arguments(args)
    
    # Start training
    logger.info("Starting supervised fine-tuning...")
    
    trainer = train_verilog_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_args=training_args,
        output_dir=args.output_dir
    )
    
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info(f"Model saved to: {args.output_dir}")
    
    # Cleanup
    if wandb.run is not None:
        wandb.finish()
    
    return trainer

if __name__ == "__main__":
    main()