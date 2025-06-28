#!/usr/bin/env python3
"""
PPO Training Script for DeepSeek R1 Verilog Generation
Reinforcement learning fine-tuning after supervised training.
"""

import sys
import logging
import argparse
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
import wandb
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM

from models.model_setup import DeepSeekVerilogModel
from training.ppo_trainer import VerilogPPOTrainer, VerilogPPOConfig, create_ppo_trainer
from training.reward_model import VerilogRewardModel, VerilogRewardConfig
from training.supervised_trainer import load_verilog_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="PPO training for DeepSeek R1 Verilog generation")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to supervised fine-tuned model")
    parser.add_argument("--base_model_name", type=str, 
                       default="deepseek-ai/deepseek-r1-distill-qwen-32b",
                       help="Base model name for reference model")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data/processed",
                       help="Directory containing processed datasets")
    parser.add_argument("--max_sequence_length", type=int, default=2048,
                       help="Maximum sequence length")
    
    # PPO Training arguments
    parser.add_argument("--output_dir", type=str, default="./checkpoints/deepseek-verilog-ppo",
                       help="Output directory for PPO checkpoints")
    parser.add_argument("--num_ppo_steps", type=int, default=1000,
                       help="Number of PPO training steps")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="PPO batch size")
    parser.add_argument("--mini_batch_size", type=int, default=4,
                       help="PPO mini-batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1.41e-5,
                       help="PPO learning rate")
    parser.add_argument("--ppo_epochs", type=int, default=4,
                       help="Number of PPO epochs per step")
    parser.add_argument("--target_kl", type=float, default=0.1,
                       help="Target KL divergence")
    parser.add_argument("--cliprange", type=float, default=0.2,
                       help="PPO clip range")
    parser.add_argument("--cliprange_value", type=float, default=0.2,
                       help="Value function clip range")
    parser.add_argument("--vf_coef", type=float, default=0.1,
                       help="Value function coefficient")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum new tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p for generation")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k for generation")
    
    # Reward model arguments
    parser.add_argument("--syntax_weight", type=float, default=0.3,
                       help="Weight for syntax correctness in reward")
    parser.add_argument("--functional_weight", type=float, default=0.4,
                       help="Weight for functional correctness in reward")
    parser.add_argument("--quality_weight", type=float, default=0.2,
                       help="Weight for code quality in reward")
    parser.add_argument("--adherence_weight", type=float, default=0.1,
                       help="Weight for specification adherence in reward")
    parser.add_argument("--reward_scale", type=float, default=1.0,
                       help="Reward scaling factor")
    parser.add_argument("--reward_clip", type=float, default=5.0,
                       help="Reward clipping value")
    
    # Evaluation arguments
    parser.add_argument("--eval_frequency", type=int, default=100,
                       help="Evaluation frequency (steps)")
    parser.add_argument("--eval_samples", type=int, default=50,
                       help="Number of samples for evaluation")
    parser.add_argument("--save_freq", type=int, default=250,
                       help="Checkpoint save frequency (steps)")
    
    # Logging arguments
    parser.add_argument("--wandb_project", type=str, default="deepseek-verilog-ppo",
                       help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="Wandb run name")
    parser.add_argument("--log_with_step", type=int, default=10,
                       help="Logging frequency (steps)")
    
    return parser.parse_args()

def create_reward_config(args) -> VerilogRewardConfig:
    """Create reward model configuration"""
    return VerilogRewardConfig(
        syntax_weight=args.syntax_weight,
        functional_weight=args.functional_weight,
        quality_weight=args.quality_weight,
        adherence_weight=args.adherence_weight,
        use_iverilog=True,
        use_verilator=False  # Can be enabled if available
    )

def create_ppo_config(args) -> VerilogPPOConfig:
    """Create PPO configuration"""
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ppo_config = VerilogPPOConfig(
        # Model settings
        model_name=args.model_path,
        
        # Training settings
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        ppo_epochs=args.ppo_epochs,
        
        # PPO-specific settings
        target_kl=args.target_kl,
        cliprange=args.cliprange,
        cliprange_value=args.cliprange_value,
        vf_coef=args.vf_coef,
        
        # Generation settings
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=True,
        
        # Reward settings
        reward_scale=args.reward_scale,
        reward_clip=args.reward_clip,
        reward_model_config={
            'syntax_weight': args.syntax_weight,
            'functional_weight': args.functional_weight,
            'quality_weight': args.quality_weight,
            'adherence_weight': args.adherence_weight
        },
        
        # Logging and evaluation
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        log_with_step=args.log_with_step,
        eval_frequency=args.eval_frequency,
        eval_samples=args.eval_samples,
        save_freq=args.save_freq,
        
        # Sequence settings
        max_sequence_length=args.max_sequence_length,
        
        # Optimization settings
        optimize_cuda_cache=True,
        early_stopping=True,
        use_score_scaling=True,
        use_score_norm=True,
        score_clip=0.5
    )
    
    return ppo_config

def load_models(args):
    """Load policy and reference models"""
    
    logger.info(f"Loading policy model from {args.model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load policy model (fine-tuned)
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    logger.info(f"Loading reference model from {args.base_model_name}")
    
    # Load reference model (base model)
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Resize embeddings if needed
    if len(tokenizer) != policy_model.config.vocab_size:
        policy_model.resize_token_embeddings(len(tokenizer))
        ref_model.resize_token_embeddings(len(tokenizer))
        
    logger.info("Models loaded successfully")
    logger.info(f"Policy model parameters: {policy_model.num_parameters():,}")
    logger.info(f"Reference model parameters: {ref_model.num_parameters():,}")
    
    return policy_model, ref_model, tokenizer

def save_training_config(args, output_dir: str):
    """Save training configuration"""
    config = vars(args)
    config_path = Path(output_dir) / "ppo_training_config.json"
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
        
    logger.info(f"PPO training configuration saved to {config_path}")

def main():
    """Main PPO training function"""
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    logger.info("=" * 80)
    logger.info("DEEPSEEK R1 VERILOG PPO TRAINING")
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
    
    # Use training dataset for PPO (it learns from the data distribution)
    ppo_dataset = train_dataset
    eval_dataset = val_dataset
    
    logger.info(f"PPO dataset: {len(ppo_dataset)} examples")
    logger.info(f"Evaluation dataset: {len(eval_dataset)} examples")
    
    # Load models
    policy_model, ref_model, tokenizer = load_models(args)
    
    # Create reward model
    logger.info("Setting up reward model...")
    reward_config = create_reward_config(args)
    reward_model = VerilogRewardModel(reward_config)
    
    # Create PPO configuration
    ppo_config = create_ppo_config(args)
    
    # Create PPO trainer
    logger.info("Setting up PPO trainer...")
    ppo_trainer = create_ppo_trainer(
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=ppo_dataset,
        ppo_config=ppo_config,
        reward_model=reward_model
    )
    
    # Test reward model
    logger.info("Testing reward model...")
    test_code = '''module test_counter(
    input clk,
    input reset,
    output reg [3:0] count
);
always @(posedge clk) begin
    if (reset)
        count <= 0;
    else
        count <= count + 1;
end
endmodule'''
    test_spec = "Create a 4-bit counter"
    test_reward = reward_model.compute_reward(test_code, test_spec)
    logger.info(f"Test reward: {test_reward['total_reward']:.3f}")
    
    # Start PPO training
    logger.info(f"Starting PPO training for {args.num_ppo_steps} steps...")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Target KL: {args.target_kl}")
    
    training_summary = ppo_trainer.train(
        num_steps=args.num_ppo_steps,
        eval_dataset=eval_dataset
    )
    
    # Save final model
    logger.info(f"Saving final PPO model to {args.output_dir}")
    ppo_trainer.save_checkpoint("final", args.output_dir)
    
    # Save training summary
    summary_path = Path(args.output_dir) / "ppo_training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(training_summary, f, indent=2)
        
    logger.info("=" * 80)
    logger.info("PPO TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info(f"Final model saved to: {args.output_dir}")
    logger.info(f"Final evaluation reward: {training_summary['final_eval_stats']['eval_mean_reward']:.3f}")
    
    # Cleanup
    if wandb.run is not None:
        wandb.finish()
    
    return ppo_trainer

if __name__ == "__main__":
    main()