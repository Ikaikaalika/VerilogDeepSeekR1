#!/usr/bin/env python3
"""
Model Evaluation Script for DeepSeek R1 Verilog Generation
Comprehensive evaluation of trained models on test datasets.
"""

import sys
import logging
import argparse
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from evaluation.evaluator import VerilogEvaluator, EvaluationConfig
from training.supervised_trainer import load_verilog_dataset
from training.reward_model import VerilogRewardModel, VerilogRewardConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate DeepSeek R1 Verilog generation model")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                       help="Path to tokenizer (defaults to model_path)")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data/processed",
                       help="Directory containing processed datasets")
    parser.add_argument("--test_file", type=str, default="test.json",
                       help="Test file name")
    parser.add_argument("--eval_split", type=str, default="test",
                       choices=["train", "val", "test", "all"],
                       help="Which split to evaluate on")
    
    # Evaluation arguments
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                       help="Output directory for evaluation results")
    parser.add_argument("--max_sequence_length", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum new tokens for generation")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to evaluate (None for all)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Evaluation batch size")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Generation arguments
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p for generation")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k for generation")
    parser.add_argument("--do_sample", action="store_true", default=True,
                       help="Use sampling for generation")
    parser.add_argument("--num_beams", type=int, default=1,
                       help="Number of beams for beam search")
    
    # Output arguments
    parser.add_argument("--save_predictions", action="store_true", default=True,
                       help="Save individual predictions")
    parser.add_argument("--save_detailed_results", action="store_true", default=True,
                       help="Save detailed evaluation results")
    parser.add_argument("--generate_report", action="store_true", default=True,
                       help="Generate human-readable report")
    
    # Reward model arguments
    parser.add_argument("--syntax_weight", type=float, default=0.3,
                       help="Weight for syntax correctness in reward")
    parser.add_argument("--functional_weight", type=float, default=0.4,
                       help="Weight for functional correctness in reward")
    parser.add_argument("--quality_weight", type=float, default=0.2,
                       help="Weight for code quality in reward")
    parser.add_argument("--adherence_weight", type=float, default=0.1,
                       help="Weight for specification adherence in reward")
    
    return parser.parse_args()

def load_model_and_tokenizer(model_path: str, tokenizer_path: Optional[str] = None):
    """Load model and tokenizer"""
    
    logger.info(f"Loading model from {model_path}")
    
    # Load tokenizer
    tokenizer_path = tokenizer_path or model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Ensure tokenizer and model vocab sizes match
    if len(tokenizer) != model.config.vocab_size:
        logger.info(f"Resizing model embeddings from {model.config.vocab_size} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
        
    logger.info(f"Model loaded successfully")
    logger.info(f"Model parameters: {model.num_parameters():,}")
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
    
    return model, tokenizer

def create_evaluation_config(args) -> EvaluationConfig:
    """Create evaluation configuration from arguments"""
    
    return EvaluationConfig(
        max_sequence_length=args.max_sequence_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=args.do_sample,
        num_beams=args.num_beams,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        seed=args.seed,
        save_predictions=args.save_predictions,
        save_detailed_results=args.save_detailed_results
    )

def create_reward_model_config(args) -> VerilogRewardConfig:
    """Create reward model configuration"""
    
    return VerilogRewardConfig(
        syntax_weight=args.syntax_weight,
        functional_weight=args.functional_weight,
        quality_weight=args.quality_weight,
        adherence_weight=args.adherence_weight,
        use_iverilog=True,
        use_verilator=False
    )

def evaluate_on_split(evaluator: VerilogEvaluator, 
                     dataset, 
                     split_name: str,
                     output_dir: str) -> Dict:
    """Evaluate on a specific dataset split"""
    
    logger.info(f"Evaluating on {split_name} split ({len(dataset)} examples)")
    
    # Create split-specific output directory
    split_output_dir = Path(output_dir) / split_name
    
    # Run evaluation
    results = evaluator.evaluate_dataset(dataset, str(split_output_dir))
    
    # Generate report
    if hasattr(evaluator.config, 'generate_report') and evaluator.config.generate_report:
        report = evaluator.generate_report(results['detailed_results'] or [])
        report_file = split_output_dir / "evaluation_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Evaluation report saved to {report_file}")
    
    return results

def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    logger.info("=" * 80)
    logger.info("DEEPSEEK R1 VERILOG MODEL EVALUATION")
    logger.info("=" * 80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save evaluation configuration
    config_file = output_dir / "evaluation_config.json"
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Evaluation configuration saved to {config_file}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.tokenizer_path)
    
    # Load datasets
    logger.info("Loading datasets...")
    try:
        train_dataset, val_dataset, test_dataset = load_verilog_dataset(args.data_dir)
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        logger.error("Please run the data preprocessing pipeline first:")
        logger.error("python scripts/download_and_preprocess.py")
        sys.exit(1)
    
    # Create evaluation configuration
    eval_config = create_evaluation_config(args)
    
    # Create reward model
    logger.info("Setting up reward model...")
    reward_config = create_reward_model_config(args)
    reward_model = VerilogRewardModel(reward_config)
    
    # Create evaluator
    logger.info("Setting up evaluator...")
    evaluator = VerilogEvaluator(
        model=model,
        tokenizer=tokenizer,
        config=eval_config,
        reward_model=reward_model
    )
    
    # Run evaluation on specified splits
    all_results = {}
    
    if args.eval_split == "all":
        splits_to_evaluate = [
            ("train", train_dataset),
            ("val", val_dataset),
            ("test", test_dataset)
        ]
    else:
        dataset_map = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset
        }
        splits_to_evaluate = [(args.eval_split, dataset_map[args.eval_split])]
    
    for split_name, dataset in splits_to_evaluate:
        logger.info(f"=" * 60)
        logger.info(f"EVALUATING {split_name.upper()} SPLIT")
        logger.info(f"=" * 60)
        
        results = evaluate_on_split(evaluator, dataset, split_name, str(output_dir))
        all_results[split_name] = results
        
        # Log summary metrics
        metrics = results['aggregate_metrics']
        logger.info(f"{split_name} Results Summary:")
        logger.info(f"  Mean total reward: {metrics.get('mean_total_reward', 0):.3f}")
        logger.info(f"  Syntax accuracy: {metrics.get('syntax_accuracy', 0):.1f}%")
        logger.info(f"  Functional accuracy: {metrics.get('functional_accuracy', 0):.1f}%")
        logger.info(f"  Proper structure rate: {metrics.get('has_proper_structure_rate', 0):.1f}%")
        
    # Save combined results
    combined_results_file = output_dir / "combined_results.json"
    with open(combined_results_file, 'w') as f:
        # Remove detailed results for space efficiency
        summary_results = {}
        for split_name, results in all_results.items():
            summary_results[split_name] = {
                'aggregate_metrics': results['aggregate_metrics'],
                'num_examples': results['num_examples']
            }
        json.dump(summary_results, f, indent=2)
    
    logger.info(f"Combined results saved to {combined_results_file}")
    
    # Generate overall summary
    logger.info("=" * 80)
    logger.info("EVALUATION COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    
    for split_name, results in all_results.items():
        metrics = results['aggregate_metrics']
        logger.info(f"{split_name.upper()} - Total Reward: {metrics.get('mean_total_reward', 0):.3f}, "
                   f"Syntax: {metrics.get('syntax_accuracy', 0):.1f}%, "
                   f"Structure: {metrics.get('has_proper_structure_rate', 0):.1f}%")
    
    logger.info(f"Detailed results saved to: {output_dir}")
    
    return all_results

if __name__ == "__main__":
    main()