#!/usr/bin/env python3
"""
Comprehensive Evaluation Framework for Verilog Code Generation
Evaluates syntax, functionality, and code quality of generated Verilog.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset
from tqdm import tqdm

from ..training.reward_model import VerilogRewardModel, VerilogRewardConfig
from ..training.data_collator import VerilogEvaluationCollator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    max_sequence_length: int = 2048
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    num_beams: int = 1
    
    # Evaluation settings
    batch_size: int = 8
    num_samples: Optional[int] = None  # None means use all
    seed: int = 42
    
    # Output settings
    save_predictions: bool = True
    save_detailed_results: bool = True


class VerilogEvaluator:
    """Comprehensive Verilog code generation evaluator"""
    
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 config: Optional[EvaluationConfig] = None,
                 reward_model: Optional[VerilogRewardModel] = None):
        
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or EvaluationConfig()
        
        # Setup reward model for quality evaluation
        if reward_model is None:
            reward_config = VerilogRewardConfig()
            self.reward_model = VerilogRewardModel(reward_config)
        else:
            self.reward_model = reward_model
            
        # Setup data collator
        self.eval_collator = VerilogEvaluationCollator(
            tokenizer=self.tokenizer,
            max_length=self.config.max_sequence_length
        )
        
        # Generation config
        self.generation_config = {
            'max_new_tokens': self.config.max_new_tokens,
            'temperature': self.config.temperature,
            'top_p': self.config.top_p,
            'top_k': self.config.top_k,
            'do_sample': self.config.do_sample,
            'num_beams': self.config.num_beams,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }
        
        # Results storage
        self.evaluation_results = []
        
    def generate_response(self, instruction: str) -> str:
        """Generate Verilog code for a single instruction"""
        
        # Format instruction
        formatted_instruction = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_instruction,
            return_tensors="pt",
            max_length=self.config.max_sequence_length // 2,
            truncation=True,
            padding=False
        )
        
        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.generation_config
            )
            
        # Decode response (remove input tokens)
        input_length = inputs['input_ids'].shape[1]
        response_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        return response.strip()
        
    def evaluate_single_example(self, 
                               instruction: str, 
                               reference: str,
                               metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Evaluate a single example"""
        
        # Generate prediction
        try:
            prediction = self.generate_response(instruction)
        except Exception as e:
            logger.warning(f"Generation failed for instruction: {str(e)[:100]}")
            prediction = ""
            
        # Compute reward-based metrics
        try:
            reward_results = self.reward_model.compute_reward(prediction, instruction)
        except Exception as e:
            logger.warning(f"Reward computation failed: {str(e)[:100]}")
            reward_results = {
                'total_reward': 0.0,
                'syntax_score': 0.0,
                'functional_score': 0.0,
                'quality_score': 0.0,
                'adherence_score': 0.0,
                'issues': [f"Reward computation error: {str(e)}"]
            }
            
        # Compute additional metrics
        exact_match = (prediction.strip() == reference.strip()) if reference else False
        
        # Length metrics
        pred_length = len(prediction.split())
        ref_length = len(reference.split()) if reference else 0
        
        # Basic code structure metrics
        has_module = 'module' in prediction.lower()
        has_endmodule = 'endmodule' in prediction.lower()
        has_proper_structure = has_module and has_endmodule
        
        # Compile results
        result = {
            'instruction': instruction,
            'reference': reference,
            'prediction': prediction,
            'metadata': metadata or {},
            
            # Reward-based metrics
            'total_reward': reward_results['total_reward'],
            'syntax_score': reward_results['syntax_score'],
            'functional_score': reward_results['functional_score'],
            'quality_score': reward_results['quality_score'],
            'adherence_score': reward_results['adherence_score'],
            
            # Basic metrics
            'exact_match': exact_match,
            'prediction_length': pred_length,
            'reference_length': ref_length,
            'has_proper_structure': has_proper_structure,
            'has_module': has_module,
            'has_endmodule': has_endmodule,
            
            # Issues and breakdown
            'issues': reward_results.get('issues', []),
            'reward_breakdown': reward_results.get('breakdown', {})
        }
        
        return result
        
    def evaluate_dataset(self, 
                        dataset: Dataset,
                        output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate on a complete dataset"""
        
        logger.info(f"Starting evaluation on {len(dataset)} examples")
        
        # Sample dataset if specified
        if self.config.num_samples and self.config.num_samples < len(dataset):
            indices = np.random.choice(len(dataset), self.config.num_samples, replace=False)
            dataset = dataset.select(indices.tolist())
            logger.info(f"Sampling {self.config.num_samples} examples for evaluation")
            
        # Process in batches for memory efficiency
        all_results = []
        
        for i in tqdm(range(0, len(dataset), self.config.batch_size), desc="Evaluating"):
            batch_end = min(i + self.config.batch_size, len(dataset))
            batch = dataset.select(range(i, batch_end))
            
            # Process batch
            for example in batch:
                instruction = example['instruction']
                reference = example.get('output', '')
                metadata = example.get('metadata', {})
                
                result = self.evaluate_single_example(instruction, reference, metadata)
                all_results.append(result)
                
        # Store results
        self.evaluation_results = all_results
        
        # Compute aggregate metrics
        aggregate_metrics = self.compute_aggregate_metrics(all_results)
        
        # Save results if output directory specified
        if output_dir:
            self.save_results(all_results, aggregate_metrics, output_dir)
            
        logger.info("Evaluation completed!")
        logger.info(f"Aggregate metrics: {aggregate_metrics}")
        
        return {
            'aggregate_metrics': aggregate_metrics,
            'detailed_results': all_results if self.config.save_detailed_results else None,
            'num_examples': len(all_results)
        }
        
    def compute_aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute aggregate metrics from individual results"""
        
        if not results:
            return {}
            
        # Extract numeric metrics
        metrics = {
            'total_reward': [r['total_reward'] for r in results],
            'syntax_score': [r['syntax_score'] for r in results],
            'functional_score': [r['functional_score'] for r in results],
            'quality_score': [r['quality_score'] for r in results],
            'adherence_score': [r['adherence_score'] for r in results],
            'exact_match': [r['exact_match'] for r in results],
            'prediction_length': [r['prediction_length'] for r in results],
            'has_proper_structure': [r['has_proper_structure'] for r in results],
            'has_module': [r['has_module'] for r in results],
            'has_endmodule': [r['has_endmodule'] for r in results]
        }
        
        # Compute statistics
        aggregate_metrics = {}
        
        for metric_name, values in metrics.items():
            if metric_name in ['exact_match', 'has_proper_structure', 'has_module', 'has_endmodule']:
                # Boolean metrics - compute percentage
                aggregate_metrics[f'{metric_name}_rate'] = np.mean(values) * 100
            else:
                # Continuous metrics - compute mean and std
                aggregate_metrics[f'mean_{metric_name}'] = np.mean(values)
                aggregate_metrics[f'std_{metric_name}'] = np.std(values)
                
        # Special metrics
        aggregate_metrics['syntax_accuracy'] = np.mean([r['syntax_score'] > 0.5 for r in results]) * 100
        aggregate_metrics['functional_accuracy'] = np.mean([r['functional_score'] > 0.5 for r in results]) * 100
        aggregate_metrics['high_quality_rate'] = np.mean([r['quality_score'] > 0.7 for r in results]) * 100
        
        # Count issues
        total_issues = sum(len(r['issues']) for r in results)
        aggregate_metrics['avg_issues_per_example'] = total_issues / len(results)
        
        return aggregate_metrics
        
    def save_results(self, 
                    results: List[Dict[str, Any]], 
                    aggregate_metrics: Dict[str, float],
                    output_dir: str):
        """Save evaluation results"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save aggregate metrics
        metrics_file = output_path / "evaluation_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(aggregate_metrics, f, indent=2)
        logger.info(f"Aggregate metrics saved to {metrics_file}")
        
        # Save detailed results if requested
        if self.config.save_detailed_results:
            detailed_file = output_path / "detailed_results.json"
            with open(detailed_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Detailed results saved to {detailed_file}")
            
        # Save predictions for manual inspection
        if self.config.save_predictions:
            predictions_data = []
            for i, result in enumerate(results):
                predictions_data.append({
                    'id': i,
                    'instruction': result['instruction'],
                    'prediction': result['prediction'],
                    'reference': result['reference'],
                    'total_reward': result['total_reward'],
                    'syntax_score': result['syntax_score'],
                    'issues': result['issues']
                })
                
            predictions_file = output_path / "predictions.json"
            with open(predictions_file, 'w') as f:
                json.dump(predictions_data, f, indent=2)
            logger.info(f"Predictions saved to {predictions_file}")
            
            # Also save as CSV for easy viewing
            df = pd.DataFrame(predictions_data)
            csv_file = output_path / "predictions.csv"
            df.to_csv(csv_file, index=False)
            logger.info(f"Predictions CSV saved to {csv_file}")
            
        # Save configuration
        config_file = output_path / "evaluation_config.json"
        with open(config_file, 'w') as f:
            config_dict = {
                'max_sequence_length': self.config.max_sequence_length,
                'max_new_tokens': self.config.max_new_tokens,
                'temperature': self.config.temperature,
                'top_p': self.config.top_p,
                'top_k': self.config.top_k,
                'do_sample': self.config.do_sample,
                'batch_size': self.config.batch_size,
                'num_samples': self.config.num_samples,
                'seed': self.config.seed
            }
            json.dump(config_dict, f, indent=2)
        logger.info(f"Evaluation configuration saved to {config_file}")
        
    def generate_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate a human-readable evaluation report"""
        
        if not results:
            return "No results to report."
            
        aggregate_metrics = self.compute_aggregate_metrics(results)
        
        report = f"""
Verilog Code Generation Evaluation Report
=========================================

Dataset Statistics:
- Total examples evaluated: {len(results)}
- Average prediction length: {aggregate_metrics.get('mean_prediction_length', 0):.1f} words

Syntax and Structure:
- Syntax accuracy (>0.5): {aggregate_metrics.get('syntax_accuracy', 0):.1f}%
- Proper module structure: {aggregate_metrics.get('has_proper_structure_rate', 0):.1f}%
- Contains 'module': {aggregate_metrics.get('has_module_rate', 0):.1f}%
- Contains 'endmodule': {aggregate_metrics.get('has_endmodule_rate', 0):.1f}%

Functionality and Quality:
- Functional accuracy (>0.5): {aggregate_metrics.get('functional_accuracy', 0):.1f}%
- High quality rate (>0.7): {aggregate_metrics.get('high_quality_rate', 0):.1f}%
- Exact match rate: {aggregate_metrics.get('exact_match_rate', 0):.1f}%

Reward Scores (0-1 scale):
- Mean total reward: {aggregate_metrics.get('mean_total_reward', 0):.3f} ± {aggregate_metrics.get('std_total_reward', 0):.3f}
- Mean syntax score: {aggregate_metrics.get('mean_syntax_score', 0):.3f} ± {aggregate_metrics.get('std_syntax_score', 0):.3f}
- Mean functional score: {aggregate_metrics.get('mean_functional_score', 0):.3f} ± {aggregate_metrics.get('std_functional_score', 0):.3f}
- Mean quality score: {aggregate_metrics.get('mean_quality_score', 0):.3f} ± {aggregate_metrics.get('std_quality_score', 0):.3f}

Issues and Errors:
- Average issues per example: {aggregate_metrics.get('avg_issues_per_example', 0):.1f}

Generation Configuration:
- Temperature: {self.config.temperature}
- Top-p: {self.config.top_p}
- Max new tokens: {self.config.max_new_tokens}
- Sampling: {self.config.do_sample}
"""
        
        return report


def create_evaluator(model: PreTrainedModel,
                    tokenizer: PreTrainedTokenizer,
                    config: Optional[EvaluationConfig] = None) -> VerilogEvaluator:
    """Factory function to create evaluator"""
    return VerilogEvaluator(model, tokenizer, config)


if __name__ == "__main__":
    pass