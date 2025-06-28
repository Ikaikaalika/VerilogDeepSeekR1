#!/usr/bin/env python3
"""
Supervised Fine-tuning Trainer for DeepSeek R1 Verilog Generation
Implements comprehensive training pipeline with evaluation and checkpointing.
"""

import os
import json
import logging
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from transformers import (
    TrainingArguments, 
    Trainer,
    EvalPrediction,
    PreTrainedTokenizer,
    PreTrainedModel
)
from datasets import Dataset, load_dataset
import wandb
from peft import PeftModel
import numpy as np
from tqdm import tqdm

from .data_collator import VerilogDataCollator, VerilogEvaluationCollator, VerilogSyntaxValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VerilogTrainingArguments(TrainingArguments):
    """Extended training arguments for Verilog fine-tuning"""
    
    # Verilog-specific arguments
    max_sequence_length: int = field(default=2048)
    validation_split: float = field(default=0.1)
    eval_steps_ratio: float = field(default=0.1)
    save_steps_ratio: float = field(default=0.1)
    
    # Generation arguments for evaluation
    max_new_tokens: int = field(default=512)
    temperature: float = field(default=0.7)
    top_p: float = field(default=0.9)
    do_sample: bool = field(default=True)
    
    # Wandb logging
    wandb_project: str = field(default="deepseek-verilog-finetune")
    wandb_run_name: Optional[str] = field(default=None)
    
    def __post_init__(self):
        super().__post_init__()
        
        # Set dynamic eval and save steps based on ratios
        if hasattr(self, 'max_steps') and self.max_steps > 0:
            if self.eval_steps is None or self.eval_steps <= 0:
                self.eval_steps = max(1, int(self.max_steps * self.eval_steps_ratio))
            if self.save_steps is None or self.save_steps <= 0:
                self.save_steps = max(1, int(self.max_steps * self.save_steps_ratio))


class VerilogMetrics:
    """Metrics computation for Verilog code generation"""
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.syntax_validator = VerilogSyntaxValidator()
        
    def compute_syntax_accuracy(self, predictions: List[str]) -> float:
        """Compute percentage of syntactically valid Verilog code"""
        valid_count = 0
        for pred in predictions:
            if self.syntax_validator(pred):
                valid_count += 1
        return valid_count / len(predictions) if predictions else 0.0
        
    def compute_bleu_score(self, predictions: List[str], references: List[str]) -> float:
        """Compute BLEU score for code similarity"""
        try:
            from evaluate import load
            bleu = load("bleu")
            
            # Tokenize for BLEU computation
            pred_tokens = [pred.split() for pred in predictions]
            ref_tokens = [[ref.split()] for ref in references]
            
            result = bleu.compute(predictions=pred_tokens, references=ref_tokens)
            return result['bleu']
        except ImportError:
            logger.warning("BLEU metric not available, skipping")
            return 0.0
        except Exception as e:
            logger.warning(f"BLEU computation failed: {e}")
            return 0.0
            
    def compute_code_quality_score(self, predictions: List[str]) -> float:
        """Compute code quality score based on Verilog best practices"""
        scores = []
        
        for pred in predictions:
            score = 1.0
            
            # Check for proper module structure
            if 'module' not in pred.lower() or 'endmodule' not in pred.lower():
                score *= 0.5
                
            # Check for proper indentation (basic check)
            lines = pred.split('\n')
            indented_lines = sum(1 for line in lines if line.startswith('    ') or line.startswith('\t'))
            if len(lines) > 5 and indented_lines / len(lines) < 0.3:
                score *= 0.8
                
            # Check for comments
            comment_lines = sum(1 for line in lines if '//' in line or '/*' in line)
            if len(lines) > 10 and comment_lines == 0:
                score *= 0.9
                
            scores.append(score)
            
        return np.mean(scores) if scores else 0.0


class VerilogTrainer(Trainer):
    """Custom trainer for Verilog code generation"""
    
    def __init__(self, 
                 eval_dataset: Optional[Dataset] = None,
                 metrics_computer: Optional[VerilogMetrics] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.eval_dataset = eval_dataset
        self.metrics_computer = metrics_computer
        self.generation_config = {
            'max_new_tokens': getattr(self.args, 'max_new_tokens', 512),
            'temperature': getattr(self.args, 'temperature', 0.7),
            'top_p': getattr(self.args, 'top_p', 0.9),
            'do_sample': getattr(self.args, 'do_sample', True),
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }
        
    def evaluate(self, eval_dataset: Optional[Dataset] = None, **kwargs) -> Dict[str, float]:
        """Enhanced evaluation with generation and Verilog-specific metrics"""
        
        # Standard evaluation
        eval_results = super().evaluate(eval_dataset=eval_dataset, **kwargs)
        
        # Generation-based evaluation
        if self.eval_dataset is not None and self.metrics_computer is not None:
            try:
                generation_metrics = self._evaluate_generation()
                eval_results.update(generation_metrics)
            except Exception as e:
                logger.warning(f"Generation evaluation failed: {e}")
                
        return eval_results
        
    def _evaluate_generation(self, num_samples: int = 100) -> Dict[str, float]:
        """Evaluate model by generating Verilog code"""
        
        logger.info(f"Running generation evaluation on {num_samples} samples")
        
        # Sample from evaluation dataset
        eval_samples = self.eval_dataset.select(range(min(num_samples, len(self.eval_dataset))))
        
        # Prepare evaluation collator
        eval_collator = VerilogEvaluationCollator(
            tokenizer=self.tokenizer,
            max_length=getattr(self.args, 'max_sequence_length', 2048)
        )
        
        # Generate predictions
        predictions = []
        references = []
        
        self.model.eval()
        batch_size = 8
        with torch.no_grad():
            for i in tqdm(range(0, len(eval_samples), batch_size), desc="Generating"):
                batch_end = min(i + batch_size, len(eval_samples))
                batch = eval_samples.select(range(i, batch_end))
                
                # Prepare batch
                batch_dict = eval_collator(batch)
                input_ids = batch_dict['input_ids'].to(self.model.device)
                attention_mask = batch_dict['attention_mask'].to(self.model.device)
                
                # Generate
                with torch.cuda.amp.autocast():
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **self.generation_config
                    )
                
                # Decode predictions
                for j, output in enumerate(outputs):
                    # Remove input tokens from output
                    input_length = input_ids[j].shape[0]
                    generated_tokens = output[input_length:]
                    
                    pred_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    predictions.append(pred_text)
                    references.append(batch_dict['references'][j])
                    
        # Compute metrics
        metrics = {}
        
        if predictions:
            metrics['eval_syntax_accuracy'] = self.metrics_computer.compute_syntax_accuracy(predictions)
            metrics['eval_bleu_score'] = self.metrics_computer.compute_bleu_score(predictions, references)
            metrics['eval_code_quality'] = self.metrics_computer.compute_code_quality_score(predictions)
            
            # Log some examples
            if hasattr(self.args, 'wandb_project') and wandb.run is not None:
                examples_table = wandb.Table(columns=["Instruction", "Reference", "Prediction"])
                
                for i in range(min(5, len(predictions))):
                    examples_table.add_data(
                        eval_samples[i]['instruction'],
                        references[i][:200] + "..." if len(references[i]) > 200 else references[i],
                        predictions[i][:200] + "..." if len(predictions[i]) > 200 else predictions[i]
                    )
                    
                wandb.log({"eval_examples": examples_table})
                
        logger.info(f"Generation evaluation completed. Metrics: {metrics}")
        return metrics
        
    def log(self, logs: Dict[str, float]) -> None:
        """Enhanced logging with Wandb integration"""
        super().log(logs)
        
        # Log to Wandb if configured
        if hasattr(self.args, 'wandb_project') and wandb.run is not None:
            wandb.log(logs)


def load_verilog_dataset(data_dir: str) -> Tuple[Dataset, Dataset, Dataset]:
    """Load preprocessed Verilog datasets"""
    
    data_path = Path(data_dir)
    
    # Load datasets
    train_path = data_path / "train.json"
    val_path = data_path / "val.json"  
    test_path = data_path / "test.json"
    
    if not all(p.exists() for p in [train_path, val_path, test_path]):
        raise FileNotFoundError(f"Missing dataset files in {data_dir}")
        
    train_dataset = Dataset.from_json(str(train_path))
    val_dataset = Dataset.from_json(str(val_path))
    test_dataset = Dataset.from_json(str(test_path))
    
    logger.info(f"Loaded datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def setup_wandb(args: VerilogTrainingArguments):
    """Setup Wandb logging"""
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=args.to_dict()
        )
        logger.info(f"Wandb initialized for project: {args.wandb_project}")


def train_verilog_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    val_dataset: Dataset,
    training_args: VerilogTrainingArguments,
    output_dir: str
) -> VerilogTrainer:
    """Main training function"""
    
    logger.info("Setting up Verilog fine-tuning training...")
    
    # Setup Wandb
    setup_wandb(training_args)
    
    # Create data collator
    data_collator = VerilogDataCollator(
        tokenizer=tokenizer,
        max_length=training_args.max_sequence_length
    )
    
    # Create metrics computer
    metrics_computer = VerilogMetrics(tokenizer)
    
    # Create trainer
    trainer = VerilogTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        metrics_computer=metrics_computer
    )
    
    logger.info("Starting training...")
    
    # Train model
    trainer.train()
    
    # Save final model
    logger.info(f"Saving final model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Final evaluation
    logger.info("Running final evaluation...")
    final_metrics = trainer.evaluate()
    
    # Save metrics
    metrics_path = Path(output_dir) / "training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
        
    logger.info(f"Training completed! Final metrics: {final_metrics}")
    
    return trainer


if __name__ == "__main__":
    pass