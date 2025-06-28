#!/usr/bin/env python3
"""
PPO Training Pipeline for Verilog Code Generation
Implements reinforcement learning fine-tuning using Proximal Policy Optimization.
"""

import torch
import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import wandb

from .reward_model import VerilogRewardModel, VerilogRewardConfig
from .data_collator import VerilogEvaluationCollator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VerilogPPOConfig(PPOConfig):
    """Extended PPO configuration for Verilog training"""
    
    # Verilog-specific settings
    max_sequence_length: int = field(default=2048)
    reward_model_config: Optional[Dict] = field(default=None)
    
    # Generation settings
    max_new_tokens: int = field(default=512)
    temperature: float = field(default=0.7)
    top_p: float = field(default=0.9)
    top_k: int = field(default=50)
    do_sample: bool = field(default=True)
    
    # Reward scaling
    reward_scale: float = field(default=1.0)
    reward_clip: float = field(default=5.0)
    
    # Wandb settings
    wandb_project: str = field(default="deepseek-verilog-ppo")
    wandb_run_name: Optional[str] = field(default=None)
    
    # Evaluation settings
    eval_frequency: int = field(default=100)
    eval_samples: int = field(default=50)
    
    def __post_init__(self):
        super().__post_init__()
        if self.reward_model_config is None:
            self.reward_model_config = {}


class VerilogPPOTrainer:
    """PPO trainer specialized for Verilog code generation"""
    
    def __init__(self,
                 model: PreTrainedModel,
                 ref_model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 ppo_config: VerilogPPOConfig,
                 dataset: Dataset,
                 reward_model: Optional[VerilogRewardModel] = None):
        
        self.ppo_config = ppo_config
        self.tokenizer = tokenizer
        self.dataset = dataset
        
        # Setup reward model
        if reward_model is None:
            reward_config = VerilogRewardConfig(**ppo_config.reward_model_config)
            self.reward_model = VerilogRewardModel(reward_config)
        else:
            self.reward_model = reward_model
            
        # Prepare models for PPO
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(ref_model)
        
        # Create PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=tokenizer,
            dataset=dataset
        )
        
        # Generation config
        self.generation_config = {
            'max_new_tokens': ppo_config.max_new_tokens,
            'temperature': ppo_config.temperature,
            'top_p': ppo_config.top_p,
            'top_k': ppo_config.top_k,
            'do_sample': ppo_config.do_sample,
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token_id': tokenizer.eos_token_id,
        }
        
        # Data collator for evaluation
        self.eval_collator = VerilogEvaluationCollator(
            tokenizer=tokenizer,
            max_length=ppo_config.max_sequence_length
        )
        
        # Training statistics
        self.training_stats = {
            'step': 0,
            'total_rewards': [],
            'syntax_scores': [],
            'functional_scores': [],
            'quality_scores': [],
            'kl_divergences': [],
            'policy_losses': [],
            'value_losses': []
        }
        
    def prepare_batch(self, batch: Dict[str, Any]) -> Tuple[List[str], List[str], torch.Tensor]:
        """Prepare batch for PPO training"""
        
        instructions = batch['instruction']
        references = batch.get('output', [''] * len(instructions))
        
        # Format instructions for generation
        formatted_instructions = []
        for instruction in instructions:
            formatted = f"### Instruction:\n{instruction}\n\n### Response:\n"
            formatted_instructions.append(formatted)
            
        # Tokenize instructions
        query_tensors = []
        for instruction in formatted_instructions:
            tokens = self.tokenizer(
                instruction,
                return_tensors="pt",
                max_length=self.ppo_config.max_sequence_length // 2,  # Leave room for response
                truncation=True,
                padding=False
            )
            query_tensors.append(tokens['input_ids'].squeeze(0))
            
        return formatted_instructions, references, query_tensors
        
    def generate_responses(self, query_tensors: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[str]]:
        """Generate responses for given queries"""
        
        response_tensors = []
        response_texts = []
        
        self.model.eval()
        with torch.no_grad():
            for query_tensor in query_tensors:
                # Generate response
                response = self.ppo_trainer.generate(
                    query_tensor.unsqueeze(0).to(self.model.pretrained_model.device),
                    **self.generation_config
                )
                
                # Extract response tokens (remove query tokens)
                response_tensor = response[0][len(query_tensor):]
                response_tensors.append(response_tensor)
                
                # Decode response text
                response_text = self.tokenizer.decode(response_tensor, skip_special_tokens=True)
                response_texts.append(response_text)
                
        return response_tensors, response_texts
        
    def compute_rewards(self, instructions: List[str], responses: List[str]) -> List[float]:
        """Compute rewards for generated responses"""
        
        rewards = []
        syntax_scores = []
        functional_scores = []
        quality_scores = []
        
        for instruction, response in zip(instructions, responses):
            try:
                # Compute comprehensive reward
                reward_results = self.reward_model.compute_reward(response, instruction)
                
                reward = reward_results['total_reward']
                
                # Apply reward scaling and clipping
                reward *= self.ppo_config.reward_scale
                reward = np.clip(reward, -self.ppo_config.reward_clip, self.ppo_config.reward_clip)
                
                rewards.append(reward)
                syntax_scores.append(reward_results['syntax_score'])
                functional_scores.append(reward_results['functional_score'])
                quality_scores.append(reward_results['quality_score'])
                
            except Exception as e:
                logger.warning(f"Reward computation failed: {e}")
                rewards.append(-1.0)  # Penalty for failed computation
                syntax_scores.append(0.0)
                functional_scores.append(0.0)
                quality_scores.append(0.0)
                
        # Store statistics
        if hasattr(self, 'training_stats'):
            self.training_stats['syntax_scores'].extend(syntax_scores)
            self.training_stats['functional_scores'].extend(functional_scores)
            self.training_stats['quality_scores'].extend(quality_scores)
        
        return rewards
        
    def training_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Perform one PPO training step"""
        
        # Prepare batch
        instructions, references, query_tensors = self.prepare_batch(batch)
        
        # Generate responses
        response_tensors, response_texts = self.generate_responses(query_tensors)
        
        # Compute rewards
        rewards = self.compute_rewards(instructions, response_texts)
        reward_tensors = [torch.tensor(r) for r in rewards]
        
        # PPO step
        stats = self.ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
        
        # Update training statistics
        self.training_stats['step'] += 1
        self.training_stats['total_rewards'].extend(rewards)
        
        if 'objective/kl' in stats:
            self.training_stats['kl_divergences'].append(stats['objective/kl'])
        if 'ppo/loss/policy' in stats:
            self.training_stats['policy_losses'].append(stats['ppo/loss/policy'])
        if 'ppo/loss/value' in stats:
            self.training_stats['value_losses'].append(stats['ppo/loss/value'])
            
        # Compute step statistics
        step_stats = {
            'step': self.training_stats['step'],
            'mean_reward': np.mean(rewards),
            'mean_syntax_score': np.mean(self.training_stats['syntax_scores'][-len(rewards):]),
            'mean_functional_score': np.mean(self.training_stats['functional_scores'][-len(rewards):]),
            'mean_quality_score': np.mean(self.training_stats['quality_scores'][-len(rewards):]),
        }
        
        step_stats.update(stats)
        
        return step_stats
        
    def evaluate(self, eval_dataset: Optional[Dataset] = None, num_samples: int = None) -> Dict[str, float]:
        """Evaluate model performance"""
        
        if eval_dataset is None:
            eval_dataset = self.dataset
            
        if num_samples is None:
            num_samples = self.ppo_config.eval_samples
            
        # Sample evaluation data
        eval_indices = np.random.choice(len(eval_dataset), min(num_samples, len(eval_dataset)), replace=False)
        eval_samples = eval_dataset.select(eval_indices.tolist())
        
        # Generate and evaluate
        eval_rewards = []
        eval_syntax_scores = []
        eval_functional_scores = []
        eval_quality_scores = []
        
        logger.info(f"Running evaluation on {len(eval_samples)} samples...")
        
        for sample in eval_samples:
            instruction = sample['instruction']
            reference = sample.get('output', '')
            
            # Generate response
            formatted_instruction = f"### Instruction:\n{instruction}\n\n### Response:\n"
            tokens = self.tokenizer(formatted_instruction, return_tensors="pt")
            
            self.model.eval()
            with torch.no_grad():
                response = self.ppo_trainer.generate(
                    tokens['input_ids'].to(self.model.pretrained_model.device),
                    **self.generation_config
                )
                
            # Decode response
            response_tokens = response[0][tokens['input_ids'].shape[1]:]
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            # Compute reward
            try:
                reward_results = self.reward_model.compute_reward(response_text, instruction)
                eval_rewards.append(reward_results['total_reward'])
                eval_syntax_scores.append(reward_results['syntax_score'])
                eval_functional_scores.append(reward_results['functional_score'])
                eval_quality_scores.append(reward_results['quality_score'])
            except Exception as e:
                logger.warning(f"Evaluation reward computation failed: {e}")
                eval_rewards.append(-1.0)
                eval_syntax_scores.append(0.0)
                eval_functional_scores.append(0.0)
                eval_quality_scores.append(0.0)
                
        eval_stats = {
            'eval_mean_reward': np.mean(eval_rewards),
            'eval_mean_syntax_score': np.mean(eval_syntax_scores),
            'eval_mean_functional_score': np.mean(eval_functional_scores),
            'eval_mean_quality_score': np.mean(eval_quality_scores),
            'eval_std_reward': np.std(eval_rewards),
        }
        
        logger.info(f"Evaluation completed. Mean reward: {eval_stats['eval_mean_reward']:.3f}")
        
        return eval_stats
        
    def train(self, num_steps: int, eval_dataset: Optional[Dataset] = None) -> Dict[str, Any]:
        """Main training loop"""
        
        logger.info(f"Starting PPO training for {num_steps} steps...")
        
        # Setup Wandb if configured
        if self.ppo_config.wandb_project:
            wandb.init(
                project=self.ppo_config.wandb_project,
                name=self.ppo_config.wandb_run_name,
                config=self.ppo_config.to_dict()
            )
            
        # Training loop
        for step in range(num_steps):
            # Sample batch from dataset
            batch_indices = np.random.choice(len(self.dataset), self.ppo_config.batch_size, replace=True)
            batch = self.dataset.select(batch_indices.tolist())
            
            # Training step
            step_stats = self.training_step(batch)
            
            # Log statistics
            if step % self.ppo_config.log_with_step == 0:
                logger.info(f"Step {step}: Mean reward = {step_stats['mean_reward']:.3f}")
                
                if wandb.run is not None:
                    wandb.log(step_stats)
                    
            # Evaluation
            if step % self.ppo_config.eval_frequency == 0 and step > 0:
                eval_stats = self.evaluate(eval_dataset)
                
                if wandb.run is not None:
                    wandb.log(eval_stats)
                    
            # Save checkpoint
            if hasattr(self.ppo_config, 'save_freq') and step % self.ppo_config.save_freq == 0 and step > 0:
                self.save_checkpoint(f"checkpoint_step_{step}")
                
        # Final evaluation
        final_eval_stats = self.evaluate(eval_dataset)
        logger.info("Training completed!")
        logger.info(f"Final evaluation: {final_eval_stats}")
        
        # Save final statistics
        training_summary = {
            'total_steps': num_steps,
            'final_eval_stats': final_eval_stats,
            'training_stats': {k: v[-100:] if isinstance(v, list) else v 
                             for k, v in self.training_stats.items()}  # Keep last 100 values
        }
        
        if wandb.run is not None:
            wandb.log(final_eval_stats)
            wandb.finish()
            
        return training_summary
        
    def save_checkpoint(self, checkpoint_name: str, output_dir: Optional[str] = None):
        """Save training checkpoint"""
        
        if output_dir is None:
            output_dir = f"./checkpoints/ppo_{checkpoint_name}"
            
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        # Save configuration
        config_path = output_path / "ppo_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.ppo_config.to_dict(), f, indent=2)
            
        # Save training statistics
        stats_path = output_path / "training_stats.json"
        with open(stats_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            stats_to_save = {}
            for k, v in self.training_stats.items():
                if isinstance(v, list):
                    stats_to_save[k] = [float(x) if isinstance(x, (np.float32, np.float64)) else x for x in v]
                else:
                    stats_to_save[k] = float(v) if isinstance(v, (np.float32, np.float64)) else v
            json.dump(stats_to_save, f, indent=2)
            
        logger.info(f"Checkpoint saved to {output_path}")


def create_ppo_trainer(
    model: PreTrainedModel,
    ref_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    ppo_config: Optional[VerilogPPOConfig] = None,
    reward_model: Optional[VerilogRewardModel] = None
) -> VerilogPPOTrainer:
    """Factory function to create PPO trainer"""
    
    if ppo_config is None:
        ppo_config = VerilogPPOConfig()
        
    return VerilogPPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        ppo_config=ppo_config,
        dataset=dataset,
        reward_model=reward_model
    )


if __name__ == "__main__":
    pass