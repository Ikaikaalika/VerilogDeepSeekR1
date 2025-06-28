#!/usr/bin/env python3
"""
DeepSeek R1 Model Setup and Configuration
Handles model loading, LoRA configuration, and tokenizer setup.
"""

import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType
from typing import Tuple, Optional, Dict, Any
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepSeekVerilogModel:
    """DeepSeek R1 model wrapper for Verilog code generation"""
    
    def __init__(self, 
                 model_name: str = "deepseek-ai/deepseek-r1-distill-qwen-32b",
                 use_lora: bool = True,
                 lora_config: Optional[Dict] = None,
                 device_map: str = "auto",
                 torch_dtype: torch.dtype = torch.bfloat16,
                 trust_remote_code: bool = True):
        
        self.model_name = model_name
        self.use_lora = use_lora
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        
        # Default LoRA configuration optimized for Verilog generation
        self.default_lora_config = {
            "r": 64,
            "lora_alpha": 32,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": TaskType.CAUSAL_LM,
            "modules_to_save": []
        }
        
        if lora_config:
            self.default_lora_config.update(lora_config)
            
        self.tokenizer = None
        self.model = None
        self.config = None
        
    def load_tokenizer(self) -> AutoTokenizer:
        """Load and configure tokenizer"""
        logger.info(f"Loading tokenizer for {self.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                padding_side="left"
            )
            
            # Add special tokens for Verilog if needed
            special_tokens = {
                "pad_token": "<|pad|>",
                "eos_token": "<|endoftext|>",
                "bos_token": "<|startoftext|>",
            }
            
            # Only add tokens that don't exist
            tokens_to_add = {}
            for token_type, token in special_tokens.items():
                if getattr(self.tokenizer, token_type) is None:
                    tokens_to_add[token_type] = token
                    
            if tokens_to_add:
                self.tokenizer.add_special_tokens(tokens_to_add)
                logger.info(f"Added special tokens: {tokens_to_add}")
                
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info(f"Tokenizer loaded successfully. Vocab size: {len(self.tokenizer)}")
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
            
        return self.tokenizer
        
    def load_model(self) -> AutoModelForCausalLM:
        """Load and configure the base model"""
        logger.info(f"Loading model {self.model_name}")
        
        try:
            # Load model configuration
            self.config = AutoConfig.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                config=self.config,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
                trust_remote_code=self.trust_remote_code,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
            )
            
            # Resize embeddings if tokenizer was modified
            if self.tokenizer and len(self.tokenizer) != self.model.config.vocab_size:
                logger.info(f"Resizing token embeddings from {self.model.config.vocab_size} to {len(self.tokenizer)}")
                self.model.resize_token_embeddings(len(self.tokenizer))
                
            logger.info(f"Model loaded successfully")
            logger.info(f"Model parameters: {self.model.num_parameters():,}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
        return self.model
        
    def apply_lora(self) -> AutoModelForCausalLM:
        """Apply LoRA to the model"""
        if not self.use_lora:
            logger.info("LoRA not enabled, skipping")
            return self.model
            
        if self.model is None:
            raise ValueError("Model must be loaded before applying LoRA")
            
        logger.info("Applying LoRA configuration")
        
        try:
            # Create LoRA configuration
            lora_config = LoraConfig(**self.default_lora_config)
            
            # Apply LoRA
            self.model = get_peft_model(self.model, lora_config)
            
            # Print trainable parameters
            self.model.print_trainable_parameters()
            
            logger.info("LoRA applied successfully")
            
        except Exception as e:
            logger.error(f"Failed to apply LoRA: {e}")
            raise
            
        return self.model
        
    def setup_for_training(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Complete model setup for training"""
        logger.info("Setting up model for training")
        
        # Load tokenizer
        self.load_tokenizer()
        
        # Load model
        self.load_model()
        
        # Apply LoRA if enabled
        if self.use_lora:
            self.apply_lora()
            
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
            
        # Prepare model for training
        self.model.train()
        
        logger.info("Model setup completed successfully")
        return self.model, self.tokenizer
        
    def save_model_config(self, output_dir: str):
        """Save model configuration"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        config = {
            "model_name": self.model_name,
            "use_lora": self.use_lora,
            "lora_config": self.default_lora_config if self.use_lora else None,
            "torch_dtype": str(self.torch_dtype),
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code
        }
        
        with open(output_path / "model_config.json", 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"Model configuration saved to {output_path}")
        
    @classmethod
    def from_config(cls, config_path: str) -> 'DeepSeekVerilogModel':
        """Load model from saved configuration"""
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Convert torch_dtype string back to dtype
        if config.get("torch_dtype"):
            config["torch_dtype"] = getattr(torch, config["torch_dtype"].split(".")[-1])
            
        return cls(**config)
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        info = {
            "model_name": self.model_name,
            "use_lora": self.use_lora,
            "torch_dtype": str(self.torch_dtype),
            "device_map": self.device_map
        }
        
        if self.model:
            info.update({
                "total_parameters": self.model.num_parameters(),
                "model_config": self.model.config.to_dict() if hasattr(self.model.config, 'to_dict') else str(self.model.config)
            })
            
            if self.use_lora and hasattr(self.model, 'peft_config'):
                info["lora_config"] = {k: str(v) for k, v in self.default_lora_config.items()}
                
        if self.tokenizer:
            info.update({
                "vocab_size": len(self.tokenizer),
                "pad_token": self.tokenizer.pad_token,
                "eos_token": self.tokenizer.eos_token,
                "bos_token": self.tokenizer.bos_token
            })
            
        return info


def create_model_for_training(model_name: str = "deepseek-ai/deepseek-r1-distill-qwen-32b",
                             lora_config: Optional[Dict] = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Convenience function to create model for training"""
    model_wrapper = DeepSeekVerilogModel(
        model_name=model_name,
        lora_config=lora_config
    )
    
    return model_wrapper.setup_for_training()


def main():
    """Test model setup"""
    logger.info("Testing model setup...")
    
    try:
        model_wrapper = DeepSeekVerilogModel()
        model, tokenizer = model_wrapper.setup_for_training()
        
        # Print model info
        info = model_wrapper.get_model_info()
        logger.info("Model information:")
        for key, value in info.items():
            logger.info(f"  {key}: {value}")
            
        # Test tokenization
        test_input = "Generate Verilog code for a 4-bit counter"
        tokens = tokenizer(test_input, return_tensors="pt")
        logger.info(f"Test tokenization successful. Input tokens: {tokens['input_ids'].shape}")
        
        logger.info("Model setup test completed successfully!")
        
    except Exception as e:
        logger.error(f"Model setup test failed: {e}")
        raise


if __name__ == "__main__":
    main()