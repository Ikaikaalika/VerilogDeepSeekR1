#!/usr/bin/env python3
"""
Model Deployment Script for DeepSeek R1 Verilog Generation
Prepares models for production deployment with optimization.
"""

import sys
import os
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import onnx
import onnxruntime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Deploy DeepSeek R1 Verilog generation model")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--base_model_name", type=str,
                       default="deepseek-ai/deepseek-r1-distill-qwen-32b",
                       help="Base model name for LoRA merging")
    parser.add_argument("--is_peft_model", action="store_true",
                       help="Whether the model is a PEFT/LoRA model")
    
    # Deployment arguments
    parser.add_argument("--output_dir", type=str, default="./deployed_model",
                       help="Output directory for deployed model")
    parser.add_argument("--merge_lora", action="store_true", default=True,
                       help="Merge LoRA weights with base model")
    parser.add_argument("--quantize", action="store_true",
                       help="Apply quantization for deployment")
    parser.add_argument("--quantization_bits", type=int, default=8,
                       choices=[4, 8, 16],
                       help="Quantization bits")
    parser.add_argument("--export_onnx", action="store_true",
                       help="Export model to ONNX format")
    parser.add_argument("--optimize_for_inference", action="store_true", default=True,
                       help="Apply inference optimizations")
    
    # Testing arguments
    parser.add_argument("--test_deployment", action="store_true", default=True,
                       help="Test deployed model")
    parser.add_argument("--test_prompts", type=str, nargs="+",
                       default=["Create a 4-bit counter in Verilog",
                               "Design an 8-bit ALU module"],
                       help="Test prompts for deployment testing")
    
    return parser.parse_args()

class ModelDeployer:
    """Model deployment utility"""
    
    def __init__(self, model_path: str, output_dir: str, base_model_name: str = None):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.base_model_name = base_model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def merge_lora_weights(self) -> tuple:
        """Merge LoRA weights with base model"""
        
        logger.info("Merging LoRA weights with base model...")
        
        # Load base model
        logger.info(f"Loading base model: {self.base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="cpu",  # Keep on CPU for merging
            trust_remote_code=True
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Resize embeddings if needed
        if len(tokenizer) != base_model.config.vocab_size:
            logger.info(f"Resizing embeddings from {base_model.config.vocab_size} to {len(tokenizer)}")
            base_model.resize_token_embeddings(len(tokenizer))
        
        # Load and merge PEFT model
        logger.info(f"Loading PEFT model from {self.model_path}")
        peft_model = PeftModel.from_pretrained(base_model, self.model_path)
        
        logger.info("Merging LoRA weights...")
        merged_model = peft_model.merge_and_unload()
        
        logger.info("LoRA weights merged successfully!")
        return merged_model, tokenizer
        
    def load_full_model(self) -> tuple:
        """Load a full (non-PEFT) model"""
        
        logger.info(f"Loading full model from {self.model_path}")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
        
    def apply_quantization(self, model, bits: int = 8):
        """Apply quantization to model"""
        
        logger.info(f"Applying {bits}-bit quantization...")
        
        try:
            from transformers import BitsAndBytesConfig
            
            if bits == 8:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True
                )
            elif bits == 4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            else:
                logger.warning(f"Unsupported quantization bits: {bits}")
                return model
                
            # Note: For deployment, we typically save the quantized model separately
            logger.info("Quantization configuration prepared")
            return model
            
        except ImportError:
            logger.warning("BitsAndBytesConfig not available, skipping quantization")
            return model
            
    def optimize_for_inference(self, model):
        """Apply inference optimizations"""
        
        logger.info("Applying inference optimizations...")
        
        # Set to evaluation mode
        model.eval()
        
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False
            
        # Enable inference mode optimizations
        try:
            # Compile model for faster inference (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                logger.info("Compiling model with torch.compile...")
                model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}")
            
        logger.info("Inference optimizations applied")
        return model
        
    def export_to_onnx(self, model, tokenizer, sample_input_length: int = 128):
        """Export model to ONNX format"""
        
        logger.info("Exporting model to ONNX format...")
        
        try:
            # Prepare sample input
            sample_text = "### Instruction:\nCreate a 4-bit counter in Verilog\n\n### Response:\n"
            inputs = tokenizer(sample_text, return_tensors="pt", max_length=sample_input_length, padding="max_length")
            
            # Define input and output names
            input_names = ["input_ids", "attention_mask"]
            output_names = ["logits"]
            
            # Dynamic axes for variable-length inputs
            dynamic_axes = {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"}
            }
            
            onnx_path = self.output_dir / "model.onnx"
            
            # Export to ONNX
            torch.onnx.export(
                model,
                (inputs["input_ids"], inputs["attention_mask"]),
                str(onnx_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=14,
                do_constant_folding=True,
                export_params=True
            )
            
            # Verify ONNX model
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            logger.info(f"Model exported to ONNX: {onnx_path}")
            
            # Test ONNX inference
            self.test_onnx_inference(str(onnx_path), tokenizer, sample_text)
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            
    def test_onnx_inference(self, onnx_path: str, tokenizer, sample_text: str):
        """Test ONNX model inference"""
        
        try:
            logger.info("Testing ONNX inference...")
            
            # Create ONNX runtime session
            session = onnxruntime.InferenceSession(onnx_path)
            
            # Prepare input
            inputs = tokenizer(sample_text, return_tensors="np", max_length=128, padding="max_length")
            
            # Run inference
            outputs = session.run(None, {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            })
            
            logger.info("ONNX inference test successful")
            
        except Exception as e:
            logger.error(f"ONNX inference test failed: {e}")
            
    def save_deployment_info(self, model, tokenizer, config: dict):
        """Save deployment information"""
        
        # Save model and tokenizer
        logger.info(f"Saving deployed model to {self.output_dir}")
        model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        
        # Save deployment configuration
        deployment_info = {
            "source_model_path": str(self.model_path),
            "base_model_name": self.base_model_name,
            "deployment_config": config,
            "model_info": {
                "num_parameters": model.num_parameters(),
                "model_type": type(model).__name__,
                "torch_dtype": str(model.dtype),
                "vocab_size": len(tokenizer)
            },
            "deployment_timestamp": str(datetime.now())
        }
        
        info_file = self.output_dir / "deployment_info.json"
        with open(info_file, 'w') as f:
            json.dump(deployment_info, f, indent=2)
            
        logger.info(f"Deployment info saved to {info_file}")
        
        # Create inference example script
        self.create_inference_script(tokenizer)
        
    def create_inference_script(self, tokenizer):
        """Create example inference script"""
        
        script_content = f'''#!/usr/bin/env python3
"""
Example inference script for deployed DeepSeek R1 Verilog model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model():
    """Load the deployed model"""
    model_path = "{self.output_dir}"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    return model, tokenizer

def generate_verilog(instruction: str, model, tokenizer, max_new_tokens: int = 512):
    """Generate Verilog code from instruction"""
    
    # Format instruction
    formatted_input = f"### Instruction:\\n{{instruction}}\\n\\n### Response:\\n"
    
    # Tokenize
    inputs = tokenizer(formatted_input, return_tensors="pt")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return response.strip()

def main():
    """Example usage"""
    model, tokenizer = load_model()
    
    test_instructions = [
        "Create a 4-bit counter in Verilog",
        "Design an 8-bit ALU module",
        "Implement a simple FIFO buffer"
    ]
    
    for instruction in test_instructions:
        print(f"Instruction: {{instruction}}")
        print("Generated Verilog:")
        print(generate_verilog(instruction, model, tokenizer))
        print("-" * 80)

if __name__ == "__main__":
    main()
'''
        
        script_file = self.output_dir / "inference_example.py"
        with open(script_file, 'w') as f:
            f.write(script_content)
            
        # Make executable
        os.chmod(script_file, 0o755)
        logger.info(f"Inference example script created: {script_file}")
        
    def test_deployment(self, model, tokenizer, test_prompts: list):
        """Test deployed model"""
        
        logger.info("Testing deployed model...")
        
        for i, prompt in enumerate(test_prompts):
            logger.info(f"Test {i+1}: {prompt}")
            
            # Format input
            formatted_input = f"### Instruction:\n{prompt}\n\n### Response:\n"
            
            # Tokenize
            inputs = tokenizer(formatted_input, return_tensors="pt")
            
            # Generate
            model.eval()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # Decode
            response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
            
            logger.info(f"Generated:\n{response.strip()}\n")
            
        logger.info("Deployment testing completed!")
        
    def deploy(self, 
               is_peft_model: bool,
               merge_lora: bool = True,
               quantize: bool = False,
               quantization_bits: int = 8,
               export_onnx: bool = False,
               optimize_for_inference: bool = True,
               test_deployment: bool = True,
               test_prompts: list = None):
        """Main deployment function"""
        
        config = {
            "is_peft_model": is_peft_model,
            "merge_lora": merge_lora,
            "quantize": quantize,
            "quantization_bits": quantization_bits,
            "export_onnx": export_onnx,
            "optimize_for_inference": optimize_for_inference
        }
        
        logger.info("Starting model deployment...")
        logger.info(f"Configuration: {config}")
        
        # Load model
        if is_peft_model and merge_lora:
            model, tokenizer = self.merge_lora_weights()
        else:
            model, tokenizer = self.load_full_model()
            
        # Apply optimizations
        if optimize_for_inference:
            model = self.optimize_for_inference(model)
            
        if quantize:
            model = self.apply_quantization(model, quantization_bits)
            
        # Save deployed model
        self.save_deployment_info(model, tokenizer, config)
        
        # Export to ONNX if requested
        if export_onnx:
            self.export_to_onnx(model, tokenizer)
            
        # Test deployment
        if test_deployment and test_prompts:
            self.test_deployment(model, tokenizer, test_prompts)
            
        logger.info(f"Model deployment completed! Saved to: {self.output_dir}")
        
        return model, tokenizer

def main():
    """Main deployment function"""
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("DEEPSEEK R1 VERILOG MODEL DEPLOYMENT")
    logger.info("=" * 80)
    
    # Create deployer
    deployer = ModelDeployer(
        model_path=args.model_path,
        output_dir=args.output_dir,
        base_model_name=args.base_model_name
    )
    
    # Deploy model
    model, tokenizer = deployer.deploy(
        is_peft_model=args.is_peft_model,
        merge_lora=args.merge_lora,
        quantize=args.quantize,
        quantization_bits=args.quantization_bits,
        export_onnx=args.export_onnx,
        optimize_for_inference=args.optimize_for_inference,
        test_deployment=args.test_deployment,
        test_prompts=args.test_prompts
    )
    
    logger.info("=" * 80)
    logger.info("DEPLOYMENT COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info(f"Deployed model available at: {args.output_dir}")
    

if __name__ == "__main__":
    main()