#!/usr/bin/env python3
"""
Custom Data Collator for Verilog Code Generation
Handles instruction-following format and Verilog-specific tokenization.
"""

import torch
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from transformers import PreTrainedTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VerilogDataCollator:
    """
    Data collator for Verilog instruction-following fine-tuning.
    Handles instruction-response format and proper masking.
    """
    
    tokenizer: PreTrainedTokenizer
    max_length: int = 2048
    instruction_template: str = "### Instruction:\n{instruction}\n\n### Response:\n"
    response_template: str = "{response}"
    ignore_index: int = -100
    
    def __post_init__(self):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def format_instruction(self, instruction: str, response: str) -> str:
        """Format instruction and response into training format"""
        formatted_instruction = self.instruction_template.format(instruction=instruction)
        formatted_response = self.response_template.format(response=response)
        return formatted_instruction + formatted_response
        
    def tokenize_example(self, example: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Tokenize a single example with proper masking"""
        
        instruction = example.get('instruction', '')
        response = example.get('output', '')
        
        # Format the full text
        full_text = self.format_instruction(instruction, response)
        
        # Tokenize instruction part only (for masking)
        instruction_formatted = self.instruction_template.format(instruction=instruction)
        instruction_tokens = self.tokenizer(
            instruction_formatted,
            add_special_tokens=False,
            return_tensors="pt"
        )
        instruction_length = instruction_tokens['input_ids'].shape[1]
        
        # Tokenize full text
        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )
        
        input_ids = tokenized['input_ids'].squeeze(0)
        attention_mask = tokenized['attention_mask'].squeeze(0)
        
        # Create labels (mask instruction part, keep response part)
        labels = input_ids.clone()
        
        # Mask instruction tokens (set to ignore_index)
        if instruction_length < len(labels):
            labels[:instruction_length] = self.ignore_index
        else:
            # If instruction is longer than max_length, mask everything
            labels[:] = self.ignore_index
            
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of examples"""
        
        batch_input_ids = []
        batch_attention_masks = []
        batch_labels = []
        
        for example in examples:
            try:
                tokenized = self.tokenize_example(example)
                batch_input_ids.append(tokenized['input_ids'])
                batch_attention_masks.append(tokenized['attention_mask'])
                batch_labels.append(tokenized['labels'])
            except Exception as e:
                logger.warning(f"Failed to tokenize example: {e}")
                # Skip problematic examples
                continue
                
        if not batch_input_ids:
            raise ValueError("No valid examples in batch")
            
        # Pad sequences to same length
        max_len = max(len(seq) for seq in batch_input_ids)
        max_len = min(max_len, self.max_length)
        
        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []
        
        for input_ids, attention_mask, labels in zip(batch_input_ids, batch_attention_masks, batch_labels):
            # Truncate if necessary
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                attention_mask = attention_mask[:max_len]
                labels = labels[:max_len]
                
            # Pad
            pad_length = max_len - len(input_ids)
            if pad_length > 0:
                pad_token_id = self.tokenizer.pad_token_id
                
                input_ids = torch.cat([
                    input_ids,
                    torch.full((pad_length,), pad_token_id, dtype=input_ids.dtype)
                ])
                
                attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros(pad_length, dtype=attention_mask.dtype)
                ])
                
                labels = torch.cat([
                    labels,
                    torch.full((pad_length,), self.ignore_index, dtype=labels.dtype)
                ])
                
            padded_input_ids.append(input_ids)
            padded_attention_masks.append(attention_mask)
            padded_labels.append(labels)
            
        return {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(padded_attention_masks),
            'labels': torch.stack(padded_labels)
        }


@dataclass
class VerilogEvaluationCollator:
    """
    Data collator for evaluation that doesn't mask the instruction part.
    Used for generation and evaluation.
    """
    
    tokenizer: PreTrainedTokenizer
    max_length: int = 2048
    instruction_template: str = "### Instruction:\n{instruction}\n\n### Response:\n"
    
    def __post_init__(self):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate examples for evaluation"""
        
        instructions = []
        references = []
        
        for example in examples:
            instruction = example.get('instruction', '')
            response = example.get('output', '')
            
            formatted_instruction = self.instruction_template.format(instruction=instruction)
            instructions.append(formatted_instruction)
            references.append(response)
            
        # Tokenize instructions
        tokenized = self.tokenizer(
            instructions,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'references': references,
            'instructions': [example.get('instruction', '') for example in examples]
        }


class VerilogSyntaxValidator:
    """
    Validates Verilog syntax in generated code.
    Used during training for early filtering of invalid outputs.
    """
    
    def __init__(self):
        self.verilog_keywords = {
            'module', 'endmodule', 'input', 'output', 'inout', 'wire', 'reg',
            'always', 'initial', 'assign', 'if', 'else', 'case', 'endcase',
            'for', 'while', 'begin', 'end', 'posedge', 'negedge'
        }
        
    def validate_basic_syntax(self, verilog_code: str) -> Dict[str, Any]:
        """Perform basic Verilog syntax validation"""
        
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'score': 1.0
        }
        
        try:
            # Check for module declaration
            if 'module' not in verilog_code.lower():
                results['errors'].append("No module declaration found")
                results['is_valid'] = False
                
            # Check for endmodule
            if 'endmodule' not in verilog_code.lower():
                results['errors'].append("No endmodule found")
                results['is_valid'] = False
                
            # Check balanced begin/end
            begin_count = verilog_code.lower().count('begin')
            end_count = verilog_code.lower().count('end') - verilog_code.lower().count('endmodule')
            
            if begin_count != end_count:
                results['warnings'].append(f"Unbalanced begin/end: {begin_count} begins, {end_count} ends")
                results['score'] *= 0.8
                
            # Check for basic Verilog keywords
            keyword_count = sum(1 for keyword in self.verilog_keywords 
                              if keyword in verilog_code.lower())
            
            if keyword_count < 3:
                results['warnings'].append("Few Verilog keywords detected")
                results['score'] *= 0.9
                
            # Update final validity
            if results['errors']:
                results['is_valid'] = False
                results['score'] = 0.0
                
        except Exception as e:
            results['errors'].append(f"Validation error: {str(e)}")
            results['is_valid'] = False
            results['score'] = 0.0
            
        return results
        
    def __call__(self, verilog_code: str) -> bool:
        """Simple callable interface for quick validation"""
        return self.validate_basic_syntax(verilog_code)['is_valid']


def create_verilog_data_collator(tokenizer: PreTrainedTokenizer,
                                max_length: int = 2048,
                                instruction_template: Optional[str] = None) -> VerilogDataCollator:
    """Factory function to create Verilog data collator"""
    
    kwargs = {
        'tokenizer': tokenizer,
        'max_length': max_length
    }
    
    if instruction_template:
        kwargs['instruction_template'] = instruction_template
        
    return VerilogDataCollator(**kwargs)


def test_data_collator():
    """Test the data collator functionality"""
    from transformers import AutoTokenizer
    
    # This is a test function - would need actual tokenizer in practice
    logger.info("Testing data collator...")
    
    # Mock test data
    examples = [
        {
            'instruction': 'Create a 4-bit counter in Verilog',
            'output': '''module counter_4bit(
    input clk,
    input reset,
    output reg [3:0] count
);

always @(posedge clk or posedge reset) begin
    if (reset)
        count <= 4'b0000;
    else
        count <= count + 1;
end

endmodule'''
        }
    ]
    
    logger.info("Example format test:")
    collator = VerilogDataCollator(tokenizer=None)  # Mock
    formatted = collator.format_instruction(examples[0]['instruction'], examples[0]['output'])
    logger.info(f"Formatted text:\n{formatted}")
    

if __name__ == "__main__":
    test_data_collator()