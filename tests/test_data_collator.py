#!/usr/bin/env python3
"""
Tests for Verilog Data Collator
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from training.data_collator import VerilogDataCollator, VerilogSyntaxValidator


class MockTokenizer:
    """Mock tokenizer for testing"""
    
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        
    def __call__(self, text, **kwargs):
        # Simple mock tokenization
        tokens = text.split()[:10]  # Limit for testing
        input_ids = list(range(len(tokens)))
        attention_mask = [1] * len(tokens)
        
        if kwargs.get('return_tensors') == 'pt':
            return {
                'input_ids': torch.tensor([input_ids]),
                'attention_mask': torch.tensor([attention_mask])
            }
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def decode(self, tokens, **kwargs):
        return f"decoded_{len(tokens)}_tokens"


class TestVerilogDataCollator:
    """Test cases for VerilogDataCollator"""
    
    @pytest.fixture
    def tokenizer(self):
        """Mock tokenizer for testing"""
        return MockTokenizer()
    
    @pytest.fixture
    def collator(self, tokenizer):
        """Create data collator for testing"""
        return VerilogDataCollator(tokenizer=tokenizer, max_length=50)
    
    @pytest.fixture
    def sample_data(self):
        """Sample training data"""
        return [
            {
                'instruction': 'Create a 4-bit counter',
                'output': 'module counter(input clk, output reg [3:0] count); endmodule'
            },
            {
                'instruction': 'Design an ALU',
                'output': 'module alu(input [7:0] a, b, output [7:0] result); endmodule'
            }
        ]
    
    def test_collator_initialization(self, collator):
        """Test collator initialization"""
        assert collator is not None
        assert collator.tokenizer is not None
        assert collator.max_length == 50
        assert collator.ignore_index == -100
    
    def test_format_instruction(self, collator):
        """Test instruction formatting"""
        instruction = "Create a counter"
        response = "module counter(); endmodule"
        
        formatted = collator.format_instruction(instruction, response)
        
        assert "### Instruction:" in formatted
        assert "### Response:" in formatted
        assert instruction in formatted
        assert response in formatted
    
    def test_tokenize_example(self, collator):
        """Test example tokenization"""
        example = {
            'instruction': 'Create a counter',
            'output': 'module counter(); endmodule'
        }
        
        result = collator.tokenize_example(example)
        
        assert 'input_ids' in result
        assert 'attention_mask' in result
        assert 'labels' in result
        assert isinstance(result['input_ids'], torch.Tensor)
        assert isinstance(result['attention_mask'], torch.Tensor)
        assert isinstance(result['labels'], torch.Tensor)
    
    def test_batch_collation(self, collator, sample_data):
        """Test batch collation"""
        batch = collator(sample_data)
        
        assert 'input_ids' in batch
        assert 'attention_mask' in batch
        assert 'labels' in batch
        
        # Check batch dimensions
        assert batch['input_ids'].shape[0] == len(sample_data)
        assert batch['attention_mask'].shape[0] == len(sample_data)
        assert batch['labels'].shape[0] == len(sample_data)
        
        # Check that sequences have same length (padded)
        assert batch['input_ids'].shape[1] == batch['attention_mask'].shape[1]
        assert batch['input_ids'].shape[1] == batch['labels'].shape[1]


class TestVerilogSyntaxValidator:
    """Test cases for VerilogSyntaxValidator"""
    
    @pytest.fixture
    def validator(self):
        """Create syntax validator for testing"""
        return VerilogSyntaxValidator()
    
    def test_validator_initialization(self, validator):
        """Test validator initialization"""
        assert validator is not None
        assert hasattr(validator, 'verilog_keywords')
        assert 'module' in validator.verilog_keywords
        assert 'endmodule' in validator.verilog_keywords
    
    def test_valid_verilog_validation(self, validator):
        """Test validation of valid Verilog"""
        valid_code = '''module test(
            input clk,
            output reg out
        );
        always @(posedge clk) begin
            out <= ~out;
        end
        endmodule'''
        
        result = validator.validate_basic_syntax(valid_code)
        
        assert result['is_valid'] is True
        assert result['score'] > 0.5
        assert len(result['errors']) == 0
    
    def test_invalid_verilog_validation(self, validator):
        """Test validation of invalid Verilog"""
        invalid_code = "this is not verilog code at all"
        
        result = validator.validate_basic_syntax(invalid_code)
        
        assert result['is_valid'] is False
        assert result['score'] == 0.0
        assert len(result['errors']) > 0
    
    def test_callable_interface(self, validator):
        """Test callable interface"""
        valid_code = "module test(); endmodule"
        invalid_code = "not verilog"
        
        assert validator(valid_code) is True
        assert validator(invalid_code) is False


if __name__ == "__main__":
    pytest.main([__file__])