#!/usr/bin/env python3
"""
Tests for Verilog Reward Model
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from training.reward_model import VerilogRewardModel, VerilogRewardConfig


class TestVerilogRewardModel:
    """Test cases for VerilogRewardModel"""
    
    @pytest.fixture
    def reward_model(self):
        """Create a reward model for testing"""
        config = VerilogRewardConfig()
        return VerilogRewardModel(config)
    
    @pytest.fixture
    def valid_verilog_code(self):
        """Sample valid Verilog code"""
        return '''module counter_4bit(
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
    
    @pytest.fixture
    def invalid_verilog_code(self):
        """Sample invalid Verilog code"""
        return '''this is not verilog code
missing module declaration
no proper syntax'''
    
    def test_reward_model_initialization(self, reward_model):
        """Test reward model initialization"""
        assert reward_model is not None
        assert reward_model.config is not None
        assert reward_model.syntax_checker is not None
        assert reward_model.functional_checker is not None
        assert reward_model.quality_analyzer is not None
    
    def test_valid_verilog_reward(self, reward_model, valid_verilog_code):
        """Test reward computation for valid Verilog"""
        specification = "Create a 4-bit counter with clock and reset"
        result = reward_model.compute_reward(valid_verilog_code, specification)
        
        assert 'total_reward' in result
        assert 'syntax_score' in result
        assert 'functional_score' in result
        assert 'quality_score' in result
        assert 'adherence_score' in result
        
        # Valid code should have positive syntax score
        assert result['syntax_score'] > 0.5
        assert result['total_reward'] > 0.0
    
    def test_invalid_verilog_reward(self, reward_model, invalid_verilog_code):
        """Test reward computation for invalid Verilog"""
        specification = "Create a 4-bit counter"
        result = reward_model.compute_reward(invalid_verilog_code, specification)
        
        assert 'total_reward' in result
        # Invalid code should have low or negative reward
        assert result['total_reward'] <= 0.0
        assert result['syntax_score'] <= 0.1
    
    def test_callable_interface(self, reward_model, valid_verilog_code):
        """Test callable interface"""
        specification = "Create a 4-bit counter"
        reward = reward_model(valid_verilog_code, specification)
        
        assert isinstance(reward, float)
        assert reward > 0.0  # Valid code should have positive reward
    
    def test_reward_config_weights(self):
        """Test custom reward configuration"""
        config = VerilogRewardConfig(
            syntax_weight=0.5,
            functional_weight=0.3,
            quality_weight=0.1,
            adherence_weight=0.1
        )
        
        assert config.syntax_weight == 0.5
        assert config.functional_weight == 0.3
        assert config.quality_weight == 0.1
        assert config.adherence_weight == 0.1
        
        reward_model = VerilogRewardModel(config)
        assert reward_model.config.syntax_weight == 0.5


if __name__ == "__main__":
    pytest.main([__file__])