#!/usr/bin/env python3
"""
Verilog Reward Model for Reinforcement Learning
Evaluates Verilog code quality for PPO training.
"""

import re
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VerilogRewardConfig:
    """Configuration for Verilog reward computation"""
    syntax_weight: float = 0.3
    functional_weight: float = 0.4
    quality_weight: float = 0.2
    adherence_weight: float = 0.1
    
    # Syntax checking
    use_iverilog: bool = True
    use_verilator: bool = False
    
    # Quality metrics
    check_indentation: bool = True
    check_comments: bool = True
    check_naming: bool = True
    check_structure: bool = True
    
    # Penalty settings
    syntax_error_penalty: float = -1.0
    incomplete_penalty: float = -0.5
    quality_bonus: float = 0.2


class VerilogSyntaxChecker:
    """Verilog syntax validation using external tools"""
    
    def __init__(self, config: VerilogRewardConfig):
        self.config = config
        self.iverilog_available = self._check_iverilog()
        self.verilator_available = self._check_verilator()
        
    def _check_iverilog(self) -> bool:
        """Check if iverilog is available"""
        try:
            result = subprocess.run(['iverilog', '-V'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
            
    def _check_verilator(self) -> bool:
        """Check if verilator is available"""
        try:
            result = subprocess.run(['verilator', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
            
    def check_syntax(self, verilog_code: str) -> Dict[str, Any]:
        """Check Verilog syntax using available tools"""
        results = {
            'is_valid': False,
            'errors': [],
            'warnings': [],
            'score': 0.0,
            'tool_used': None
        }
        
        # Basic syntax checks first
        basic_check = self._basic_syntax_check(verilog_code)
        if not basic_check['is_valid']:
            results.update(basic_check)
            return results
            
        # Try external tools
        if self.config.use_iverilog and self.iverilog_available:
            tool_result = self._check_with_iverilog(verilog_code)
            if tool_result is not None:
                results.update(tool_result)
                results['tool_used'] = 'iverilog'
                return results
                
        if self.config.use_verilator and self.verilator_available:
            tool_result = self._check_with_verilator(verilog_code)
            if tool_result is not None:
                results.update(tool_result)
                results['tool_used'] = 'verilator'
                return results
                
        # Fallback to basic check
        results.update(basic_check)
        results['tool_used'] = 'basic'
        return results
        
    def _basic_syntax_check(self, verilog_code: str) -> Dict[str, Any]:
        """Basic syntax validation without external tools"""
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'score': 1.0
        }
        
        try:
            # Check for module declaration
            if not re.search(r'\bmodule\s+\w+', verilog_code, re.IGNORECASE):
                results['errors'].append("No module declaration found")
                results['is_valid'] = False
                
            # Check for endmodule
            if not re.search(r'\bendmodule\b', verilog_code, re.IGNORECASE):
                results['errors'].append("No endmodule found")
                results['is_valid'] = False
                
            # Check balanced parentheses
            paren_count = verilog_code.count('(') - verilog_code.count(')')
            if paren_count != 0:
                results['errors'].append(f"Unbalanced parentheses: {paren_count}")
                results['is_valid'] = False
                
            # Check balanced begin/end
            begin_count = len(re.findall(r'\bbegin\b', verilog_code, re.IGNORECASE))
            end_count = len(re.findall(r'\bend\b', verilog_code, re.IGNORECASE))
            end_count -= len(re.findall(r'\bendmodule\b', verilog_code, re.IGNORECASE))  # Exclude endmodule
            
            if begin_count != end_count:
                results['warnings'].append(f"Unbalanced begin/end: {begin_count} begins, {end_count} ends")
                results['score'] *= 0.8
                
            # Check for basic Verilog constructs
            verilog_keywords = ['input', 'output', 'wire', 'reg', 'always', 'assign']
            keyword_count = sum(1 for keyword in verilog_keywords 
                              if re.search(rf'\b{keyword}\b', verilog_code, re.IGNORECASE))
            
            if keyword_count < 2:
                results['warnings'].append("Few Verilog keywords detected")
                results['score'] *= 0.9
                
            if results['errors']:
                results['is_valid'] = False
                results['score'] = 0.0
                
        except Exception as e:
            results['errors'].append(f"Syntax check error: {str(e)}")
            results['is_valid'] = False
            results['score'] = 0.0
            
        return results
        
    def _check_with_iverilog(self, verilog_code: str) -> Optional[Dict[str, Any]]:
        """Check syntax using iverilog"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as f:
                f.write(verilog_code)
                temp_file = f.name
                
            # Run iverilog
            result = subprocess.run([
                'iverilog', '-t', 'null', temp_file
            ], capture_output=True, text=True, timeout=30)
            
            # Clean up
            Path(temp_file).unlink()
            
            is_valid = result.returncode == 0
            errors = []
            warnings = []
            
            if result.stderr:
                lines = result.stderr.strip().split('\n')
                for line in lines:
                    if 'error:' in line.lower():
                        errors.append(line.strip())
                    elif 'warning:' in line.lower():
                        warnings.append(line.strip())
                        
            score = 1.0 if is_valid else 0.0
            if warnings:
                score *= (1.0 - 0.1 * len(warnings))  # Reduce score for warnings
                
            return {
                'is_valid': is_valid,
                'errors': errors,
                'warnings': warnings,
                'score': max(0.0, score)
            }
            
        except Exception as e:
            logger.debug(f"iverilog check failed: {e}")
            return None
            
    def _check_with_verilator(self, verilog_code: str) -> Optional[Dict[str, Any]]:
        """Check syntax using verilator"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as f:
                f.write(verilog_code)
                temp_file = f.name
                
            # Run verilator
            result = subprocess.run([
                'verilator', '--lint-only', temp_file
            ], capture_output=True, text=True, timeout=30)
            
            # Clean up
            Path(temp_file).unlink()
            
            is_valid = result.returncode == 0
            errors = []
            warnings = []
            
            if result.stderr:
                lines = result.stderr.strip().split('\n')
                for line in lines:
                    if '%Error:' in line:
                        errors.append(line.strip())
                    elif '%Warning:' in line:
                        warnings.append(line.strip())
                        
            score = 1.0 if is_valid else 0.0
            if warnings:
                score *= (1.0 - 0.05 * len(warnings))  # Small penalty for warnings
                
            return {
                'is_valid': is_valid,
                'errors': errors,
                'warnings': warnings,
                'score': max(0.0, score)
            }
            
        except Exception as e:
            logger.debug(f"verilator check failed: {e}")
            return None


class VerilogFunctionalChecker:
    """Check functional correctness of Verilog code"""
    
    def __init__(self, config: VerilogRewardConfig):
        self.config = config
        
    def check_functionality(self, verilog_code: str, specification: str) -> Dict[str, Any]:
        """Check if Verilog code meets functional specification"""
        results = {
            'is_functional': False,
            'score': 0.0,
            'issues': []
        }
        
        try:
            # Extract expected functionality from specification
            expected_features = self._extract_expected_features(specification)
            
            # Check if code implements expected features
            implemented_features = self._extract_implemented_features(verilog_code)
            
            # Compare features
            missing_features = expected_features - implemented_features
            extra_features = implemented_features - expected_features
            
            # Calculate functional score
            if expected_features:
                match_ratio = len(expected_features & implemented_features) / len(expected_features)
                results['score'] = match_ratio
                results['is_functional'] = match_ratio >= 0.7  # Threshold
                
                if missing_features:
                    results['issues'].append(f"Missing features: {missing_features}")
                if extra_features:
                    results['issues'].append(f"Extra features: {extra_features}")
            else:
                # If no expected features extracted, give partial credit
                results['score'] = 0.5
                results['is_functional'] = len(implemented_features) > 0
                
        except Exception as e:
            logger.debug(f"Functional check failed: {e}")
            results['issues'].append(f"Functional check error: {str(e)}")
            
        return results
        
    def _extract_expected_features(self, specification: str) -> set:
        """Extract expected features from specification text"""
        features = set()
        
        spec_lower = specification.lower()
        
        # Common Verilog features
        feature_patterns = {
            'counter': r'\bcounter\b',
            'adder': r'\badder?\b',
            'multiplexer': r'\b(mux|multiplexer)\b',
            'decoder': r'\bdecoder\b',
            'register': r'\bregister\b',
            'memory': r'\bmemory\b',
            'fifo': r'\bfifo\b',
            'state_machine': r'\b(state\s+machine|fsm)\b',
            'alu': r'\balu\b',
            'clock': r'\b(clock|clk)\b',
            'reset': r'\breset\b',
            'enable': r'\benable\b'
        }
        
        for feature, pattern in feature_patterns.items():
            if re.search(pattern, spec_lower):
                features.add(feature)
                
        return features
        
    def _extract_implemented_features(self, verilog_code: str) -> set:
        """Extract implemented features from Verilog code"""
        features = set()
        
        code_lower = verilog_code.lower()
        
        # Look for common patterns in the code
        if re.search(r'\bcount\w*\s*<=.*\+', code_lower):
            features.add('counter')
        if re.search(r'\+', code_lower) and 'assign' in code_lower:
            features.add('adder')
        if re.search(r'\bcase\b.*\bendcase\b', code_lower, re.DOTALL):
            features.add('multiplexer')
        if re.search(r'\balways\s*@.*posedge', code_lower):
            features.add('clock')
        if re.search(r'\breset\b', code_lower):
            features.add('reset')
        if re.search(r'\benable\b', code_lower):
            features.add('enable')
        if re.search(r'\breg\s*\[.*\]', code_lower):
            features.add('register')
            
        return features


class VerilogQualityAnalyzer:
    """Analyze Verilog code quality"""
    
    def __init__(self, config: VerilogRewardConfig):
        self.config = config
        
    def analyze_quality(self, verilog_code: str) -> Dict[str, Any]:
        """Analyze code quality"""
        results = {
            'quality_score': 0.0,
            'issues': [],
            'bonuses': []
        }
        
        score = 1.0
        
        try:
            if self.config.check_indentation:
                indent_score = self._check_indentation(verilog_code)
                score *= indent_score
                if indent_score < 1.0:
                    results['issues'].append("Poor indentation")
                    
            if self.config.check_comments:
                comment_score = self._check_comments(verilog_code)
                if comment_score > 1.0:
                    results['bonuses'].append("Good commenting")
                score *= comment_score
                
            if self.config.check_naming:
                naming_score = self._check_naming(verilog_code)
                score *= naming_score
                if naming_score < 1.0:
                    results['issues'].append("Poor naming conventions")
                    
            if self.config.check_structure:
                structure_score = self._check_structure(verilog_code)
                score *= structure_score
                if structure_score < 1.0:
                    results['issues'].append("Poor code structure")
                    
            results['quality_score'] = max(0.0, min(1.0 + self.config.quality_bonus, score))
            
        except Exception as e:
            logger.debug(f"Quality analysis failed: {e}")
            results['issues'].append(f"Quality analysis error: {str(e)}")
            results['quality_score'] = 0.5  # Neutral score on error
            
        return results
        
    def _check_indentation(self, verilog_code: str) -> float:
        """Check code indentation quality"""
        lines = verilog_code.split('\n')
        total_lines = len([line for line in lines if line.strip()])
        
        if total_lines < 5:
            return 1.0  # Too short to judge
            
        indented_lines = 0
        for line in lines:
            if line.strip():  # Non-empty line
                if line.startswith('    ') or line.startswith('\t'):
                    indented_lines += 1
                    
        indentation_ratio = indented_lines / total_lines
        
        # Expect at least 30% of lines to be indented
        if indentation_ratio >= 0.3:
            return 1.0
        elif indentation_ratio >= 0.1:
            return 0.8
        else:
            return 0.6
            
    def _check_comments(self, verilog_code: str) -> float:
        """Check commenting quality"""
        lines = verilog_code.split('\n')
        total_lines = len([line for line in lines if line.strip()])
        
        if total_lines < 10:
            return 1.0  # Too short to require comments
            
        comment_lines = 0
        for line in lines:
            if '//' in line or '/*' in line:
                comment_lines += 1
                
        comment_ratio = comment_lines / total_lines
        
        if comment_ratio >= 0.1:  # At least 10% commented
            return 1.1  # Bonus for good commenting
        elif comment_ratio > 0:
            return 1.0
        else:
            return 0.9  # Small penalty for no comments
            
    def _check_naming(self, verilog_code: str) -> float:
        """Check naming conventions"""
        # Look for meaningful signal names
        signal_names = re.findall(r'\b(input|output|wire|reg)\s+(?:\[.*?\]\s+)?(\w+)', 
                                verilog_code, re.IGNORECASE)
        
        if not signal_names:
            return 1.0
            
        meaningful_names = 0
        for _, name in signal_names:
            if len(name) >= 3 and not name.lower() in ['a', 'b', 'c', 'x', 'y', 'z']:
                meaningful_names += 1
                
        naming_ratio = meaningful_names / len(signal_names)
        
        if naming_ratio >= 0.8:
            return 1.0
        elif naming_ratio >= 0.5:
            return 0.9
        else:
            return 0.8
            
    def _check_structure(self, verilog_code: str) -> float:
        """Check code structure quality"""
        score = 1.0
        
        # Check for proper module structure
        if not re.search(r'module\s+\w+\s*\(', verilog_code, re.IGNORECASE):
            score *= 0.8
            
        # Check for port declarations
        if not re.search(r'(input|output|inout)', verilog_code, re.IGNORECASE):
            score *= 0.9
            
        # Check for proper statement termination
        lines = [line.strip() for line in verilog_code.split('\n') if line.strip()]
        statements = [line for line in lines if not line.startswith('//') and 
                     not line.startswith('/*') and not line.endswith('*/')]
        
        if statements:
            terminated_statements = sum(1 for stmt in statements 
                                      if stmt.endswith(';') or stmt.endswith(')') or 
                                         any(keyword in stmt.lower() for keyword in 
                                             ['begin', 'end', 'module', 'endmodule', 'case', 'endcase']))
            
            termination_ratio = terminated_statements / len(statements)
            if termination_ratio < 0.7:
                score *= 0.8
                
        return score


class VerilogRewardModel:
    """Main reward model for Verilog code evaluation"""
    
    def __init__(self, config: Optional[VerilogRewardConfig] = None):
        self.config = config or VerilogRewardConfig()
        self.syntax_checker = VerilogSyntaxChecker(self.config)
        self.functional_checker = VerilogFunctionalChecker(self.config)
        self.quality_analyzer = VerilogQualityAnalyzer(self.config)
        
    def compute_reward(self, verilog_code: str, specification: str) -> Dict[str, Any]:
        """Compute comprehensive reward for Verilog code"""
        
        results = {
            'total_reward': 0.0,
            'syntax_score': 0.0,
            'functional_score': 0.0,
            'quality_score': 0.0,
            'adherence_score': 0.0,
            'breakdown': {},
            'issues': []
        }
        
        try:
            # Syntax evaluation
            syntax_results = self.syntax_checker.check_syntax(verilog_code)
            results['syntax_score'] = syntax_results['score']
            results['breakdown']['syntax'] = syntax_results
            
            if not syntax_results['is_valid']:
                # Heavy penalty for syntax errors
                results['total_reward'] = self.config.syntax_error_penalty
                results['issues'].extend(syntax_results['errors'])
                return results
                
            # Functional evaluation
            functional_results = self.functional_checker.check_functionality(verilog_code, specification)
            results['functional_score'] = functional_results['score']
            results['breakdown']['functional'] = functional_results
            results['issues'].extend(functional_results['issues'])
            
            # Quality evaluation
            quality_results = self.quality_analyzer.analyze_quality(verilog_code)
            results['quality_score'] = quality_results['quality_score']
            results['breakdown']['quality'] = quality_results
            results['issues'].extend(quality_results['issues'])
            
            # Specification adherence (simple keyword matching for now)
            adherence_score = self._compute_adherence_score(verilog_code, specification)
            results['adherence_score'] = adherence_score
            
            # Compute weighted total reward
            total_reward = (
                results['syntax_score'] * self.config.syntax_weight +
                results['functional_score'] * self.config.functional_weight +
                results['quality_score'] * self.config.quality_weight +
                results['adherence_score'] * self.config.adherence_weight
            )
            
            results['total_reward'] = total_reward
            
        except Exception as e:
            logger.error(f"Reward computation failed: {e}")
            results['total_reward'] = self.config.syntax_error_penalty
            results['issues'].append(f"Reward computation error: {str(e)}")
            
        return results
        
    def _compute_adherence_score(self, verilog_code: str, specification: str) -> float:
        """Compute how well code adheres to specification"""
        try:
            # Simple keyword matching approach
            spec_words = set(re.findall(r'\b\w+\b', specification.lower()))
            code_words = set(re.findall(r'\b\w+\b', verilog_code.lower()))
            
            # Remove common words
            common_words = {
                'the', 'and', 'or', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been'
            }
            spec_words -= common_words
            code_words -= common_words
            
            if not spec_words:
                return 0.5  # Neutral score if no meaningful words
                
            # Compute overlap
            overlap = len(spec_words & code_words)
            adherence_score = overlap / len(spec_words)
            
            return min(1.0, adherence_score)
            
        except Exception as e:
            logger.debug(f"Adherence computation failed: {e}")
            return 0.5
            
    def __call__(self, verilog_code: str, specification: str) -> float:
        """Simple callable interface that returns just the total reward"""
        return self.compute_reward(verilog_code, specification)['total_reward']


def create_reward_model(config: Optional[VerilogRewardConfig] = None) -> VerilogRewardModel:
    """Factory function to create reward model"""
    return VerilogRewardModel(config)


def main():
    """Test reward model"""
    logger.info("Testing Verilog reward model...")
    
    # Test code
    test_code = '''module counter_4bit(
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
    
    test_spec = "Create a 4-bit counter with clock and reset inputs"
    
    reward_model = VerilogRewardModel()
    results = reward_model.compute_reward(test_code, test_spec)
    
    logger.info("Reward computation results:")
    for key, value in results.items():
        if key != 'breakdown':
            logger.info(f"  {key}: {value}")
            
    logger.info(f"Total reward: {results['total_reward']:.3f}")


if __name__ == "__main__":
    main()