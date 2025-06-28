#!/usr/bin/env python3
"""
Verilog Dataset Preprocessor
Processes raw Verilog datasets into instruction-following format for fine-tuning.
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VerilogPreprocessor:
    def __init__(self, raw_data_dir: str = "data/raw", processed_data_dir: str = "data/processed"):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Verilog keywords and patterns for quality filtering
        self.verilog_keywords = {
            'module', 'endmodule', 'input', 'output', 'inout', 'wire', 'reg',
            'always', 'initial', 'assign', 'if', 'else', 'case', 'endcase',
            'for', 'while', 'begin', 'end', 'posedge', 'negedge', 'and', 'or',
            'not', 'xor', 'nand', 'nor', 'xnor'
        }
        
    def extract_module_info(self, verilog_code: str) -> Optional[Dict]:
        """Extract module name, ports, and description from Verilog code"""
        try:
            # Find module declaration
            module_pattern = r'module\s+(\w+)\s*(?:\#.*?)?\s*\((.*?)\)\s*;'
            module_match = re.search(module_pattern, verilog_code, re.DOTALL)
            
            if not module_match:
                return None
                
            module_name = module_match.group(1)
            ports_str = module_match.group(2)
            
            # Extract ports
            ports = self._parse_ports(verilog_code, ports_str)
            
            # Extract comments as description
            description = self._extract_description(verilog_code)
            
            return {
                'module_name': module_name,
                'ports': ports,
                'description': description,
                'code': verilog_code.strip()
            }
            
        except Exception as e:
            logger.debug(f"Failed to extract module info: {e}")
            return None
            
    def _parse_ports(self, full_code: str, ports_str: str) -> Dict:
        """Parse port declarations"""
        ports = {'input': [], 'output': [], 'inout': []}
        
        # Look for port declarations in the module body
        port_patterns = [
            r'input\s+(?:\[.*?\]\s+)?(\w+)',
            r'output\s+(?:\[.*?\]\s+)?(\w+)',
            r'inout\s+(?:\[.*?\]\s+)?(\w+)'
        ]
        
        for pattern in port_patterns:
            port_type = pattern.split(r'\s+')[0]
            matches = re.findall(pattern, full_code)
            ports[port_type].extend(matches)
            
        return ports
        
    def _extract_description(self, verilog_code: str) -> str:
        """Extract description from comments"""
        # Look for comments at the beginning of the file
        comment_patterns = [
            r'//\s*(.*?)(?=module|\n\s*$)',
            r'/\*\s*(.*?)\s*\*/',
        ]
        
        descriptions = []
        for pattern in comment_patterns:
            matches = re.findall(pattern, verilog_code, re.DOTALL)
            descriptions.extend(matches)
            
        if descriptions:
            # Clean up the description
            desc = ' '.join(descriptions).strip()
            desc = re.sub(r'\s+', ' ', desc)
            return desc
            
        # Generate basic description from module name
        module_match = re.search(r'module\s+(\w+)', verilog_code)
        if module_match:
            module_name = module_match.group(1)
            return f"Verilog module implementing {module_name}"
            
        return "Verilog hardware description"
        
    def is_valid_verilog(self, code: str) -> bool:
        """Check if code is valid Verilog"""
        if len(code.strip()) < 20:
            return False
            
        # Must have module declaration
        if not re.search(r'module\s+\w+', code):
            return False
            
        # Must have endmodule
        if 'endmodule' not in code:
            return False
            
        # Should contain some Verilog keywords
        keyword_count = sum(1 for keyword in self.verilog_keywords if keyword in code.lower())
        if keyword_count < 3:
            return False
            
        # Check for balanced begin/end
        begin_count = len(re.findall(r'\bbegin\b', code, re.IGNORECASE))
        end_count = len(re.findall(r'\bend\b', code, re.IGNORECASE))
        if begin_count != end_count:
            return False
            
        return True
        
    def process_verilog_eval(self) -> List[Dict]:
        """Process VerilogEval dataset"""
        logger.info("Processing VerilogEval dataset...")
        dataset = []
        
        verilog_eval_dir = self.raw_data_dir / "verilog-eval"
        if not verilog_eval_dir.exists():
            logger.warning("VerilogEval directory not found")
            return dataset
            
        # Look for problem files
        problems_dir = verilog_eval_dir / "problems"
        if problems_dir.exists():
            for problem_file in problems_dir.rglob("*.json"):
                try:
                    with open(problem_file, 'r') as f:
                        problem_data = json.load(f)
                        
                    if 'description' in problem_data and 'solution' in problem_data:
                        entry = {
                            'instruction': problem_data['description'],
                            'output': problem_data['solution'],
                            'source': 'verilog_eval',
                            'difficulty': problem_data.get('difficulty', 'unknown')
                        }
                        dataset.append(entry)
                        
                except Exception as e:
                    logger.debug(f"Failed to process {problem_file}: {e}")
                    
        # Also process Verilog files directly
        for verilog_file in verilog_eval_dir.rglob("*.v"):
            try:
                with open(verilog_file, 'r', encoding='utf-8', errors='ignore') as f:
                    code = f.read()
                    
                if self.is_valid_verilog(code):
                    module_info = self.extract_module_info(code)
                    if module_info:
                        entry = {
                            'instruction': f"Generate Verilog code for: {module_info['description']}",
                            'output': module_info['code'],
                            'source': 'verilog_eval',
                            'module_name': module_info['module_name']
                        }
                        dataset.append(entry)
                        
            except Exception as e:
                logger.debug(f"Failed to process {verilog_file}: {e}")
                
        logger.info(f"Processed {len(dataset)} entries from VerilogEval")
        return dataset
        
    def process_hdlbits(self) -> List[Dict]:
        """Process HDL-Bits dataset"""
        logger.info("Processing HDL-Bits dataset...")
        dataset = []
        
        hdlbits_dir = self.raw_data_dir / "hdlbits"
        if not hdlbits_dir.exists():
            logger.warning("HDL-Bits directory not found")
            return dataset
            
        for verilog_file in hdlbits_dir.rglob("*.v"):
            try:
                with open(verilog_file, 'r', encoding='utf-8', errors='ignore') as f:
                    code = f.read()
                    
                if self.is_valid_verilog(code):
                    module_info = self.extract_module_info(code)
                    if module_info:
                        # Use filename as hint for instruction
                        problem_name = verilog_file.stem.replace('_', ' ').title()
                        instruction = f"Implement a Verilog module for {problem_name}: {module_info['description']}"
                        
                        entry = {
                            'instruction': instruction,
                            'output': module_info['code'],
                            'source': 'hdlbits',
                            'module_name': module_info['module_name']
                        }
                        dataset.append(entry)
                        
            except Exception as e:
                logger.debug(f"Failed to process {verilog_file}: {e}")
                
        logger.info(f"Processed {len(dataset)} entries from HDL-Bits")
        return dataset
        
    def process_openroad(self) -> List[Dict]:
        """Process OpenROAD dataset"""
        logger.info("Processing OpenROAD dataset...")
        dataset = []
        
        openroad_dir = self.raw_data_dir / "OpenROAD"
        if not openroad_dir.exists():
            logger.warning("OpenROAD directory not found")
            return dataset
            
        for verilog_file in openroad_dir.rglob("*.v"):
            try:
                with open(verilog_file, 'r', encoding='utf-8', errors='ignore') as f:
                    code = f.read()
                    
                if self.is_valid_verilog(code):
                    module_info = self.extract_module_info(code)
                    if module_info:
                        entry = {
                            'instruction': f"Create a Verilog module: {module_info['description']}",
                            'output': module_info['code'],
                            'source': 'openroad',
                            'module_name': module_info['module_name']
                        }
                        dataset.append(entry)
                        
            except Exception as e:
                logger.debug(f"Failed to process {verilog_file}: {e}")
                
        logger.info(f"Processed {len(dataset)} entries from OpenROAD")
        return dataset
        
        
    def create_instruction_format(self, entry: Dict) -> Dict:
        """Convert entry to instruction-following format"""
        instruction = entry['instruction']
        output = entry['output']
        
        # Create the instruction-following format
        formatted_entry = {
            'instruction': f"Generate Verilog code for the following specification:\n\n{instruction}",
            'input': "",
            'output': output,
            'source': entry.get('source', 'unknown'),
            'metadata': {
                'module_name': entry.get('module_name'),
                'difficulty': entry.get('difficulty'),
                'original_instruction': instruction
            }
        }
        
        return formatted_entry
        
    def split_dataset(self, dataset: List[Dict], train_ratio: float = 0.8, 
                     val_ratio: float = 0.1, test_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split dataset into train/validation/test sets"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        # Shuffle dataset
        random.shuffle(dataset)
        
        n = len(dataset)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        train_set = dataset[:train_size]
        val_set = dataset[train_size:train_size + val_size]
        test_set = dataset[train_size + val_size:]
        
        return train_set, val_set, test_set
        
    def save_dataset(self, dataset: List[Dict], filename: str):
        """Save dataset to JSON file"""
        output_path = self.processed_data_dir / filename
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        logger.info(f"Saved {len(dataset)} entries to {output_path}")
        
    def process_all_datasets(self):
        """Process all datasets and create train/val/test splits"""
        logger.info("Starting comprehensive dataset processing...")
        
        # Process individual datasets
        all_datasets = []
        
        verilog_eval_data = self.process_verilog_eval()
        all_datasets.extend(verilog_eval_data)
        
        hdlbits_data = self.process_hdlbits()
        all_datasets.extend(hdlbits_data)
        
        openroad_data = self.process_openroad()
        all_datasets.extend(openroad_data)
        
        # Convert to instruction format
        formatted_dataset = []
        for entry in tqdm(all_datasets, desc="Formatting entries"):
            try:
                formatted_entry = self.create_instruction_format(entry)
                formatted_dataset.append(formatted_entry)
            except Exception as e:
                logger.debug(f"Failed to format entry: {e}")
                
        logger.info(f"Total formatted entries: {len(formatted_dataset)}")
        
        # Split dataset
        train_set, val_set, test_set = self.split_dataset(formatted_dataset)
        
        # Save datasets
        self.save_dataset(train_set, "train.json")
        self.save_dataset(val_set, "val.json")
        self.save_dataset(test_set, "test.json")
        
        # Save complete dataset
        self.save_dataset(formatted_dataset, "complete_dataset.json")
        
        # Save statistics
        stats = {
            'total_entries': len(formatted_dataset),
            'train_entries': len(train_set),
            'val_entries': len(val_set),
            'test_entries': len(test_set),
            'sources': {source: sum(1 for entry in formatted_dataset if entry['source'] == source) 
                       for source in set(entry['source'] for entry in formatted_dataset)}
        }
        
        with open(self.processed_data_dir / "dataset_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
            
        logger.info("Dataset processing completed!")
        logger.info(f"Statistics: {stats}")
        
        return train_set, val_set, test_set


def main():
    """Main function to process all datasets"""
    preprocessor = VerilogPreprocessor()
    train_set, val_set, test_set = preprocessor.process_all_datasets()
    print("Dataset preprocessing completed!")


if __name__ == "__main__":
    main()