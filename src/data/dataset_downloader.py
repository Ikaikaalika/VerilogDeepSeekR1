#!/usr/bin/env python3
"""
Dataset Downloader for Verilog Code Generation Fine-tuning
Downloads and organizes multiple Verilog datasets for training.
"""

import subprocess
import urllib.request
import zipfile
import json
import logging
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VerilogDatasetDownloader:
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_verilog_eval(self) -> Path:
        """Download VerilogEval benchmark dataset"""
        logger.info("Downloading VerilogEval dataset...")
        repo_dir = self.data_dir / "verilog-eval"
        
        if repo_dir.exists():
            logger.info("VerilogEval already exists, skipping download")
            return repo_dir
            
        try:
            subprocess.run([
                "git", "clone", 
                "https://github.com/NVlabs/verilog-eval.git",
                str(repo_dir)
            ], check=True, cwd=self.data_dir.parent)
            logger.info("VerilogEval downloaded successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download VerilogEval: {e}")
            raise
            
        return repo_dir
        
    def download_hdlbits(self) -> Path:
        """Download HDL-Bits dataset"""
        logger.info("Downloading HDL-Bits dataset...")
        zip_path = self.data_dir / "hdlbits.zip"
        extract_dir = self.data_dir / "hdlbits"
        
        if extract_dir.exists():
            logger.info("HDL-Bits already exists, skipping download")
            return extract_dir
            
        try:
            urllib.request.urlretrieve(
                "https://github.com/alinush/hdlbits-verilog/archive/main.zip",
                zip_path
            )
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
                
            # Rename extracted folder
            extracted_folder = self.data_dir / "hdlbits-verilog-main"
            if extracted_folder.exists():
                extracted_folder.rename(extract_dir)
                
            zip_path.unlink()  # Remove zip file
            logger.info("HDL-Bits downloaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to download HDL-Bits: {e}")
            raise
            
        return extract_dir
        
    def download_openroad(self) -> Path:
        """Download OpenROAD Verilog examples"""
        logger.info("Downloading OpenROAD dataset...")
        repo_dir = self.data_dir / "OpenROAD"
        
        if repo_dir.exists():
            logger.info("OpenROAD already exists, skipping download")
            return repo_dir
            
        try:
            subprocess.run([
                "git", "clone", "--depth", "1",
                "https://github.com/The-OpenROAD-Project/OpenROAD.git",
                str(repo_dir)
            ], check=True, cwd=self.data_dir.parent)
            logger.info("OpenROAD downloaded successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download OpenROAD: {e}")
            raise
            
        return repo_dir
        
    def download_ivtest(self) -> Path:
        """Download Icarus Verilog test suite"""
        logger.info("Downloading ivtest dataset...")
        repo_dir = self.data_dir / "ivtest"
        
        if repo_dir.exists():
            logger.info("ivtest already exists, skipping download")
            return repo_dir
            
        try:
            subprocess.run([
                "git", "clone", "--depth", "1",
                "https://github.com/steveicarus/ivtest.git",
                str(repo_dir)
            ], check=True, cwd=self.data_dir.parent)
            logger.info("ivtest downloaded successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download ivtest: {e}")
            raise
            
        return repo_dir
        
    def download_yosys_examples(self) -> Path:
        """Download Yosys example designs"""
        logger.info("Downloading Yosys examples...")
        repo_dir = self.data_dir / "yosys"
        
        if repo_dir.exists():
            logger.info("Yosys already exists, skipping download")
            return repo_dir
            
        try:
            subprocess.run([
                "git", "clone", "--depth", "1",
                "https://github.com/YosysHQ/yosys.git",
                str(repo_dir)
            ], check=True, cwd=self.data_dir.parent)
            logger.info("Yosys downloaded successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download Yosys: {e}")
            raise
            
        return repo_dir
        
    def find_verilog_files(self, directory: Path) -> List[Path]:
        """Find all Verilog files in a directory"""
        verilog_extensions = ['.v', '.sv', '.vh', '.svh']
        verilog_files = []
        
        for ext in verilog_extensions:
            verilog_files.extend(directory.rglob(f"*{ext}"))
            
        return verilog_files
        
    def create_file_inventory(self) -> Dict[str, List[str]]:
        """Create an inventory of all downloaded Verilog files"""
        inventory = {}
        
        for dataset_dir in self.data_dir.iterdir():
            if dataset_dir.is_dir():
                verilog_files = self.find_verilog_files(dataset_dir)
                inventory[dataset_dir.name] = [str(f) for f in verilog_files]
                
        # Save inventory
        inventory_path = self.data_dir / "file_inventory.json"
        with open(inventory_path, 'w') as f:
            json.dump(inventory, f, indent=2)
            
        logger.info(f"File inventory saved to {inventory_path}")
        return inventory
        
    def download_all(self) -> Dict[str, Path]:
        """Download all datasets"""
        logger.info("Starting download of all Verilog datasets...")
        
        datasets = {}
        
        try:
            datasets['verilog_eval'] = self.download_verilog_eval()
            datasets['hdlbits'] = self.download_hdlbits()
            datasets['openroad'] = self.download_openroad()
            datasets['ivtest'] = self.download_ivtest()
            datasets['yosys'] = self.download_yosys_examples()
            
            # Create file inventory
            inventory = self.create_file_inventory()
            
            logger.info("All datasets downloaded successfully!")
            logger.info(f"Total datasets: {len(datasets)}")
            
            for name, files in inventory.items():
                logger.info(f"  {name}: {len(files)} Verilog files")
                
        except Exception as e:
            logger.error(f"Failed to download datasets: {e}")
            raise
            
        return datasets


def main():
    """Main function to download all datasets"""
    downloader = VerilogDatasetDownloader()
    datasets = downloader.download_all()
    print("Dataset download completed!")
    

if __name__ == "__main__":
    main()