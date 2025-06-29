#!/usr/bin/env python3
"""
Thunder Compute Setup and Management Script
Handles Thunder Compute instance creation, configuration, and job submission.
"""

import subprocess
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThunderManager:
    """Manages Thunder Compute instances for DeepSeek R1 training"""
    
    def __init__(self):
        self.check_tnr_cli()
        
    def check_tnr_cli(self):
        """Check if TNR CLI is installed and authenticated"""
        try:
            result = subprocess.run(['tnr', 'status'], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("TNR CLI not authenticated or not working properly")
                logger.info("Please run: tnr login")
                sys.exit(1)
        except FileNotFoundError:
            logger.error("TNR CLI not found. Please install Thunder Compute CLI first.")
            sys.exit(1)
            
    def create_instance(self, 
                       gpu_type: str = "a100",
                       num_gpus: int = 8,
                       vcpus: int = 16,
                       mode: str = "production") -> str:
        """Create a new Thunder Compute instance"""
        logger.info(f"Creating Thunder instance with {num_gpus}x{gpu_type} GPUs...")
        
        cmd = [
            'tnr', 'create',
            '--gpu', gpu_type,
            '--num-gpus', str(num_gpus),
            '--vcpus', str(vcpus),
            '--mode', mode
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # Extract instance ID from output
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'Instance ID' in line or 'Created instance' in line:
                    # Parse instance ID (typically the last word or number)
                    instance_id = line.split()[-1]
                    logger.info(f"Created Thunder instance: {instance_id}")
                    return instance_id
            
            # Fallback: get status and return first instance
            status_result = subprocess.run(['tnr', 'status'], capture_output=True, text=True)
            if status_result.returncode == 0:
                # Parse status output to get instance ID
                logger.info("Instance created, checking status for ID...")
                return "0"  # Default instance ID
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create instance: {e.stderr}")
            raise
            
    def wait_for_instance(self, instance_id: str, timeout: int = 300):
        """Wait for instance to be running"""
        logger.info(f"Waiting for instance {instance_id} to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                result = subprocess.run(['tnr', 'status'], capture_output=True, text=True)
                if result.returncode == 0 and 'running' in result.stdout.lower():
                    logger.info(f"Instance {instance_id} is ready!")
                    return True
                time.sleep(10)
            except subprocess.CalledProcessError:
                logger.warning("Error checking instance status, retrying...")
                time.sleep(10)
                
        logger.error(f"Instance {instance_id} not ready after {timeout} seconds")
        return False
        
    def setup_training_environment(self, instance_id: str):
        """Setup the training environment on the instance"""
        logger.info(f"Setting up training environment on instance {instance_id}...")
        
        setup_commands = [
            "sudo apt-get update",
            "sudo apt-get install -y git wget",
            "conda create -n deepseek_verilog python=3.10 -y",
            "conda activate deepseek_verilog",
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
            "pip install transformers datasets accelerate deepspeed",
            "pip install wandb peft trl",
            "pip install numpy pandas tqdm",
        ]
        
        for cmd in setup_commands:
            logger.info(f"Running: {cmd}")
            try:
                result = subprocess.run([
                    'tnr', 'connect', instance_id, '--', 'bash', '-c', cmd
                ], capture_output=True, text=True, check=True)
                logger.info("Command completed successfully")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Command failed: {e.stderr}")
                
    def upload_project(self, instance_id: str, local_path: str = "."):
        """Upload project files to the instance"""
        logger.info(f"Uploading project to instance {instance_id}...")
        
        try:
            subprocess.run([
                'tnr', 'scp', '-r', local_path, f'{instance_id}:/workspace/'
            ], check=True)
            logger.info("Project uploaded successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to upload project: {e}")
            raise
            
    def run_training(self, 
                    instance_id: str,
                    phase: str = "both",
                    wandb_project: str = "deepseek-verilog-thunder"):
        """Run training on the Thunder instance"""
        logger.info(f"Starting {phase} training on instance {instance_id}...")
        
        training_cmd = f"""
        cd /workspace && \
        conda activate deepseek_verilog && \
        bash scripts/train_thunder.sh \
            --phase {phase} \
            --wandb-project {wandb_project} \
            --num-gpus $(nvidia-smi -L | wc -l)
        """
        
        try:
            # Run training with port forwarding for monitoring
            result = subprocess.run([
                'tnr', 'connect', instance_id, '-t', '8080', '-t', '6006',
                '--', 'bash', '-c', training_cmd
            ], check=True)
            logger.info("Training completed successfully!")
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e}")
            raise
            
    def download_results(self, instance_id: str, local_path: str = "./results"):
        """Download training results from the instance"""
        logger.info(f"Downloading results from instance {instance_id}...")
        
        try:
            subprocess.run([
                'tnr', 'scp', '-r', 
                f'{instance_id}:/workspace/checkpoints', 
                f'{instance_id}:/workspace/logs',
                local_path
            ], check=True)
            logger.info(f"Results downloaded to {local_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download results: {e}")
            raise
            
    def cleanup_instance(self, instance_id: str, delete: bool = False):
        """Stop or delete the Thunder instance"""
        if delete:
            logger.info(f"Deleting instance {instance_id}...")
            subprocess.run(['tnr', 'delete', instance_id], check=True)
        else:
            logger.info(f"Stopping instance {instance_id}...")
            subprocess.run(['tnr', 'stop', instance_id], check=True)

def main():
    parser = argparse.ArgumentParser(description="Thunder Compute setup for DeepSeek R1 training")
    parser.add_argument('--action', choices=['create', 'setup', 'train', 'download', 'cleanup', 'full'], 
                       default='full', help='Action to perform')
    parser.add_argument('--instance-id', help='Existing instance ID to use')
    parser.add_argument('--gpu-type', default='a100', choices=['t4', 'a100', 'a100xl'],
                       help='GPU type to use')
    parser.add_argument('--num-gpus', type=int, default=8, help='Number of GPUs')
    parser.add_argument('--vcpus', type=int, default=16, help='Number of vCPUs')
    parser.add_argument('--phase', default='both', choices=['sft', 'ppo', 'both'],
                       help='Training phase')
    parser.add_argument('--wandb-project', default='deepseek-verilog-thunder',
                       help='Wandb project name')
    parser.add_argument('--delete-after', action='store_true',
                       help='Delete instance after completion')
    
    args = parser.parse_args()
    
    manager = ThunderManager()
    instance_id = args.instance_id
    
    try:
        if args.action in ['create', 'full']:
            instance_id = manager.create_instance(
                gpu_type=args.gpu_type,
                num_gpus=args.num_gpus,
                vcpus=args.vcpus
            )
            manager.wait_for_instance(instance_id)
            
        if args.action in ['setup', 'full']:
            if not instance_id:
                logger.error("Instance ID required for setup")
                return
            manager.setup_training_environment(instance_id)
            manager.upload_project(instance_id)
            
        if args.action in ['train', 'full']:
            if not instance_id:
                logger.error("Instance ID required for training")
                return
            manager.run_training(instance_id, args.phase, args.wandb_project)
            
        if args.action in ['download', 'full']:
            if not instance_id:
                logger.error("Instance ID required for download")
                return
            manager.download_results(instance_id)
            
        if args.action in ['cleanup', 'full'] or args.delete_after:
            if not instance_id:
                logger.error("Instance ID required for cleanup")
                return
            manager.cleanup_instance(instance_id, delete=args.delete_after)
            
        logger.info("All operations completed successfully!")
        
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        if instance_id and args.delete_after:
            logger.info("Cleaning up instance due to failure...")
            manager.cleanup_instance(instance_id, delete=True)
        sys.exit(1)

if __name__ == "__main__":
    main()