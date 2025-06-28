#!/usr/bin/env python3
"""
Thunder Compute Launch Script for DeepSeek R1 Verilog Fine-tuning
Handles resource allocation and multi-node distributed training setup.
"""

import os
import sys
import json
import subprocess
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThunderLauncher:
    """Launcher for Thunder Compute distributed training"""
    
    def __init__(self, 
                 nodes: int = 1,
                 gpus_per_node: int = 8,
                 memory_per_gpu: int = 80,  # GB
                 timeout_hours: int = 48):
        
        self.nodes = nodes
        self.gpus_per_node = gpus_per_node
        self.total_gpus = nodes * gpus_per_node
        self.memory_per_gpu = memory_per_gpu
        self.timeout_hours = timeout_hours
        
        # Thunder-specific configurations - UPDATE THESE FOR YOUR CLUSTER
        self.partition = "gpu"  # Configure based on your Thunder cluster partition names
        self.qos = "normal"     # Configure based on your Thunder cluster QoS policies
        
    def generate_slurm_script(self, 
                             job_name: str,
                             training_command: str,
                             output_dir: str,
                             log_dir: str) -> str:
        """Generate SLURM job script for Thunder Compute"""
        
        script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={self.partition}
#SBATCH --qos={self.qos}
#SBATCH --nodes={self.nodes}
#SBATCH --ntasks-per-node={self.gpus_per_node}
#SBATCH --gres=gpu:{self.gpus_per_node}
#SBATCH --mem={self.memory_per_gpu * self.gpus_per_node}G
#SBATCH --time={self.timeout_hours:02d}:00:00
#SBATCH --output={log_dir}/slurm_%j.out
#SBATCH --error={log_dir}/slurm_%j.err
#SBATCH --exclusive

# Environment setup
echo "Job started at: $(date)"
echo "Node: $SLURM_NODEID"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: {self.gpus_per_node}"
echo "Total GPUs: {self.total_gpus}"

# Load modules - UPDATE THESE FOR YOUR THUNDER ENVIRONMENT
module load cuda/12.1
module load python/3.10
module load nccl/2.18

# Activate conda environment
source ~/.bashrc
conda activate deepseek_verilog

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

# Set distributed training environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE={self.total_gpus}
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "World size: $WORLD_SIZE"
echo "Rank: $RANK"
echo "Local rank: $LOCAL_RANK"

# Create output directories
mkdir -p {output_dir}
mkdir -p {log_dir}

# Monitor GPU usage
nvidia-smi &

# Run training command
echo "Starting training command:"
echo "{training_command}"
echo ""

{training_command}

# Job completion
exit_code=$?
echo "Job completed at: $(date)"
echo "Exit code: $exit_code"

exit $exit_code
"""
        return script_content
        
    def launch_training(self,
                       phase: str,
                       model_name: str,
                       project_dir: str,
                       run_name: str,
                       additional_args: Optional[List[str]] = None) -> str:
        """Launch training job on Thunder Compute"""
        
        project_path = Path(project_dir)
        output_dir = project_path / "checkpoints"
        log_dir = project_path / "logs" / "thunder"
        
        # Create directories
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate job name
        job_name = f"deepseek-verilog-{phase}-{run_name}"
        
        # Build training command
        script_path = project_path / "scripts" / "train_thunder.sh"
        
        training_command = f"bash {script_path}"
        training_command += f" --phase {phase}"
        training_command += f" --model-name '{model_name}'"
        training_command += f" --num-gpus {self.total_gpus}"
        training_command += f" --wandb-project deepseek-verilog-thunder"
        training_command += f" --run-name {run_name}"
        training_command += f" --output-base {output_dir}"
        
        if additional_args:
            training_command += " " + " ".join(additional_args)
            
        # Generate SLURM script
        slurm_script = self.generate_slurm_script(
            job_name=job_name,
            training_command=training_command,
            output_dir=str(output_dir),
            log_dir=str(log_dir)
        )
        
        # Save SLURM script
        script_file = log_dir / f"submit_{job_name}.sh"
        with open(script_file, 'w') as f:
            f.write(slurm_script)
            
        # Make script executable
        os.chmod(script_file, 0o755)
        
        logger.info(f"SLURM script generated: {script_file}")
        logger.info(f"Job configuration:")
        logger.info(f"  Nodes: {self.nodes}")
        logger.info(f"  GPUs per node: {self.gpus_per_node}")
        logger.info(f"  Total GPUs: {self.total_gpus}")
        logger.info(f"  Memory per GPU: {self.memory_per_gpu}GB")
        logger.info(f"  Timeout: {self.timeout_hours} hours")
        
        return str(script_file)
        
    def submit_job(self, script_file: str) -> Optional[str]:
        """Submit job to SLURM scheduler"""
        
        try:
            result = subprocess.run(
                ["sbatch", script_file],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Extract job ID from sbatch output
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines:
                if "Submitted batch job" in line:
                    job_id = line.split()[-1]
                    logger.info(f"Job submitted successfully: {job_id}")
                    return job_id
                    
            logger.warning("Could not extract job ID from sbatch output")
            return None
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to submit job: {e}")
            logger.error(f"Error output: {e.stderr}")
            return None
            
    def monitor_job(self, job_id: str):
        """Monitor job status"""
        
        logger.info(f"Monitoring job {job_id}")
        logger.info("Use the following commands to monitor:")
        logger.info(f"  squeue -j {job_id}")
        logger.info(f"  sacct -j {job_id}")
        logger.info(f"  scancel {job_id}  (to cancel)")
        
    def estimate_resources(self, model_size: str = "32B") -> Dict[str, float]:
        """Estimate resource requirements"""
        
        # Rough estimates for different model sizes
        estimates = {
            "7B": {"memory_per_gpu": 40, "training_time_hours": 24},
            "13B": {"memory_per_gpu": 60, "training_time_hours": 36},
            "32B": {"memory_per_gpu": 80, "training_time_hours": 48},
            "70B": {"memory_per_gpu": 80, "training_time_hours": 72}
        }
        
        estimate = estimates.get(model_size, estimates["32B"])
        
        logger.info(f"Resource estimates for {model_size} model:")
        logger.info(f"  Memory per GPU: {estimate['memory_per_gpu']}GB")
        logger.info(f"  Estimated training time: {estimate['training_time_hours']} hours")
        logger.info(f"  Recommended GPUs: {self.total_gpus}")
        
        return estimate


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Launch DeepSeek R1 Verilog training on Thunder Compute")
    
    # Resource arguments
    parser.add_argument("--nodes", type=int, default=1,
                       help="Number of compute nodes")
    parser.add_argument("--gpus-per-node", type=int, default=8,
                       help="Number of GPUs per node")
    parser.add_argument("--memory-per-gpu", type=int, default=80,
                       help="Memory per GPU in GB")
    parser.add_argument("--timeout-hours", type=int, default=48,
                       help="Job timeout in hours")
    
    # Training arguments
    parser.add_argument("--phase", type=str, default="both",
                       choices=["sft", "ppo", "both"],
                       help="Training phase")
    parser.add_argument("--model-name", type=str,
                       default="deepseek-ai/deepseek-r1-distill-qwen-32b",
                       help="Base model name")
    parser.add_argument("--run-name", type=str,
                       default=None,
                       help="Run name for tracking")
    parser.add_argument("--project-dir", type=str,
                       default=".",
                       help="Project directory")
    
    # Job management
    parser.add_argument("--submit", action="store_true",
                       help="Submit job immediately")
    parser.add_argument("--dry-run", action="store_true",
                       help="Generate scripts but don't submit")
    
    return parser.parse_args()


def main():
    """Main launcher function"""
    args = parse_args()
    
    # Set default run name
    if args.run_name is None:
        import datetime
        args.run_name = f"thunder-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    logger.info("=" * 80)
    logger.info("DEEPSEEK R1 VERILOG THUNDER COMPUTE LAUNCHER")
    logger.info("=" * 80)
    
    # Create launcher
    launcher = ThunderLauncher(
        nodes=args.nodes,
        gpus_per_node=args.gpus_per_node,
        memory_per_gpu=args.memory_per_gpu,
        timeout_hours=args.timeout_hours
    )
    
    # Estimate resources
    launcher.estimate_resources()
    
    # Generate launch script
    script_file = launcher.launch_training(
        phase=args.phase,
        model_name=args.model_name,
        project_dir=args.project_dir,
        run_name=args.run_name
    )
    
    if args.dry_run:
        logger.info("Dry run completed. Script generated but not submitted.")
        logger.info(f"To submit manually: sbatch {script_file}")
        return
        
    if args.submit:
        # Submit job
        job_id = launcher.submit_job(script_file)
        
        if job_id:
            launcher.monitor_job(job_id)
        else:
            logger.error("Job submission failed")
            sys.exit(1)
    else:
        logger.info("Job script generated. Use --submit to submit immediately.")
        logger.info(f"To submit manually: sbatch {script_file}")


if __name__ == "__main__":
    main()