#!/usr/bin/env python3
"""
Training Monitoring Script for DeepSeek R1 Verilog Fine-tuning
Monitors training progress, GPU usage, and model performance.
"""

import sys
import os
import time
import json
import logging
import argparse
import psutil
import GPUtil
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingMonitor:
    """Monitor training progress and system resources"""
    
    def __init__(self, 
                 checkpoint_dir: str,
                 log_dir: str,
                 monitoring_interval: int = 60,
                 save_plots: bool = True):
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.monitoring_interval = monitoring_interval
        self.save_plots = save_plots
        
        # Create monitoring directory
        self.monitor_dir = self.log_dir / "monitoring"
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.metrics_history = []
        self.system_history = []
        
        # Monitoring state
        self.start_time = datetime.now()
        self.is_monitoring = False
        
    def get_system_stats(self) -> Dict:
        """Get current system statistics"""
        
        stats = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'disk_usage_percent': psutil.disk_usage('/').percent
        }
        
        # GPU statistics
        try:
            gpus = GPUtil.getGPUs()
            gpu_stats = []
            for i, gpu in enumerate(gpus):
                gpu_stats.append({
                    'gpu_id': i,
                    'gpu_name': gpu.name,
                    'gpu_load': gpu.load * 100,
                    'gpu_memory_used': gpu.memoryUsed,
                    'gpu_memory_total': gpu.memoryTotal,
                    'gpu_memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'gpu_temperature': gpu.temperature
                })
            stats['gpus'] = gpu_stats
            stats['total_gpu_memory_used'] = sum(gpu['gpu_memory_used'] for gpu in gpu_stats)
            stats['avg_gpu_load'] = sum(gpu['gpu_load'] for gpu in gpu_stats) / len(gpu_stats) if gpu_stats else 0
            
        except Exception as e:
            logger.warning(f"Failed to get GPU stats: {e}")
            stats['gpus'] = []
            stats['total_gpu_memory_used'] = 0
            stats['avg_gpu_load'] = 0
            
        return stats
        
    def get_training_metrics(self) -> Optional[Dict]:
        """Extract training metrics from logs/checkpoints"""
        
        metrics = {}
        
        # Check for trainer_state.json (Transformers)
        trainer_state_file = self.checkpoint_dir / "trainer_state.json"
        if trainer_state_file.exists():
            try:
                with open(trainer_state_file, 'r') as f:
                    trainer_state = json.load(f)
                    
                if 'log_history' in trainer_state:
                    log_history = trainer_state['log_history']
                    if log_history:
                        latest_log = log_history[-1]
                        metrics.update({
                            'step': latest_log.get('step', 0),
                            'epoch': latest_log.get('epoch', 0),
                            'learning_rate': latest_log.get('learning_rate', 0),
                            'train_loss': latest_log.get('train_loss'),
                            'eval_loss': latest_log.get('eval_loss'),
                            'eval_syntax_accuracy': latest_log.get('eval_syntax_accuracy'),
                            'eval_code_quality': latest_log.get('eval_code_quality')
                        })
                        
            except Exception as e:
                logger.debug(f"Failed to read trainer state: {e}")
                
        # Check for wandb logs if available
        if WANDB_AVAILABLE:
            try:
                # This would require wandb API setup
                pass
            except Exception as e:
                logger.debug(f"Failed to get wandb metrics: {e}")
                
        # Check training log files
        log_files = list(self.log_dir.glob("*.log"))
        if log_files:
            latest_log_file = max(log_files, key=lambda x: x.stat().st_mtime)
            try:
                # Parse recent log entries for metrics
                with open(latest_log_file, 'r') as f:
                    lines = f.readlines()
                    
                # Look for recent metric lines
                for line in reversed(lines[-100:]):  # Check last 100 lines
                    if 'loss:' in line.lower() or 'reward:' in line.lower():
                        # Simple parsing - could be improved
                        if 'Step' in line and 'reward' in line:
                            try:
                                parts = line.split()
                                for i, part in enumerate(parts):
                                    if 'reward' in part.lower() and i+1 < len(parts):
                                        metrics['latest_reward'] = float(parts[i+1].strip(','))
                                        break
                            except (ValueError, IndexError):
                                pass
                                
            except Exception as e:
                logger.debug(f"Failed to parse log file: {e}")
                
        return metrics if metrics else None
        
    def save_metrics(self, system_stats: Dict, training_metrics: Optional[Dict]):
        """Save metrics to files"""
        
        # Add timestamp
        timestamp = datetime.now().isoformat()
        
        # Store system stats
        system_entry = {'timestamp': timestamp, **system_stats}
        self.system_history.append(system_entry)
        
        # Store training metrics
        if training_metrics:
            metrics_entry = {'timestamp': timestamp, **training_metrics}
            self.metrics_history.append(metrics_entry)
            
        # Save to files
        system_file = self.monitor_dir / "system_stats.json"
        with open(system_file, 'w') as f:
            json.dump(self.system_history, f, indent=2)
            
        if self.metrics_history:
            metrics_file = self.monitor_dir / "training_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
                
    def generate_plots(self):
        """Generate monitoring plots"""
        
        if not self.save_plots or not self.system_history:
            return
            
        try:
            # System resource plots
            df_system = pd.DataFrame(self.system_history)
            df_system['timestamp'] = pd.to_datetime(df_system['timestamp'])
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('System Resource Monitoring', fontsize=16)
            
            # CPU Usage
            axes[0, 0].plot(df_system['timestamp'], df_system['cpu_percent'])
            axes[0, 0].set_title('CPU Usage (%)')
            axes[0, 0].set_ylabel('CPU %')
            axes[0, 0].grid(True)
            
            # Memory Usage
            axes[0, 1].plot(df_system['timestamp'], df_system['memory_percent'])
            axes[0, 1].set_title('Memory Usage (%)')
            axes[0, 1].set_ylabel('Memory %')
            axes[0, 1].grid(True)
            
            # GPU Usage (if available)
            if 'avg_gpu_load' in df_system.columns:
                axes[1, 0].plot(df_system['timestamp'], df_system['avg_gpu_load'])
                axes[1, 0].set_title('Average GPU Load (%)')
                axes[1, 0].set_ylabel('GPU Load %')
                axes[1, 0].grid(True)
            else:
                axes[1, 0].text(0.5, 0.5, 'No GPU Data', ha='center', va='center')
                axes[1, 0].set_title('GPU Load (N/A)')
                
            # GPU Memory (if available)
            if 'total_gpu_memory_used' in df_system.columns:
                axes[1, 1].plot(df_system['timestamp'], df_system['total_gpu_memory_used'])
                axes[1, 1].set_title('Total GPU Memory Used (MB)')
                axes[1, 1].set_ylabel('GPU Memory (MB)')
                axes[1, 1].grid(True)
            else:
                axes[1, 1].text(0.5, 0.5, 'No GPU Data', ha='center', va='center')
                axes[1, 1].set_title('GPU Memory (N/A)')
                
            plt.tight_layout()
            plt.savefig(self.monitor_dir / "system_monitoring.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Training metrics plots (if available)
            if self.metrics_history:
                df_metrics = pd.DataFrame(self.metrics_history)
                df_metrics['timestamp'] = pd.to_datetime(df_metrics['timestamp'])
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle('Training Progress Monitoring', fontsize=16)
                
                # Training Loss
                if 'train_loss' in df_metrics.columns:
                    loss_data = df_metrics.dropna(subset=['train_loss'])
                    if not loss_data.empty:
                        axes[0, 0].plot(loss_data['timestamp'], loss_data['train_loss'])
                        axes[0, 0].set_title('Training Loss')
                        axes[0, 0].set_ylabel('Loss')
                        axes[0, 0].grid(True)
                    else:
                        axes[0, 0].text(0.5, 0.5, 'No Loss Data', ha='center', va='center')
                        
                # Learning Rate
                if 'learning_rate' in df_metrics.columns:
                    lr_data = df_metrics.dropna(subset=['learning_rate'])
                    if not lr_data.empty:
                        axes[0, 1].plot(lr_data['timestamp'], lr_data['learning_rate'])
                        axes[0, 1].set_title('Learning Rate')
                        axes[0, 1].set_ylabel('Learning Rate')
                        axes[0, 1].grid(True)
                    else:
                        axes[0, 1].text(0.5, 0.5, 'No LR Data', ha='center', va='center')
                        
                # Evaluation Metrics
                if 'eval_syntax_accuracy' in df_metrics.columns:
                    eval_data = df_metrics.dropna(subset=['eval_syntax_accuracy'])
                    if not eval_data.empty:
                        axes[1, 0].plot(eval_data['timestamp'], eval_data['eval_syntax_accuracy'])
                        axes[1, 0].set_title('Eval Syntax Accuracy')
                        axes[1, 0].set_ylabel('Accuracy')
                        axes[1, 0].grid(True)
                    else:
                        axes[1, 0].text(0.5, 0.5, 'No Eval Data', ha='center', va='center')
                        
                # Reward (for PPO)
                if 'latest_reward' in df_metrics.columns:
                    reward_data = df_metrics.dropna(subset=['latest_reward'])
                    if not reward_data.empty:
                        axes[1, 1].plot(reward_data['timestamp'], reward_data['latest_reward'])
                        axes[1, 1].set_title('Latest Reward')
                        axes[1, 1].set_ylabel('Reward')
                        axes[1, 1].grid(True)
                    else:
                        axes[1, 1].text(0.5, 0.5, 'No Reward Data', ha='center', va='center')
                else:
                    axes[1, 1].text(0.5, 0.5, 'No Reward Data', ha='center', va='center')
                    
                plt.tight_layout()
                plt.savefig(self.monitor_dir / "training_monitoring.png", dpi=150, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            logger.error(f"Failed to generate plots: {e}")
            
    def print_status(self, system_stats: Dict, training_metrics: Optional[Dict]):
        """Print current status"""
        
        runtime = datetime.now() - self.start_time
        
        print(f"\n{'='*60}")
        print(f"Training Monitor Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Runtime: {runtime}")
        print(f"{'='*60}")
        
        # System Stats
        print(f"System Resources:")
        print(f"  CPU: {system_stats['cpu_percent']:.1f}%")
        print(f"  Memory: {system_stats['memory_percent']:.1f}% ({system_stats['memory_used_gb']:.1f}/{system_stats['memory_total_gb']:.1f} GB)")
        
        if system_stats.get('gpus'):
            print(f"  GPUs: {len(system_stats['gpus'])} available")
            print(f"  Avg GPU Load: {system_stats['avg_gpu_load']:.1f}%")
            print(f"  Total GPU Memory: {system_stats['total_gpu_memory_used']:.0f} MB")
            
        # Training Metrics
        if training_metrics:
            print(f"\nTraining Progress:")
            if 'step' in training_metrics:
                print(f"  Step: {training_metrics['step']}")
            if 'epoch' in training_metrics:
                print(f"  Epoch: {training_metrics['epoch']:.2f}")
            if 'train_loss' in training_metrics and training_metrics['train_loss']:
                print(f"  Train Loss: {training_metrics['train_loss']:.4f}")
            if 'eval_loss' in training_metrics and training_metrics['eval_loss']:
                print(f"  Eval Loss: {training_metrics['eval_loss']:.4f}")
            if 'latest_reward' in training_metrics:
                print(f"  Latest Reward: {training_metrics['latest_reward']:.3f}")
        else:
            print(f"\nTraining Progress: No metrics available")
            
        print(f"{'='*60}\n")
        
    def monitor_loop(self):
        """Main monitoring loop"""
        
        logger.info(f"Starting training monitoring...")
        logger.info(f"Checkpoint dir: {self.checkpoint_dir}")
        logger.info(f"Log dir: {self.log_dir}")
        logger.info(f"Monitoring interval: {self.monitoring_interval} seconds")
        
        self.is_monitoring = True
        
        try:
            while self.is_monitoring:
                # Get current stats
                system_stats = self.get_system_stats()
                training_metrics = self.get_training_metrics()
                
                # Save metrics
                self.save_metrics(system_stats, training_metrics)
                
                # Generate plots
                if len(self.system_history) % 10 == 0:  # Update plots every 10 intervals
                    self.generate_plots()
                    
                # Print status
                self.print_status(system_stats, training_metrics)
                
                # Wait for next interval
                time.sleep(self.monitoring_interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
        finally:
            self.is_monitoring = False
            self.generate_plots()  # Final plot generation
            logger.info("Monitoring completed")
            
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        
    def generate_final_report(self) -> str:
        """Generate final monitoring report"""
        
        if not self.system_history:
            return "No monitoring data available"
            
        runtime = datetime.now() - self.start_time
        
        # Calculate averages
        avg_cpu = sum(entry['cpu_percent'] for entry in self.system_history) / len(self.system_history)
        avg_memory = sum(entry['memory_percent'] for entry in self.system_history) / len(self.system_history)
        avg_gpu_load = sum(entry.get('avg_gpu_load', 0) for entry in self.system_history) / len(self.system_history)
        
        report = f"""
Training Monitoring Final Report
===============================

Monitoring Period: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Runtime: {runtime}
Data Points Collected: {len(self.system_history)}

System Resource Usage (Average):
- CPU: {avg_cpu:.1f}%
- Memory: {avg_memory:.1f}%
- GPU Load: {avg_gpu_load:.1f}%

Files Generated:
- System stats: {self.monitor_dir}/system_stats.json
- Training metrics: {self.monitor_dir}/training_metrics.json
- System plots: {self.monitor_dir}/system_monitoring.png
- Training plots: {self.monitor_dir}/training_monitoring.png

Monitoring completed successfully!
"""
        
        return report


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Monitor DeepSeek R1 Verilog training")
    
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                       help="Directory containing training checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs",
                       help="Directory containing training logs")
    parser.add_argument("--interval", type=int, default=60,
                       help="Monitoring interval in seconds")
    parser.add_argument("--no_plots", action="store_true",
                       help="Disable plot generation")
    
    return parser.parse_args()

def main():
    """Main monitoring function"""
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("DEEPSEEK R1 VERILOG TRAINING MONITOR")
    logger.info("=" * 80)
    
    # Create monitor
    monitor = TrainingMonitor(
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        monitoring_interval=args.interval,
        save_plots=not args.no_plots
    )
    
    try:
        # Start monitoring
        monitor.monitor_loop()
        
    finally:
        # Generate final report
        report = monitor.generate_final_report()
        print(report)
        
        # Save report
        report_file = monitor.monitor_dir / "monitoring_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Final report saved to {report_file}")

if __name__ == "__main__":
    main()