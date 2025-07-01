#!/usr/bin/env python3
"""
Thunder Compute Cost Calculator for DeepSeek R1 Training
Estimates training costs based on different configurations
"""

import argparse
from typing import Dict, Tuple

class ThunderCostCalculator:
    """Calculate Thunder Compute costs for different training scenarios"""
    
    # Thunder Compute pricing (approximate USD/hour per GPU)
    GPU_PRICING = {
        "t4": 0.40,      # T4 16GB
        "a100": 2.50,    # A100 40GB  
        "a100xl": 3.50,  # A100 80GB
    }
    
    def __init__(self):
        self.model_params = 32_000_000_000  # 32B parameters
        
    def estimate_training_time(self, 
                             gpu_type: str, 
                             num_gpus: int,
                             batch_size: int,
                             dataset_size: int = 100_000,
                             epochs: int = 3) -> float:
        """Estimate training time in hours"""
        
        # Throughput estimates (tokens/second per GPU)
        throughput = {
            "t4": 100,
            "a100": 800, 
            "a100xl": 1200
        }
        
        tokens_per_sample = 2048  # Average Verilog sequence length
        total_tokens = dataset_size * tokens_per_sample * epochs
        
        # Account for gradient accumulation reducing effective throughput
        effective_throughput = throughput[gpu_type] * num_gpus * 0.8  # 80% efficiency
        
        training_hours = total_tokens / effective_throughput / 3600
        return max(training_hours, 1.0)  # Minimum 1 hour
        
    def calculate_cost(self, 
                      gpu_type: str,
                      num_gpus: int, 
                      training_hours: float) -> Dict[str, float]:
        """Calculate total training cost"""
        
        gpu_cost_per_hour = self.GPU_PRICING[gpu_type] * num_gpus
        total_gpu_cost = gpu_cost_per_hour * training_hours
        
        # Additional costs (storage, networking, etc.)
        additional_cost = total_gpu_cost * 0.15  # 15% overhead
        
        return {
            "gpu_cost": total_gpu_cost,
            "additional_cost": additional_cost,
            "total_cost": total_gpu_cost + additional_cost,
            "cost_per_hour": gpu_cost_per_hour + (additional_cost / training_hours)
        }
        
    def compare_configurations(self) -> Dict[str, Dict]:
        """Compare different cost-optimized configurations"""
        
        configs = {
            "development": {
                "gpu_type": "t4",
                "num_gpus": 1,
                "batch_size": 1,
                "dataset_size": 1000,
                "epochs": 1
            },
            "small_scale": {
                "gpu_type": "a100", 
                "num_gpus": 2,
                "batch_size": 4,
                "dataset_size": 10000,
                "epochs": 2
            },
            "cost_optimized": {
                "gpu_type": "a100",
                "num_gpus": 4, 
                "batch_size": 8,
                "dataset_size": 50000,
                "epochs": 3
            },
            "standard": {
                "gpu_type": "a100",
                "num_gpus": 8,
                "batch_size": 16, 
                "dataset_size": 100000,
                "epochs": 3
            },
            "large_scale": {
                "gpu_type": "a100xl",
                "num_gpus": 8,
                "batch_size": 32,
                "dataset_size": 200000, 
                "epochs": 3
            }
        }
        
        results = {}
        for name, config in configs.items():
            training_time = self.estimate_training_time(
                config["gpu_type"],
                config["num_gpus"], 
                config["batch_size"],
                config["dataset_size"],
                config["epochs"]
            )
            
            cost_info = self.calculate_cost(
                config["gpu_type"],
                config["num_gpus"],
                training_time
            )
            
            results[name] = {
                **config,
                "training_hours": training_time,
                **cost_info
            }
            
        return results
        
    def optimize_for_budget(self, budget: float) -> Dict[str, any]:
        """Find best configuration within budget"""
        
        configs = self.compare_configurations()
        viable_configs = {
            name: config for name, config in configs.items() 
            if config["total_cost"] <= budget
        }
        
        if not viable_configs:
            return {"error": f"No configuration fits budget of ${budget:.2f}"}
            
        # Select configuration with best performance within budget
        best_config = max(
            viable_configs.items(),
            key=lambda x: x[1]["dataset_size"] * x[1]["num_gpus"]
        )
        
        return {"recommended": best_config[0], "config": best_config[1]}

def main():
    parser = argparse.ArgumentParser(description="Thunder Compute cost calculator")
    parser.add_argument('--budget', type=float, help='Maximum budget in USD')
    parser.add_argument('--compare', action='store_true', help='Compare all configurations')
    parser.add_argument('--gpu-type', choices=['t4', 'a100', 'a100xl'], help='Specific GPU type')
    parser.add_argument('--num-gpus', type=int, help='Number of GPUs')
    parser.add_argument('--dataset-size', type=int, default=100000, help='Dataset size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    
    args = parser.parse_args()
    calculator = ThunderCostCalculator()
    
    if args.compare or not any([args.gpu_type, args.budget]):
        print("=== Thunder Compute Cost Comparison ===\n")
        results = calculator.compare_configurations()
        
        for name, config in results.items():
            print(f"{name.upper()}:")
            print(f"  GPUs: {config['num_gpus']}x {config['gpu_type']}")
            print(f"  Training time: {config['training_hours']:.1f} hours")
            print(f"  Total cost: ${config['total_cost']:.2f}")
            print(f"  Cost per hour: ${config['cost_per_hour']:.2f}")
            print()
            
    if args.budget:
        print(f"\n=== Budget Optimization (${args.budget:.2f}) ===")
        result = calculator.optimize_for_budget(args.budget)
        
        if "error" in result:
            print(result["error"])
        else:
            config = result["config"]
            print(f"Recommended: {result['recommended'].upper()}")
            print(f"  GPUs: {config['num_gpus']}x {config['gpu_type']}")
            print(f"  Training time: {config['training_hours']:.1f} hours") 
            print(f"  Total cost: ${config['total_cost']:.2f}")
            
    if args.gpu_type and args.num_gpus:
        training_time = calculator.estimate_training_time(
            args.gpu_type, args.num_gpus, 
            batch_size=4,  # Default
            dataset_size=args.dataset_size,
            epochs=args.epochs
        )
        
        cost_info = calculator.calculate_cost(args.gpu_type, args.num_gpus, training_time)
        
        print(f"\n=== Custom Configuration ===")
        print(f"GPUs: {args.num_gpus}x {args.gpu_type}")
        print(f"Training time: {training_time:.1f} hours")
        print(f"Total cost: ${cost_info['total_cost']:.2f}")

if __name__ == "__main__":
    main()