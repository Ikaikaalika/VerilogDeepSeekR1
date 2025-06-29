# Thunder Compute Integration Guide

This guide explains how to use Thunder Compute for training the DeepSeek R1 Verilog model.

## Prerequisites

1. **Install Thunder Compute CLI**
   ```bash
   # Install TNR CLI (follow Thunder Compute documentation)
   # Authenticate with your Thunder account
   tnr login
   ```

2. **Verify Authentication**
   ```bash
   tnr status
   ```

## Quick Start

### Option 1: Full Automated Workflow
```bash
# Complete end-to-end training on Thunder
make thunder-full
```
This will:
- Create an A100 instance with 8 GPUs
- Setup the training environment
- Upload project files
- Run both SFT and PPO training
- Download results

### Option 2: Step-by-Step Process

1. **Create Thunder Instance**
   ```bash
   make thunder-create
   # Or customize:
   python scripts/setup_thunder.py --action create --gpu-type a100xl --num-gpus 4
   ```

2. **Setup Environment**
   ```bash
   # Use the instance ID from step 1
   make thunder-setup INSTANCE_ID=your_instance_id
   ```

3. **Run Training**
   ```bash
   make thunder-train INSTANCE_ID=your_instance_id
   # Or customize phase:
   python scripts/setup_thunder.py --action train --instance-id your_instance_id --phase sft
   ```

4. **Download Results**
   ```bash
   python scripts/setup_thunder.py --action download --instance-id your_instance_id
   ```

5. **Cleanup**
   ```bash
   python scripts/setup_thunder.py --action cleanup --instance-id your_instance_id --delete-after
   ```

## Configuration Options

### GPU Types
- `t4`: NVIDIA T4 (16GB VRAM) - Best for testing
- `a100`: NVIDIA A100 (40GB VRAM) - Default for training
- `a100xl`: NVIDIA A100 (80GB VRAM) - For largest models

### Training Phases
- `sft`: Supervised fine-tuning only
- `ppo`: PPO training only (requires SFT model)
- `both`: Complete pipeline (default)

## Example Workflows

### Development/Testing
```bash
python scripts/setup_thunder.py \
    --action full \
    --gpu-type t4 \
    --num-gpus 1 \
    --phase sft \
    --delete-after
```

### Production Training
```bash
python scripts/setup_thunder.py \
    --action full \
    --gpu-type a100xl \
    --num-gpus 8 \
    --phase both \
    --wandb-project my-verilog-project
```

### Resume Training
```bash
# Create instance and setup environment
python scripts/setup_thunder.py --action create --gpu-type a100 --num-gpus 8
python scripts/setup_thunder.py --action setup --instance-id INSTANCE_ID

# Upload existing checkpoints if needed
tnr scp -r ./checkpoints INSTANCE_ID:/workspace/

# Run specific training phase
python scripts/setup_thunder.py --action train --instance-id INSTANCE_ID --phase ppo
```

## Monitoring

The setup script automatically forwards ports for monitoring:
- Port 8080: Training dashboard
- Port 6006: TensorBoard (if enabled)

Access via `localhost:8080` when training is running.

## Cost Optimization

1. **Use Snapshots**: Save instance state for reuse
   ```bash
   tnr stop INSTANCE_ID
   tnr snapshot INSTANCE_ID my-verilog-env
   
   # Later, create new instance from snapshot
   tnr create --template my-verilog-env --gpu a100
   ```

2. **Stop vs Delete**: Stop instances when not in use to save compute costs
   ```bash
   tnr stop INSTANCE_ID  # Keeps storage, stops compute billing
   tnr start INSTANCE_ID # Resume when needed
   ```

3. **Right-size Resources**: Match GPU/CPU to your needs
   - Development: 1-2 T4 GPUs
   - Small training: 4 A100 GPUs  
   - Large training: 8 A100XL GPUs

## Troubleshooting

### TNR Command Not Found
```bash
# Install Thunder Compute CLI
# Follow installation guide at thunder.ai/docs

# Verify installation
which tnr
tnr --version
```

### Authentication Issues
```bash
# Re-authenticate
tnr logout
tnr login

# Check token
cat ~/.thunder/token
```

### Instance Issues
```bash
# Check instance status
tnr status

# Connect to instance for debugging
tnr connect INSTANCE_ID

# View instance logs
tnr connect INSTANCE_ID -- tail -f /workspace/logs/*.log
```

### Training Failures
```bash
# Connect and check logs
tnr connect INSTANCE_ID
cd /workspace
ls logs/

# Download logs for local inspection
tnr scp INSTANCE_ID:/workspace/logs ./local_logs
```

## Advanced Usage

### Custom Training Scripts
```bash
# Upload custom training script
tnr scp ./my_custom_script.py INSTANCE_ID:/workspace/

# Run custom training
tnr connect INSTANCE_ID -- python /workspace/my_custom_script.py
```

### Multi-Instance Training
For extremely large models, you can coordinate multiple instances:
```bash
# Create multiple instances
python scripts/setup_thunder.py --action create --gpu-type a100xl --num-gpus 8
# Repeat for additional instances

# Setup distributed training (advanced - requires custom configuration)
```

## Support

- Thunder Compute Documentation: [thunder.ai/docs](https://thunder.ai/docs)
- CLI Reference: See `Thunder_Compute_CLI_ref.txt` in this repository
- Issues: Report problems in the project GitHub issues