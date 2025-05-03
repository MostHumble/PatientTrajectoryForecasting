#!/bin/bash
#SBATCH --job-name=Patient_Traj_Pred
#SBATCH --partition=gpu
#SBATCH --nodes=1 # Use 1 node to avoid comm overhead stinky af
#SBATCH --ntasks-per-node=7 # must match the number of GPUs
#SBATCH --gres=gpu:7     # must match this
#SBATCH --cpus-per-task=5
#SBATCH --time=7-00:00:00
#SBATCH --output=stdout_%j.out
#SBATCH --error=stderr_%j.err

# Load Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch-2.2

# Set distributed training environment variables
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12345
export WORLD_SIZE=$SLURM_NTASKS

# Optional: Set CUDA_VISIBLE_DEVICES for each process (if not handled in Python)
# SLURM_LOCALID can help assign GPUs per task on each node
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Run the Python script with srun
srun python cross_val_train_ddp.py