#!/bin/bash

#SBATCH --job-name      frame2_training
#SBATCH --partition=gpu_long
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
##SBATCH --mem-per-gpu=2G
#SBATCH --time          13:15:00
#SBATCH --output        slogs/train_frame2.%j.out
#SBATCH --error         slogs/train_frame2.%j.err
#SBATCH --exclude       compg009,compg010,compg011,compg013
##SBATCH --constraint    "a100"

# Cell Simulation Diffusion Model - Frame-2 Training
#
# Trains the temporally-conditioned model:
#   p( I_{t+1} | C_{t+1}, I_t )
# where the 4-channel conditioning is:
#   channels 0-2 : C_{t+1} geometry maps  (heatmap, distance, boundary)
#   channel  3   : I_t                     (previous frame, clean)
#
# Prerequisites:
#   1. Run slurm_build_temporal_dataset.sl to produce
#      data_multiz/processed/pairs/{train,val,test}/
#
# Run with:
#   sbatch SLURM/slurm_train_frame2.sl
#
# To resume from a checkpoint, set RESUME below.

echo "========================================"
echo "Cell Simulation - Frame-2 Training"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
echo "CPUs:   $SLURM_CPUS_PER_TASK"
echo "Start:  $(date)"
echo "========================================"

WORK_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff"
CONFIG="configs/frame2.yaml"

# Set to a checkpoint path to resume, or leave empty to train from scratch
RESUME=""

cd $WORK_DIR

mkdir -p slogs
mkdir -p checkpoints/frame2
mkdir -p visualizations/frame2

# Load environment
module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/cell_simul_env/bin/activate

echo ""
echo "Python version:"
python --version
echo ""
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo ""
echo "CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"
echo ""
echo "GPU info:"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU')"
python -c "import torch; print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB' if torch.cuda.is_available() else '')"
echo ""
echo "========================================"
echo "Training Configuration"
echo "========================================"
echo "Config: $CONFIG"
echo ""
echo "========================================"

# Build command
CMD="python -u -m scripts.train_frame2"
CMD="$CMD --config $CONFIG"
CMD="$CMD --device cuda"

if [ -n "$RESUME" ]; then
    CMD="$CMD --resume $RESUME"
    echo "Resuming from: $RESUME"
fi

echo "Starting training..."
echo ""
eval $CMD

TRAIN_EXIT_CODE=$?

echo ""
echo "========================================"
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo ""
    echo "Latest checkpoints:"
    ls -lht checkpoints/frame2/*.pt 2>/dev/null | head -5
    echo ""
else
    echo "Training FAILED  (exit code: $TRAIN_EXIT_CODE)"
fi
echo "End time: $(date)"
echo "========================================"

exit $TRAIN_EXIT_CODE
