#!/bin/bash

#SBATCH --job-name      cell_simul_training
#SBATCH --partition=gpu_long
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-gpu=8G
#SBATCH --time          13:59:00
#SBATCH --output        slogs/train_frame1.%j.out
#SBATCH --error         slogs/train_frame1.%j.err
#SBATCH --exclude       compg009,compg010,compg011,compg013
##SBATCH --constraint    "a100"


# Cell Simulation Diffusion Model - Training
# Trains a centre-conditioned diffusion model on preprocessed patches

echo "========================================"
echo "Cell Simulation - Diffusion Training"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "========================================"

# Setup paths
WORK_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff"
CONFIG="configs/frame1.yaml"

# Set to a checkpoint path to resume, or leave empty to train from scratch
RESUME=""

cd $WORK_DIR


# Load Python environment
echo ""
echo "Loading environment..."
module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/cell_simul_env/bin/activate


# Verify GPU availability
echo ""
echo "========================================"
echo "System Information"
echo "========================================"
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

# Run training
echo ""
echo "Starting training..."
echo ""

CMD="python scripts/train.py"
CMD="$CMD --config $CONFIG"
CMD="$CMD --device cuda"

if [ -n "$RESUME" ]; then
    CMD="$CMD --resume $RESUME"
    echo "Resuming from: $RESUME"
fi

eval $CMD

# Capture exit status
TRAIN_EXIT_CODE=$?

echo ""
echo "========================================"
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo ""
    echo "Latest checkpoints:"
    ls -lht checkpoints/frame1/*.pt 2>/dev/null | head -5
    echo ""
else
    echo "Training failed with exit code: $TRAIN_EXIT_CODE"
fi
echo "End time: $(date)"
echo "========================================"

exit $TRAIN_EXIT_CODE
