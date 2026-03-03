#!/bin/bash

#SBATCH --job-name      cell3d_train
#SBATCH --partition=gpu_long
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=10G
#SBATCH --time          23:59:00
#SBATCH --output        slogs/train_frame1_3d.%j.out
#SBATCH --error         slogs/train_frame1_3d.%j.err
#SBATCH --exclude       compg009,compg010,compg011,compg013
##SBATCH --constraint    "a100"


# 3D Centre-Conditioned Diffusion — Training
# ConditionalUNet3D (base_channels=32) + DDPM3D, 128^3 patches

echo "========================================"
echo "Cell Simulation 3D - Training"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
echo "CPUs:   $SLURM_CPUS_PER_TASK"
echo "Start:  $(date)"
echo "========================================"

WORK_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff"
CONFIG="configs/frame1_3d.yaml"

# Set to a checkpoint path to resume, or leave empty to start fresh
RESUME="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/checkpoints/frame1_3d_combined_noD_noZ_raw/best.pt"

cd $WORK_DIR

echo ""
echo "Loading environment..."
module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/cell_simul_env/bin/activate

echo ""
echo "========================================"
echo "System Information"
echo "========================================"
echo "Python:  $(python --version)"
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
python -c "import torch; print('VRAM: %.1f GB' % (torch.cuda.get_device_properties(0).total_memory/1e9) if torch.cuda.is_available() else '')"

echo ""
echo "========================================"
echo "Training Configuration"
echo "========================================"
echo "Config: $CONFIG"
echo ""

CMD="python scripts/train3d.py --config $CONFIG --device cuda"

if [ -n "$RESUME" ]; then
    CMD="$CMD --resume $RESUME"
    echo "Resuming from: $RESUME"
fi

echo "Command: $CMD"
echo "========================================"
echo ""

eval $CMD

TRAIN_EXIT_CODE=$?

echo ""
echo "========================================"
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training FAILED with exit code $TRAIN_EXIT_CODE"
fi
echo "End: $(date)"
echo "========================================"

exit $TRAIN_EXIT_CODE
