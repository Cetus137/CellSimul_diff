#!/bin/bash

#SBATCH --job-name      frame2_3d_train
#SBATCH --partition=gpu_long
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=12G
#SBATCH --time          13:59:00
#SBATCH --output        slogs/train_frame2_3d.%j.out
#SBATCH --error         slogs/train_frame2_3d.%j.err
#SBATCH --exclude       compg009,compg010,compg011,compg013
#SBATCH --constraint    "a100"


# 3D Temporally-Conditioned Diffusion — Training
#
# Model: ConditionalUNet3D + DDPM3D
# Conditioning: 3 channels = [heatmap_{t+1}, distance_{t+1}, volume_t]
# Target:       volume_{t+1}
#
# This script trains the frame2_3d model which generates the next frame
# conditioned on both the future cell positions AND the previous frame's appearance.
#
# Prerequisites:
#   1. Preprocessed temporal pairs in data_live_node1_3d/pairs/
#      (run SLURM/slurm_build_temporal_dataset3d.sl first)
#
#   2. A trained frame1_3d checkpoint (recommended but not strictly required)
#      This proves that the base 3D architecture works before adding temporal conditioning.
#
# Run with:
#   sbatch SLURM/slurm_train_frame2_3d.sl

echo "========================================"
echo "Cell Simulation 3D Frame2 - Training"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
echo "CPUs:   $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Start:  $(date)"
echo "========================================"

WORK_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff"
CONFIG="configs/frame2_3d.yaml"

# Set to a checkpoint path to resume, or leave empty to start fresh
# Example: RESUME="checkpoints/frame2_3d/latest.pt"
RESUME=""

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

CMD="python scripts/train_frame2_3d.py --config $CONFIG --device cuda"

if [ -n "$RESUME" ]; then
    CMD="$CMD --resume $RESUME"
    echo "Resuming from: $RESUME"
else
    echo "Starting fresh training (no checkpoint resume)"
fi

echo ""
echo "Command: $CMD"
echo "========================================"
echo ""

eval $CMD

TRAIN_EXIT_CODE=$?

echo ""
echo "========================================"
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo ""
    echo "Checkpoints saved to: checkpoints/frame2_3d/"
    echo "Visualizations in: visualizations/frame2_3d/"
    echo ""
    echo "Next steps:"
    echo "  1. Generate two-frame samples:"
    echo "     sbatch SLURM/slurm_generate_two_frame3d.sl"
    echo "  2. Or run manual sampling:"
    echo "     python -m sampling.sample_two_frame3d \\"
    echo "       --frame1_ckpt checkpoints/frame1_3d/best.pt \\"
    echo "       --frame2_ckpt checkpoints/frame2_3d/best.pt \\"
    echo "       --centres_t synthetic_cells/synthetic_0000_centres.npy \\"
    echo "       --centres_t1 synthetic_cells/synthetic_0001_centres.npy \\"
    echo "       --out_dir output_3d/"
else
    echo "Training FAILED with exit code $TRAIN_EXIT_CODE"
fi
echo "End: $(date)"
echo "========================================"

exit $TRAIN_EXIT_CODE
