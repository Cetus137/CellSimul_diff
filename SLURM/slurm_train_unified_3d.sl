#!/bin/bash

#SBATCH --job-name      unified_3d_train
#SBATCH --partition=gpu_long
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
##SBATCH --mem=70G
#SBATCH --mem-per-gpu=80G
#SBATCH --time          23:59:00
#SBATCH --output        slogs/train_unified_3d.%j.out
#SBATCH --error         slogs/train_unified_3d.%j.err
#SBATCH --exclude       compg009,compg010,compg011,compg013
#SBATCH --constraint    "a100"


# 3D Unified Autoregressive Diffusion — Training
#
# Model: ConditionalUNet3D + DDPM3D
# Conditioning: 2 channels = [heatmap_t, volume_{t-1}]
#   Channel 0: Gaussian blob heatmap of target-frame centres
#   Channel 1: Previous frame V_{t-1} in [0,1]  (zeros for frame 0)
#
# Trains jointly on:
#   - Single-frame patches  (channel 1 = zeros)
#   - Temporal pairs        (channel 1 = V_{t-1}, 15% prev-frame dropout)
#
# This replaces both the frame1_3d and frame2_3d models with a single checkpoint.
# At inference:
#   Frame 0:    pass zeros for channel 1
#   Frame t+1:  pass V_t (re-ranged to [0,1]) for channel 1
#
# Prerequisites:
#   1. Preprocessed patches in data_live_node1_3d/{train,val,test}/
#      and data_live_node2_3d/{train,val,test}/
#      (run slurm_build_patches3d.sl if not already done)
#   2. Preprocessed temporal pairs in data_live_node1_3d/pairs/
#      and data_live_node2_3d/pairs/
#      (run slurm_build_temporal_dataset3d.sl if not already done)
#
# Run with:
#   sbatch SLURM/slurm_train_unified_3d.sl

echo "========================================"
echo "Cell Simulation 3D Unified — Training"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
echo "CPUs:   $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Start:  $(date)"
echo "========================================"

WORK_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff"
CONFIG="configs/unified_3d.yaml"

# Set to a checkpoint path to resume, or leave empty to start fresh
# Example: RESUME="checkpoints/unified_3d/latest.pt"
RESUME= "" #"/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/checkpoints/unified_3d_node2_D_C/best.pt"

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

CMD="python scripts/train_unified_3d.py --config $CONFIG --device cuda"

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
    echo "Checkpoints saved to: checkpoints/unified_3d/"
    echo "Visualizations in:    visualizations/unified_3d/"
    echo ""
    echo "Next steps:"
    echo "  Generate a timeseries using the unified checkpoint:"
    echo "    python -m scripts.generate_timeseries3d_img2img \\"
    echo "      --unified_ckpt checkpoints/unified_3d/best.pt \\"
    echo "      --centres_dir  synthetic_cells_3d/ \\"
    echo "      --out_dir      timeseries_output/"
else
    echo "Training FAILED with exit code $TRAIN_EXIT_CODE"
fi
echo "End: $(date)"
echo "========================================"

exit $TRAIN_EXIT_CODE
