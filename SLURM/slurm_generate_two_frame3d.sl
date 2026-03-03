#!/bin/bash
#SBATCH --job-name=generate_two_frame_3d
#SBATCH --partition=gpu_short
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=4G
#SBATCH --time          00:30:00
#SBATCH --output        slogs/generate_two_frame_3d.%j.out
#SBATCH --error         slogs/generate_two_frame_3d.%j.err
#SBATCH --exclude       compg009,compg010,compg011,compg013
##SBATCH --constraint    "a100"

# Cell Simulation Diffusion Model - 3D Two-Frame Generation
#
# Runs the temporally consistent 3D two-frame sampling pipeline:
#
#   Step 1:  V_t     ~ p( V    | C_t )            [frame1_3d model]
#   Step 2:  V_{t+1} ~ p( V    | C_{t+1}, V_t )  [frame2_3d model]
#
# where C_t, C_{t+1} are 2-channel conditioning maps (heatmap + distance)
# derived from 3D cell-centre positions at times t and t+1.
#
# Prerequisites:
#   1. Trained frame1_3d checkpoint  (checkpoints/frame1_3d/best.pt)
#   2. Trained frame2_3d checkpoint  (checkpoints/frame2_3d/best.pt)
#   3. Centre .npy files for time t and t+1  (N,3) arrays in (z,y,x) order
#
# Run with:
#   sbatch SLURM/slurm_generate_two_frame3d.sl
#
# CPU fallback (slow but no GPU needed):
#   Change --partition=gpu_short  →  --partition=short
#   Remove or comment --gpus-per-node=1
#   Set BATCH_SIZE=1 below for manageable memory use


module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/cell_simul_env/bin/activate

WORK_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff"
cd $WORK_DIR

# ---- Configuration  --------------------------------------------------
FRAME1_CKPT="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/checkpoints/frame1_3d/best.pt"
FRAME2_CKPT="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/checkpoints/frame2_3d/best.pt"
FRAME1_CONFIG="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/configs/frame1_3d.yaml"
FRAME2_CONFIG="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/configs/frame2_3d.yaml"

# Cell-centre files for time t and t+1
# Only used when METHOD=from_file.  In realistic mode centres are generated
# automatically from the centre_generation_3d block in FRAME1_CONFIG.
METHOD="realistic"          # realistic | from_file
DISPLACEMENT_SIGMA=0.0      # Gaussian σ (voxels) for inter-frame cell displacement

CENTRES_T="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/data_live_node1_3d/pairs/test/pair3d_p00128_01_centres_t0.npy"
CENTRES_T1="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/data_live_node1_3d/pairs/test/pair3d_p00128_01_centres_t1.npy"

OUTPUT_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/two_frame_3d_output"
VOLUME_SIZE=128
BATCH_SIZE=1

# Classifier-free guidance (set USE_CFG=true to enable)
USE_CFG=true
GUIDANCE_SCALE=2.0

# Load EMA weights from checkpoints (recommended for best quality)
USE_EMA=false

# Histogram matching: match V_{t+1} intensity distribution to V_t
# Reduces inter-frame SNR differences caused by model bias
MATCH_HISTOGRAMS=true

# Heatmap sigma (voxels) and distance percentile for conditioning
HEATMAP_SIGMA=0.0
DISTANCE_PERCENTILE=95.0
# ----------------------------------------------------------------------

mkdir -p $OUTPUT_DIR

echo "================================================"
echo "3D Two-Frame Temporally Linked Generation"
echo "================================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "GPU:           ${CUDA_VISIBLE_DEVICES:-none}"
echo "Start:         $(date)"
echo "================================================"
echo "Frame-1 ckpt:  $FRAME1_CKPT"
echo "Frame-2 ckpt:  $FRAME2_CKPT"
echo "Method:        $METHOD"
if [ "$METHOD" = "from_file" ]; then
    echo "Centres t:     $CENTRES_T"
    echo "Centres t+1:   $CENTRES_T1"
else
    echo "Displacement σ: ${DISPLACEMENT_SIGMA} voxels"
fi
echo "Output dir:    $OUTPUT_DIR"
echo "Volume size:   ${VOLUME_SIZE}³"
echo "Batch size:    $BATCH_SIZE"
echo "CFG:           $USE_CFG  (scale=$GUIDANCE_SCALE)"
echo "EMA:           $USE_EMA"
echo "================================================"
echo ""

# Auto-detect device — falls back to CPU if no GPU is allocated.
if [ -n "$CUDA_VISIBLE_DEVICES" ] && [ "$CUDA_VISIBLE_DEVICES" != "NoDevFiles" ]; then
    DEVICE="cuda"
    echo "Device: CUDA (GPU $CUDA_VISIBLE_DEVICES)"
else
    DEVICE="cpu"
    echo "Device: CPU (no GPU allocated — generation will be slow)"
fi

echo ""
echo "Python version:"
python --version

echo ""
echo "Starting 3D two-frame generation..."
echo ""

CMD="python -u -m sampling.sample_two_frame3d \
    --frame1_ckpt \"$FRAME1_CKPT\" \
    --frame2_ckpt \"$FRAME2_CKPT\" \
    --frame1_config \"$FRAME1_CONFIG\" \
    --frame2_config \"$FRAME2_CONFIG\" \
    --method $METHOD \
    --volume_size $VOLUME_SIZE \
    --out_dir \"$OUTPUT_DIR\" \
    --batch_size $BATCH_SIZE \
    --heatmap_sigma $HEATMAP_SIGMA \
    --distance_percentile $DISTANCE_PERCENTILE \
    --device $DEVICE"

# Append centres args only for from_file mode
if [ "$METHOD" = "from_file" ]; then
    CMD="$CMD --centres_t \"$CENTRES_T\" --centres_t1 \"$CENTRES_T1\""
else
    CMD="$CMD --displacement_sigma $DISPLACEMENT_SIGMA"
fi

if [ "$USE_EMA" = "true" ]; then
    CMD="$CMD --use_ema"
fi

if [ "$USE_CFG" = "true" ]; then
    CMD="$CMD --cfg --guidance_scale $GUIDANCE_SCALE"
fi

if [ "$MATCH_HISTOGRAMS" = "true" ]; then
    CMD="$CMD --match_histograms"
fi

eval $CMD

GEN_EXIT_CODE=$?

echo ""
echo "================================================"
if [ $GEN_EXIT_CODE -eq 0 ]; then
    echo "Generation completed successfully!"
    echo ""
    echo "Output saved to: $OUTPUT_DIR"
    echo "  frame_t_sample***.tif   — volumes at time t"
    echo "  frame_t1_sample***.tif  — volumes at time t+1"
    echo "  pair_sample***.tif      — stacked pairs (2, D, H, W)"
    echo ""
    echo "Visualize with:"
    echo "  napari $OUTPUT_DIR/frame_t_sample000.tif"
    echo "  napari $OUTPUT_DIR/pair_sample000.tif"
else
    echo "Generation FAILED with exit code $GEN_EXIT_CODE"
    echo "Check the log above for errors."
fi
echo "End time: $(date)"
echo "================================================"

exit $GEN_EXIT_CODE
