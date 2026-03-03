#!/bin/bash

#SBATCH --job-name      morph3d_arr
#SBATCH --cpus-per-task 1
#SBATCH --mem           2G
#SBATCH --time          00:05:00
#SBATCH --output        slogs/morph3d/morph3d_arr.%A_%a.out
#SBATCH --error         slogs/morph3d/morph3d_arr.%A_%a.err
#SBATCH --exclude       compg009,compg010,compg011,compg013
# Submit as:  sbatch --array=0-$((N-1))%30 SLURM/slurm_eval_morphology3d__arr.sl
# where N = number of real + synthetic *_masks.tif files combined.

module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/cell_simul_env/bin/activate

cd /users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff

File=${SLURM_ARRAY_TASK_ID}

python3 -m evaluation.morphology3d \
    --real_dir   /users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/data_live_node1_3d \
    --syn_dir    /users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/data_synthetic_3D_frame1_volumes/no_z_raw \
    --index      ${File} \
    --stats_dir  /users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/evaluation/results_noZ_raw/morphology \
    --min_volume 500 \
    --max_volume 500000 \
    --z_start    24 \
    --z_end      152 \
    --tile_size  128
