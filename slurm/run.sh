#!/bin/bash
#SBATCH --array=0-644
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --job-name=robustness
#SBATCH --mem=8GB
#SBATCH -t 5:00:00
#SBATCH -D ./log/
#SBATCH --partition=cbmm
#SBATCH --gres=gpu:1
#SBATCH --exclude=node104



cd /om2/user/xboix/src/convex_adversarial/

hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER

singularity exec -B /om:/om -B /om2:/om2 -B /scratch/user/xboix:/vast --nv /om/user/xboix/singularity/xboix-tensorflow2.8.0.simg \
python3 main.py \
--experiment_id=$((644 +${SLURM_ARRAY_TASK_ID})) \
--run=train \
--gpu_id=0

singularity exec -B /om:/om -B /om2:/om2 -B /scratch/user/xboix:/vast --nv /om/user/xboix/singularity/xboix-tensorflow2.8.0.simg \
python3 main.py \
--experiment_id=$((644 +${SLURM_ARRAY_TASK_ID})) \
--run=test \
--gpu_id=0

