#!/bin/bash
#SBATCH --nodes=1
# #SBATCH --ntasks-per-node=2
#SBATCH --job-name=ann
#SBATCH --time=2:00:00

date
hostname
echo 

echo "All jobs in this array have:"
echo "- SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}"
echo "- SLURM_ARRAY_TASK_COUNT=${SLURM_ARRAY_TASK_COUNT}"
echo "- SLURM_ARRAY_TASK_MIN=${SLURM_ARRAY_TASK_MIN}"
echo "- SLURM_ARRAY_TASK_MAX=${SLURM_ARRAY_TASK_MAX}"
echo 

echo "This job in the array has:"
echo "- SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "- SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo 

echo "Script Directory:"
echo "- DIRNAME=${DIRNAME}"
echo 

FILES=($(ls -1 ${DIRNAME}/*.inp))
INFILE=${FILES[$SLURM_ARRAY_TASK_ID]}
echo "Files:"
echo "- INFILE=${INFILE}"
echo 

bash "${INFILE}"
