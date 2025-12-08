#!/usr/bin/env bash
TASK_NAME="${TASK_NAME:-Template-Rob6323-Go2-Direct-v0}"
ssh -o StrictHostKeyChecking=accept-new burst \
    "cd ~/rob6323_go2_project && \
    export TASK_NAME=\"$TASK_NAME\" && \
    sbatch --export=ALL,TASK_NAME --job-name='rob6323_${USER}' --mail-user='${USER}@nyu.edu' \
    train.slurm '$@'"
