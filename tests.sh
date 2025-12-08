#!/usr/bin/env bash
mkdir logs
ssh -o StrictHostKeyChecking=accept-new burst \
"cd ~/rob6323_go2_project && \
sbatch --job-name='rob6323_${USER}' --mail-user='${USER}@nyu.edu' \
tests.slurm '$@'"
