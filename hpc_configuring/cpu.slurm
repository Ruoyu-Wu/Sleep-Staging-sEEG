#!/bin/bash

#SBATCH --job-name=tutorial

#SBATCH --output=%j.out
#SBATCH --error=%j.err

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=1

#SBATCH --nodes=1

#SBATCH --mem=20G

#SBATCH --partition=debug

#SBATCH --time=00:30:00
module load miniconda3
source activate project1
python AttnSleep_config.py
