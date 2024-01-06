#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=foo
#SBATCH --output=foo.out

python train.py overwrite=True
