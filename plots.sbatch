#!/bin/sh
#SBATCH -J dust_plots
#SBATCH -p skx-normal
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A TG-AST120025
#SBATCH --mail-user=cchoban@ucsd.edu
#SBATCH --mail-type=all
#SBATCH -o dust_plots.log
#SBATCH --export=ALL
#SBATCH -D .
set -e
pwd
date


python paper_plots.py
