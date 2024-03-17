#!/bin/bash
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH -p cp1 
#SBTACH -J pdes

source activate pdes

yhrun python main.py
