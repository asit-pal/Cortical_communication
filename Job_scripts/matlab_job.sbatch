#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=MATLAB
#SBATCH --output=matlab_job_%j.out

module purge
module load matlab/2024a

# Change to the software directory
cd /home/ap6603/communication-subspace/Data_crcns/v1-v2_gratings/software

# Run MATLAB script with addpath commands and run startup.m first
matlab -nodisplay -nodesktop -r "addpath('/home/ap6603/communication-subspace/Data_crcns/v1-v2_gratings'); addpath('/home/ap6603/communication-subspace/Data_crcns/v1-v2_gratings/software'); cd('/home/ap6603/communication-subspace/Data_crcns/v1-v2_gratings'); startup; cd('/home/ap6603/communication-subspace/Data_crcns/v1-v2_gratings/software'); example; exit" 