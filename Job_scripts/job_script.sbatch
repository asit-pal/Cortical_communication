#!/bin/bash
# vim: syntax=bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=14GB
#SBATCH --job-name=Power_Spectra
#SBATCH --output=job.%j.out

module purge # Clear any existing modules

# Define base paths
BASE_DIR="/home/ap6603/communication-subspace"  # Changed underscore to hyphen

# Run the singularity container
singularity exec --overlay /home/ap6603/Pytorch/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
    /bin/bash << EOF
source /ext3/env.sh
conda activate torch-nn
cd "${BASE_DIR}"


# python "${BASE_DIR}/Data_crcns/v1-v2_gratings/software/example_pca.py"  # Using full path to python script
python "${BASE_DIR}/example.py"
# python "${BASE_DIR}/kohn_data_spont.py"
# python "${BASE_DIR}/plot_neural_activity.py"

EOF
