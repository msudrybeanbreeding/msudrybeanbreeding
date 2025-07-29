#!/bin/bash --login
#SBATCH --job-name=train                 # Name of the job
#SBATCH --output=job_logs/%x_%j.out      # Output log file (%x=job name, %j=job ID)
#SBATCH --error=job_logs/%x_%j.err       # Error log file
#SBATCH --ntasks=1                       # Number of tasks (usually 1 for single job)
#SBATCH --cpus-per-task=1                # Number of CPU cores per task
#SBATCH --gpus=v100:1                    # Type and number of GPUs
#SBATCH --mem=16G                        # Total memory
#SBATCH --time=01:00:00                  # Time limit (hh:mm:ss)
#SBATCH --mail-type=ALL                  # Email notifications
#SBATCH --mail-user=lellaom@msu.edu      # Your email address

# Activate your virtual environment
source /mnt/scratch/lellaom/msudrybeanbreeding/Canned-Beans-Training-Pipeline/venv/bin/activate

# Navigate to your project directory
cd /mnt/scratch/lellaom/msudrybeanbreeding/Canned-Beans-Training-Pipeline

# Run your Python script
python train.py