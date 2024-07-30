#!/bin/bash
#SBATCH --job-name=speech-classifier  # Job name
#SBATCH --output=output.txt           # Standard output file
#SBATCH --error=error.txt             # Standard error file
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=1             # Number of CPU cores per task
#SBATCH --time=15:00:00                # Maximum runtime (D-HH:MM:SS)
#SBATCH --mail-type=END               # Send email at job completion
#SBATCH --mail-user=arseneausm@gmail.com    # Email address for notifications

echo "Start Job $SLURM_ARRAY_TASK_ID on $HOSTNAME"  # Display job start information

module load anaconda
conda activate mlenv

echo "Using dataset $1"
echo "Sentence chunking param: $2"
echo "Score weighting param: $3"

python src/scoring.py $1  --sentence-chunking=$2 --score-weighting=$3
