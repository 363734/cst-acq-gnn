#! /bin/bash
# Submission script for Lemaitre4
#SBATCH --job-name=step1-1-generationDataBench
#SBATCH --output=log/log_step1-1-generationDataBench_%A_%a.txt
#SBATCH --array=1-334
#SBATCH --time=00:30:00 # hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1000 # megabytes
#SBATCH --partition=batch
#
#SBATCH --mail-user=helene.verhaeghe27+cism@gmail.com
#SBATCH --mail-type=ALL

if [ -z "${SLURM_ARRAY_TASK_ID}" ]
then # run on laptop
  a=$1
  c=""
else # run with job array
  a=$SLURM_ARRAY_TASK_ID
  c="srun "
fi

# Step 1: creation/loading of datasets (creating of the pickle file for each models)
# Step 1-1: for each set of parameter allow generation of the benchmark

datadir="../target/data"
scriptdir="../src"
configfile="fulldatasetconfig.txt"

opts=$(awk -F "\t" -v JOBID="$a" '$1==JOBID {print $2}' "${datadir}/${configfile}")

printf "+++ Generation with options \"${opts}\" +++\n"

$c python ${scriptdir}/benchmark/create.py -d $datadir ${opts}

printf "+++ Generation DONE +++\n"


