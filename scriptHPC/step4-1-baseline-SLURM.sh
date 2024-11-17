#! /bin/bash
# Submission script for Lemaitre4
#SBATCH --job-name=step4-1-baseline
#SBATCH --output=log/log_step4-1-baseline_%A_%a.txt
#SBATCH --array=1-60
#SBATCH --time=02:00:00 # hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10000 # megabytes
#SBATCH --partition=batch
#
#SBATCH --mail-user=helene.verhaeghe27+cism@gmail.com
#SBATCH --mail-type=ALL

module load Python/3.11.3-GCCcore-12.3.0
module load virtualenv
source ../venv/bin/activate

if [ -z "${SLURM_ARRAY_TASK_ID}" ]
then # run on laptop
  a=$1
  c=""
else # run with job array
  a=$SLURM_ARRAY_TASK_ID
  c="srun "
fi

# Step 3: evaluation of a model with a given benchmark
# Step 3-1: models from the SK family

workingdir="../target"
scriptdir="../src"
datadir="${workingdir}/data"
ldir="${workingdir}/logs"
outputdir="${ldir}/resultsCA"
configfile="baselineconfig.txt"

caalgoname=$(awk -F "\t" -v JOBID="$a" '$1==JOBID {print $2}' "${outputdir}/${configfile}")
id=$(awk -F "\t" -v JOBID="$a" '$1==JOBID {print $6}' "${outputdir}/${configfile}")
caalgo=$(awk -F "\t" -v JOBID="$a" '$1==JOBID {print $3}' "${outputdir}/${configfile}")
instancenameCA=$(awk -F "\t" -v JOBID="$a" '$1==JOBID {print $4}' "${outputdir}/${configfile}")
instancenamePRIOR=$(awk -F "\t" -v JOBID="$a" '$1==JOBID {print $5}' "${outputdir}/${configfile}")

resultdir="${outputdir}/${caalgoname}/baseline"
mkdir -p ${resultdir}

logfile="${resultdir}/log_b[${instancenamePRIOR}]_[${id}].txt"

# BASELINE

printf "+++ Baseline for CA model ${caalgoname} [complete parameters = ${caalgo}, instance = ${instancenamePRIOR}, run = ${id}] +++\n"

$c python ${scriptdir}/cstAcqAlgos/main.py -a ${caalgo} -b ${instancenameCA} > ${logfile}

printf "+++ Baseline DONE +++\n"






