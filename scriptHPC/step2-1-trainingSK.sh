#! /bin/bash
# Submission script for Lemaitre4
#SBATCH --job-name=step2-1-trainingSK
#SBATCH --output=log/log_step2-1-trainingSK_%A_%a.txt
#SBATCH --array=1-400
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

# Step 2: training predictors with a given benchmark
# Step 2-1: training scikit-learn predictor

workingdir="../target"
scriptdir="../src"
datadir="${workingdir}/data"
trainingsetdir="${workingdir}/benchmarks/training_sets"
ldir="${workingdir}/logs"
configfile="trainingSKconfig.txt"

algo=$(awk -F "\t" -v JOBID="$a" '$1==JOBID {print $2}' "${ldir}/${configfile}")
id=$(awk -F "\t" -v JOBID="$a" '$1==JOBID {print $6}' "${ldir}/${configfile}")
trainingset=$(awk -F "\t" -v JOBID="$a" '$1==JOBID {print $3}' "${ldir}/${configfile}")
modelname="$(awk -F "\t" -v JOBID="$a" '$1==JOBID {print $4}' "${ldir}/${configfile}")[${id}]"
featuresset=$(awk -F "\t" -v JOBID="$a" '$1==JOBID {print $5}' "${ldir}/${configfile}")

modeldir="${workingdir}/models/${algo}"
logdir="${ldir}/training_${algo}"


logfile="${logdir}/model_${modelname}.txt"

# TRAINING

printf "+++ Training model ${modelname} [algo = ${algo}, training set = ${trainingset}, feature set = ${featuresset}] +++\n"

$c python ${scriptdir}/prior/training.py -"${algo}" -fs "${featuresset}" -dd "${datadir}" -bm "${trainingsetdir}/${trainingset}.txt" -mf "${modeldir}/${modelname}.pickle" > "${logfile}"

printf "+++ Training DONE +++\n"

