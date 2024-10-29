#! /bin/bash
# Submission script for Lemaitre4
#SBATCH --job-name=step3-1-evaluationSK
#SBATCH --output=log/log_step3-1-evaluationSK_%A_%a.txt
#SBATCH --array=1-400
#SBATCH --time=02:00:00 # hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10000 # megabytes
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

# Step 3: evaluation of a model with a given benchmark
# Step 3-1: models from the SK family

workingdir="../target"
scriptdir="../src"
datadir="${workingdir}/data"
evaluationsetdir="${workingdir}/benchmarks/testing_sets"
ldir="${workingdir}/logs"
configfile="testingSKconfig.txt"

algo=$(awk -F "\t" -v JOBID="$a" '$1==JOBID {print $2}' "${ldir}/${configfile}")
id=$(awk -F "\t" -v JOBID="$a" '$1==JOBID {print $4}' "${ldir}/${configfile}")
modelname="$(awk -F "\t" -v JOBID="$a" '$1==JOBID {print $3}' "${ldir}/${configfile}")[${id}]"
testingset=$(awk -F "\t" -v JOBID="$a" '$1==JOBID {print $5}' "${ldir}/${configfile}")



modeldir="${workingdir}/models/${algo}"
logdir="${ldir}/eval_${algo}"
mkdir -p ${logdir}

logfile="${logdir}/model_${modelname}_[${testingset//\//-}].txt"

# TRAINING

printf "+++ Evaluating model ${modelname} [algo = ${algo}, testing set = ${testingset}] +++\n"

python ${scriptdir}/prior/eval.py -${algo} -dd ${datadir} -bm "${evaluationsetdir}/${testingset}.txt" -mf "${modeldir}/${modelname}.pickle" > ${logfile}

printf "+++ Evaluation DONE +++\n"






