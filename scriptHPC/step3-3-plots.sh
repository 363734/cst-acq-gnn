#! /bin/bash
# Submission script for Lemaitre4
#SBATCH --job-name=step3-3-plotsevaluation
#SBATCH --output=log/log_step3-3-plotsevaluation_%A_%a.txt
#SBATCH --array=1-40
#SBATCH --time=00:10:00 # hh:mm:ss
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

# Step 3: evaluation a neural network with a given benchmark
# Step 3-3: plots the results

workingdir="../target"
scriptdir="../src"
datadir="${workingdir}/data"
evaluationsetdir="${workingdir}/benchmarks/testing_sets"
ldir="${workingdir}/logs"
configfile="testingSKconfiggraph.txt"

algo=$(awk -F "\t" -v JOBID="$a" '$1==JOBID {print $2}' "${ldir}/${configfile}")
modelname="$(awk -F "\t" -v JOBID="$a" '$1==JOBID {print $3}' "${ldir}/${configfile}")[{}]"
testingset=$(awk -F "\t" -v JOBID="$a" '$1==JOBID {print $4}' "${ldir}/${configfile}")
echo ${modelname}
echo ${testingset}

logdir="${ldir}/eval_${algo}"
logfile="${logdir}/model_${modelname}_[${testingset//\//-}].txt"

echo ${logfile}

printf "+++ Plots for model ${modelname} [algo = ${algo}, testing set = ${testingset}] +++\n"

outputgraphmetric="${logdir}/model_${modelname}_[${testingset//\//-}]_plot_metric.pdf"
python ${scriptdir}/log_analysis/produce_graph.py -eval_met_multi -lf ${logfile} -of ${outputgraphmetric}

printf "+++ Plots DONE +++\n"


