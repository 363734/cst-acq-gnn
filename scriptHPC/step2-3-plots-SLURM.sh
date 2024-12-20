#! /bin/bash
# Submission script for Lemaitre4
#SBATCH --job-name=step2-3-plottraining
#SBATCH --output=log/log_step2-3-plottraining_%A_%a.txt
#SBATCH --array=1-40
#SBATCH --time=00:10:00 # hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1000 # megabytes
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
# Step 2-3: Creating plots for SK config
workingdir="../target"
scriptdir="../src"
datadir="${workingdir}/data"
trainingsetdir="${workingdir}/benchmarks/training_sets"
ldir="${workingdir}/logs"
configfile="trainingSKconfiggraph.txt"

algo=$(awk -F "\t" -v JOBID="$a" '$1==JOBID {print $2}' "${ldir}/${configfile}")
modelname="$(awk -F "\t" -v JOBID="$a" '$1==JOBID {print $4}' "${ldir}/${configfile}")[{}]"

modeldir="${workingdir}/models/${algo}"
logdir="${ldir}/training_${algo}"
logfile="${logdir}/model_${modelname}.txt"

if [ "${algo}" == "nn" ]
then
  printf "+++ Generating plot loss model ${modelname} [algo = ${algo}] +++\n"
  outputgraphloss="${logdir}/model_${modelname}_plot_loss.pdf"
  $c python ${scriptdir}/log_analysis/produce_graph.py -loss_multi -lf "${logfile}" -of ${outputgraphloss}
  printf "+++ Generating plot loss DONE +++\n"
else
  printf "+++ No plot loss model ${modelname} [algo = ${algo}] +++\n"
fi

printf "+++ Generating plot metric model ${modelname} [algo = ${algo}] +++\n"
outputgraphmetric="${logdir}/model_${modelname}_plot_metric.pdf"
$c python ${scriptdir}/log_analysis/produce_graph.py -train_met_multi -lf "${logfile}" -of ${outputgraphmetric}
printf "+++ Generating plot metric DONE +++\n"
