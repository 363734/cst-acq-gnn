#! /bin/bash
# Step 2: training a neural network with a given benchmark

workingdir="../target"
scriptdir="../src"

algo="nn"

datadir="${workingdir}/data"
trainingsetdir="${workingdir}/benchmarks/training_sets"
modeldir="${workingdir}/models/${algo}"
logdir="${workingdir}/logs/training_${algo}"
mkdir -p ${logdir}

featuresset="aaai24"

trainingset="training_set_sudoku"
modelname="${algo}_sudoku"

trainingset="classical_CA/training_set_classical_ca"
modelname="${algo}_classical_ca"

logfile="${logdir}/model_${modelname}.txt"

# TRAINING

python ${scriptdir}/prior/training.py -${algo} -fs ${featuresset} -dd ${datadir} -bm "${trainingsetdir}/${trainingset}.txt" -mf "${modeldir}/${modelname}.pickle" > ${logfile}

# PLOTS

outputgraphloss="${logdir}/model_${modelname}_plot_loss.pdf"
python ${scriptdir}/log_analysis/produce_graph.py -loss -lf ${logfile} -of ${outputgraphloss}

outputgraphmetric="${logdir}/model_${modelname}_plot_metric.pdf"
python ${scriptdir}/log_analysis/produce_graph.py -train_met -lf ${logfile} -of ${outputgraphmetric}