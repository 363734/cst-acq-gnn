#! /bin/bash
# Step 3: evaluation a neural network with a given benchmark

workingdir="../target"
scriptdir="../src"

algo="nn"

datadir="${workingdir}/data"
evaluationsetdir="${workingdir}/benchmarks/testing_sets"
modeldir="${workingdir}/models/${algo}"
logdir="${workingdir}/logs/eval_${algo}"
mkdir -p ${logdir}

#testingset="testing_set_sudoku"
#modelname="${algo}_sudoku"

testingset="classical_CA/testing_set_classical_ca_same_family"
modelname="${algo}_classical_ca"

logfile="${logdir}/model_${modelname}.txt"

# TRAINING

python ${scriptdir}/prior/eval.py -${algo} -dd ${datadir} -bm "${evaluationsetdir}/${testingset}.txt" -mf "${modeldir}/${modelname}.pickle" > ${logfile}

# PLOTS

outputgraphmetric="${logdir}/model_${modelname}_plot_metric.pdf"
python ${scriptdir}/log_analysis/produce_graph.py -eval_met -lf ${logfile} -of ${outputgraphmetric}






