#! /bin/bash
# To be executed after step 1

workingdir="../target"
scriptdir="../src"

algo="rf"

datadir="${workingdir}/data"
trainingsetdir="${workingdir}/benchmarks/training_sets/classical_CA"
evaluationsetdir="${workingdir}/benchmarks/testing_sets/classical_CA"
modeldir="${workingdir}/models/${algo}"
logdir="${workingdir}/logs/experimentallbutone_${algo}"
mkdir -p ${logdir}

featuresset="aaai24"

for one in "sudoku_9" "jsudoku_9" "NRA_3_7_18_5" "JSS_10_3_15_0" "TTS_8_6_3_3_10" "Random_100_5_200"
do
  trainingset="training_set_classical_ca_allbut_${one}"
  evaluationset="training_set_classical_ca_only_${one}"
  modelname="${algo}_allbut_${one}"

  echo

  python ${scriptdir}/prior/training.py -${algo} -fs ${featuresset} -dd ${datadir} -bm "${trainingsetdir}/${trainingset}.txt" -mf "${modeldir}/${modelname}.pickle" > "${logdir}/training_model_${modelname}.txt"

  python ${scriptdir}/prior/eval.py -${algo} -dd ${datadir} -bm "${evaluationsetdir}/${evaluationset}.txt" -mf "${modeldir}/${modelname}.pickle" > "${logdir}/testing_model_${modelname}.txt"

done

# PLOT

benchmarks="${trainingsetdir}/training_set_classical_ca.txt"
outputgraph="${logdir}/plot.pdf"
python ${scriptdir}/log_analysis/produce_graph.py -all_but -lf "${logdir}/training_model_${algo}_allbut_{}.txt---${logdir}/testing_model_${algo}_allbut_{}.txt" -of ${outputgraph} -bm ${benchmarks}



