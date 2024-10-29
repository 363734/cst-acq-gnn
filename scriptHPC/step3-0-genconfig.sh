#! /bin/bash
# Step 3: evaluation of a model with a given benchmark
# Step 3-0: generate the file containing all the parameters set to be generated

workingdir="../target"
scriptdir="../src"
datadir="${workingdir}/data"
trainingsetdir="${workingdir}/benchmarks/testing_sets"
logdir="${workingdir}/logs"
mkdir -p ${logdir}

outputfile1="testingSKconfig.txt"
outputfile2="testingGNNconfig.txt"

printf "JOBID\tALGO\tMODELNAME\tRUNID\tTESTSET\n" > "${logdir}/${outputfile1}"
printf "JOBID\tALGO\tMODELNAME\tRUNID\tTESTSET\n" > "${logdir}/${outputfile2}"

jobid1=1
jobid2=1

modelnames=(
  "sudoku"
  "classical_ca"
  "classical_ca_allbut_JSS_10_3_15_0"
  "classical_ca_allbut_jsudoku_9"
  "classical_ca_allbut_NRA_3_7_18_5"
  "classical_ca_allbut_Random_100_5_200"
  "classical_ca_allbut_sudoku_9"
  "classical_ca_allbut_TTS_8_6_3_3_10"
)
testsets=(
  "testing_set_sudoku"
  "classical_CA/testing_set_classical_ca_same_family"
  "classical_CA/training_set_classical_ca_only_JSS_10_3_15_0"
  "classical_CA/training_set_classical_ca_only_jsudoku_9"
  "classical_CA/training_set_classical_ca_only_NRA_3_7_18_5"
  "classical_CA/training_set_classical_ca_only_Random_100_5_200"
  "classical_CA/training_set_classical_ca_only_sudoku_9"
  "classical_CA/training_set_classical_ca_only_TTS_8_6_3_3_10"
)
algos=("nn" "rf" "cnb" "gnb" "svm")
for i in "${!modelnames[@]}"
do
  for runid in {0..9}
  do
    for algo in "${algos[@]}"
    do
      printf "${jobid1}\t${algo}\t${algo}_${modelnames[$i]}\t${runid}\t${testsets[i]}\n" >> "${logdir}/${outputfile1}"
      jobid1=$((jobid1 + 1))
    done
    printf "${jobid2}\tgnn\tgnn_${modelnames[$i]}\t${runid}\t${testsets[i]}\n" >> "${logdir}/${outputfile2}"
    jobid2=$((jobid2 + 1))
  done
done



outputfile1g="testingSKconfiggraph.txt"
outputfile2g="testingGNNconfiggraph.txt"

printf "JOBID\tALGO\tMODELNAME\tTESTSET\n" > "${logdir}/${outputfile1g}"
printf "JOBID\tALGO\tMODELNAME\tTESTSET\n" > "${logdir}/${outputfile2g}"

jobid1=1
jobid2=1
for i in "${!modelnames[@]}"
do
  for algo in "${algos[@]}"
  do
    printf "${jobid1}\t${algo}\t${algo}_${modelnames[$i]}\t${testsets[i]}\n" >> "${logdir}/${outputfile1g}"
    jobid1=$((jobid1 + 1))
  done
  printf "${jobid2}\tgnn\tgnn_${modelnames[$i]}\t${testsets[i]}\n" >> "${logdir}/${outputfile2g}"
  jobid2=$((jobid2 + 1))
done
