#! /bin/bash
# Step 2: training predictors with a given benchmark
# Step 2-0: generate the file containing all the parameters set to be generated

workingdir="../target"
scriptdir="../src"
datadir="${workingdir}/data"
trainingsetdir="${workingdir}/benchmarks/training_sets"
logdir="${workingdir}/logs"
mkdir -p ${logdir}

outputfile1="trainingSKconfig.txt"
outputfile2="trainingGNNconfig.txt"

printf "JOBID\tALGO\tTRAININGSET\tMODELNAME\tFEATURESET\tRUNID\n" > "${logdir}/${outputfile1}"
printf "JOBID\tALGO\tTRAININGSET\tMODELNAME\tFEATURESET\tRUNID\n" > "${logdir}/${outputfile2}"

jobid1=1
jobid2=1

trainsets=(
  "training_set_sudoku"
  "classical_CA/training_set_classical_ca"
  "classical_CA/training_set_classical_ca_allbut_JSS_10_3_15_0"
  "classical_CA/training_set_classical_ca_allbut_jsudoku_9"
  "classical_CA/training_set_classical_ca_allbut_NRA_3_7_18_5"
  "classical_CA/training_set_classical_ca_allbut_Random_100_5_200"
  "classical_CA/training_set_classical_ca_allbut_sudoku_9"
  "classical_CA/training_set_classical_ca_allbut_TTS_8_6_3_3_10"
)
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
algos=("nn" "rf" "cnb" "gnb" "svm")
for i in "${!modelnames[@]}"
do
  for featureset in "aaai24"
  do
    for runid in {0..9}
    do
      for algo in "${algos[@]}"
      do
        printf "${jobid1}\t${algo}\t${trainsets[$i]}\t${algo}_${modelnames[$i]}\t${featureset}\t${runid}\n" >> "${logdir}/${outputfile1}"
        jobid1=$((jobid1 + 1))
      done
      printf "${jobid2}\tgnn\t${trainsets[$i]}\tgnn_${modelnames[$i]}\t${featureset}\t${runid}\n" >> "${logdir}/${outputfile2}"
      jobid2=$((jobid2 + 1))
    done
  done
done


outputfile1g="trainingSKconfiggraph.txt"
outputfile2g="trainingGNNconfiggraph.txt"

printf "JOBID\tALGO\tTRAININGSET\tMODELNAME\n" > "${logdir}/${outputfile1g}"
printf "JOBID\tALGO\tTRAININGSET\tMODELNAME\n" > "${logdir}/${outputfile2g}"

jobid1=1
jobid2=1
for i in "${!modelnames[@]}"
do
  for algo in "${algos[@]}"
  do
    printf "${jobid1}\t${algo}\t${trainsets[$i]}\t${algo}_${modelnames[$i]}\n" >> "${logdir}/${outputfile1g}"
    jobid1=$((jobid1 + 1))
  done
  printf "${jobid2}\tgnn\t${trainsets[$i]}\tgnn_${modelnames[$i]}\n" >> "${logdir}/${outputfile2g}"
  jobid2=$((jobid2 + 1))
done