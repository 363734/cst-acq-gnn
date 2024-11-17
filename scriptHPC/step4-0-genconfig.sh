#! /bin/bash

module load Python/3.11.3-GCCcore-12.3.0
module load virtualenv
source ../venv/bin/activate

# Step 4: evaluation of a prior when doing constraint acquisition
# Step 4-0: generate the file containing all the parameters set to be generated

workingdir="../target"
scriptdir="../src"
datadir="${workingdir}/data"
trainingsetdir="${workingdir}/benchmarks/testing_sets"
logdir="${workingdir}/logs"
mkdir -p ${logdir}


caalgo="growacq -ia mquacq2-a -qg pqgen -gqg -o proba -c random_forest"
caalgoname="growacqmquacq2-a"

is=( "sudoku_9" "jsudoku_9" "NRA_3_7_18_5" "JSS_10_3_15_0" "TTS_8_6_3_3_10" "Random_100_5_200" )
js=( "9sudoku" "jsudoku" "nurse_rostering_advanced -nspd 3 -ndfs 7 -nn 18 -nps 5" "job_shop_scheduling -nj 10 -nm 3 -hor 15 -s 0" "exam_timetabling_simple -ns 8 -ncps 6 -nr 3 -ntpd 3 -ndfe 10" "random122" )

algos=("nn" "rf" "cnb" "gnb" "svm")

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

lambdas=(
  "0.25"
  "0.5"
  "0.75"
  "1"
)

outputdir="${logdir}/resultsCA"
mkdir -p ${outputdir}

outputfile1="baselineconfig.txt"
outputfile2="withpriorpafconfig.txt"
outputfile3="withpriorpamconfig.txt"

printf "JOBID\tALGOCANAME\tALGOCA\tINSTNAMECA\tINSTNAMEPRIOR\tRUNID\n" > "${outputdir}/${outputfile1}"
printf "JOBID\tALGOCANAME\tALGOCA\tPRIORARCH\tPRIORMODEL\tINSTNAMECA\tINSTNAMEPRIOR\tRUNID\n" > "${outputdir}/${outputfile2}"
printf "JOBID\tALGOCANAME\tALGOCA\tPRIORARCH\tPRIORMODEL\tINSTNAMECA\tINSTNAMEPRIOR\tRUNID\tLAMBDA\n" > "${outputdir}/${outputfile3}"

jobid1=1
jobid2=1
jobid3=1

for idx in "${!is[@]}"; do
  for runid in {0..9}
  do
    instancenameCA=${js[$idx]}
    instancenamePRIOR=${is[$idx]}
    printf "${jobid1}\t${caalgoname}\t${caalgo}\t${instancenameCA}\t${instancenamePRIOR}\t${runid}\t\n" >> "${outputdir}/${outputfile1}"
    jobid1=$((jobid1 + 1))
  done
done

for idx in "${!is[@]}"; do
  for i in "${!modelnames[@]}"
  do
    for algo in "${algos[@]}"
    do
      for runid in {0..9}
      do
        instancenameCA=${js[$idx]}
        instancenamePRIOR=${is[$idx]}
        printf "${jobid2}\t${caalgoname}\t${caalgo}\t${algo}\t${algo}_${modelnames[$i]}\t${instancenameCA}\t${instancenamePRIOR}\t${runid}\t\n" >> "${outputdir}/${outputfile2}"
        jobid2=$((jobid2 + 1))
        for lambda in "${lambdas[@]}"
        do
          printf "${jobid3}\t${caalgoname}\t${caalgo}\t${algo}\t${algo}_${modelnames[$i]}\t${instancenameCA}\t${instancenamePRIOR}\t${runid}\t${lambda}\t\n" >> "${outputdir}/${outputfile3}"
          jobid3=$((jobid3 + 1))
        done
      done
    done
  done
done




