#! /bin/bash
# Step 4: run

workingdir="../target"
scriptdir="../src"

caalgo="growacq -ia mquacq2-a -qg pqgen -gqg -o proba -c random_forest"
caalgoname="growacqmquacq2-a"
#caalgo=mquacq

outputdir="${workingdir}/logs/resultsCA/${caalgoname}/paf"
mkdir -p ${outputdir}

is=( "sudoku_9" "jsudoku_9" "NRA_3_7_18_5" "JSS_10_3_15_0" "TTS_8_6_3_3_10" "Random_100_5_200" )
js=( "9sudoku" "jsudoku" "nurse_rostering_advanced -nspd 3 -ndfs 7 -nn 18 -nps 5" "job_shop_scheduling -nj 10 -nm 3 -hor 15 -s 0" "exam_timetabling_simple -ns 8 -ncps 6 -nr 3 -ntpd 3 -ndfe 10" "random122" )
for idx in "${!is[@]}"; do
  modnameprior=${is[$idx]}
  modnameca=${js[$idx]}
  echo "bench ${modnameca}"

  for arch in "nn" "rf"; do
    echo " - with arch ${arch}"

    priorname="${arch}_allbut_${modnameprior}"
    logfile="${outputdir}/log_withprior_paf_p[${priorname}]_b[${modnameprior}].txt"
    priorfile="${workingdir}/models/${arch}/${priorname}.pickle"

    echo "   - with prior ${priorname}"
    python ${scriptdir}/cstAcqAlgos/main.py -a ${caalgo} -b ${modnameca} -paf -pf ${priorfile} > ${logfile}

    priorname="${arch}_classical_ca"
    logfile="${outputdir}/log_withprior_paf_p[${priorname}]_b[${modnameprior}].txt"
    priorfile="${workingdir}/models/${arch}/${priorname}.pickle"

    echo "   - with prior ${priorname}"
    python ${scriptdir}/cstAcqAlgos/main.py -a ${caalgo} -b ${modnameca} -paf -pf ${priorfile} > ${logfile}

  done

done