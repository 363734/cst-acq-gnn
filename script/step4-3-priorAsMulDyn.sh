#! /bin/bash
# Step 4: run

workingdir="../target"
scriptdir="../src"

caalgo="growacq -ia mquacq2-a"
caalgoname="growacqmquacq2-a"
#caalgo=mquacq

outputdir="${workingdir}/logs/resultsCA/${caalgoname}/pamd"
mkdir -p ${outputdir}

is=( "sudoku_9" "jsudoku_9" "NRA_3_7_18_5" "JSS_10_3_15_0" "TTS_8_6_3_3_10" "Random_100_5_200" )
js=( "9sudoku" "jsudoku" "nurse_rostering_advanced -nspd 3 -ndfs 7 -nn 18 -nps 5" "job_shop_scheduling -nj 10 -nm 3 -hor 15 -s 0" "exam_timetabling_simple -ns 8 -ncps 6 -nr 3 -ntpd 3 -ndfe 10" "random122" )
for idx in "${!is[@]}"; do
  modnameprior=${is[$idx]}
  modnameca=${js[$idx]}
  echo "bench ${modnameca}"

  for arch in "nn" "rf"; do
    echo " - with arch ${arch}"
    for dec in "0.99" "0.95" "0.9"; do
      echo " - with decay ${dec}"
      lam=1

      priorname="${arch}_allbut_${modnameprior}"
      logfile="${outputdir}/log_withprior_pamd_${lam}-${dec}_p[${priorname}]_b[${modnameprior}].txt"
      priorfile="${workingdir}/models/${arch}/${priorname}.pickle"

      echo "   - with prior ${priorname}"
      python ${scriptdir}/cstAcqAlgos/main.py -a ${caalgo} -b ${modnameca} -pamd -lam ${lam} -dec ${dec} -pf ${priorfile}> ${logfile}

      priorname="${arch}_classical_ca"
      logfile="${outputdir}/log_withprior_pamd_${lam}-${dec}_p[${priorname}]_b[${modnameprior}].txt"
      priorfile="${workingdir}/models/${arch}/${priorname}.pickle"

      echo "   - with prior ${priorname}"
      python ${scriptdir}/cstAcqAlgos/main.py -a ${caalgo} -b ${modnameca} -pamd -lam ${lam} -dec ${dec} -pf ${priorfile} > ${logfile}

    done
  done

done