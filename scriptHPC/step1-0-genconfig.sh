#! /bin/bash
# Step 1: creation/loading of datasets (creating of the pickle file for each models)
# Step 1-0: generate the file containing all the parameters set to be generated

datadir="../target/data"
scriptdir="../src"
mkdir -p ${datadir}

outputfile="fulldatasetconfig.txt"

jobid=1

### Classical CA instances ###

printf "JOBID\tOPTIONS\n" > "${datadir}/${outputfile}"

#sudoku instances
for GS in 4 9 16 25
do
  printf "${jobid}\t-n sudoku_${GS} -b sudoku -gs ${GS}\n" >> "${datadir}/${outputfile}"
  jobid=$((jobid + 1))
done


#rsudoku instances
for dim1 in 1 2 3 4 5
do
  for dim2 in 1 2 3 4 5
  do
    if [ "$dim1" != "$dim2" ]
    then
      printf "${jobid}\t-n \"rsudoku_${dim1}_${dim2}\" -b rsudoku -dim1 ${dim1} -dim2 ${dim2}\n" >> "${datadir}/${outputfile}"
      jobid=$((jobid + 1))
    fi
  done
done

#jsudoku instances
printf "${jobid}\t-n \"jsudoku_9\" -b jsudoku\n" >> "${datadir}/${outputfile}"
jobid=$((jobid + 1))

#nurse rostering instances
for numshiftsperday in 3 4 # CA=3
do
  for numdaysforschedule in 5 7 10 # CA=7
  do
    for numnurses in 16 18 20 # CA=18
    do
      for nursespershift in 4 5 6 # CA=5
      do
        printf "${jobid}\t-n \"NRA_${numshiftsperday}_${numdaysforschedule}_${numnurses}_${nursespershift}\" -b nurse_rostering_adv -nspd ${numshiftsperday} -ndfs ${numdaysforschedule} -nn ${numnurses} -nps ${nursespershift}\n" >> "${datadir}/${outputfile}"
        jobid=$((jobid + 1))
      done
    done
  done
done

#exam timetabling instances
for numsemesters in 4 8 12 # CA=8
do
  for numcoursespersemester in 3 6 9 # CA=6
  do
    for numrooms in 3 5 7 # CA=3
    do
      for numtimeslotsperday in 3 4 # CA=3
      do
        for numdaysforexams in 5 10 15 # CA=10
        do
          printf "${jobid}\t-n \"TTS_${numsemesters}_${numcoursespersemester}_${numrooms}_${numtimeslotsperday}_${numdaysforexams}\" -b exam_timetabling_simple -ns ${numsemesters} -ncps ${numcoursespersemester} -nr ${numrooms} -ntpd ${numtimeslotsperday} -ndfe ${numdaysforexams}\n" >> "${datadir}/${outputfile}"
          jobid=$((jobid + 1))
        done
      done
    done
  done
done

#job shop scheduling instances
for numjobs in 10 15 20 # CA=10
do
  for nummachines in 3 4 5 # CA=3
  do
    for horizon in 10 15 20 # CA=15
    do
      for seed in 0 1 2 # CA=0
      do
        printf "${jobid}\t-n \"JSS_${numjobs}_${nummachines}_${horizon}_${seed}\" -b job_shop_scheduling -nj ${numjobs} -nm ${nummachines} -hor ${horizon} -s ${seed}\n" >> "${datadir}/${outputfile}"
        jobid=$((jobid + 1))
      done
    done
  done
done

#random instances
for numbervars in 50 100 # CA=100
do
  for sizedomain in 5 10 # CA=5
  do
    for numbercst in 200 300 400 # CA=200
    do
      printf "${jobid}\t-n \"Random_${numbervars}_${sizedomain}_${numbercst}\" -b random -nv ${numbervars} -sd ${sizedomain} -nc ${numbercst}\n" >> "${datadir}/${outputfile}"
      jobid=$((jobid + 1))
    done
  done
done