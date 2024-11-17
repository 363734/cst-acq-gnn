#! /bin/bash
# Submission script for Lemaitre4
#SBATCH --job-name=step4-3-prior-as-multiplier
#SBATCH --output=log/log_step4-2-prior-as-multiplier_%A_%a.txt
#SBATCH --array=1-2400
#SBATCH --time=02:00:00 # hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10000 # megabytes
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

# Step 4:
# Step 4-3:

workingdir="../target"
scriptdir="../src"
datadir="${workingdir}/data"
ldir="${workingdir}/logs"
outputdir="${ldir}/resultsCA"
configfile="withpriorpamconfig.txt"

caalgoname=$(awk -F "\t" -v JOBID="$a" '$1==JOBID {print $2}' "${outputdir}/${configfile}")
id=$(awk -F "\t" -v JOBID="$a" '$1==JOBID {print $8}' "${outputdir}/${configfile}")
caalgo=$(awk -F "\t" -v JOBID="$a" '$1==JOBID {print $3}' "${outputdir}/${configfile}")
arch=$(awk -F "\t" -v JOBID="$a" '$1==JOBID {print $4}' "${outputdir}/${configfile}")
priorname="$(awk -F "\t" -v JOBID="$a" '$1==JOBID {print $5}' "${outputdir}/${configfile}")[${id}]"
instancenameCA=$(awk -F "\t" -v JOBID="$a" '$1==JOBID {print $6}' "${outputdir}/${configfile}")
instancenamePRIOR=$(awk -F "\t" -v JOBID="$a" '$1==JOBID {print $7}' "${outputdir}/${configfile}")
lambda=$(awk -F "\t" -v JOBID="$a" '$1==JOBID {print $9}' "${outputdir}/${configfile}")

resultdir="${outputdir}/${caalgoname}/pam"
mkdir -p ${resultdir}

logfile="${resultdir}/log_withprior_pam_${lambda}_p[${priorname}]_b[${instancenamePRIOR}].txt"

priorfile="${workingdir}/models/${arch}/${priorname}.pickle"

# Multiplier

printf "+++ Prior as a multiplier for CA model ${caalgoname} [complete parameters = ${caalgo}, instance = ${instancenamePRIOR}, prior model = ${priorname}, lambda = ${lambda}] +++\n"

$c python ${scriptdir}/cstAcqAlgos/main.py -a ${caalgo} -b ${instancenameCA} -pam -lam ${lambda} -pf ${priorfile} > ${logfile}

printf "+++ With prior DONE +++\n"






