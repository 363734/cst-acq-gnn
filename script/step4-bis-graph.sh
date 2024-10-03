#! /bin/bash
# Step 4: run

workingdir="../target"
scriptdir="../src"
trainingsetdir="${workingdir}/benchmarks/training_sets/classical_CA"

caalgoname="growacqmquacq2-a"

outputdir="${workingdir}/logs/resultsCA/${caalgoname}"
mkdir -p ${outputdir}

priorname='nn_classical_ca'
#priorname='nn_allbut'
#priorname='rf_classical_ca'
#priorname='rf_allbut'



benchmarks="${trainingsetdir}/training_set_classical_ca.txt"
outputname="output_[${priorname}]"
outputgraph="${outputdir}/${outputname}"

python ${scriptdir}/log_analysis/produce_graph.py -results_ca -bm "${benchmarks}" -of "${outputgraph}.tex" -lf ${outputdir} -pn ${priorname}

pdflatex "${outputgraph}.tex"

mv "${outputname}.pdf" "${outputgraph}.pdf"
rm "${outputname}.aux"
rm "${outputname}.log"


