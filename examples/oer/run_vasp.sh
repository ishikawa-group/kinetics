#!/bin/sh 
#------ pjsub option --------#
#PJM -L rscgrp=n22240a
#PJM -L node=1
#PJM -L elapse=20:00:00 
#PJM -g n22240
#PJM -j
#------- Program execution -------#
NUM_NODES=${PJM_VNODES}
NUM_CORES=40
NUM_PROCS=`expr $NUM_NODES "*" $NUM_CORES`

module load intel

PRG=${HOME}/vasp/vasp.6.4.3/bin/vasp_std
#MOL=`echo ${INP} | cut -d . -f 1`
DIRNAME=`pwd | rev | cut -d "/" -f 1 | rev`
#OUT=stdout_$$
OUT=${DIRNAME}.out

# remove old results
rm stdout* stderr*

mpiexec.hydra -n ${NUM_PROCS} ${PRG} >& ${OUT}

