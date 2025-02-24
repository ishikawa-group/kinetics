#!/bin/sh 
#------ pjsub option --------#
#PJM -L rscgrp=n22240a
#PJM -L node=1
#PJM -L elapse=24:00:00 
#PJM -g n22240
#PJM -j
#------- Program execution -------#
NUM_NODES=${PJM_VNODES}
NUM_CORES=40
NUM_PROCS=`expr $NUM_NODES "*" $NUM_CORES`

module load intel

python test.py

