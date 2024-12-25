#!/bin/sh 
#------ pjsub option --------#
#PJM -L rscgrp=n22240a
#PJM -L node=2
#PJM -L elapse=48:00:00 
#PJM -g n22240
#PJM -j
#------- Program execution -------#
NNODES=${PJM_VNODES}
NCORES=40
NPROCS=`expr $NNODES "*" $NCORES`

module load intel

# vasp_script="${HOME}/ase/run_vasp.py"
# echo "import os" > $vasp_script
# echo "exitcode = os.system(\"mpirun -np ${NUM_PROCS} ${PRG}\")" >> $vasp_script

# python -m microkinetics_toolkit
# python orr.py --unique_id=$$ --replace_percent=75

PRG="${HOME}/vasp/vasp.6.4.3/bin/vasp_std"
export ASE_VASP_COMMAND="mpiexec.hydra -ppn ${NCORES} -n ${NPROCS} $PRG"
export VASP_PP_PATH="${HOME}/vasp/potentials"

python orr.py --unique_id=$$

