#!/bin/sh
#$ -cwd
#$ -l cpu_8=1
#$ -l h_rt=01:00:00
#$ -N flatmpi

NCORE=8

source /etc/profile.d/modules.sh

module load intel
module load intel-mpi

PRG="/home/1/uk02411/vasp/vasp.6.4.3/bin/vasp_std"

export VASP_PP_PATH="/home/1/uk02411/vasp/potential"
export ASE_VASP_COMMAND="mpiexec.hydra -ppn $NCORE -n $NCORE $PRG"

python test_bandgap.py

