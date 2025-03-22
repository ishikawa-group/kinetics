#!/bin/sh
#$ -cwd
#$ -l cpu_40=1
#$ -l h_rt=10:00:00
#$ -N flatmpi

NCORE=40
# 160, 80, 40, 16, 8, 4

source /etc/profile.d/modules.sh

module load intel
module load intel-mpi

PRG="/home/1/uk02411/vasp/vasp.6.4.3/bin/vasp_std"
export VASP_PP_PATH="/home/1/uk02411/vasp/potentials"
export ASE_VASP_COMMAND="mpiexec.hydra -ppn $NCORE -n $NCORE $PRG"

python orr.py --dirname=$$ --start=0 --end=10

