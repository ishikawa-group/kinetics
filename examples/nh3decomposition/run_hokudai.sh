#!/bin/sh 
#------ pjsub option --------#
#PBS -q n22240c
#PBS -l select=1:nsockets=1:mpiprocs=32
#PBS -l walltime=40:00:00
#PBS -W group_list=n22240
#PBS -j oe
#------- Program execution -------#
module purge
module load intel impi

PRG="${HOME}/vasp/vasp.6.4.3/bin/vasp_std"
OUT=stdout_$$

export ASE_VASP_COMMAND="mpiexec.hydra -n ${MPI_PROC} ${PRG} >& ${OUT}"
export VASP_PP_PATH="${HOME}/vasp/potentials"

cd ${PBS_O_WORKDIR}

clean.sh
python prepare_initial.py
python nh3decomp.py --calculator="vasp" --yaml_path="vasp.yaml"

