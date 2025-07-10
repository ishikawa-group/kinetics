#!/bin/sh 
#------ pjsub option --------#
#PBS -q n22240c
#PBS -l select=1:nsockets=2:mpiprocs=64
#PBS -l walltime=01:00:00
#PBS -W group_list=n22240
#PBS -j oe
#------- Program execution -------#

module purge
module load intel impi

# vasp_script="${HOME}/ase/run_vasp.py"
# echo "import os" > $vasp_script
# echo "exitcode = os.system(\"mpirun -np ${NUM_PROCS} ${PRG}\")" >> $vasp_script

PRG=${HOME}/vasp/vasp.6.4.3/bin/vasp_std
export ASE_VASP_COMMAND="mpiexec.hydra -n ${MPI_PROC} $PRG > stdout.out"
export VASP_PP_PATH="${HOME}/vasp/potentials"

cd ${PBS_O_WORKDIR}

# clean up
clean.sh

# calculation
python oer.py --max_sample=1 --calculator="vasp" --vacuum=7.5

