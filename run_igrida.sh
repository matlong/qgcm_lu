#!/bin/bash
#OAR -l {host='igrida13-01.irisa.fr' AND dedicated='serpico'}/nodes=1,walltime=24:00:00 
#OAR -t besteffort
#OAR -t idempotent
#OAR -O /srv/tempdd/loli/output/make.%jobid%.output
#OAR -E /srv/tempdd/loli/output/make.%jobid%.error

QGCMDIR=/srv/tempdd/${USER}/qgcm-lu/
rcopt=1 # option for recompilation

# Load modules
. /etc/profile.d/modules.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/srv/soft/spack/opt/spack/linux-debian10-x86_64/gcc-8.3.0/netcdf-fortran-4.5.3-kcighyw4a7jd5wberlf3ic3znjoko3my/lib:/srv/soft/spack/opt/spack/linux-debian10-x86_64/gcc-8.3.0/hdf5-1.10.7-6wclfm4i64jkuc7uwdfqio2c7xufkdoy/lib
module load spack/netcdf-fortran/4.5.3/gcc-8.3.0-kcighyw

# Compile q-gcm (optional)
cd $QGCMDIR
mkdir -p outdata
if [[ "$rcopt" == "1" ]]; then
  echo "Compiling q-gcm"
  rm q-gcm
  cd ./src
  make clean
  rm make.macro make.config parameters_data.F luparam_data.F q-gcm
  ln -s ../make.macro_igrida make.macro
  ln -s ../make.config .
  ln -s ../parameters_data.F .
  ln -s ../luparam_data.F .
  make &> make.log
  cat make.log
  cd ..
  ln -s ./src/q-gcm .
  python create_forcing.py
else
  echo "Do not require recompilation"
fi

# Excute q-gcm
echo "Running q-gcm"
ulimit -s hard
./q-gcm &> qgcm.output
cat qgcm.output
mv qgcm.output outdata/

echo "Done"
exit 1
