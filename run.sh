#!/bin/bash

set -x
ulimit -s hard
export OMP_NUM_THREADS=4

rcopt=1 # option for recompilation

mkdir -p outdata
if [[ "$rcopt" == "1" ]]; then
  echo "Compiling q-gcm"
  rm q-gcm
  cd ./src
  make clean
  rm make.macro make.config parameters_data.F luparam_data.F q-gcm
  ln -s ../make.macro .
  ln -s ../make.config .
  ln -s ../parameters_data.F .
  ln -s ../luparam_data.F .
  make 
  cd ..
  ln -s ./src/q-gcm .
  python create_forcing.py
else
  echo "Do not require recompilation"
fi

# Excute q-gcm
echo "Running q-gcm"
ulimit -s hard
./q-gcm
mv qgcm.output outdata/