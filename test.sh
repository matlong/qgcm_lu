#!/bin/bash

ulimit -s hard
export OMP_NUM_THREADS=4

# Fortran compilator with flags
FC="gfortran -ffixed-line-length-80 -fimplicit-none -std=legacy -w -fallow-argument-mismatch -O2"
FFLAGS="-O3 -mtune=native -Wunused -Wuninitialized -Waliasing -Wsurprising -ffpe-trap=invalid,zero,overflow -fbacktrace -g -fopenmp -w -fallow-argument-mismatch -O2"
LAPACK="src/lasubs.f"

# NetCDF dir and link
NCIDIR="-I/usr/local/opt/netcdf/include"
NCLINK="-L/usr/local/opt/netcdf/lib -lnetcdff -lnetcdf"

# Compile and excute
rm test
$FC $FFLAGS $NCIDIR $LAPACK test.F -o test $NCLINK
./test
