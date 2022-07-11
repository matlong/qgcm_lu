#!/bin/bash

ulimit -s hard
export OMP_NUM_THREADS=4

# Compile
make

# Excute
./main

# Clean
make clean
