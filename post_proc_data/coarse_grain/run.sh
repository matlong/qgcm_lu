#!/bin/bash
ulimit -s hard

# Compile
make

# Excute
./main

# Clean
make clean
