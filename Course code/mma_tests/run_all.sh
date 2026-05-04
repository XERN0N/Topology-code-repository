#!/bin/bash

python3 test0.py
python3 test1.py
python3 test2.py
python3 test3.py
python3 test4.py
python3 test5.py
python3 test7.py

OMPI_MCA_btl_vader_single_copy_mechanism=none mpirun -n 2 python3 test3pet.py
OMPI_MCA_btl_vader_single_copy_mechanism=none mpirun -n 2 python3 test4pet.py
OMPI_MCA_btl_vader_single_copy_mechanism=none mpirun -n 2 python3 test5pet.py
OMPI_MCA_btl_vader_single_copy_mechanism=none mpirun -n 2 python3 test7pet.py
