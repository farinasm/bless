import boolevard as blv
import pandas as pd
import zipfile
import os
import re
import numpy as np
import sys
from mpi4py import MPI
import shutil
from .BLESS_functions import PathsBlessFull_HPC

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

zips   = sys.argv[1]


PathsBlessFull_HPC(zips)
