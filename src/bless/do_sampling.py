import boolevard as blv
import pandas as pd
import zipfile
import os
import re
import sys
from pyeda.inter import expr
from mpi4py import MPI
import shutil
from .BLESS_functions import SampleModels

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

zip_path = sys.argv[1]
SampleModels(zip_path, sample_size=10, th=300)
