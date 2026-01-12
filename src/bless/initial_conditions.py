import boolevard as blv
import pandas as pd
import zipfile
import os
import re
import sys
from pyeda.inter import expr
from mpi4py import MPI
import shutil
from .BLESS_functions import AddMedia, InitialConditions

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

zips   = sys.argv[1]
targets = ["EGFR", "IGF1R", "MET"]

AddMedia(zips, targets)

comm.Barrier()

InitialConditions(zips)

comm.Barrier()
