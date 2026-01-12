import boolevard as blv
import pandas as pd
import zipfile
import os
import re
import numpy as np
import sys
from mpi4py import MPI
import shutil
from pyeda.inter import expr, ExprNot
from .BLESS_functions import NeedsPerturbation, PertModels_HPC, PathsBless_HPC, Synergies

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

zips   = sys.argv[1]

need = NeedsPerturbation(zips)
comm.Barrier()

if need:
    PertModels_HPC(zips)
else:
    if rank == 0:
        print("✅ Perturbations already exist, skipping perturbation step.")
comm.Barrier()

PathsBless_HPC(zips)

comm.Barrier()

zfiles = sorted(f for f in os.listdir(zips) if f.endswith(".zip"))
for i, zfile in enumerate(zfiles):
    if i % size == rank:
        zip_path = os.path.join(zips, zfile)
        groupname = zfile.replace(".zip", "")
        data = Synergies(zip_path)
        tsv = data.to_csv(sep="\t", index=True).encode("utf-8")
        with zipfile.ZipFile(zip_path, "a") as z:
            arcname = f"Results/SynergyExcess_{groupname}.tsv"
            z.writestr(arcname, tsv)
        print(f"✅ Synergy excess data added to {arcname} in {zfile}")
