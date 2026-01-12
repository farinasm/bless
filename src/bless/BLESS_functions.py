import boolevard as blv
import pandas as pd
import zipfile
import os
import re
import time
import numpy as np
import sys
from pyeda.inter import expr, ExprNot
from mpi4py import MPI
import shutil
import warnings
import random
import signal
from multiprocessing import Process, Queue

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def AddMedia(zips, targets):
    zfiles = sorted(f for f in os.listdir(zips) if f.endswith(".zip"))
    
    for zfile in zfiles:

        zip_path = os.path.join(zips, zfile)
        tempdir = os.path.join(zips, f"tempdir_{os.path.splitext(zfile)[0]}")

        if rank == 0:
            os.makedirs(tempdir, exist_ok = True)
            with zipfile.ZipFile(zip_path, 'r') as z:
                model_files = [m for m in z.namelist() if m.startswith("Models/")]
                for f in model_files:
                    z.extract(f, path = tempdir)
        comm.Barrier()

        model_dir = os.path.join(tempdir, "Models")
        modelfiles = sorted(os.path.join(model_dir, fn) for fn in os.listdir(model_dir) if fn.endswith(".bnet"))
        mfiles = [modelfiles[i] for i in range(rank, len(modelfiles), size)]
        
        for mfile in mfiles:
            model = blv.Load(mfile)
            pmodel = model

            for target in targets:
                if f"{target}_MEDIA" not in pmodel.Nodes:
                    pmodel = pmodel.Pert(f"{target}%ACT", additive = False)
                else:
                    print(f"🧫 {target}_MEDIA already in model, skipping...")

            pmodel.Nodes = [n[:-4] + "_MEDIA" if isinstance(n, str) and n.endswith("_ACT") else n for n in pmodel.Nodes]
            pmodel.Info.index = pmodel.Nodes
            for col in ("DNF", "NDNF"):
                pmodel.Info[col] = pmodel.Info[col].astype(str).str.replace(r"_ACT\b", "_MEDIA", regex = True).apply(expr)

            pmodel._IsPert = False
            base = os.path.basename(mfile)
            out_path = os.path.join(model_dir, base)
            pmodel.Export(out_path)
        comm.Barrier()

        if rank == 0:
            tmp_zip = zip_path + ".tmp"
            with zipfile.ZipFile(zip_path, "r") as zin, zipfile.ZipFile(tmp_zip, "w") as zout:
                for item in zin.infolist():
                    if not item.filename.startswith("Models/"):
                        zout.writestr(item, zin.read(item.filename))
                for root, _, files in os.walk(model_dir):
                    for fn in files:
                        full = os.path.join(root, fn)
                        arc = os.path.relpath(full, tempdir)
                        zout.write(full, arc)
            os.replace(tmp_zip, zip_path)
            shutil.rmtree(tempdir, ignore_errors = True)
            print(f"✅️ Media constraints added to {os.path.splitext(zfile)[0]}.zip")  
        comm.Barrier()

def InitialConditions(zips):
    zfiles = sorted(f for f in os.listdir(zips) if f.endswith(".zip"))

    for zfile in zfiles:

        base = os.path.splitext(zfile)[0]
        zip_path = os.path.join(zips, zfile)
        tempdir = os.path.join(zips, f"tempdir_{os.path.splitext(zfile)[0]}")

        if rank == 0:
            os.makedirs(tempdir, exist_ok = True)
            with zipfile.ZipFile(zip_path, "r") as z:
                model_files = [m for m in z.namelist() if m.startswith("Models/")]
                for f in model_files:
                    z.extract(f, path = tempdir)
                z.extract(f"src/{base}_training", path = tempdir)
        comm.Barrier()

        model_dir = os.path.join(tempdir, "Models")
        modelfiles = [os.path.join(model_dir, fn) for fn in os.listdir(model_dir) if fn.endswith(".bnet")]
        mfiles = [modelfiles[i] for i in range(rank, len(modelfiles), size)]

        train_path = os.path.join(tempdir, "src", f"{base}_training")

        with open(train_path) as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            if line.strip() == "Response":
                resp_line = lines[i + 1].strip()
                break

        training = {k: int(float(v)) for k, v in (e.split(":") for e in resp_line.split("\t") if e)}
        minfo = blv.Load(modelfiles[0]).Info
        inputs = minfo[minfo["DNF"].astype(str).eq(minfo.index)].index
        itraining = {k: v for k, v in training.items() if k in inputs}
        perts = [f"{k}%{'ACT' if v == 1 else 'INH'}" for k, v in itraining.items()]

        for mfile in mfiles:
            model = blv.Load(mfile)
            pmodel = model

            for pert in perts:
                if f"{pert.split('%')[0]}_ACT" not in pmodel.Nodes and f"{pert.split('%')[0]}_iCOND" not in pmodel.Nodes:
                    pmodel = pmodel.Pert(pert, additive = False)
                else:
                    print(f"{pert.split('%')[0]}_iCOND already in model, skipping...")

            pmodel.Nodes = [n[:-4] + "_iCOND" if isinstance(n, str) and (n.endswith("_ACT") or n.endswith("_INH")) else n for n in pmodel.Nodes]
            pmodel.Info.index = pmodel.Nodes
            for col in ("DNF", "NDNF"):
                pmodel.Info[col] = pmodel.Info[col].astype(str).str.replace(r"_(ACT|INH)\b", "_iCOND", regex = True).apply(expr)
 
            pmodel._IsPert = False
            base_name = os.path.basename(mfile)
            out_path = os.path.join(model_dir, base_name)
            pmodel.Export(out_path)
        comm.Barrier()

        if rank == 0:
            tmp_zip = zip_path + ".tmp"
            with zipfile.ZipFile(zip_path, "r") as zin, zipfile.ZipFile(tmp_zip, "w") as zout:
                for item in zin.infolist():
                    if not item.filename.startswith("Models/"):
                        zout.writestr(item, zin.read(item.filename))
                for root, _, files in os.walk(model_dir):
                    for fn in files:
                        full = os.path.join(root, fn)
                        arc = os.path.relpath(full, tempdir)
                        zout.write(full, arc)
            os.replace(tmp_zip, zip_path)
            shutil.rmtree(tempdir, ignore_errors = True)
            print(f"✅️ Initial conditions added to {os.path.splitext(zfile)[0]}.zip")
        comm.Barrier()

##################################################################################


def SampleModels(zip_path, sample_size=10, th=120, master_timeout=125):

    # === Paths ===
    zips_dir = os.path.dirname(zip_path)
    zfile    = os.path.basename(zip_path)

    tempdir    = os.path.join(zips_dir, f"tempdir_{os.path.splitext(zfile)[0]}")
    models_dir = os.path.join(tempdir, "Models")
    mo_path    = os.path.join(tempdir, "src", "modeloutputs")

    # === Prepare tempdir and extract ===
    if rank == 0:
        os.makedirs(tempdir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            model_files_in_zip = [
                m for m in z.namelist()
                if m.startswith("Models/") and m.endswith(".bnet")
            ]
            for f in model_files_in_zip:
                z.extract(f, path=tempdir)
            z.extract("src/modeloutputs", tempdir)
    comm.Barrier()

    # === Load outputs and model list ===
    mo = pd.read_csv(mo_path, sep="\t", comment = "#", header = None).iloc[:, 0].tolist()
    model_files = sorted(
        os.path.join(models_dir, fn)
        for fn in os.listdir(models_dir)
        if fn.endswith(".bnet")
    )
    modelcount  = len(model_files)
    target_size = min(sample_size, modelcount)

    indices = list(range(modelcount))
    if rank == 0:
        random.shuffle(indices)
    indices = comm.bcast(indices, root=0)

    # ======================================================================
    # MASTER
    # ======================================================================
    def master_sampling():
        sampled_idxs = []
        next_ptr = 0
        status = MPI.Status()

        print(f"\n[MASTER] Sampling for {zfile} (target = {target_size}, th = {th}s)")
        sys.stdout.flush()

        # Send initial tasks
        active_workers = 0
        for r in range(1, size):
            if next_ptr < modelcount:
                comm.send(indices[next_ptr], dest=r, tag=11)
                next_ptr += 1
                active_workers += 1
            else:
                comm.send(-1, dest=r, tag=11)

        start_master = time.time()
        # puedes ajustar esto si quieres, pero visto lo que dices:
        master_timeout = th   # por ejemplo, mismo que th

        while active_workers > 0 and len(sampled_idxs) < target_size:
            # Check global timeout for this zip
            if time.time() - start_master > master_timeout:
                print(f"[MASTER] Master timeout ({master_timeout}s) reached, stopping.")
                sys.stdout.flush()
                break

            # Non-blocking check for incoming messages
            if comm.Iprobe(source=MPI.ANY_SOURCE, tag=22):
                idx, elapsed, ok = comm.recv(source=MPI.ANY_SOURCE, tag=22, status=status)
                src = status.Get_source()

                if ok and len(sampled_idxs) < target_size:
                    sampled_idxs.append(idx)
                    print(
                        f"[MASTER] +1 model OK ({len(sampled_idxs)}/{target_size}) "
                        f" -> {os.path.basename(model_files[idx])} "
                        f"(elapsed={elapsed:.1f}s)"
                    )
                    sys.stdout.flush()

                # Target reached → stop this worker and break
                if len(sampled_idxs) >= target_size:
                    comm.send(-1, dest=src, tag=11)
                    active_workers -= 1
                    break

                # Assign next model if available
                if next_ptr < modelcount:
                    comm.send(indices[next_ptr], dest=src, tag=11)
                    next_ptr += 1
                else:
                    # No more models → shut down worker
                    comm.send(-1, dest=src, tag=11)
                    active_workers -= 1
            else:
                # No message ready, give a tiny breather
                time.sleep(0.1)

        # Write whatever we got (even if < target_size)
        sampled_models = [model_files[i] for i in sampled_idxs]
        base = os.path.splitext(zfile)[0]
        txt_name = f"SampledModels_{base}.txt"
        local_txt = os.path.join(tempdir, txt_name)

        # Write txt inside tempdir
        with open(local_txt, "w", encoding="utf-8") as f:
            for m in sampled_models:
                f.write(os.path.basename(m) + "\n")

        print(f"\n[MASTER] Final sample for {zfile}: {len(sampled_models)} models")
        for m in sampled_models:
            print("   ", os.path.basename(m))
        print(f"[MASTER] Written locally: {local_txt}")
        sys.stdout.flush()

        # Insert txt into the original zip under src/
        with zipfile.ZipFile(zip_path, "a") as z:
            z.write(local_txt, arcname=f"src/{txt_name}")

        # Remove temporary folder
        shutil.rmtree(tempdir, ignore_errors=True)

        MPI.COMM_WORLD.Abort(0)

    # ======================================================================
    # WORKER (subprocess timeout)
    # ======================================================================
    def worker_sampling():
        status = MPI.Status()

        def run_countpaths(mpath, mo, q):
            try:
                model = blv.Load(mpath)
                subset = model.Info[
                    model.Info.index.str.endswith(
                        ("_ACT", "_INH", "_MEDIA", "_iCOND", "_event")
                    )
                ]
                constraints_on = subset.columns[subset.ne(0).all(axis=0)]
                model.Info = model.Info[constraints_on]
                model.Info.columns = (
                    list(range(len(model.Info.columns) - 2))
                    + model.Info.columns[-2:].to_list()
                )
                if len(model.Info.columns) <= 2:
                    q.put((0.0, False))
                    return
                t0 = time.time()
                model.CountPaths(mo, ss_wise=True)
                elapsed = time.time() - t0
                q.put((elapsed, True))
            except Exception:
                q.put((0.0, False))

        while True:
            idx = comm.recv(source=0, tag=11, status=status)
            if idx < 0:
                break

            mpath = model_files[idx]
            q = Queue()
            p = Process(target=run_countpaths, args=(mpath, mo, q))
            p.start()
            p.join(timeout=th)

            if p.is_alive():
                p.terminate()
                p.join()
                elapsed = th
                ok = False
            else:
                try:
                    elapsed, ok = q.get_nowait()
                    ok = ok and (elapsed <= th)
                except Exception:
                    elapsed = 0.0
                    ok = False

            comm.send((idx, elapsed, ok), dest=0, tag=22)

    if rank == 0:
        master_sampling()
    else:
        worker_sampling()


##################################################################################



def NeedsPerturbation(zips):
    zfiles = sorted(f for f in os.listdir(zips) if f.endswith(".zip"))
    first = zfiles[0]
    with zipfile.ZipFile(os.path.join(zips, first), "r") as z:
        namelist = z.namelist()
    if any(name.startswith("PertModels/") for name in namelist):
        return False
    else:
        return True

def PertModels_HPC(zips):
    zfiles = sorted(f for f in os.listdir(zips) if f.endswith(".zip"))
    for zfile in zfiles:
        zip_path = os.path.join(zips, zfile)
        tempdir = os.path.join(zips, f"tempdir_{os.path.splitext(zfile)[0]}")
        pert_dir = os.path.join(tempdir, "PertModels")
        
        if rank == 0:
            os.makedirs(tempdir, exist_ok = True)
            os.makedirs(pert_dir, exist_ok = True)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(tempdir)
        comm.Barrier()
        
        dp_path = os.path.join(tempdir, "src", "drugpanel")
        perts_path = os.path.join(tempdir, "src", "perturbations")
        mo_path = os.path.join(tempdir, "src", "modeloutputs")
        model_dir = os.path.join(tempdir, "Models")
        subset_file = os.path.join(tempdir, "src", f"SampledModels_{os.path.splitext(zfile)[0]}.txt")
        
        if os.path.exists(subset_file):
            with open(subset_file) as f:
                subset_bases = {line.strip() for line in f if line.strip()}
        else:
            subset_bases = None
        all_modelfiles = [os.path.join(model_dir, fn) for fn in os.listdir(model_dir) if fn.endswith(".bnet")]
        if subset_bases is not None:
            modelfiles = [mf for mf in all_modelfiles if os.path.basename(mf) in subset_bases]
        else:
            modelfiles = all_modelfiles
        dpcols = ["PD_profile", "action"] + [f"target_{i}" for i in range(1, 21)]
        drugpanel = pd.read_csv(dp_path, sep = "\t", header = None, names = dpcols, skiprows = 1)
        drugpanel = drugpanel.dropna(axis = 1, how = "all")
        targetcolsdp = [c for c in drugpanel.columns if c.startswith("target_")]
        def join_targets(row):
            vals = [str(row[c]) for c in targetcolsdp if c in row and pd.notna(row[c]) and str(row[c]).strip() != ""]
            return ",".join(vals)
        drugpanel["node_targets"] = drugpanel.apply(join_targets, axis = 1)
        drugpanel["moa"] = (drugpanel["action"].map({"inhibits": "INH", "activates": "ACT"}).fillna("INH"))
        drugpanel["node_targets"] = (drugpanel["node_targets"].astype(str).str.replace("[", "").str.replace("]", "").str.replace("'", "").str.replace(" ", ""))
        perts = pd.read_csv(perts_path, header = None)[0].str.replace("\t", "_").tolist()
        mo = pd.read_csv(mo_path, sep = "\t", header = None).iloc[:, 0].tolist()
        drugdict = {
            row["PD_profile"]:{
                "pert": [f"{t}%{row['moa']}" for t in row["node_targets"].split(",") if t]
            }
            for _, row in drugpanel.iterrows()
        }
        drugcombos = [p for p in perts if p.count("PD") == 2]
        for combo in drugcombos:
            p1, p2 = combo.split("_")[1], combo.split("_")[3]
            drugdict[combo] = {
                "pert": drugdict[f"PD_{p1}"]["pert"] + drugdict[f"PD_{p2}"]["pert"]
            }
        tasks = [(mfile, key) for mfile in modelfiles for key in drugdict.keys()]
        my_tasks = [tasks[i] for i in range(rank, len(tasks), size)]
        cached_path = None
        cached_model = None
        for mfile, key in my_tasks:
            if mfile != cached_path:
                cached_model = blv.Load(mfile)
                cached_path = mfile
            info = drugdict[key]
            pmodel = cached_model
            for pert in info["pert"]:
                pmodel = pmodel.Pert(pert)
                pertID = pert.replace("%", "_")
                pnode  = f"{key}_{pert.split('%')[1]}"
                pat = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(pertID)}(?![A-Za-z0-9_])")
                pmodel._bnet = [pat.sub(pnode, line) for line in pmodel._bnet]
                seen = set()
                new_bnet = []
                for line in pmodel._bnet:
                    if line not in seen:
                        seen.add(line)
                        new_bnet.append(line)
                pmodel._bnet = new_bnet
                nodes = []; DNFs = []; NDNFs = []
                for line in pmodel._bnet:
                    if line.strip() and not line.startswith("#") and "targets" not in line and "target" not in line:
                        node, rule = line.split(",", 1)
                        node = node.strip()
                        rule = rule.strip()
                        nodes.append(node)
                        DNFs.append(expr(rule.replace(" ", "").replace("!", "~")).to_dnf())
                        NDNFs.append(ExprNot(expr(rule.replace(" ", "").replace("!", "~"))).to_dnf())
                pmodel.Info = pmodel.Info.reindex(index=nodes)
                pmodel.Info["DNF"] = DNFs
                pmodel.Info["NDNF"] = NDNFs
                pmodel.Nodes = nodes
                pmodel.DNFs = dict(zip(nodes, DNFs))
                pmodel.NDNFs = dict(zip(nodes, NDNFs))
                pmodel.SS = pmodel.Info.drop(["DNF", "NDNF"], axis=1)
            base = os.path.splitext(os.path.basename(mfile))[0]
            out_name = f"{base}_{key}.bnet"
            out_path = os.path.join(pert_dir, out_name)
            pmodel.Export(out_path)
        comm.Barrier()

        if rank == 0:
            with zipfile.ZipFile(zip_path, "a") as z:
                for root, _, files in os.walk(pert_dir):
                    for fn in files:
                        full = os.path.join(root, fn)
                        arc = os.path.relpath(full, tempdir)
                        z.write(full, arc)
                if subset_bases is not None:
                    for mfile in modelfiles:
                        arc = os.path.join("SampModels", os.path.basename(mfile))
                        z.write(mfile, arc)
            shutil.rmtree(tempdir, ignore_errors=True)
            print(f"✅ Perturbations added to {os.path.splitext(zfile)[0]}.zip")
        comm.Barrier()


def PathsBless_HPC(zips, limit = None):

    if limit is not None:
        TIME_LIMIT_SECS = limit * 60
    else:
        TIME_LIMIT_SECS = 1000 * 60 # default 1000 minutes

    def master():
        next_idx = 0; finished = 0; results_accum = header; status = MPI.Status()
        for r in range(1, size):
            if next_idx < modelcount and time.time() < deadline:
                comm.send(next_idx, dest = r, tag = 11)
                next_idx += 1
            else:
                comm.send(-1, dest = r, tag = 11)
                finished += 1
        while finished < size - 1:
            data = comm.recv(source = MPI.ANY_SOURCE, tag = 12, status = status)
            src = status.Get_source()
            if data:
                results_accum += data
            if next_idx < modelcount and time.time() < deadline:
                comm.send(next_idx, dest = src, tag = 11)
                next_idx += 1
            else:
                comm.send(-1, dest = src, tag = 11)
                finished += 1
        out_path = os.path.join(results_dir, f"PathCounts_{os.path.splitext(zfile)[0]}.tsv")
        with open(out_path, "w", encoding = "utf-8") as f:
            f.write(results_accum)
        with zipfile.ZipFile(zip_path, "a") as z:
            arcname = os.path.join("Results", f"PathCounts_{os.path.splitext(zfile)[0]}.tsv")
            z.write(out_path, arcname)

    def worker():
        import signal
        status = MPI.Status()
        while True:
            idx = comm.recv(source = 0, tag = 11, status = status)
            if idx < 0:
                break
            modelname = model_files[idx]
            modelpath = os.path.join(models_dir, modelname)
            with open(modelpath, "r") as f:
                lines = f.readlines()[1:]
            with open(modelpath, "w") as f:
                f.writelines(lines)
            bmodel = blv.Load(modelpath)
            subset = bmodel.Info[bmodel.Info.index.str.endswith(("_ACT", "_INH", "_MEDIA", "_iCOND", "_event"))]
            constraints_on = subset.columns[subset.ne(0).all(axis = 0)]
            bmodel.Info = bmodel.Info[constraints_on]
            bmodel.Info.columns = list(range(len(bmodel.Info.columns) - 2)) + bmodel.Info.columns[-2:].to_list()
            if len(bmodel.Info.columns) > 2:
                remaining = max(1, int(deadline - time.time()))
                signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
                signal.alarm(remaining)
                try:
                    res = bmodel.CountPaths(mo, ss_wise = True)
                except TimeoutError:
                    res = [["NA"] * len(mo)]
                finally:
                    signal.alarm(0)
            else:
                res = [["NA"] * len(mo)]
            result_lines = ""
            for ss in range(len(res)):
                values = res[ss]
                try:
                    signs = [0 if float(x) < 0 else 1 for x in values]
                except:
                    signs = ["NA"] * len(values)
                line = "\t".join(str(x) for x in ([f"{os.path.splitext(zfile)[0]}_{modelname.replace('.bnet', '').replace('PertModels/', '')}"] + [ss] + values + signs))
                result_lines += line + "\n"
            comm.send(result_lines, dest = 0, tag = 12)

    zfiles = sorted([f for f in os.listdir(zips) if f.endswith(".zip")])
    for zfile in zfiles:
        zip_path = os.path.join(zips, zfile)
        tempdir  = os.path.join(zips, f"tempdir_{os.path.splitext(zfile)[0]}")
        models_dir = os.path.join(tempdir, "Models")
        results_dir = os.path.join(tempdir, "Results")
        mo_path = os.path.join(tempdir, "src", "modeloutputs")
        deadline = time.time() + TIME_LIMIT_SECS
        if rank == 0:
            os.makedirs(tempdir, exist_ok = True)
            os.makedirs(models_dir, exist_ok = True)
            os.makedirs(results_dir, exist_ok = True)
            with zipfile.ZipFile(zip_path, "r") as z:
                nl = z.namelist()
                has_samp = any(m.startswith("SampModels/") for m in nl)
                if has_samp:
                    to_extract = [
                        m for m in nl
                        if (m.startswith("PertModels/") or m.startswith("SampModels/"))
                        and m.endswith(".bnet")
                    ]
                else:
                    to_extract = [
                        m for m in nl
                        if (m.startswith("PertModels/") or m.startswith("Models/"))
                        and m.endswith(".bnet")
                    ]
                for member in to_extract:
                    filename = os.path.basename(member)
                    dest = os.path.join(models_dir, filename)
                    with z.open(member) as src, open(dest, "wb") as dst:
                        dst.write(src.read())
                z.extract("src/modeloutputs", tempdir)
        comm.Barrier()
        mo = pd.read_csv(mo_path, sep = "\t", comment = "#", header = None).iloc[:, 0].tolist()
        model_files = sorted(fn for fn in os.listdir(models_dir) if fn.endswith(".bnet"))
        modelcount = len(model_files)
        header = "\t".join(["Model ID", "SS"] + 2*mo) + "\n"
        if rank == 0:
            master()
            print(f"✅ Finished processing {zfile}")
        else:
            worker()
        if rank == 0:
            shutil.rmtree(tempdir, ignore_errors = True)
        comm.Barrier()


def Synergies(zip_path):

    groupname = os.path.splitext(os.path.basename(zip_path))[0]
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if name.startswith("Results/PathCounts_") and name.endswith(".tsv"):
                with z.open(name) as f:
                    data = pd.read_csv(f, sep = "\t", header = 0)
            if name.startswith("src/modeloutputs"):
                with z.open(name) as f:
                    mo = pd.read_csv(f, sep = "\t", header = None, comment = "#")
    
    mo.columns = ["Node", "Weight"]
    mo.set_index("Node", inplace = True)
    moNodes = data.columns[data.columns.str.endswith(".1")].str.replace(".1", "").tolist()
    posNodes = [n for n in moNodes if mo.loc[n, "Weight"] == 1]
    negNodes = [n for n in moNodes if mo.loc[n, "Weight"] == -1]
    data.dropna(inplace = True)
    data = data.groupby("Model ID").mean()
    data["Perturbation"] = data.index.to_series().apply(lambda ID: "unperturbed" if "_PD_" not in ID else re.findall(r'(PD_\d+(?:_PD_\d+)*)$', ID)[0])
    data["ID"] = data.index.to_series().apply(lambda ID: re.sub(f"{groupname}_", "", re.sub(r'_PD_\d+(?:_PD_\d+)*$', '', ID)))
    data["Viability"] = (data[posNodes].sum(axis = 1) - data[negNodes].sum(axis = 1))
    #max_is_control = data.groupby("ID", group_keys=False)[["Viability", "Perturbation"]].apply(lambda g: g.loc[g["Viability"].idxmax(), "Perturbation"] == "unperturbed")
    #valid = max_is_control[max_is_control].index
    #data = data[data["ID"].isin(valid)]

    s_ref = data.groupby("ID")["Viability"].max().median()
    s_min, s_max = s_ref/30, s_ref*30
    s_grid = np.geomspace(s_min, s_max, 150)
    best_s, best_var, best_bliss = None, -1, None

    for s in s_grid:
        tmp = data.copy()
        tmp["Viability_norm"] = (np.tanh(tmp["Viability"]/s) + 1)/2
        ctrl = tmp.loc[tmp["Perturbation"] == "unperturbed"].set_index("ID")["Viability_norm"]
        tmp["Viability_norm"] = tmp["Viability_norm"].clip(upper=tmp["ID"].map(ctrl))
        tmp["Viability_norm"] = tmp["Viability_norm"]/tmp["ID"].map(ctrl)
        v = tmp.groupby("Perturbation")["Viability_norm"].mean()
        bliss = {
            k: v[k] - v[d1]*v[d2]
            for k in v.index if "_PD_" in k
            for d1, d2 in [(k.split("_PD_")[0].strip(), "PD_" + k.split("_PD_")[1].strip())]
            if d1 in v.index and d2 in v.index
        }
        var_abs = np.var(np.abs(list(bliss.values())))
        if var_abs > best_var:
            best_s, best_var, best_bliss = s, var_abs, bliss
    bliss = best_bliss
    bliss = pd.DataFrame.from_dict(bliss, orient="index", columns=["Excess"])
    bliss.index.name = "Perturbation"
    
    return bliss

def PathwaysBless(zips, perturbations = "all"):
    
    zfiles = sorted([f for f in os.listdir(zips) if f.endswith(".zip")])

    for zfile in zfiles:
        zip_path = os.path.join(zips, zfile)
        tempdir = os.path.join(zips, f"tempdir_{os.path.splitext(zfile)[0]}")
        models_dir = os.path.join(tempdir, "Models")
        results_dir = os.path.join(tempdir, "Results")
        nodeHGNC_path = os.path.join(tempdir, "src", "NodeHGNC.csv")
        kpw_path = os.path.join(tempdir, "src", "kegg_pathway.csv")

        # --- file extraction at rank 0 ---
        if rank == 0:
            os.makedirs(tempdir, exist_ok = True)
            os.makedirs(models_dir, exist_ok = True)
            os.makedirs(results_dir, exist_ok = True)
            with zipfile.ZipFile(zip_path, "r") as z:
                all_candidates = [m for m in z.namelist()
                            if (m.startswith("PertModels/") or m.startswith("Models/"))
                            and m.endswith(".bnet")]
                controls = [m for m in all_candidates if "PD_" not in os.path.basename(m)]
                perturbations_all = [m for m in all_candidates if "PD_" in os.path.basename(m)]
                if perturbations == "all":
                    to_extract = controls + perturbations_all
                elif isinstance(perturbations, (list, tuple, set)):
                    selected = [] 
                    for m in perturbations_all:
                        base = os.path.splitext(os.path.basename(m))[0] 
                        parts = re.findall(r"PD_\d+", base)
                        pert = "_".join(parts)
                        if pert in perturbations:
                            selected.append(m)
                    to_extract = controls + selected
                for member in to_extract:
                    filename = os.path.basename(member)
                    dest = os.path.join(models_dir, filename)
                    with z.open(member) as src, open(dest, "wb") as dst:
                        dst.write(src.read())
                z.extract("src/NodeHGNC.csv", tempdir)
                z.extract("src/kegg_pathway.csv", tempdir)
        comm.Barrier()

        # --- metadata load ---
        nodeHGNC = (
            pd.read_csv(nodeHGNC_path)
            .set_index("node_name")["HGNC_symbol"]
            .apply(lambda x: [s.strip() for s in str(x).split(", ")])
            .to_dict()
        )
        kpw = pd.read_csv(kpw_path)

        model_HGNC = sorted(set([gene for genes in nodeHGNC.values() for gene in genes]))
        model_kpw = kpw[kpw["HGNC"].isin(model_HGNC)].copy()
        model_kpw["Node"] = model_kpw["HGNC"].map(
            {gene: node for node, genes in nodeHGNC.items() for gene in genes}
        )
        model_kpw = (
            model_kpw.groupby(["Pathway_ID", "Pathway_Name", "Node"], as_index = False)
            .agg({"HGNC": lambda x: ", ".join(sorted(set(x)))})
        )
        nodes_in_pathways = sorted(set(model_kpw["Node"]))
        
        models = sorted([m for m in os.listdir(models_dir) if m.endswith(".bnet")])
        modelcount = len(models)

        # ----------------- MASTER -----------------
        if rank == 0:
            next_idx = 0
            finished = 0
            results_accum = model_kpw.copy()
            status = MPI.Status()

            # start sending models to workers
            for r in range(1, size):
                if next_idx < modelcount:
                    comm.send(next_idx, dest = r, tag = 11)
                    next_idx += 1
                else:
                    comm.send(-1, dest = r, tag = 11)
                    finished += 1

            # receive results
            while finished < size - 1:
                data = comm.recv(source = MPI.ANY_SOURCE, tag = 12, status = status)
                src = status.Get_source()
                if data is not None:
                    results_accum = results_accum.merge(data, on = "Node", how = "left")
                if next_idx < modelcount:
                    comm.send(next_idx, dest = src, tag = 11)
                    next_idx += 1
                else:
                    comm.send(-1, dest = src, tag = 11)
                    finished += 1
            
            # finalize results
            results_accum[results_accum.columns[4:]] = results_accum[
                results_accum.columns[4:]
            ].apply(pd.to_numeric, errors = "coerce")

            model_kpw_gene_wise = (
                results_accum.sort_values("Pathway_Name").reset_index(drop = True)
            )
            model_kpw_pathway_wise = (
                results_accum.groupby(["Pathway_ID", "Pathway_Name"], as_index = False)
                .agg({
                    "Node": lambda x: ", ".join(sorted(set(x))),
                    "HGNC": lambda x: ", ".join(sorted(set(x))),
                    **{col: "sum" for col in results_accum.columns[4:]}
                })
            )

            out_excel = os.path.join(results_dir, f"Pathways_{os.path.splitext(zfile)[0]}.xlsx")
            with pd.ExcelWriter(out_excel, engine = "openpyxl") as writer:
                model_kpw_gene_wise.to_excel(writer, sheet_name = f"GeneWise_{os.path.splitext(zfile)[0]}", index = False)
                model_kpw_pathway_wise.to_excel(writer, sheet_name = f"PathwayWise_{os.path.splitext(zfile)[0]}", index = False)
            with zipfile.ZipFile(zip_path, "a") as z:
                arcname = os.path.join("Results", os.path.basename(out_excel))
                z.write(out_excel, arcname)

            print(f"✅ Finished processing {zfile}")

        # ----------------- WORKER -----------------
        else:
            status = MPI.Status()
            while True:
                idx = comm.recv(source = 0, tag = 11, status = status)
                if idx < 0:
                    break
                modelname = models[idx]
                modelpath = os.path.join(models_dir, modelname)
                with open(modelpath, "r") as f:
                    lines = f.readlines()[1:]
                with open(modelpath, "w") as f:
                    f.writelines(lines)
                bmodel = blv.Load(modelpath)
                if len(set(nodes_in_pathways) - set(bmodel.Nodes)) > 0:
                    warnings.warn(
                        f"{set(nodes_in_pathways) - set(bmodel.Nodes)} not in {modelname} model!"
                    )
                tNodes = [n for n in bmodel.Nodes if n in nodes_in_pathways]
                subset = bmodel.Info[bmodel.Info.index.str.endswith(("_ACT", "_INH", "_MEDIA", "_iCOND"))]
                constraints_on = subset.columns[subset.ne(0).all(axis = 0)]
                bmodel.Info = bmodel.Info[constraints_on]
                bmodel.Info.columns = list(range(len(bmodel.Info.columns) - 2)) + bmodel.Info.columns[-2:].to_list()
                if len(bmodel.Info.columns) > 2:
                    counts = bmodel.CountPaths(tNodes)
                else:
                    counts = ["NA"]*len(tNodes)
                paths = pd.DataFrame({
                    "Node": tNodes,
                    f"{modelname.replace('.bnet', '')}": counts
                })
                comm.send(paths, dest = 0, tag = 12)
        
        comm.Barrier()
        if rank == 0:
            shutil.rmtree(tempdir, ignore_errors = True)



######################
def PathsBlessFull_HPC(zips, limit = None):

    if limit is not None:
        TIME_LIMIT_SECS = limit * 60
    else:
        TIME_LIMIT_SECS = 1000 * 60 # default 1000 minutes

    def master():
        next_idx = 0; finished = 0; results_accum = header; status = MPI.Status()
        for r in range(1, size):
            if next_idx < modelcount and time.time() < deadline:
                comm.send(next_idx, dest = r, tag = 11)
                next_idx += 1
            else:
                comm.send(-1, dest = r, tag = 11)
                finished += 1
        while finished < size - 1:
            data = comm.recv(source = MPI.ANY_SOURCE, tag = 12, status = status)
            src = status.Get_source()
            if data:
                results_accum += data
            if next_idx < modelcount and time.time() < deadline:
                comm.send(next_idx, dest = src, tag = 11)
                next_idx += 1
            else:
                comm.send(-1, dest = src, tag = 11)
                finished += 1
        out_path = os.path.join(results_dir, f"PathCountsFull_{os.path.splitext(zfile)[0]}.tsv")
        with open(out_path, "w", encoding = "utf-8") as f:
            f.write(results_accum)
        with zipfile.ZipFile(zip_path, "a") as z:
            arcname = os.path.join("Results", f"PathCountsFull_{os.path.splitext(zfile)[0]}.tsv")
            z.write(out_path, arcname)

    def worker():
        import signal
        status = MPI.Status()
        while True:
            idx = comm.recv(source = 0, tag = 11, status = status)
            if idx < 0:
                break
            modelname = model_files[idx]
            modelpath = os.path.join(models_dir, modelname)
            with open(modelpath, "r") as f:
                lines = f.readlines()[1:]
            with open(modelpath, "w") as f:
                f.writelines(lines)
            bmodel = blv.Load(modelpath)
            subset = bmodel.Info[bmodel.Info.index.str.endswith(("_ACT", "_INH", "_MEDIA", "_iCOND", "_event"))]
            constraints_on = subset.columns[subset.ne(0).all(axis = 0)]
            bmodel.Info = bmodel.Info[constraints_on]
            bmodel.Info.columns = list(range(len(bmodel.Info.columns) - 2)) + bmodel.Info.columns[-2:].to_list()
            if len(bmodel.Info.columns) > 2:
                remaining = max(1, int(deadline - time.time()))
                signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
                signal.alarm(remaining)
                try:
                    res = bmodel.CountPaths(bmodel.Nodes, ss_wise = True)
                except TimeoutError:
                    res = [["NA"] * len(bmodel.Nodes)]
                finally:
                    signal.alarm(0)
            else:
                res = [["NA"] * len(bmodel.Nodes)]
            result_lines = ""
            for ss in range(len(res)):
                values = res[ss]
                try:
                    signs = [0 if float(x) < 0 else 1 for x in values]
                except:
                    signs = ["NA"] * len(values)
                line = "\t".join(str(x) for x in ([f"{os.path.splitext(zfile)[0]}_{modelname.replace('.bnet', '').replace('PertModels/', '')}"] + [ss] + values + signs))
                result_lines += line + "\n"
            comm.send(result_lines, dest = 0, tag = 12)

    zfiles = sorted([f for f in os.listdir(zips) if f.endswith(".zip")])
    for zfile in zfiles:
        zip_path = os.path.join(zips, zfile)
        tempdir  = os.path.join(zips, f"tempdir_{os.path.splitext(zfile)[0]}")
        models_dir = os.path.join(tempdir, "Models")
        results_dir = os.path.join(tempdir, "Results")
        deadline = time.time() + TIME_LIMIT_SECS
        if rank == 0:
            os.makedirs(tempdir, exist_ok = True)
            os.makedirs(models_dir, exist_ok = True)
            os.makedirs(results_dir, exist_ok = True)
            with zipfile.ZipFile(zip_path, "r") as z:
                nl = z.namelist()
                has_samp = any(m.startswith("SampModels/") for m in nl)
                if has_samp:
                    to_extract = [
                        m for m in nl
                        if (m.startswith("PertModels/") or m.startswith("SampModels/"))
                        and m.endswith(".bnet")
                    ]
                else:
                    to_extract = [
                        m for m in nl
                        if (m.startswith("PertModels/") or m.startswith("Models/"))
                        and m.endswith(".bnet")
                    ]
                for member in to_extract:
                    filename = os.path.basename(member)
                    dest = os.path.join(models_dir, filename)
                    with z.open(member) as src, open(dest, "wb") as dst:
                        dst.write(src.read())
        comm.Barrier()
        model_files = sorted(fn for fn in os.listdir(models_dir) if fn.endswith(".bnet"))
        modelcount = len(model_files)

        if rank == 0:
            guide = os.path.join(models_dir, model_files[0])
            tmp_model = blv.Load(guide)
            node_names = list(tmp_model.Nodes)
        else:
            node_names = None
        node_names = comm.bcast(node_names, root = 0)
        header = "\t".join(["Model ID", "SS"] + 2*node_names) + "\n"
        if rank == 0:
            master()
            print(f"✅ Finished processing {zfile}")
        else:
            worker()
        if rank == 0:
            shutil.rmtree(tempdir, ignore_errors = True)
        comm.Barrier()