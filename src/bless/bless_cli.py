#!/usr/bin/env python3

import subprocess
import argparse
import os
import sys
import zipfile
from pathlib import Path
import textwrap
from datetime import datetime

# Python 3.11+ includes tomllib in stdlib
try:
    import tomllib
except ModuleNotFoundError:
    import toml as tomllib


def load_config(config_path: str | None) -> dict:
    """
    Load the TOML configuration file.
    If config_path is None, defaults to 'config.toml'
    """
    if config_path is None:
        config_path = "config.toml"
    config_file = Path(config_path)
    if not config_file.is_file():
        print(f"[ERROR] Config file not found: {config_file}", file = sys.stderr)
        sys.exit(1)
    with config_file.open("rb") as f:
        config = tomllib.load(f)
    return config

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog = "bless",
        description = "BLESS pipeline launcher (Step 1: configuration parsing).",
    )
    # Required logical arguments
    parser.add_argument(
        "--zips",
        required = True,
        help = "Path to the directory containing .zip files.",
    )
    parser.add_argument(
        "--mode",
        required = True,
        choices = ["synergies", "countpaths", "full"],
        help = "Pipeline mode: synergies / countpaths / full.",
    )
    # Optional pipeline parameters
    parser.add_argument(
        "--sampling",
        type = int,
        default = None,
        help = "Sampling size (0 = no sampling). If omitted, config default is used.",
    )
    parser.add_argument(
        "--timeout-sampling",
        type = int,
        default = None,
        help = "Timeout in seconds for sampling. If omitted, config value is used.",
    )
    parser.add_argument(
        "--timeout-paths",
        type = int,
        default = None,
        help = "Timeout in seconds for path counting. If omitted, config value is used.",
    )
    parser.add_argument(
        "--skip-initial",
        action = "store_true",
        help = "Skip the initial conditions step (MEDIA + iCOND)."
    )
    # Infrastructure / control
    parser.add_argument(
        "--config",
        default = None,
        help = "Path to TOML config file (default: config.toml).",
    )
    parser.add_argument(
        "--dry-run",
        action = "store_true",
        help = "Do not submit or execute anything (summary only).",
    )
    parser.add_argument(
        "--verbose",
        action = "store_true",
        help = "Print additional debug information.",
    )
    return parser

def build_sbatch_script(slurm_cfg: dict, env_cfg: dict, body: str) -> str:
    """
    Build a generic Slurm sbatch script with the given body.
    `body` is a shell snippet (bash) that will be the payload of the job.
    """
    header_lines = []
    partition = slurm_cfg.get("partition")
    if partition:
        header_lines.append(f"#SBATCH -p {partition}")
    account = slurm_cfg.get("account")
    if account:
        header_lines.append(f"#SBATCH -J {account}")
    time_limit = slurm_cfg.get("time")
    if time_limit:
        header_lines.append(f"#SBATCH -t {time_limit}")
    nodes = slurm_cfg.get("nodes")
    if nodes is not None:
        header_lines.append(f"#SBATCH -N {nodes}")
    ntasks_per_node = slurm_cfg.get("ntasks_per_node")
    if ntasks_per_node is not None:
        header_lines.append(f"#SBATCH --ntasks-per-node={ntasks_per_node}")
    output = slurm_cfg.get("output")
    if output:
        header_lines.append(f"#SBATCH -o {output}")
    error = slurm_cfg.get("error")
    if error:
        header_lines.append(f"#SBATCH -e {error}")
    
    preamble = env_cfg.get("preamble", "").rstrip("\n")
    script_lines = [
        "#!/bin/bash",
        *header_lines,
        "",
        "# --- Environment setup ---",
        preamble,
        "",
        "# --- BLESS job ---",
        body,
        "",
    ]
    return "\n".join(script_lines)

def submit_sbatch(script_path: Path, dependency: str | None = None) -> str | None:
    """
    Submit an sbatch script. If `dependency` is given, it will be used as
    afterok dependency job id. Returns the job id as a string (or None).
    """
    cmd = ["sbatch"]
    if dependency is not None:
        cmd.extend(["--dependency", f"afterok:{dependency}"])
    cmd.append(str(script_path))

    result = subprocess.run(cmd, capture_output = True, text = True)
    if result.returncode != 0:
        print(f"[ERROR] sbatch failed for {script_path.name}:", file = sys.stderr)
        print(result.stderr, file = sys.stderr)
        return None
    out = result.stdout.strip()
    print(f"[INFO] sbatch output: {out}")
    job_id = None
    for token in out.split():
        if token.isdigit():
            job_id = token
            break
    if job_id:
        print(f"[INFO] Job ID: {job_id}")
    else:
        print("[WARN] Could not parse job id from sbatch output.")
    return job_id

def run_synergies_pipeline(
        zips_dir: str,
        sampling: int,
        timeout_sampling: int | None,
        timeout_paths: int | None,
        media_targets: list[str],
        skip_initial: bool,
) -> None:
    """
    Run the 'synergies' pipeline inside a Slurm/MPI job.
    Steps:
        1) Add MEDIA and initial conditions (optional if skip_initial).
        2) Sampling (optional if sampling == 0).
        3) Generate perturbed models if needed.
        4) Count paths with perturbations.
        5) Compute synergies for each zip
    """

    # Import MPI and BLESS functions
    from mpi4py import MPI
    from .BLESS_functions import (
        AddMedia,
        InitialConditions,
        SampleModels,
        NeedsPerturbation,
        PertModels_HPC,
        PathsBless_HPC,
        PathsBlessFull_HPC,
        Synergies,
    )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("[PIPELINE] Mode = synergies")
        print(f"[PIPELINE] Zips directory: {zips_dir}")
        print(f"[PIPELINE] Sampling size : {sampling}")
        print(f"[PIPELINE] Skip initial  : {skip_initial}")
        sys.stdout.flush()
    # 1) Initial conditions (MEDIA + iCOND)
    if not skip_initial:
        if rank == 0:
            print("[PIPELINE] Step 1: AddMedia + InitialConditions")
            sys.stdout.flush()
        AddMedia(zips_dir, media_targets)
        comm.Barrier()
        InitialConditions(zips_dir)
        comm.Barrier()
    else:
        if rank == 0:
            print("[PIPELINE] Skipping initial conditions step")
            sys.stdout.flush()
    # 2) Sampling (if sampling > 0)
    if sampling > 0:
        if rank == 0:
            print(f"[PIPELINE] Step 2: Sampling (SampleModels) with size = {sampling}")
            sys.stdout.flush()
        # Loop over zip files and call SampleModels on each
        zfiles = sorted(f for f in os.listdir(zips_dir) if f.endswith(".zip"))
        for zfile in zfiles:
            zip_path = os.path.join(zips_dir, zfile)
            if rank == 0:
                print(f"[PIPELINE]  Sampling zip: {zfile}")
                sys.stdout.flush()
            SampleModels(
                zip_path,
                sample_size = sampling,
                th = timeout_sampling if timeout_sampling is not None else 300,
            )
            comm.Barrier()
    else:
        if rank == 0:
            print("[PIPELINE] Sampling disabled")
            sys.stdout.flush()
        # 3) Generate perturbed models if needed
        if rank == 0:
            print("[PIPELINE] Step 3: Perturbing models")
            sys.stdout.flush()
        if NeedsPerturbation(zips_dir):
            PertModels_HPC(zips_dir)
        comm.Barrier()
        # 4) Count paths with perturbations
        if rank == 0:
            print("[PIPELINE] Step 4: PathsBless (counting paths)")
            print(f"[PIPELINE] Timeout for paths = {timeout_paths}")
            sys.stdout.flush()
        PathsBless_HPC(zips_dir)
        comm.Barrier()
        # 5) Compute synergies per zip
        if rank == 0:
            print("[PIPELINE] Step 5: Synergies per zip")
            sys.stdout.flush()
        zfiles = sorted(f for f in os.listdir(zips_dir) if f.endswith(".zip"))
        for i, zfile in enumerate(zfiles):
            if i % size == rank:
                zip_path = os.path.join(zips_dir, zfile)
                groupname = os.path.splitext(zfile)[0]
                if rank == 0:
                    print(f"[PIPELINE]  Computing synergies for {zfile}")
                    sys.stdout.flush()
                data = Synergies(zip_path)
                tsv = data.to_csv(sep = "\t", index = True).encode("utf-8")
                with zipfile.ZipFile(zip_path, "a") as z:
                    arcname = f"Results/SynergyExcess_{groupname}.tsv"
                    z.writestr(arcname, tsv)
                print(f"[PIPELINE]  SynergyExcess writen to {arcname} in {zfile}")
                sys.stdout.flush()
        comm.Barrier()
        if rank == 0:
            print("[PIPELINE] Synergies pipeline completed.")
            sys.stdout.flush()

def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # 1) Decide which config path is actually used
    config_path_used = args.config or "config.toml"
    config = load_config(config_path_used)

    slurm_cfg = config.get("slurm", {})
    env_cfg = config.get("env", {})
    bless_cfg = config.get("bless", {})

    # 2) Resolve effective values (CLI overrides config)
    sampling = args.sampling
    if sampling is None:
        sampling = bless_cfg.get("default_sampling", 0)

    timeout_sampling = args.timeout_sampling
    if timeout_sampling is None:
        timeout_sampling = bless_cfg.get("timeout_sampling", None)

    timeout_paths = args.timeout_paths
    if timeout_paths is None:
        timeout_paths = bless_cfg.get("timeout_paths", None)

    media_targets = bless_cfg.get("media_targets", [])

    # 3) Detect if running inside a Slurm job
    in_slurm = "SLURM_JOB_ID" in os.environ

    # 4) Print summary
    print("====================================")
    print("  BLESS CLI - SUMMARY")
    print("====================================")
    print(f"ZIPs directory    : {args.zips}")
    print(f"Mode              : {args.mode}")
    print(f"Sampling          : {sampling}")
    print(f"Skip initial      : {args.skip_initial}")
    print(f"Timeout sampling  : {timeout_sampling}")
    print(f"Timeout paths     : {timeout_paths}")
    print()
    print(f"Config file       : {args.config or 'config.toml'}")
    print(f"Media targets     : {media_targets}")
    print()
    print("Slurm context     :", "INSIDE job" if in_slurm else "OUTSIDE (login node)")
    print()
    print("Slurm settings (from config):")
    print(f"  partition        : {slurm_cfg.get('partition')}")
    print(f"  account          : {slurm_cfg.get('account')}")
    print(f"  time             : {slurm_cfg.get('time')}")
    print(f"  nodes            : {slurm_cfg.get('nodes')}")
    print(f"  ntasks_per_node  : {slurm_cfg.get('ntasks_per_node')}")
    print(f"  output           : {slurm_cfg.get('output')}")
    print(f"  error            : {slurm_cfg.get('error')}")
    print(f"  auto_submit      : {slurm_cfg.get('auto_submit')}")
    print()
    print("Environment preamble (from config):")
    preamble = env_cfg.get("preamble", "").strip()
    if preamble:
        for line in preamble.splitlines():
            print(" ", line)
    else:
        print("  <empty>")
    print()

    auto_submit = bool(slurm_cfg.get("auto_submit", False))

    if args.dry_run:
        print("Dry-run flag detected: no sbatch submission will be performed.")
    print()

    # 5) Do not orchestrate anything if inside Slurm
    if in_slurm:
        print("[WARN] bless_cli is designed to be run from the login node, not inside a Slurm job.")
        print("       No jobs will be submitted from within an existing Slurm allocation.")
        return

    # 6) From here: we are on the login node and will generate sbatch scripts

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Normalise path for zips (but we keep what the user passed in scripts)
    zips_dir = args.zips
    zips_path = Path(zips_dir).resolve()

    # Run directory + subfolders
    run_root = zips_path.parent / f"bless_run_{timestamp}"
    sbatch_dir = run_root / "sbatch"
    logs_dir = run_root / "logs"

    sbatch_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] BLESS run directory: {run_root}")
    print(f"[INFO]   sbatch scripts -> {sbatch_dir}")
    print(f"[INFO]   logs           -> {logs_dir}")
    print()

    # Clone Slurm config and redirect output/error to logs_dir
    slurm_run_cfg = dict(slurm_cfg)
    out_pattern = slurm_run_cfg.get("output") or "bless_%j.out"
    err_pattern = slurm_run_cfg.get("error") or "bless_%j.err"
    slurm_run_cfg["output"] = str(logs_dir / out_pattern)
    slurm_run_cfg["error"] = str(logs_dir / err_pattern)

    initial_job_id: str | None = None
    sampling_job_id: str | None = None
    pipeline_job_id: str | None = None

    initial_script_name: str | None = None
    sampling_script_name: str | None = None
    pipeline_script_name: str | None = None

    # 6a) Initial conditions job (MEDIA + iCOND), unless user skipped it
    if not args.skip_initial:
        initial_body = textwrap.dedent(
            f"""\
            zip_dir="{zips_dir}"
            echo ">>> [INITIAL] Starting AddMedia + InitialConditions on $zip_dir"
            mpirun python3 -m bless.initial_conditions "$zip_dir"
            echo ">>> [INITIAL] Finished AddMedia + InitialConditions on $zip_dir"
            """
        ).rstrip()

        initial_script_name = f"bless_initial_{timestamp}.sbatch"
        initial_script_path = sbatch_dir / initial_script_name
        initial_script_text = build_sbatch_script(slurm_run_cfg, env_cfg, initial_body)
        initial_script_path.write_text(initial_script_text, encoding="utf-8")

        print(f"[INFO] Initial conditions sbatch script written to: {initial_script_path}")

        if auto_submit and not args.dry_run:
            print("[INFO] Submitting initial conditions job...")
            initial_job_id = submit_sbatch(initial_script_path)

        print()
    else:
        print("[INFO] Skipping initial conditions step (per --skip-initial).")
        print()

    # 6b) Sampling job, only if sampling > 0
    if sampling and sampling > 0:
        sampling_body = textwrap.dedent(
            f"""\
            zip_dir="{zips_dir}"
            for z in "$zip_dir"/*.zip; do
                echo ">>> [SAMPLING] Starting radical sampling for: $z"
                mpirun python3 -m bless.do_sampling "$z"
                echo ">>> [SAMPLING] Finished radical sampling for: $z"
            done
            """
        ).rstrip()

        sampling_script_name = f"bless_sampling_{timestamp}.sbatch"
        sampling_script_path = sbatch_dir / sampling_script_name
        sampling_script_text = build_sbatch_script(slurm_run_cfg, env_cfg, sampling_body)
        sampling_script_path.write_text(sampling_script_text, encoding="utf-8")

        print(f"[INFO] Sampling sbatch script written to: {sampling_script_path}")

        if auto_submit and not args.dry_run:
            dep_for_sampling = initial_job_id  # may be None
            if dep_for_sampling:
                print(f"[INFO] Submitting sampling job with dependency afterok:{dep_for_sampling} ...")
            else:
                print("[INFO] Submitting sampling job without dependency...")
            sampling_job_id = submit_sbatch(sampling_script_path, dependency=dep_for_sampling)

        print()
    else:
        print("[INFO] Sampling disabled (sampling == 0). No sampling job will be created.")
        print()

    # 6c) Pipeline job(s) depending on mode
    if args.mode == "synergies":
        pipeline_body = textwrap.dedent(
            f"""\
            zip_dir="{zips_dir}"
            echo ">>> [PIPELINE] Starting synergies pipeline (compute_synergies.py) on $zip_dir"
            mpirun python3 -m bless.compute_synergies "$zip_dir"
            echo ">>> [PIPELINE] Finished synergies pipeline on $zip_dir"
            """
        ).rstrip()

    elif args.mode == "countpaths":
        pipeline_body = textwrap.dedent(
            f"""\
            zip_dir="{zips_dir}"
            echo ">>> [PIPELINE] Starting full path counting (count_allpaths.py) on $zip_dir"
            mpirun python3 bless.count_allpaths "$zip_dir"
            echo ">>> [PIPELINE] Finished full path counting on $zip_dir"
            """
        ).rstrip()

    elif args.mode == "full":
        pipeline_body = textwrap.dedent(
            f"""\
            zip_dir="{zips_dir}"
            echo ">>> [PIPELINE] Starting synergies (compute_synergies.py) on $zip_dir"
            mpirun python3 -m bless.compute_synergies "$zip_dir"
            echo ">>> [PIPELINE] Synergies finished, starting full path counting (count_allpaths.py)"
            mpirun python3 -m bless.count_allpaths "$zip_dir"
            echo ">>> [PIPELINE] Finished full mode (synergies + countpaths) on $zip_dir"
            """
        ).rstrip()
    else:
        print(f"[ERROR] Unsupported mode '{args.mode}'. This should not happen.")
        sys.exit(1)

    pipeline_script_name = f"bless_{args.mode}_{timestamp}.sbatch"
    pipeline_script_path = sbatch_dir / pipeline_script_name
    pipeline_script_text = build_sbatch_script(slurm_run_cfg, env_cfg, pipeline_body)
    pipeline_script_path.write_text(pipeline_script_text, encoding="utf-8")

    print(f"[INFO] Pipeline sbatch script written to: {pipeline_script_path}")

    if auto_submit and not args.dry_run:
        # Dependency chain: pipeline waits for sampling if present, else initial if present.
        dep_for_pipeline = sampling_job_id or initial_job_id
        if dep_for_pipeline:
            print(f"[INFO] Submitting pipeline job with dependency afterok:{dep_for_pipeline} ...")
        else:
            print("[INFO] Submitting pipeline job without dependency...")
        pipeline_job_id = submit_sbatch(pipeline_script_path, dependency=dep_for_pipeline)

    print()
    print("====================================")
    print("  BLESS CLI - ORCHESTRATION SUMMARY")
    print("====================================")
    print(f"Run directory       : {run_root}")
    print(f"Initial job script  : {('none (skipped)' if args.skip_initial else sbatch_dir / initial_script_name)}")
    print(f"Sampling job script : {('none (sampling == 0)' if not (sampling and sampling > 0) else sbatch_dir / sampling_script_name)}")
    print(f"Pipeline job script : {sbatch_dir / pipeline_script_name}")
    if auto_submit and not args.dry_run:
        print()
        print("Submitted jobs:")
        if not args.skip_initial:
            print(f"  initial   : {initial_job_id}")
        if sampling and sampling > 0:
            print(f"  sampling  : {sampling_job_id}")
        print(f"  pipeline  : {pipeline_job_id}")
    else:
        print()
        print("auto_submit is disabled or dry-run was used.")
        print("You can submit the jobs manually with:")
        if not args.skip_initial and initial_script_name is not None:
            print(f"  sbatch {sbatch_dir / initial_script_name}")
        if sampling and sampling > 0 and sampling_script_name is not None:
            if not args.skip_initial:
                print(f"  sbatch --dependency=afterok:<initial_job_id> {sbatch_dir / sampling_script_name}")
            else:
                print(f"  sbatch {sbatch_dir / sampling_script_name}")
        dep_hint = "<sampling_job_id>" if (sampling and sampling > 0) else "<initial_job_id>"
        print(f"  sbatch --dependency=afterok:{dep_hint} {sbatch_dir / pipeline_script_name}")

if __name__ == "__main__":
    main()
