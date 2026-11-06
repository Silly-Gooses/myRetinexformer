"""Bounded execution, atomic reports, and small report exports."""
import hashlib
import csv
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np

STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("Stop requested: saving at the next safe boundary.", flush=True)


def sha_file(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            d.update(block)
    return d.hexdigest()


def runtime_info(device):
    from model_d.cli import runtime_info as baseline_runtime_info
    from .checkpoint import PINNED_COMMIT, SOURCE_SHA256
    info = baseline_runtime_info(device)
    info["backbone_source_commit"] = info.pop("reference_commit")
    info["baseline_commit"] = PINNED_COMMIT
    info["original_script_sha256"] = SOURCE_SHA256
    return info


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def append_csv(path, row):
    path = Path(path)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()


class BudgetStop(RuntimeError):
    pass


class Budget:
    def __init__(self, minutes, reserve_seconds=90):
        self.started = time.monotonic()
        self.deadline = self.started + minutes * 60 - reserve_seconds

    def check(self):
        if STOP_REQUESTED or time.monotonic() >= self.deadline:
            raise BudgetStop("session_time_budget_or_stop_request")

    def remaining(self):
        return max(0.0, self.deadline - time.monotonic())


def export_small_report(run):
    target = run / "DPro_Pilot_Report.zip"
    temp = run / "DPro_Pilot_Report.zip.tmp"
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(run.rglob("*")):
            if path.is_file() and path.suffix in {".json", ".csv", ".log", ".txt", ".py"}:
                z.write(path, path.relative_to(run))
        for path in sorted((run / "comparisons").glob("*.png")):
            z.write(path, path.relative_to(run))
    os.replace(temp, target)
    return target


def launch_training(cfg, session_minutes=45):
    """Isolated worker with the notebook's terminate/kill watchdog and log stream."""
    import subprocess
    import sys
    import tempfile
    import threading
    import uuid
    from .checkpoint import ROOT
    from .train import protect_output, validate_config

    validate_config(cfg, session_minutes)
    run = Path(cfg["output_root"]) / cfg["run_name"]
    protect_output(run, [cfg["data_root"], cfg["model_b"], cfg["head"]])
    session_id = uuid.uuid4().hex
    env = dict(os.environ, DPRO_SESSION_ID=session_id)
    fired = threading.Event()
    kill_timer = None
    with tempfile.TemporaryDirectory(prefix="dpro_worker_") as temporary:
        config_path = Path(temporary) / "config.json"
        write_json(config_path, cfg)
        command = [sys.executable, "-u", "-m", "model_d_pro.cli", "train", "--worker",
                   "--config", str(config_path), "--session-minutes", str(session_minutes)]
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)

        def hard_stop():
            nonlocal kill_timer
            if proc.poll() is None:
                fired.set()
                proc.terminate()
                def kill_if_needed():
                    if proc.poll() is None:
                        proc.kill()
                kill_timer = threading.Timer(10, kill_if_needed)
                kill_timer.daemon = True
                kill_timer.start()

        timer = threading.Timer(session_minutes * 60, hard_stop)
        timer.daemon = True
        timer.start()
        log_path = Path(temporary) / "training.log"
        try:
            with log_path.open("w", encoding="utf-8") as log:
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
            code = proc.wait()
        finally:
            timer.cancel()
            if kill_timer is not None:
                kill_timer.cancel()
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            proc.stdout.close()
            # A rejected existing run must remain byte-for-byte untouched.
            runtime_path = run / "runtime.json"
            owned = (runtime_path.is_file()
                     and json.loads(runtime_path.read_text()).get("session_id") == session_id)
            if owned:
                with (run / "training.log").open("a", encoding="utf-8") as log:
                    log.write(f"\n=== Modular session {session_id} ===\n")
                    log.write(log_path.read_text(encoding="utf-8"))
                if fired.is_set():
                    write_json(run / "watchdog.json", dict(outer_time_cap_fired=True,
                               minutes=session_minutes, session_id=session_id))
                export_small_report(run)
        return code
