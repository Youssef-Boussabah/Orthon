"""Bounded process-tree cleanup for every distributed specification workload."""

import os
import signal
import subprocess


def run_bounded(command, *, payload=None, env=None, timeout=45):
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, env=env,
                               start_new_session=True)
    try:
        stdout, stderr = process.communicate(payload, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=5)
        raise TimeoutError(f"distributed workload exceeded {timeout}s; process tree terminated") from error
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
