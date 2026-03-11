#!/usr/bin/env python3
"""
loop_kcpp.py - Cross-platform monitor and auto-restart for worker processes.

USAGE:
    python loop_kcpp.py --start-command "python worker.py" --string "MyWorkerName" [options]

USE CASE:
    Monitors an AI Horde worker (or similar API-registered service) by periodically
    querying an API endpoint. If the worker's name disappears from the response
    (indicating it crashed or became unresponsive), automatically restarts it.
    Also halts operation when battery is low (useful for laptops/portable setups).

EXAMPLES:
    # Basic: restart worker if "MyWorker" not found in API response
    python loop_kcpp.py --start-command "./run_worker.sh" --string "MyWorker"

    # Custom API endpoint and check interval
    python loop_kcpp.py --start-command "python kobold_worker.py" \\
        --string "MyWorker" --url "https://api.example.com/workers" --interval 60

    # With battery threshold (halt if battery <= 30%)
    python loop_kcpp.py --start-command "./worker.sh" --string "MyWorker" --battery-threshold 30

OPTIONS:
    --start-command   Command to start the worker (required)
    --string          Substring to search for in API response (worker name)
    --url             API endpoint to check (default: https://aihorde.net/api/v2/workers)
    --interval        Seconds between checks (default: 120)
    --battery-threshold  Halt worker if battery <= this % (default: 58)
    --no-shell        Run command without shell (safer, use with simple commands)
    --stop-kills-worker  Also terminate worker when monitor stops (Ctrl+C)

REQUIREMENTS:
    - Python 3.8+
    - psutil (pip install psutil)
"""
from __future__ import annotations

import argparse
import glob
import os
import platform
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from typing import Optional

import psutil  # required for reliable process tree termination


# Global tracking for the last time the worker printed something
last_activity_time = time.time()


def log(msg: str) -> None:
    BLUE = "\033[94m"
    RESET = "\033[0m"
    print(f"{time.asctime()} - {BLUE}{msg}{RESET}")

def log_to_file(msg: str, log_file: str = "_loop-kcpp.log") -> None:
    """Log message to file with timestamp."""
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{time.asctime()} - {msg}\n")
    except Exception as e:
        print(f"{time.asctime()} - Error writing to log file {log_file}: {e}", file=sys.stderr)

def get_battery_percent():
    # try psutil if available
    try:
        import psutil
        b = psutil.sensors_battery()
        if b is not None and b.percent is not None:
            return int(b.percent)
    except Exception:
        pass

    sysname = platform.system()
    # Windows fallback via ctypes GetSystemPowerStatus
    if sysname == "Windows":
        try:
            import ctypes
            class SYSTEM_POWER_STATUS(ctypes.Structure):
                _fields_ = [
                    ("ACLineStatus", ctypes.c_byte),
                    ("BatteryFlag", ctypes.c_byte),
                    ("BatteryLifePercent", ctypes.c_byte),
                    ("Reserved1", ctypes.c_byte),
                    ("BatteryLifeTime", ctypes.c_ulong),
                    ("BatteryFullLifeTime", ctypes.c_ulong)
                ]
            status = SYSTEM_POWER_STATUS()
            if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
                percent = status.BatteryLifePercent
                if percent != 255:
                    return int(percent)
        except Exception:
            pass
        return None

    # Linux fallback
    if sysname == "Linux":
        try:
            for path in glob.glob("/sys/class/power_supply/BAT*/capacity"):
                with open(path, "r") as f:
                    txt = f.read().strip()
                    m = re.search(r"(\d+)", txt)
                    if m:
                        return int(m.group(1))
        except Exception:
            pass
        return None

    # macOS fallback
    if sysname == "Darwin":
        try:
            out = subprocess.check_output(["pmset", "-g", "batt"], stderr=subprocess.DEVNULL).decode()
            m = re.search(r"(\d+)%", out)
            if m:
                return int(m.group(1))
        except Exception:
            pass
        return None

    return None


def stream_process_output(cmd: str, use_shell: bool = True, start_reason: str = "Unknown") -> subprocess.Popen:
    """
    Starts the process and creates a thread that prints stdout/stderr in real time.
    Returns the Popen object. Creates new process-group/session to facilitate killing only that group.
    """
    log(f"Starting command: {cmd}")
    log_to_file(f"BEGIN inner script run - Reason: {start_reason} - Command: {cmd}")
    kwargs = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "encoding": "utf-8",
        "errors": "replace", 
        "text": True,
        "bufsize": 1
        }

    if platform.system() == "Windows":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True

    if use_shell:
        process = subprocess.Popen(cmd, shell=True, **kwargs)  # type: ignore[arg-type]
    else:
        cmd_list = shlex.split(cmd) if isinstance(cmd, str) else cmd
        process = subprocess.Popen(cmd_list, shell=False, **kwargs)

    def reader(pipe):
        global last_activity_time
        try:
            for line in iter(pipe.readline, ""):
                if not line:
                    break
                last_activity_time = time.time()  # Update activity timestamp
                sys.stdout.write(f"KOBOLD: {line}")
                sys.stdout.flush()
        except Exception as e:
            print(f"{time.asctime()} - Error reading worker output: {e}", file=sys.stderr)
        finally:
            try:
                pipe.close()
            except Exception:
                pass
            log("Worker output stream closed")

    thread = threading.Thread(target=reader, args=(process.stdout,), daemon=True)
    thread.start()
    log(f"Process started (PID {process.pid})")
    return process


def _send_ctrl_break_windows(p: subprocess.Popen) -> bool:
    """
    Attempts to send CTRL_BREAK_EVENT to the process.
    Requires CREATE_NEW_PROCESS_GROUP and being in console mode.
    """
    try:
        # os.kill with CTRL_BREAK_EVENT sends to process group on Windows
        if hasattr(signal, "CTRL_BREAK_EVENT"):
            os.kill(p.pid, signal.CTRL_BREAK_EVENT)
            return True
    except Exception:
        pass
    return False


def ensure_terminate_process(p: Optional[subprocess.Popen], timeout: int = 8, try_graceful_windows: bool = True, end_reason: str = "Unknown") -> None:
    """
    Ensures that process `p` and all its children terminate using psutil.
    - Kills entire process tree to prevent orphaned children (critical when shell=True).
    - First attempts graceful termination, then forces kill if needed.
    - On Windows, tries CTRL_BREAK_EVENT for graceful shutdown.
    """
    if not p:
        return

    log_to_file(f"END inner script run - Reason: {end_reason} - PID: {p.pid}")

    if p.poll() is not None:
        log(f"Process PID {p.pid} already terminated")
        return

    try:
        parent = psutil.Process(p.pid)
        children = parent.children(recursive=True)

        # Send graceful shutdown signal based on platform
        if platform.system() == "Windows":
            if try_graceful_windows:
                _send_ctrl_break_windows(p)
            time.sleep(0.5)  # Brief pause to allow signal handling
        else:
            # On Linux/Unix, give parent time to propagate signals to children
            time.sleep(0.5)

        # Refresh children list in case new ones appeared during the graceful window
        try:
            if parent.is_running():
                new_children = parent.children(recursive=True)
                for nc in new_children:
                    if nc not in children:
                        children.append(nc)
        except Exception:
            pass

        # Terminate children first (leaf to root order)
        for child in reversed(children):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass

        # Terminate parent
        try:
            parent.terminate()
        except psutil.NoSuchProcess:
            pass

        # Wait for all processes to terminate
        all_procs = children + [parent]
        gone, alive = psutil.wait_procs(all_procs, timeout=timeout)

        if alive:
            log(f"Force killing {len(alive)} processes that did not terminate gracefully")
            log_to_file(f"Force killing {len(alive)} processes that did not terminate gracefully")

            # Windows: Aggressive tree kill if parent is still around
            taskkill_success = False
            if platform.system() == "Windows" and parent in alive:
                try:
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], 
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                    taskkill_success = True
                except Exception:
                    pass
            
            # Individual kill for remaining (or if taskkill skipped/failed)
            for proc in alive:
                try:
                    # If we ran taskkill, verify it's actually running before killing again
                    if taskkill_success and not proc.is_running():
                        continue
                    proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            # Wait again after kill
            gone2, alive2 = psutil.wait_procs(alive, timeout=3)
            if alive2:
                print(f"{time.asctime()} - WARNING: {len(alive2)} processes still running after kill", file=sys.stderr)
                log_to_file(f"WARNING: {len(alive2)} processes still running after kill")

        log(f"Process tree rooted at PID {p.pid} terminated ({len(gone)} graceful, {len(alive)} forced)")
        log_to_file(f"Process tree rooted at PID {p.pid} terminated ({len(gone)} graceful, {len(alive)} forced)")

    except psutil.NoSuchProcess:
        log(f"Process PID {p.pid} no longer exists")
    except Exception as e:
        print(f"{time.asctime()} - Error ensuring process termination: {e}", file=sys.stderr)
        log_to_file(f"Error ensuring process termination: {e}")


def kill_process_by_name_safe(name: str, exclude_pids: Optional[set[int]] = None, allow_kill_python: bool = False) -> None:
    """
    Kills processes matching `name` with safety precautions:
      - Uses psutil to identify and terminate only matching processes.
      - Excludes PIDs in `exclude_pids`.
      - Does not kill `python`/`python.exe` unless allow_kill_python is True.
    Use with CAUTION: by default this is NOT called automatically, only as fallback if explicitly requested.
    """
    exclude_pids = exclude_pids or set()
    lowered = name.lower()

    try:
        for p in psutil.process_iter(["pid", "name", "exe", "cmdline"]):
            try:
                pid = p.info.get("pid")
                if pid in exclude_pids:
                    continue
                pname = (p.info.get("name") or "").lower()
                exe = (p.info.get("exe") or "").lower()
                cmd = " ".join(p.info.get("cmdline") or []).lower()

                match = False
                # match exact exe name or substring in cmdline
                if pname == lowered or exe.endswith(lowered) or lowered in cmd:
                    match = True

                # avoid killing global python unless explicitly allowed (check both Windows and Linux patterns)
                if match and (pname.startswith("python") or exe.endswith("python.exe") or "/python" in exe or "python" in cmd):
                    if not allow_kill_python:
                        log(f"Skipping python process PID {pid} (name match) because allow_kill_python=False")
                        continue

                if match:
                    log(f"Terminating matched process PID {pid} ({pname})")
                    try:
                        p.terminate()
                        p.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        log(f"Process PID {pid} did not terminate, force killing")
                        try:
                            p.kill()
                        except psutil.NoSuchProcess:
                            pass
                    except psutil.NoSuchProcess:
                        pass
            except psutil.NoSuchProcess:
                continue
            except psutil.AccessDenied as e:
                log(f"Access denied for process PID {p.info.get('pid')}: {e}")
                continue
    except Exception as e:
        print(f"{time.asctime()} - Error in kill_process_by_name_safe: {e}", file=sys.stderr)
        log_to_file(f"Error in kill_process_by_name_safe: {e}")


# ---------------- Main monitor loop ----------------

stop_event = threading.Event()


def interruptible_sleep(total_seconds: int, sleep_interval: int, proc: Optional[subprocess.Popen] = None) -> bool:
    """
    Sleep for total_seconds in chunks of sleep_interval, checking for stop_event.
    If proc is provided, also wakes early if the process has exited (but returns False so
    the main loop continues and can restart it rather than shutting down the monitor).
    Returns True only if stop_event is set (monitor should stop), False otherwise.
    """
    for _ in range(max(1, total_seconds // sleep_interval)):
        if stop_event.is_set():
            return True
        if proc is not None and proc.poll() is not None:
            log(f"Worker process exited unexpectedly (code {proc.returncode}), waking monitor early.")
            return False
        time.sleep(sleep_interval)
    return False


def handle_signal(sig, frame):
    log(f"Signal {sig} received — stopping monitor")
    stop_event.set()


def main() -> None:
    """Main function that sets up and runs the worker monitor."""
    parser = argparse.ArgumentParser(description="Monitor and auto-restart worker processes")
    parser.add_argument("--start-command", required=True, help="Command to start the worker (string)")
    parser.add_argument("--no-shell", action="store_true", help="Execute start-command without shell (recommended when passing list)")
    parser.add_argument("--interval", type=int, default=300, help="Interval in seconds between status checks (default: 300)")
    parser.add_argument("--sleep-interval", type=int, default=10, help="Sleep interval in seconds for sub-loops (default: 10)")
    parser.add_argument("--start-wait", type=int, default=None, help="Seconds to wait after starting the worker before first activity check (default: interval)")
    parser.add_argument("--battery-threshold", type=int, default=58, help="Battery level threshold below which the worker will be halted (default: 58)")
    parser.add_argument("--process-name", default=None, help="Process name (exact/executable) for fallback kill by name (not used by default)")
    parser.add_argument("--allow-name-kill", action="store_true", help="Allow using kill by name as fallback (CAUTION: does not kill python by default)")
    parser.add_argument("--stop-kills-worker", action="store_true", help="When stopping the monitor, also terminate the worker")
    parser.add_argument("--activity-timeout", type=int, default=300, help="Seconds of silence before considering the worker stuck (default: 300)")
    
    # Deprecated/unused arguments kept for compatibility
    parser.add_argument("--string", help="Substring to search for (deprecated, unused)")
    parser.add_argument("--url", help="API URL to check (deprecated, unused)")
    parser.add_argument("--local-url", help="Local API URL (deprecated, unused)")
    
    args = parser.parse_args()

    signal.signal(signal.SIGINT, handle_signal)
    try:
        signal.signal(signal.SIGTERM, handle_signal)
    except Exception:
        pass

    current_proc: Optional[subprocess.Popen] = None
    log(f"Monitor started: interval={args.interval}s, activity_timeout={args.activity_timeout}s")

    # Start initial worker
    try:
        battery_level = get_battery_percent()
        if battery_level is None or battery_level > args.battery_threshold:
            current_proc = stream_process_output(args.start_command, use_shell=not args.no_shell, start_reason="Initial startup")
            log("Initial start command executed")
        else:
            log(f"Low battery ({battery_level}% <= {args.battery_threshold}%), halting...")
    except Exception as e:
        print(f"{time.asctime()} - Error starting worker initially: {e}", file=sys.stderr)

    # Initial wait to let worker start
    start_wait = args.start_wait if args.start_wait is not None else args.interval
    if interruptible_sleep(start_wait, args.sleep_interval, current_proc):
        log("Interrupted during initial wait")

    # Main monitoring loop
    while not stop_event.is_set():
        try:
            # Check if the worker process exited on its own — restart immediately
            if current_proc is not None and current_proc.poll() is not None:
                log(f"Worker process exited unexpectedly (code {current_proc.returncode}). Restarting...")
                current_proc = None
                try:
                    current_proc = stream_process_output(args.start_command, use_shell=not args.no_shell, start_reason="Unexpected worker exit")
                    log("Restart command executed")
                except Exception as e:
                    print(f"{time.asctime()} - Error restarting worker: {e}", file=sys.stderr)
                if interruptible_sleep(start_wait, args.sleep_interval, current_proc):
                    break
                continue

            battery_level = get_battery_percent()
            if battery_level is not None and battery_level <= args.battery_threshold:
                log(f"Low battery ({battery_level}% <= {args.battery_threshold}%), halting...")
                ensure_terminate_process(current_proc, timeout=8, try_graceful_windows=True, end_reason=f"Low battery ({battery_level}%)")
                current_proc = None

                if args.process_name and args.allow_name_kill:
                    exclude_pids = {os.getpid()}
                    kill_process_by_name_safe(args.process_name, exclude_pids=exclude_pids, allow_kill_python=False)
                
                if interruptible_sleep(args.interval, args.sleep_interval, current_proc):
                    break
                continue

            # Check activity based on last log time
            time_since_last_log = time.time() - last_activity_time
            if time_since_last_log < args.activity_timeout:
                log(f"Worker appears active. Last log: {int(time_since_last_log)}s ago.")
                if interruptible_sleep(args.interval, args.sleep_interval, current_proc):
                    break
                continue

            log(f"Worker confirmed stuck (no logs for {int(time_since_last_log)}s). Restarting...")
            ensure_terminate_process(current_proc, timeout=8, try_graceful_windows=True, end_reason="Worker confirmed stuck (no activity)")
            current_proc = None

            if args.process_name and args.allow_name_kill:
                exclude_pids = {os.getpid()}
                kill_process_by_name_safe(args.process_name, exclude_pids=exclude_pids, allow_kill_python=False)

            # Restart the worker
            try:
                current_proc = stream_process_output(args.start_command, use_shell=not args.no_shell, start_reason="Worker activity restart")
                log("Restart command executed")
            except Exception as e:
                print(f"{time.asctime()} - Error restarting worker: {e}", file=sys.stderr)

            if interruptible_sleep(start_wait, args.sleep_interval, current_proc):
                break
        except Exception as e:
            print(f"{time.asctime()} - Error in main loop: {e}", file=sys.stderr)
            time.sleep(args.sleep_interval)

    # Shutdown sequence
    log("Monitor stopping: ensuring worker termination (if requested)...")
    try:
        if args.stop_kills_worker:
            ensure_terminate_process(current_proc, timeout=6, try_graceful_windows=True, end_reason="Monitor shutdown")
        else:
            log("stop_kills_worker is False: leaving worker running")
    except Exception:
        pass
    log("Monitor stopped.")


if __name__ == "__main__":
    main()
