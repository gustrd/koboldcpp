#!/usr/bin/env python3
"""
loop-kcpp.py - Cross-platform monitor and auto-restart for worker processes.

Main features:
 - Checks if a worker appears in an API response (searches for substring in JSON/text response).
 - Starts the worker via provided command and maintains process handle (Popen).
 - When worker is detected offline (substring not found), restarts ONLY the monitored process.
 - Avoids killing global python.exe processes: kill by handle is the default approach.
 - Options for fallback by process name (only if explicitly permitted).
 - Sends CTRL_BREAK_EVENT on Windows as graceful attempt before forced kill.
 - Handles SIGINT/SIGTERM for graceful shutdown.
 - Simple logging to stdout with timestamps.
 - Battery monitoring to prevent operation on low battery.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from typing import Optional

try:
    import psutil  # optional, used for safer process-by-name handling
except Exception:
    psutil = None


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
    # tenta psutil se disponível
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


def _search_json_for_string(obj, substring: str) -> bool:
    """Busca recursiva de substring em JSON (keys/values)."""
    try:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if substring.lower() in str(k).lower():
                    return True
                if _search_json_for_string(v, substring):
                    return True
        elif isinstance(obj, list):
            for it in obj:
                if _search_json_for_string(it, substring):
                    return True
        else:
            if substring.lower() in str(obj).lower():
                return True
    except Exception:
        return False
    return False


def http_contains(url: str, substring: str, timeout: int = 10) -> bool:
    """
    Performs GET request to URL, attempts to parse JSON and searches for substring.
    Returns False only if the substring is not found in a valid response.
    Returns True for timeout errors (assumes worker is still online).
    """

    headers = {
        "X-Fields": "name",
        "User-Agent": "Mozilla/5.0",    # prevents bot blocking
        "Accept": "application/json",   # tells server what you expect  
    }

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            try:
                text = raw.decode("utf-8", errors="ignore")
            except Exception:
                text = str(raw)
            content_type = (resp.headers.get("Content-Type") or "").lower()
            if "application/json" in content_type or text.lstrip().startswith(("{", "[")):
                try:
                    json_data = json.loads(text)
                    if _search_json_for_string(json_data, substring):
                        return True
                except Exception:
                    pass
            return substring.lower() in text.lower()

    except Exception as e:
        print(f"{time.asctime()} - HTTP error when querying {url}: {e}. Will try again.", file=sys.stderr)
        return True


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
        try:
            for line in iter(pipe.readline, ""):
                if not line:
                    break
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
    Ensures that process `p` terminates; tries graceful methods before forced kill.
    - First attempts to terminate (p.terminate()).
    - On Windows, if specified, tries CTRL_BREAK_EVENT.
    - Waits `timeout` seconds; if not terminated, calls p.kill().
    """
    if not p:
        return
    
    log_to_file(f"END inner script run - Reason: {end_reason} - PID: {p.pid}")
    try:
        if p.poll() is not None:
            return  # Already terminated
        # Graceful attempt
        try:
            p.terminate()
        except Exception:
            pass

        # On Windows, sending CTRL_BREAK_EVENT may allow cleanup (if p was created in new process group)
        if try_graceful_windows and platform.system() == "Windows":
            try:
                _send_ctrl_break_windows(p)
            except Exception:
                pass

        try:
            p.wait(timeout=timeout)
            log(f"Process PID {p.pid} terminated cleanly")
            log_to_file(f"Process PID {p.pid} terminated cleanly")
            return
        except subprocess.TimeoutExpired:
            print(f"{time.asctime()} - Process PID {p.pid} did not exit after {timeout}s; killing...", file=sys.stderr)
            log_to_file(f"Process PID {p.pid} did not exit after {timeout}s; force killing")
            try:
                p.kill()
            except Exception:
                pass
            try:
                p.wait(timeout=5)
            except Exception:
                pass
    except Exception as e:
        print(f"{time.asctime()} - Error ensuring process termination: {e}", file=sys.stderr)
        log_to_file(f"Error ensuring process termination: {e}")


def kill_process_by_name_safe(name: str, exclude_pids: Optional[set[int]] = None, allow_kill_python: bool = False) -> None:
    """
    Mata processos que casem `name`, mas com precauções:
      - Se psutil disponível, usa-o para identificar processos e terminar apenas correspondentes.
      - Exclui PIDs em `exclude_pids`.
      - Não mata `python`/`python.exe` a menos que allow_kill_python True.
    Use com CUIDADO: por padrão este NÃO é chamado automaticamente, apenas como fallback se você pedir.
    """
    exclude_pids = exclude_pids or set()
    lowered = name.lower()

    if psutil:
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

                    # avoid killing global python unless explicitly allowed
                    if match and (pname.startswith("python") or exe.endswith("python.exe") or "python" in cmd):
                        if not allow_kill_python:
                            log(f"Skipping python process PID {pid} (name match) because allow_kill_python=False")
                            continue

                    if match:
                        log(f"Terminating matched process PID {pid} ({pname})")
                        try:
                            p.terminate()
                            p.wait(timeout=5)
                        except Exception:
                            try:
                                p.kill()
                            except Exception:
                                pass
                except Exception:
                    continue
            return
        except Exception as e:
            log(f"psutil fallback failed: {e}")

    # Fallback: use system commands, but avoid catching generic 'python' unless explicitly allowed
    sysname = platform.system()
    if sysname == "Windows":
        if not allow_kill_python and name.lower().startswith("python"):
            log("Skipping taskkill for python names (allow_kill_python=False)")
            return
        try:
            subprocess.run(["taskkill", "/F", "/IM", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        if not name.lower().endswith(".exe"):
            try:
                subprocess.run(["taskkill", "/F", "/IM", name + ".exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
    else:
        # on unix, avoid `pkill -f python` unless allowed
        if not allow_kill_python and "python" in name.lower():
            log("Skipping pkill for python names (allow_kill_python=False)")
            return
        try:
            subprocess.run(["pkill", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


# ---------------- Main monitor loop ----------------

stop_event = threading.Event()


def interruptible_sleep(total_seconds: int, sleep_interval: int) -> bool:
    """
    Sleep for total_seconds in chunks of sleep_interval, checking for stop_event.
    Returns True if interrupted by stop_event, False if completed normally.
    """
    for _ in range(max(1, total_seconds // sleep_interval)):
        if stop_event.is_set():
            return True
        time.sleep(sleep_interval)
    return False


def handle_signal(sig, frame):
    log(f"Signal {sig} received — stopping monitor")
    stop_event.set()


def main() -> None:
    """Main function that sets up and runs the worker monitor."""
    parser = argparse.ArgumentParser(description="Monitor and auto-restart worker processes (Horde/API)")
    parser.add_argument("--string", help="Substring to search for in the API response")
    parser.add_argument("--url", default="https://aihorde.net/api/v2/workers", help="API URL to check for worker status")
    parser.add_argument("--start-command", required=True, help="Command to start the worker (string)")
    parser.add_argument("--no-shell", action="store_true", help="Execute start-command without shell (recommended when passing list)")
    parser.add_argument("--interval", type=int, default=120, help="Interval in seconds between status checks")
    parser.add_argument("--sleep-interval", type=int, default=10, help="Sleep interval in seconds for sub-loops (default: 10)")
    parser.add_argument("--start-wait", type=int, default=5, help="Seconds to wait after starting the worker")
    parser.add_argument("--battery-threshold", type=int, default=58, help="Battery level threshold below which the worker will be halted (default: 58)")
    parser.add_argument("--process-name", default=None, help="Process name (exact/executable) for fallback kill by name (not used by default)")
    parser.add_argument("--allow-name-kill", action="store_true", help="Allow using kill by name as fallback (CAUTION: does not kill python by default)")
    parser.add_argument("--stop-kills-worker", action="store_true", help="When stopping the monitor, also terminate the worker")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, handle_signal)
    try:
        signal.signal(signal.SIGTERM, handle_signal)
    except Exception:
        pass

    current_proc: Optional[subprocess.Popen] = None
    log(f"Monitor started: url={args.url}, string={args.string}, interval={args.interval}s")

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

    # Initial wait to let worker register
    if interruptible_sleep(args.start_wait, args.sleep_interval):
        log("Interrupted during initial wait")

    # Main monitoring loop
    while not stop_event.is_set():
        try:
            battery_level = get_battery_percent()
            if battery_level is not None and battery_level <= args.battery_threshold:
                log(f"Low battery ({battery_level}% <= {args.battery_threshold}%), halting...")
                # Try graceful termination by handle first
                ensure_terminate_process(current_proc, timeout=8, try_graceful_windows=True, end_reason=f"Low battery ({battery_level}%)")
                current_proc = None

                # Fallback: only if user explicitly allowed, kill by name (careful)
                if args.process_name and args.allow_name_kill:
                    # Exclude current monitor PID so we don't accidentally kill ourselves
                    exclude_pids = {os.getpid()}
                    kill_process_by_name_safe(args.process_name, exclude_pids=exclude_pids, allow_kill_python=False)
                
                if interruptible_sleep(args.interval, args.sleep_interval):
                    break
                continue

            online = http_contains(args.url, args.string)
            if online:
                log("Worker appears online, or request failed.")
                if interruptible_sleep(args.interval, args.sleep_interval):
                    break
                continue

            # Not online -> restart the worker process we launched (only that)
            log("Worker offline (substring not found). Restarting monitored worker...")
            # Try graceful termination by handle first
            ensure_terminate_process(current_proc, timeout=8, try_graceful_windows=True, end_reason="Worker offline (substring not found)")
            current_proc = None

            # Fallback: only if user explicitly allowed, kill by name (careful)
            if args.process_name and args.allow_name_kill:
                # Exclude current monitor PID so we don't accidentally kill ourselves
                exclude_pids = {os.getpid()}
                kill_process_by_name_safe(args.process_name, exclude_pids=exclude_pids, allow_kill_python=False)

            # Restart the worker
            try:
                current_proc = stream_process_output(args.start_command, use_shell=not args.no_shell, start_reason="Worker offline restart")
                log("Restart command executed")
            except Exception as e:
                print(f"{time.asctime()} - Error restarting worker: {e}", file=sys.stderr)

            if interruptible_sleep(args.interval, args.sleep_interval):
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
