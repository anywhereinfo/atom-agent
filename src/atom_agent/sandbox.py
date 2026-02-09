import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class SandboxResult(dict):
    """
    Simple dict result:
      - ok: bool
      - exit_code: int
      - timed_out: bool
      - stdout: str
      - stderr: str
      - cmd: List[str]
    """
    pass


def _cap_output(data: bytes, limit_bytes: int) -> bytes:
    if len(data) <= limit_bytes:
        return data
    # Keep the tail (often most useful for tracebacks)
    return b"...(truncated)...\n" + data[-limit_bytes:]


def run_in_bubblewrap(
    attempt_dir: Path,
    script_path: Path,
    *,
    python_bin: str = "/usr/bin/python3",
    args: Optional[List[str]] = None,
    timeout_s: int = 10,
    stdout_limit_bytes: int = 200_000,
    stderr_limit_bytes: int = 200_000,
    extra_env: Optional[Dict[str, str]] = None,
) -> SandboxResult:
    """
    Run `python script_path [args...]` inside bubblewrap.

    attempt_dir: host path that will be bind-mounted as /work (RW)
    script_path: host path to script; must be inside attempt_dir for simplest usage
    """
    attempt_dir = attempt_dir.resolve()
    script_path = script_path.resolve()

    if not attempt_dir.exists():
        raise FileNotFoundError(f"attempt_dir does not exist: {attempt_dir}")
    if not script_path.exists():
        raise FileNotFoundError(f"script_path does not exist: {script_path}")

    # Strongly recommended: keep the script inside attempt_dir so you don't accidentally mount more.
    try:
        script_path.relative_to(attempt_dir)
    except ValueError:
        raise ValueError(
            f"script_path must be under attempt_dir for this simple sandbox.\n"
            f"attempt_dir={attempt_dir}\nscript_path={script_path}"
        )

    args = args or []
    extra_env = extra_env or {}

    # Minimal clean environment (avoid leaking secrets/proxies/credentials)
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/work",
        "TMPDIR": "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        **extra_env,
    }

    # Map your attempt_dir -> /work inside the sandbox
    # Use ro-binds for system dirs; /tmp is tmpfs; /work is RW
    #
    # Note: we *do not* unshare net, since your WSL disallows netns anyway.
    # Also note: bubblewrap must be installed and available as "bwrap".
    bwrap_cmd = [
        "bwrap",
        "--unshare-user", "--unshare-pid", "--unshare-uts", "--unshare-ipc",
        "--new-session",
        "--die-with-parent",
        # system dirs (read-only)
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/bin", "/bin",
        "--ro-bind", "/lib", "/lib",
        "--ro-bind", "/lib64", "/lib64",
        "--ro-bind", "/etc", "/etc",
        "--ro-bind", "/sbin", "/sbin",
        # proc/dev/tmp
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        # create /home to avoid odd apps failing if they expect it
        "--dir", "/home",
        # your working dir (RW)
        "--bind", str(attempt_dir), "/work",
        "--chdir", "/work",
        "--clearenv",
        "--setenv", "PATH", env["PATH"],
        "--setenv", "HOME", env["HOME"],
        "--setenv", "TMPDIR", env["TMPDIR"],
        "--setenv", "LANG", env["LANG"],
        "--setenv", "LC_ALL", env["LC_ALL"],
    ]

    # Add any extra env vars explicitly (still safe because we're not inheriting host env)
    for k, v in extra_env.items():
        bwrap_cmd += ["--setenv", k, v]

    # We run through bash to apply ulimits without needing extra tooling.
    # You can tighten these over time.
    #
    # -t CPU seconds
    # -v virtual memory KB (~512MB)
    # -n file descriptors
    #
    # NOTE: ulimit is a shell builtin, so we use bash -lc.
    inner_cmd = (
        f"ulimit -t 30 -v 524288 -n 128; "
        f"{shlex.quote(python_bin)} {shlex.quote(str(script_path.relative_to(attempt_dir)))}"
    )
    if args:
        inner_cmd += " " + " ".join(shlex.quote(a) for a in args)

    full_cmd = bwrap_cmd + ["--", "bash", "-lc", inner_cmd]

    try:
        proc = subprocess.run(
            full_cmd,
            env={},  # IMPORTANT: env is set via --setenv; don't leak host env
            cwd=str(attempt_dir),  # only affects host-side; sandbox uses --chdir /work
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
        stdout_b = _cap_output(proc.stdout, stdout_limit_bytes)
        stderr_b = _cap_output(proc.stderr, stderr_limit_bytes)

        return SandboxResult(
            ok=(proc.returncode == 0),
            exit_code=proc.returncode,
            timed_out=False,
            stdout=stdout_b.decode("utf-8", errors="replace"),
            stderr=stderr_b.decode("utf-8", errors="replace"),
            cmd=full_cmd,
        )

    except subprocess.TimeoutExpired as e:
        out = e.stdout or b""
        err = e.stderr or b""
        return SandboxResult(
            ok=False,
            exit_code=124,
            timed_out=True,
            stdout=_cap_output(out, stdout_limit_bytes).decode("utf-8", errors="replace"),
            stderr=_cap_output(err, stderr_limit_bytes).decode("utf-8", errors="replace"),
            cmd=full_cmd,
        )
