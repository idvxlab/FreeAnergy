#!/usr/bin/env python3
"""JSONL sidecar for vfe_exp3.DockerEnv.

Protocol (stdin/stdout, one JSON object per line):
  --> {"id":1,"op":"start","task":"circuit-fibsqrt","rep":2105}
  <-- {"id":1,"ok":true,"container":"tb2run-circuit-fibsqrt-r2105"}
  --> {"id":2,"op":"exec","container":"...","cmd":"ls","timeout":120}
  <-- {"id":2,"ok":true,"output":"...","exit_code":0}
  <-- {"id":2,"ok":true,"output":"...","exit_code":1,"infra":true}
  --> {"id":5,"op":"exec_tests","container":"...","task":"...","cmd":"...","timeout":600}
  <-- {"id":5,"ok":true,"output":"...","exit_code":0}
  --> {"id":3,"op":"restart","container":"...","task":"...","rep":99980}
  <-- {"id":3,"ok":true,"container":"...","mode":"started"}
  --> {"id":4,"op":"stop","container":"..."}
  <-- {"id":4,"ok":true}
  --> {"id":6,"op":"start_verify","task":"...","rep":999,"src":"tb2run-..."}
  <-- {"id":6,"ok":true,"container":"tb2verify-..."}
"""
from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vfe_exp3 import DockerEnv  # noqa: E402

_envs: dict[str, DockerEnv] = {}
# container name -> (task, rep) for restart recreate without client metadata
_env_meta: dict[str, tuple[str, int]] = {}
# verify container -> committed image tag (removed on stop; see start_verify)
_verify_images: dict[str, str] = {}
# Serialize dict mutations across worker threads (exec/restart/start may overlap).
_env_lock = threading.Lock()
# One reply at a time to keep stdout JSONL framing intact.
_out_lock = threading.Lock()

# Host-side preclones (network workaround): task -> [(host_src, container_dst)]
# Populated under /tmp/vfe_preclone/<task>/ when github.com is unreachable.
PRECLONES: dict[str, list[tuple[str, str]]] = {
    "build-cython-ext": [
        ("/tmp/vfe_preclone/pyknotid", "/app/pyknotid"),
        ("/tmp/vfe_preclone/build-cython-ext", "/opt/vfe_seed"),
        ("/tmp/vfe_preclone/build-cython-ext", "/tmp"),
    ],
    "build-pov-ray": [
        ("/tmp/vfe_preclone/build-pov-ray", "/app"),
        ("/tmp/vfe_preclone/build-pov-ray", "/tmp"),
    ],
    "install-windows-3.11": [
        ("/tmp/vfe_preclone/install-windows-3.11/seed", "/opt/vfe_seed"),
        ("/tmp/vfe_preclone/install-windows-3.11/seed", "/tmp"),
    ],
    "filter-js-from-html": [
        ("/tmp/vfe_preclone/filter-js-from-html/seed", "/opt/vfe_seed"),
        ("/tmp/vfe_preclone/filter-js-from-html/seed", "/tmp"),
    ],
    "pytorch-model-cli": [
        ("/tmp/vfe_preclone/pytorch-model-cli/seed", "/opt/vfe_seed"),
        ("/tmp/vfe_preclone/pytorch-model-cli/seed", "/tmp"),
    ],
    "qemu-alpine-ssh": [
        ("/tmp/vfe_preclone/qemu-alpine-ssh/seed", "/opt/vfe_seed"),
        ("/tmp/vfe_preclone/qemu-alpine-ssh/seed", "/tmp"),
        ("/tmp/vfe_preclone/qemu-alpine-ssh/site", "/usr/local/lib/python3.9/dist-packages"),
    ],
    "query-optimize": [
        ("/tmp/vfe_preclone/query-optimize/seed", "/opt/vfe_seed"),
        ("/tmp/vfe_preclone/query-optimize/seed", "/tmp"),
    ],
    "reshard-c4-data": [
        ("/tmp/vfe_preclone/reshard-c4-data/seed", "/opt/vfe_seed"),
        ("/tmp/vfe_preclone/reshard-c4-data/seed", "/tmp"),
    ],
    "sam-cell-seg": [
        ("/tmp/vfe_preclone/sam-cell-seg/seed", "/opt/vfe_seed"),
        ("/tmp/vfe_preclone/sam-cell-seg/seed", "/tmp"),
        ("/tmp/vfe_preclone/sam-cell-seg/site", "/usr/local/lib/python3.11/site-packages"),
        ("/tmp/vfe_preclone/sam-cell-seg/app", "/app"),
    ],
    # Tasks without a task-specific preclone: seed the shared pytest wheel set so
    # the bootstrap never needs a network round-trip.
    "winning-avg-corewars": [
        ("/tmp/vfe_preclone/_generic_seed", "/opt/vfe_seed"),
        ("/tmp/vfe_preclone/_generic_seed", "/tmp"),
    ],
    "build-pmars": [
        ("/tmp/vfe_preclone/_generic_seed", "/opt/vfe_seed"),
        ("/tmp/vfe_preclone/_generic_seed", "/tmp"),
    ],
    "caffe-cifar-10": [
        ("/tmp/vfe_preclone/_generic_seed", "/opt/vfe_seed"),
        ("/tmp/vfe_preclone/_generic_seed", "/tmp"),
        # 170MB cifar tarball (~25KB/s from cs.toronto.edu) + wget shim that
        # serves it locally instead of re-downloading.
        ("/tmp/vfe_preclone/caffe-cifar-10/data", "/opt/vfe_seed"),
        ("/tmp/vfe_preclone/caffe-cifar-10/shim/usr/local/bin", "/usr/local/bin"),
        ("/tmp/vfe_preclone/caffe-cifar-10/root", "/root"),
    ],
    "torch-tensor-parallelism": [
        ("/tmp/vfe_preclone/_generic_seed", "/opt/vfe_seed"),
        ("/tmp/vfe_preclone/_generic_seed", "/tmp"),
    ],
    "torch-pipeline-parallelism": [
        ("/tmp/vfe_preclone/_generic_seed", "/opt/vfe_seed"),
        ("/tmp/vfe_preclone/_generic_seed", "/tmp"),
    ],
}

# Dirs that must not be rm -rf wiped (shared/base image content).
_SEED_KEEP_DSTS = {"/app", "/tmp", "/usr", "/usr/local/bin", "/etc", "/var", "/opt", "/root"}

# Index/PATH configs: distro agnostic, safe to copy into every container.
GENERIC_CFGS: list[tuple[str, str]] = [
    ("/tmp/vfe_preclone/build-pov-ray/pip.conf", "/etc/pip.conf"),
    ("/tmp/vfe_preclone/build-cython-ext/pip.conf", "/etc/pip.conf"),
    ("/tmp/vfe_preclone/filter-js-from-html/seed/pip.conf", "/etc/pip.conf"),
    ("/tmp/vfe_preclone/filter-js-from-html/seed/etc/profile.d/vfe-net.sh", "/etc/profile.d/vfe-net.sh"),
    ("/tmp/vfe_preclone/filter-js-from-html/seed/root/.local/bin/uv", "/root/.local/bin/uv"),
    ("/tmp/vfe_preclone/filter-js-from-html/seed/root/.local/bin/uvx", "/root/.local/bin/uvx"),
    ("/tmp/vfe_preclone/filter-js-from-html/seed/root/.local/bin/env", "/root/.local/bin/env"),
    ("/tmp/vfe_preclone/install-windows-3.11/seed/pip.conf", "/etc/pip.conf"),
    ("/tmp/vfe_preclone/pytorch-model-cli/seed/pip.conf", "/etc/pip.conf"),
    ("/tmp/vfe_preclone/pytorch-model-cli/seed/etc/profile.d/vfe-net.sh", "/etc/profile.d/vfe-net.sh"),
    ("/tmp/vfe_preclone/pytorch-model-cli/seed/root/.local/bin/uv", "/root/.local/bin/uv"),
    ("/tmp/vfe_preclone/pytorch-model-cli/seed/root/.local/bin/uvx", "/root/.local/bin/uvx"),
    ("/tmp/vfe_preclone/pytorch-model-cli/seed/root/.local/bin/env", "/root/.local/bin/env"),
    ("/tmp/vfe_preclone/qemu-alpine-ssh/seed/pip.conf", "/etc/pip.conf"),
    ("/tmp/vfe_preclone/query-optimize/seed/pip.conf", "/etc/pip.conf"),
    ("/tmp/vfe_preclone/reshard-c4-data/seed/pip.conf", "/etc/pip.conf"),
    ("/tmp/vfe_preclone/sam-cell-seg/seed/pip.conf", "/etc/pip.conf"),
]

# Warm uv caches (download cache + python installs) extracted from a completed
# official run on the host. uvx in test.sh ignores pip.conf, so without these it
# re-downloads cpython/numpy on every container start (slow → flaky official
# verifier). Seeded only when absent: the paths are uv's defaults, and a fresh
# container has none.
GENERIC_DIRS: list[tuple[str, str]] = [
    ("/tmp/vfe_preclone/_generic_seed/uv_cache", "/root/.cache/uv"),
    ("/tmp/vfe_preclone/_generic_seed/uv_python", "/root/.local/share/uv"),
]

# APT sources are distro-specific. The previous flat list copied every task's
# suites into every container, which put Ubuntu noble + bullseye lines on the
# Debian trixie sam image and broke `apt-get install` there; scope them per task.
APT_CFGS: dict[str, list[tuple[str, str]]] = {
    "build-cython-ext": [
        ("/tmp/vfe_preclone/build-cython-ext/sources.list", "/etc/apt/sources.list.d/vfe-sources.list"),
    ],
    "build-pov-ray": [
        ("/tmp/vfe_preclone/build-pov-ray/sources.list", "/etc/apt/sources.list.d/vfe-sources.list"),
        ("/tmp/vfe_preclone/build-pov-ray/vfe-ubuntu.sources", "/etc/apt/sources.list.d/vfe-ubuntu.sources"),
    ],
    "filter-js-from-html": [
        (
            "/tmp/vfe_preclone/filter-js-from-html/seed/etc/apt/sources.list.d/vfe-sources.list",
            "/etc/apt/sources.list.d/vfe-sources.list",
        ),
    ],
    "install-windows-3.11": [
        ("/tmp/vfe_preclone/install-windows-3.11/seed/sources.list", "/etc/apt/sources.list.d/vfe-sources.list"),
        ("/tmp/vfe_preclone/install-windows-3.11/seed/vfe-ubuntu.sources", "/etc/apt/sources.list.d/vfe-ubuntu.sources"),
    ],
    "pytorch-model-cli": [
        (
            "/tmp/vfe_preclone/pytorch-model-cli/seed/etc/apt/sources.list.d/vfe-sources.list",
            "/etc/apt/sources.list.d/vfe-sources.list",
        ),
    ],
    "qemu-alpine-ssh": [
        ("/tmp/vfe_preclone/qemu-alpine-ssh/seed/sources.list", "/etc/apt/sources.list"),
        ("/tmp/vfe_preclone/qemu-alpine-ssh/seed/sources.list", "/etc/apt/sources.list.d/vfe-sources.list"),
    ],
    "query-optimize": [
        ("/tmp/vfe_preclone/query-optimize/seed/sources.list", "/etc/apt/sources.list.d/vfe-sources.list"),
    ],
    "reshard-c4-data": [
        (
            "/tmp/vfe_preclone/reshard-c4-data/seed/etc/apt/sources.list.d/vfe-sources.list",
            "/etc/apt/sources.list.d/vfe-sources.list",
        ),
    ],
    # sam-cell-seg: Debian trixie. The image's own deb.debian.org source serves
    # ~100KB/s here (60MB of libgl1 would take 10min); swap the URIs to tuna
    # (~7MB/s) by overwriting the deb822 file in place.
    "sam-cell-seg": [
        (
            "/tmp/vfe_preclone/sam-cell-seg/seed/etc/apt/sources.list.d/debian.sources",
            "/etc/apt/sources.list.d/debian.sources",
        ),
    ],
    # Newly batched tasks: their solutions run apt-get install, so give them the
    # same tuna swap (deb.debian.org/archive.ubuntu.com serve ~100KB/s here).
    "build-pmars": [
        (
            "/tmp/vfe_preclone/_generic/etc/apt/sources.list.d/debian.sources",
            "/etc/apt/sources.list.d/debian.sources",
        ),
    ],
    "winning-avg-corewars": [
        (
            "/tmp/vfe_preclone/_generic/etc/apt/sources.list.d/debian.sources",
            "/etc/apt/sources.list.d/debian.sources",
        ),
    ],
    "caffe-cifar-10": [
        (
            "/tmp/vfe_preclone/_generic/etc/apt/sources.list.d/ubuntu.sources",
            "/etc/apt/sources.list.d/ubuntu.sources",
        ),
    ],
    # Newly batched Ubuntu 24.04 tasks (default archive.ubuntu.com is slow here).
    # Minimal ubuntu:24.04 images ship no CA certs → https apt fails. Use http,
    # then provisioning installs ca-certificates (see _seed_toolbox).
    "compile-compcert": [
        ("/tmp/vfe_preclone/_generic/etc/apt/sources.list.d/ubuntu-http.sources", "/etc/apt/sources.list.d/ubuntu.sources"),
    ],
    "dna-assembly": [
        ("/tmp/vfe_preclone/_generic/etc/apt/sources.list.d/ubuntu-http.sources", "/etc/apt/sources.list.d/ubuntu.sources"),
    ],
    "mailman": [
        ("/tmp/vfe_preclone/_generic/etc/apt/sources.list.d/ubuntu-http.sources", "/etc/apt/sources.list.d/ubuntu.sources"),
    ],
    "financial-document-processor": [
        ("/tmp/vfe_preclone/_generic/etc/apt/sources.list.d/ubuntu-http.sources", "/etc/apt/sources.list.d/ubuntu.sources"),
    ],
    "mcmc-sampling-stan": [
        ("/tmp/vfe_preclone/_generic/etc/apt/sources.list.d/ubuntu-http.sources", "/etc/apt/sources.list.d/ubuntu.sources"),
    ],
    "merge-diff-arc-agi-task": [
        ("/tmp/vfe_preclone/_generic/etc/apt/sources.list.d/ubuntu-http.sources", "/etc/apt/sources.list.d/ubuntu.sources"),
    ],
}


def _seed_preclones(container: str, task: str) -> list[str]:
    """docker cp host preclones into a freshly started container. Best-effort.

    For protected destinations (/app, /tmp, ...) do NOT rm -rf; copy over.
    """
    seeded: list[str] = []
    for src, dst in PRECLONES.get(task, []):
        if not Path(src).exists():
            continue
        try:
            if dst not in _SEED_KEEP_DSTS:
                subprocess.run(
                    ["docker", "exec", container, "sh", "-c", f"rm -rf {dst}"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
            subprocess.run(
                ["docker", "exec", container, "sh", "-c", f"mkdir -p {dst}"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            r = subprocess.run(
                ["docker", "cp", f"{src}/.", f"{container}:{dst}"],
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            if r.returncode == 0:
                seeded.append(dst)
        except Exception:
            continue
    # Best-effort index/distro mirror configs if present in seed dirs.
    # mkdir -p the parent first: docker cp fails when it is absent in the image
    # (stripped images ship no /root/.local/bin, so uv/uvx seeding silently no-oped
    # and test.sh's astral.sh download became load-bearing).
    for host_cfg, container_cfg in GENERIC_CFGS + APT_CFGS.get(task, []):
        if not Path(host_cfg).exists():
            continue
        try:
            parent = str(Path(container_cfg).parent)
            subprocess.run(
                ["docker", "exec", container, "sh", "-c", f"mkdir -p '{parent}'"],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            subprocess.run(
                ["docker", "cp", host_cfg, f"{container}:{container_cfg}"],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except Exception:
            continue
    # Warm uv caches (~200MB) — skip when already present to avoid re-copying
    # on every start/reseed.
    for host_dir, container_dir in GENERIC_DIRS:
        if not Path(host_dir).exists():
            continue
        try:
            chk = subprocess.run(
                ["docker", "exec", container, "sh", "-c", f"[ -d '{container_dir}' ]"],
                capture_output=True,
                timeout=20,
                check=False,
            )
            if chk.returncode == 0:
                continue
            subprocess.run(
                ["docker", "exec", container, "sh", "-c", f"mkdir -p '{container_dir}'"],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            r = subprocess.run(
                ["docker", "cp", f"{host_dir}/.", f"{container}:{container_dir}"],
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            if r.returncode == 0:
                seeded.append(container_dir)
        except Exception:
            continue
    return seeded


def _seed_toolbox(container: str) -> None:
    """Ensure a minimal process/net toolbox at container start.

    Stripped task images ship without procps/iproute2/psmisc, so the agent's
    kill/port commands silently no-op and it burns steps discovering that.
    One-shot via /tmp/.vfe_toolbox; NOTE this augments the task container
    environment (uniform across tasks; disclose as an env change), not just the
    harness. Marker is only written on success so a failed apt can retry.
    """
    cmd = (
        "if [ ! -f /tmp/.vfe_toolbox ]; then "
        "if ! command -v pkill >/dev/null 2>&1 || [ ! -e /etc/ssl/certs/ca-certificates.crt ]; then "
        "export DEBIAN_FRONTEND=noninteractive; "
        "apt-get update -qq >/dev/null 2>&1 || true; "
        'for p in ca-certificates procps iproute2 psmisc; do apt-get install -y -qq "$p" >/dev/null 2>&1 || true; done; '
        "fi; "
        "touch /tmp/.vfe_toolbox; fi"
    )
    try:
        subprocess.run(
            ["docker", "exec", container, "sh", "-lc", cmd],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except Exception:
        pass


def _xfer(src: str, dst: str, path: str, dst_path: str | None = None, timeout: int = 600) -> bool:
    """Copy one dir path from container src into dst_path (same path by default).

    docker cp refuses container->container copies, so stage on the host (the
    daemon-side tar means no tar binary is needed inside stripped images).
    """
    target = dst_path or path
    name = path.strip("/").replace("/", "_") or "root"
    try:
        stage = Path(tempfile.mkdtemp(prefix=f"vfe-xfer-{name}-"))
    except Exception:
        return False
    try:
        r1 = subprocess.run(
            ["docker", "cp", f"{src}:{path}", str(stage)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if r1.returncode != 0:
            return False
        item = stage / Path(path).name
        if not item.is_dir():
            return False
        subprocess.run(
            ["docker", "exec", dst, "sh", "-c", f"mkdir -p '{target}'"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        r2 = subprocess.run(
            ["docker", "cp", f"{item}/.", f"{dst}:{target}/"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return r2.returncode == 0
    except Exception:
        return False
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _replace_app(dst: str, src: str) -> bool:
    """Give dst the src container's final agent /app, discarding image /app.

    The fresh verifier must judge only the agent's files: residue must never
    reach it — neither the in-loop pytest's files outside /app (headless /server)
    nor the in-container official run's own writes inside /app (headless
    /app/vim.txt). settle snapshots /app to /tmp/.app_final before the official
    run (that is the agent-final tree); runPytest's /tmp/.app_snap (pre-pytest,
    restored after every settle) is the fallback, then plain /app.
    """
    try:
        probe = subprocess.run(
            ["docker", "exec", src, "sh", "-c",
             'echo "F=$([ -d /tmp/.app_final ] && echo 1 || echo 0) S=$([ -d /tmp/.app_snap ] && echo 1 || echo 0)"'],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        out = probe.stdout or ""
        source = "/app"
        if "F=1" in out:
            source = "/tmp/.app_final"
        elif "S=1" in out:
            source = "/tmp/.app_snap"
        subprocess.run(
            ["docker", "exec", dst, "sh", "-c", "rm -rf /app && mkdir -p /app"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        return _xfer(src, dst, source, "/app")
    except Exception:
        return False


def _seed_verifier_caches(dst: str, src: str) -> list[str]:
    """Copy package caches (pip/uv/uv-python installs) from the agent container.

    Caches are environment, not agent state: they let the fresh container's
    test.sh provision pytest/uv without depending on the network.
    """
    copied: list[str] = []
    for path in ("/root/.cache/pip", "/root/.cache/uv", "/root/.local/share/uv", "/root/.config/uv"):
        try:
            chk = subprocess.run(
                ["docker", "exec", src, "sh", "-c", f"[ -d '{path}' ]"],
                capture_output=True,
                timeout=20,
                check=False,
            )
            if chk.returncode != 0:
                continue
        except Exception:
            continue
        if _xfer(src, dst, path):
            copied.append(path)
    return copied


def _drop_image(tag: str) -> None:
    """Best-effort removal of a committed verify image (disk hygiene)."""
    try:
        subprocess.run(
            ["docker", "rmi", "-f", tag],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except Exception:
        pass


def reply(req_id, **payload) -> None:
    obj = {"id": req_id, **payload}
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with _out_lock:
        sys.stdout.write(line)
        sys.stdout.flush()


def _fake_env(container: str, task: str | None) -> DockerEnv:
    env = DockerEnv(task or container, rep=0)
    env._container_name = container
    env._started = True
    return env


def handle(req: dict) -> None:
    rid = req.get("id")
    op = req.get("op")
    if op == "start":
        task = req["task"]
        rep = int(req.get("rep") or 0)
        # Stale verify image from a previous run of the same task+rep (its
        # prepare() commit was never consumed): drop it, the new run will
        # commit fresh. Bound the leak to at most one tag per task+rep.
        vname = f"tb2verify-{task}".replace("_", "-") + (f"-r{rep}" if rep else "")
        with _env_lock:
            stale = _verify_images.pop(vname, None)
        if stale:
            _drop_image(stale)
        env = DockerEnv(task, rep=rep)
        asyncio.run(env.start())
        with _env_lock:
            _envs[env._container_name] = env
            _env_meta[env._container_name] = (task, rep)
        seeded = _seed_preclones(env._container_name, task)
        _seed_toolbox(env._container_name)
        reply(
            rid,
            ok=True,
            container=env._container_name,
            task=env.task_name,
            **({"preseeded": seeded} if seeded else {}),
        )
        return
    if op == "commit_verify":
        # Pre-official snapshot of the agent environment as the verify image.
        # MUST run before the official suite: the official run writes real-fs
        # state (/server, /app/vim.txt) that must not leak into the fresh
        # re-verify (its own mkdir /server is non-idempotent), while the agent's
        # environment (system-python packages like pexpect — test.sh does not
        # install them) must survive there. Consumed by start_verify.
        task = req["task"]
        rep = int(req.get("rep") or 0)
        src = req.get("src") or ""
        if not src:
            reply(rid, ok=False, error="commit_verify requires src (agent container)")
            return
        vname = f"tb2verify-{task}".replace("_", "-") + (f"-r{rep}" if rep else "")
        tag = f"vfe-verify-{task}-{rep}".replace("_", "-").lower()
        try:
            r = subprocess.run(
                ["docker", "commit", "--pause=false", src, tag],
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
            )
        except Exception as exc:
            reply(rid, ok=False, error=str(exc))
            return
        if r.returncode != 0:
            reply(rid, ok=False, error=(r.stderr or "docker commit failed")[:300])
            return
        with _env_lock:
            _verify_images[vname] = tag
        reply(rid, ok=True, tag=tag)
        return
    if op == "start_verify":
        # Fresh verifier container. Preferred: commit the agent container as the
        # verify image — the official run in it installs system packages/site-
        # packages that a pristine image lacks (false fresh failures: e.g.
        # headless's pexpect lives in system python). Environment carries over,
        # test state does not: in-loop pytest runs under isolate=True (overlay,
        # zero residue) and /app is still replaced with the agent-final tree, so
        # only environment survives, not test writes. Fallback when commit fails:
        # pristine image + seeded caches (previous behavior).
        task = req["task"]
        rep = int(req.get("rep") or 0)
        src = req.get("src") or ""
        if not src:
            reply(rid, ok=False, error="start_verify requires src (agent container)")
            return
        env = DockerEnv(task, rep=rep)
        env._container_name = f"tb2verify-{env.task_name}".replace("_", "-") + (f"-r{rep}" if rep else "")
        # Prefer the pre-official commit (commit_verify): it is free of the
        # official run's own test residue. Fall back to committing the src now
        # (may carry residue, but keeps the agent environment) and finally to the
        # pristine image + seeded caches.
        with _env_lock:
            pre_tag = _verify_images.get(env._container_name)
        commit_tag = pre_tag or f"vfe-verify-{env.task_name}-{rep}".replace("_", "-").lower()
        committed = bool(pre_tag)
        if not committed:
            try:
                r = subprocess.run(
                    ["docker", "commit", "--pause=false", src, commit_tag],
                    capture_output=True,
                    text=True,
                    timeout=600,
                    check=False,
                )
                committed = r.returncode == 0
            except Exception:
                committed = False
        if committed:
            env._image = commit_tag
            with _env_lock:
                _verify_images[env._container_name] = commit_tag
        try:
            asyncio.run(env.start())
        except Exception as exc:
            with _env_lock:
                _verify_images.pop(env._container_name, None)
            if committed:
                _drop_image(commit_tag)
            reply(rid, ok=False, error=f"start_verify failed: {exc}")
            return
        container = env._container_name
        with _env_lock:
            _envs[container] = env
            _env_meta[container] = (task, rep)
        seeded: list[str] = []
        caches: list[str] = []
        if committed:
            # Committed fs already has seeds/caches/toolbox from the src container.
            pass
        else:
            seeded = _seed_preclones(container, task)
            _seed_toolbox(container)
        if not _replace_app(container, src):
            with _env_lock:
                _envs.pop(container, None)
                _env_meta.pop(container, None)
                tag = _verify_images.pop(container, None)
            asyncio.run(env.stop())
            if tag:
                _drop_image(tag)
            reply(rid, ok=False, error=f"copy /app from {src} failed")
            return
        if not committed:
            caches = _seed_verifier_caches(container, src)
        reply(
            rid,
            ok=True,
            container=container,
            task=task,
            **({"preseeded": seeded} if seeded else {}),
            **({"caches": caches} if caches else {}),
            **({"committed": True} if committed else {}),
        )
        return
    if op == "exec":
        container = req["container"]
        with _env_lock:
            env = _envs.get(container) or _fake_env(container, req.get("task"))
            _envs[container] = env
        timeout = int(req.get("timeout") or 120)
        result = asyncio.run(env.exec(req["cmd"], timeout_sec=timeout))
        reply(
            rid,
            ok=True,
            output=result.get("output") or "",
            exit_code=int(result.get("exit_code", -1)),
            **({"infra": True} if result.get("infra") else {}),
        )
        return
    if op == "exec_tests":
        # 私有 mount namespace 注入 tests/ 后执行判分/反馈命令（见 DockerEnv.exec_tests）。
        # isolate=false: 官方判分跑真实 rootfs（与官方 TB 同语义）；
        # 默认 true: in-loop 反馈在 overlay 里跑，零残留。
        container = req["container"]
        with _env_lock:
            env = _envs.get(container) or _fake_env(container, req.get("task"))
            _envs[container] = env
        timeout = int(req.get("timeout") or 120)
        isolate = bool(req.get("isolate", True))
        result = asyncio.run(env.exec_tests(req["cmd"], timeout_sec=timeout, isolate=isolate))
        reply(
            rid,
            ok=True,
            output=result.get("output") or "",
            exit_code=int(result.get("exit_code", -1)),
            **({"infra": True} if result.get("infra") else {}),
        )
        return
    if op == "restart":
        container = req["container"]
        task = req.get("task")
        rep = int(req.get("rep") or 0)
        if not task:
            with _env_lock:
                meta = _env_meta.get(container)
            if meta:
                task, rep = meta
        if not task:
            reply(rid, ok=False, error="restart requires task (for recreate)")
            return
        with _env_lock:
            env = _envs.get(container)
            if env is None or env.task_name != task:
                env = DockerEnv(task, rep=rep)
                env._container_name = container
            _envs[container] = env
            _env_meta[container] = (task, rep)
        mode = asyncio.run(env.restart())
        if mode == "recreated":
            _seed_preclones(container, task)
            _seed_toolbox(container)
        reply(rid, ok=True, container=env._container_name, mode=mode, task=task)
        return
    if op == "stop":
        container = req["container"]
        with _env_lock:
            env = _envs.pop(container, None)
            meta = _env_meta.pop(container, None)
            tag = _verify_images.pop(container, None)
            if tag is None and meta is not None:
                # Main container stopping: drop the (already consumed or never
                # consumed) pre-official verify image of the same task+rep.
                task_m, rep_m = meta
                vname = f"tb2verify-{task_m}".replace("_", "-") + (f"-r{rep_m}" if rep_m else "")
                tag = _verify_images.pop(vname, None)
        if env is None:
            env = _fake_env(container, req.get("task"))
        asyncio.run(env.stop())
        if tag:
            _drop_image(tag)
        reply(rid, ok=True)
        return
    if op == "ping":
        reply(rid, ok=True)
        return
    reply(rid, ok=False, error=f"unknown op: {op}")


def _safe_handle(req: dict) -> None:
    try:
        handle(req)
    except Exception as exc:
        reply(req.get("id"), ok=False, error=str(exc), traceback=traceback.format_exc())


def main() -> None:
    # Concurrent workers: a long exec must not block subsequent requests
    # (serialized handling caused client-side sidecar timeouts → false infra).
    pool = ThreadPoolExecutor(max_workers=8)
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except Exception as exc:
                reply(None, ok=False, error=f"bad json: {exc}")
                continue
            pool.submit(_safe_handle, req)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
