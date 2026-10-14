"""vfe_exp3.DockerEnv — 为 run_tb21_v4flash_harness.py 提供终端环境执行接口。

对 Terminal-Bench 2.1 任务库的每个任务，用其 task.toml 里 [environment].docker_image
启动一个独立容器，工作目录 /app（Dockerfile WORKDIR）。
exec() 在容器内执行 shell 命令并返回 {"output", "exit_code"}。
exec_tests() 在一个私有 mount namespace 的 tmpfs 里注入 tests/ 并执行命令：agent
阶段容器里没有 /tests，判分/反馈时测试也不会出现在 agent 可见的文件系统上。
"""
from __future__ import annotations

import asyncio
import io
import os
import shlex
import subprocess
import tarfile
import time
import tomllib
from pathlib import Path


def _run_sync(cmd: list[str], timeout_sec: int = 120) -> subprocess.CompletedProcess:
    """在事件循环中安全地跑一个阻塞 subprocess（用 to_thread）。"""
    async def _inner():
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: subprocess.run(cmd, capture_output=True, text=True,
                                   errors="replace", timeout=timeout_sec),
        )
    return asyncio.run(_inner())


async def _run_async(cmd: list[str], timeout_sec: int = 120) -> subprocess.CompletedProcess:
    """异步跑 subprocess，带超时。

    errors="replace"（关键，泛化）：命令输出常含二进制（ELF/压缩数据/xxd），
    严格 UTF-8 解码会抛 UnicodeDecodeError 并**丢掉整段输出**——等于 agent 什么都看不到。
    用 replace 把非法字节替换为 U+FFFD，保证观测永远返回。"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: subprocess.run(cmd, capture_output=True, text=True,
                               errors="replace", timeout=timeout_sec),
    )


async def _run_bytes_input(cmd: list[str], timeout_sec: int, data: bytes) -> subprocess.CompletedProcess:
    """异步跑 subprocess 并把原始字节喂给 stdin（tar 流用）。

    text=False：tar 是二进制，不能用 text=True 编码；输出在调用方按 replace 解码。"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: subprocess.run(cmd, input=data, capture_output=True, timeout=timeout_sec),
    )


# ── 宿主机代理探测 ──────────────────────────────────────────────
# 容器里 apt/pip 走 NO_PROXY 直连（archive.ubuntu.com / tuna 实测可用）；只有
# github.com 直连不通时才注入宿主机代理。旧代码写死 172.17.0.1:7899，而 7899
# 中继当前没有任何进程监听 → 容器内所有 apt 请求 Connection refused → pytest
# 供给失败 → 误判 [ENV-IRRECOVERABLE]。改成启动前探测，探测失败就不注入代理。
_PROXY_CANDIDATES = ("http://172.17.0.1:7897", "http://172.17.0.1:7899", "http://172.17.0.1:7890")
_PROXY_PROBE_URL = "https://github.com/"
_PROXY_TTL_SEC = 300
_proxy_cache: dict[str, object] = {"proxy": None, "ts": 0.0}

# 镜像站/apt 源强制直连：代理只用于 github，避免 http 明文被代理 502 拖垮 apt。
_NO_PROXY = (
    "localhost,127.0.0.1,pypi.tuna.tsinghua.edu.cn,hf-mirror.com,"
    "registry.npmmirror.com,.tsinghua.edu.cn,.hf-mirror.com,.npmmirror.com,"
    "archive.ubuntu.com,security.ubuntu.com,ports.ubuntu.com,172.17.0.0/16"
)


async def _probe_url(url: str, proxy: str | None, timeout_sec: int = 6) -> bool:
    cmd = ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "-m", str(timeout_sec)]
    cmd += ["-x", proxy] if proxy else ["--noproxy", "*"]
    cmd.append(url)
    try:
        r = await _run_async(cmd, timeout_sec=timeout_sec + 5)
        code = r.stdout.strip()
        return r.returncode == 0 and code.isdigit() and 200 <= int(code) < 400
    except Exception:
        return False


async def pick_proxy() -> str | None:
    """返回应注入容器的代理 URL；直连 github 可用或无可用代理时返回 None。"""
    now = time.time()
    if now - float(_proxy_cache["ts"]) < _PROXY_TTL_SEC:
        cached = _proxy_cache["proxy"]
        return cached if isinstance(cached, str) else None
    proxy: str | None = None
    if not await _probe_url(_PROXY_PROBE_URL, None):
        for candidate in _PROXY_CANDIDATES:
            if await _probe_url(_PROXY_PROBE_URL, candidate, timeout_sec=5):
                proxy = candidate
                break
    _proxy_cache["proxy"] = proxy
    _proxy_cache["ts"] = now
    print(f"[proxy] chosen={proxy or 'direct'}", flush=True)
    return proxy


class DockerEnv:
    """单任务终端执行环境（Terminal-Bench 2.1 容器）。"""

    def __init__(self, path: str, rep: int = 0):
        # path 形如任务目录名或路径；任务根目录由 VFE_TASKS_DIR 指定
        p = Path(path)
        self.task_name = p.name                      # 任务目录名，如 build-cython-ext
        self.image_tag = f"tb2-{self.task_name}"     # 代码会覆盖（容器名/镜像tag）
        base = p
        # 解析绝对任务目录：VFE_TASKS_DIR 环境变量 > 相对当前路径
        tasks_root = os.environ.get("VFE_TASKS_DIR", "").strip()
        if tasks_root:
            self._task_dir = Path(tasks_root) / self.task_name
        else:
            self._task_dir = p
        if not self._task_dir.exists():
            # 尝试相对当前路径
            self._task_dir = p if p.exists() else self._task_dir
        self._image = self._read_docker_image()
        self._container_name = f"tb2run-{self.task_name}".replace("_", "-")
        if rep:
            self._container_name += f"-r{rep}"
        self._started = False

    # ── 配置解析 ──────────────────────────────────────────────
    def _read_docker_image(self) -> str:
        toml = self._task_dir / "task.toml"
        if toml.exists():
            try:
                with open(toml, "rb") as f:
                    cfg = tomllib.load(f)
                img = (cfg.get("environment") or {}).get("docker_image")
                if img:
                    return img
            except Exception:
                pass
        # 兜底：找 Dockerfile？否则用一个默认占位（若无镜像，启动会失败）
        return f"alexgshaw/{self.task_name}:20251031"

    # ── 生命周期 ──────────────────────────────────────────────
    async def _docker_retry(self, cmd: list[str], timeout_sec: int,
                            retries: int = 3) -> subprocess.CompletedProcess | None:
        """跑 docker 命令，超时/失败自动重试；retries 次仍失败返回 None 不抛异常。

        修复（2026-09-11）：docker daemon 受 /data 磁盘写满拖累时会整条命令 hang，
        start() 里旧代码直接抛 TimeoutExpired 让任务 crash。这里统一兜底，让调用方决定。
        """
        for i in range(retries):
            try:
                r = await _run_async(cmd, timeout_sec=timeout_sec)
                if r.returncode == 0:
                    return r
            except subprocess.TimeoutExpired:
                pass
            except Exception:
                pass
            if i < retries - 1:
                await asyncio.sleep(3 * (i + 1))
        print(f"[docker-retry] {cmd[1] if len(cmd) > 1 else cmd} 失败 {retries} 次，放弃本命令", flush=True)
        return None

    async def start(self):
        """拉镜像（如需）并启动容器，/tests 只读挂载。"""
        # 1) 拉取（带 tag 别名便于删除历史镜像）；本地已有则跳过，避免网络抖动卡 600s
        base_image = self._image
        insp_r = await self._docker_retry(["docker", "image", "inspect", base_image], 30, 2)
        if insp_r is None or insp_r.returncode != 0:
            # 拉不到不 crash：run 环节会再试/落到本地构建兜底
            await self._docker_retry(["docker", "pull", base_image],
                                     int(os_getattr("TB2_PULL_TIMEOUT", 600)), 2)
        # 2) 删除同名容器（残留清理，只针对本任务命名空间；删不掉不阻塞启动）
        await self._docker_retry(["docker", "rm", "-f", self._container_name], 30, 3)
        # 3) 启动容器：sleep 保持运行。测试不再常驻挂载——agent 阶段容器里根本没有
        #    /tests，答案键只在 exec_tests 的私有 mount namespace(tmpfs) 里短暂存在。
        #    --cap-add SYS_ADMIN + seccomp/apparmor=unconfined 是创建该 mount namespace
        #    的前提（Docker 默认 seccomp/apparmor 会拒绝 unshare/mount）。
        # HF_ENDPOINT：huggingface.co 在本机被 DNS 污染（解析到 108.160.166.57 →
        #    EHOSTUNREACH），镜像源可达；官方 test.sh/solution 直接 load_dataset 需要它。
        # PYTHONPATH=/opt/vfe_seed：让 seed 里的 sitecustomize.py 生效（hf-mirror 的
        #    Link 分页头指回 huggingface.co，需改写为 HF_ENDPOINT）。
        # github.com 直连在本网络下不稳定（GnuTLS 断连/超时轮番出现），官方
        # verifier 里 git clone / urlretrieve github 的任务会被误判 0。代理启动前
        # 探测（pick_proxy），7899 中继已下线时不再注入死代理；镜像站/apt 源走
        # NO_PROXY 直连。
        proxy = await pick_proxy()
        proxy_env: list[str] = []
        if proxy:
            proxy_env = [
                "-e", f"HTTP_PROXY={proxy}",
                "-e", f"HTTPS_PROXY={proxy}",
                "-e", f"http_proxy={proxy}",
                "-e", f"https_proxy={proxy}",
            ]
        run_cmd = ["docker", "run", "-d", "--name", self._container_name,
                   "--cap-add", "SYS_ADMIN",
                   "--security-opt", "seccomp=unconfined",
                   "--security-opt", "apparmor=unconfined",
                   "-e", "HF_ENDPOINT=https://hf-mirror.com",
                   "-e", "HF_HUB_OFFLINE=0",
                   "-e", "HF_DATASETS_OFFLINE=0",
                   "-e", "PYTHONPATH=/opt/vfe_seed",
                   "-e", "UV_PYTHON_INSTALL_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone",
                   # uv/uvx 不读 pip.conf；官方 test.sh 里的 uvx 走 pypi.org 直连，
                   # 本机网络下又慢又 flaky（official verifier 判成 undefined 的根因）。
                   # 清华镜像 + 种子热缓存（/root/.cache/uv、/root/.local/share/uv，
                   # 由 _seed_preclones 拷入）把 uvx 拉包变成秒级。
                   "-e", "UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple",
                   "-e", "UV_HTTP_TIMEOUT=60",
                   *proxy_env,
                   "-e", f"NO_PROXY={_NO_PROXY}",
                   "-e", f"no_proxy={_NO_PROXY}",
                   base_image,
                   "sh", "-c", "sleep infinity"]
        run = await self._docker_retry(run_cmd, timeout_sec=120, retries=2)
        if run is None or run.returncode != 0:
            run_stderr = (run.stderr if run is not None else "") or ""
            # 预构建镜像可能不含本任务最新代码，尝试用环境 Dockerfile 本地构建
            dockerfile = self._task_dir / "environment" / "Dockerfile"
            if dockerfile.exists():
                build = await _run_async(
                    ["docker", "build", "-t", base_image,
                     "-f", str(dockerfile.resolve()),
                     str((self._task_dir / "environment").resolve())],
                    timeout_sec=600,
                )
                if build.returncode != 0:
                    raise RuntimeError(f"容器启动失败且本地构建失败: {run_stderr} | {build.stderr}")
                run = await self._docker_retry(run_cmd, timeout_sec=120, retries=2)
                if run is None or run.returncode != 0:
                    raise RuntimeError(f"本地构建后启动仍失败: {run_stderr}")
            else:
                raise RuntimeError(f"容器启动失败: {run_stderr}")
        # 4) 等容器就绪
        for _ in range(30):
            insp = await _run_async(
                ["docker", "inspect", "-f", "{{.State.Running}}", self._container_name],
                timeout_sec=15)
            if insp.returncode == 0 and insp.stdout.strip() == "true":
                self._started = True
                return
            await asyncio.sleep(1)
        raise RuntimeError(f"容器未在预期时间内就绪: {self._container_name}")

    async def _is_running(self) -> bool:
        try:
            insp = await _run_async(
                ["docker", "inspect", "-f", "{{.State.Running}}", self._container_name],
                timeout_sec=10)
            return insp.returncode == 0 and insp.stdout.strip() == "true"
        except Exception:
            return False

    async def exec(self, cmd: str, timeout_sec: int = 120):
        """在容器内执行 shell 命令，返回 {"output", "exit_code", "infra"?: true}。

        infra=true 表示基础设施故障（容器未运行/不存在），不是任务域命令失败。
        """
        # 用 sh -lc 执行，避免外层 shell 转义破坏；命令经 shlex 安全包裹
        inner = cmd
        cmd_line = ["docker", "exec", "-i", self._container_name,
                    "sh", "-lc", inner]
        try:
            proc = await _run_async(cmd_line, timeout_sec=timeout_sec)
        except subprocess.TimeoutExpired:
            return {"output": f"[TIMEOUT after {timeout_sec}s]", "exit_code": -1}
        except Exception as e:
            return {"output": f"[EXEC_ERROR] {e}", "exit_code": -2}
        out = (proc.stdout or "") + (proc.stderr or "")
        result: dict = {"output": out, "exit_code": proc.returncode}
        if proc.returncode != 0:
            if not await self._is_running() or _looks_infra(out):
                result["infra"] = True
                self._started = False
        return result

    def _tests_tar(self, tests_dir: Path) -> bytes:
        """把任务 tests/ 目录打成内存 tar（agent 阶段不落盘）。"""
        buf = io.BytesIO()
        td = Path(tests_dir)
        with tarfile.open(fileobj=buf, mode="w") as tf:
            if td.is_dir():
                for p in sorted(td.rglob("*")):
                    try:
                        tf.add(str(p), arcname=str(p.relative_to(td)), recursive=False)
                    except Exception:
                        continue
        return buf.getvalue()

    async def exec_tests(self, cmd: str, timeout_sec: int = 120,
                         tests_dir: Path | None = None, isolate: bool = True):
        """在私有 mount namespace 里执行 cmd，并把 tests/ 以 tmpfs 注入 /tests。

        测试文件只在 unshare(CLONE_NEWNS) 出来的 tmpfs 里存在，容器原 mount namespace
        里的 agent 进程看不到；/logs 同样用 tmpfs 遮蔽（防预写 reward.txt），
        /etc/ld.so.preload 用 /dev/null 覆盖（防预植 so）。cmd 由调用方保证是测试/判分命令。

        isolate=True（in-loop pytest / agent run_test）：在 overlay
        (lowerdir=/ 只读视图 + tmpfs upper) 里执行，测试对文件系统的任何写入
        （headless 的 mkdir /server、official 在 /app 留 vim.txt、pip 安装）
        都落在用完即弃的 upper 层，容器真实 rootfs 零残留 —— 否则同容器里
        下一轮 pytest / 官方判分会被上一轮的残留打挂（假失败）。
        isolate=False（官方判分 test.sh）：在真实 rootfs 上跑一次，语义与官方
        TB 一致（每次 run 都是全新容器，不存在跨 run 残留）。
        """
        if tests_dir is None:
            tests_dir = self._task_dir / "tests"
        data = self._tests_tar(tests_dir)
        if isolate:
            inner = "cd /app 2>/dev/null || true; " + cmd
            ns_script = (
                "OV=/var/tmp/.vfeov.$$; "
                # 上一轮超时残留的 overlay 目录（>70 分钟）顺手清掉，防积累。
                "find /var/tmp -maxdepth 1 -name '.vfeov.*' -mmin +70 -exec rm -rf {} + 2>/dev/null || true; "
                "mkdir -p \"$OV/t\" \"$OV/m\" || { echo VFE_OVL_FAIL; exit 82; }; "
                "mount -t tmpfs -o size=8192m tmpfs \"$OV/t\" 2>/dev/null "
                "|| { echo VFE_OVL_TMPFS_FAIL; exit 83; }; "
                "mkdir -p \"$OV/t/u\" \"$OV/t/w\"; "
                "mount -t overlay vfeov -o lowerdir=/,upperdir=$OV/t/u,workdir=$OV/t/w \"$OV/m\" 2>/dev/null "
                "|| { echo VFE_OVL_FAIL; exit 84; }; "
                # docker 在 /etc 下的运行时 bind mount 不进 overlay lower，必须补挂，
                # 否则 chroot 内解析不了 localhost/外网（resolv.conf/hosts/hostname）。
                "touch \"$OV/m/etc/resolv.conf\" \"$OV/m/etc/hosts\" \"$OV/m/etc/hostname\" 2>/dev/null || true; "
                "for f in resolv.conf hosts hostname; do "
                "mount --bind /etc/$f \"$OV/m/etc/$f\" 2>/dev/null || true; done; "
                "mount --rbind /proc \"$OV/m/proc\" 2>/dev/null || true; "
                "mount --rbind /dev \"$OV/m/dev\" 2>/dev/null || true; "
                "mount --rbind /sys \"$OV/m/sys\" 2>/dev/null || true; "
                # 包缓存挂回真实目录：下载缓存跨次保留，安装本身（写 site-packages）
                # 仍落在 upper 被丢弃。
                "mkdir -p \"$OV/m/root/.cache\" \"$OV/m/var/lib/apt\" \"$OV/m/var/cache/apt\"; "
                "mount --rbind /root/.cache \"$OV/m/root/.cache\" 2>/dev/null || true; "
                "mount --rbind /var/lib/apt \"$OV/m/var/lib/apt\" 2>/dev/null || true; "
                "mount --rbind /var/cache/apt \"$OV/m/var/cache/apt\" 2>/dev/null || true; "
                "mkdir -p \"$OV/m/tests\"; "
                "mount -t tmpfs tmpfs \"$OV/m/tests\" 2>/dev/null || { echo VFE_MOUNT_TESTS_FAIL; exit 80; }; "
                "mkdir -p \"$OV/m/logs\"; "
                "mount -t tmpfs tmpfs \"$OV/m/logs\" 2>/dev/null || true; "
                "mkdir -p \"$OV/m/logs/verifier\"; "
                "touch \"$OV/m/etc/ld.so.preload\" 2>/dev/null || true; "
                "mount --bind /dev/null \"$OV/m/etc/ld.so.preload\" 2>/dev/null || true; "
                "tar -x -C \"$OV/m/tests\" -f - >/dev/null 2>&1; "
                "unset LD_PRELOAD; "
                "chroot \"$OV/m\" /bin/sh -c " + shlex.quote(inner) + "; "
                # 懒卸载后移除：遗留进程若还持有 namespace，写入会失败而非阻塞。
                "umount -l \"$OV/m\" 2>/dev/null; "
                "umount -l \"$OV/t\" 2>/dev/null; "
                "rm -rf \"$OV\" 2>/dev/null; true"
            )
        else:
            ns_script = (
                "mkdir -p /tests; "
                "mount -t tmpfs tmpfs /tests 2>/dev/null || { echo VFE_MOUNT_TESTS_FAIL; exit 80; }; "
                "mkdir -p /logs/verifier; mount -t tmpfs tmpfs /logs 2>/dev/null || true; mkdir -p /logs/verifier; "
                "touch /etc/ld.so.preload 2>/dev/null || true; "
                "mount --bind /dev/null /etc/ld.so.preload 2>/dev/null || true; "
                "tar -x -C /tests -f - >/dev/null 2>&1; "
                "unset LD_PRELOAD; "
                "cd /app 2>/dev/null || true; " + cmd + "; "
                # 退出前卸载并移除挂载点，避免在共享 rootfs 上留下一个空的 /tests。
                "umount /tests 2>/dev/null; rmdir /tests 2>/dev/null; "
                "umount /logs 2>/dev/null; true"
            )
        wrapper = "unshare -m --propagation unchanged sh -c " + shlex.quote(ns_script)
        cmd_line = ["docker", "exec", "-i", self._container_name, "sh", "-lc", wrapper]
        try:
            proc = await _run_bytes_input(cmd_line, timeout_sec, data)
        except subprocess.TimeoutExpired:
            return {"output": f"[TIMEOUT after {timeout_sec}s]", "exit_code": -1, "infra": True}
        except Exception as e:
            return {"output": f"[EXEC_ERROR] {e}", "exit_code": -2, "infra": True}
        out = (proc.stdout or b"").decode("utf-8", "replace") + (proc.stderr or b"").decode("utf-8", "replace")
        result: dict = {"output": out, "exit_code": proc.returncode}
        # 只有真正失败（非 0 退出）才谈 infra：ns_script 成功路径以 `; true` 结尾
        # 恒为 0，而 harness 自己的失败命令行（含 VFE_* 标记）会被 /app 下影子
        # 标准库的文件回显进 stdout —— rc==0 时的标记只是被打印出来的文本。
        if proc.returncode != 0 and (
            "VFE_MOUNT_TESTS_FAIL" in out or _looks_tests_infra(out)
        ):
            result["infra"] = True
        elif proc.returncode != 0:
            if not await self._is_running() or _looks_infra(out):
                result["infra"] = True
                self._started = False
        return result

    async def restart(self) -> str:
        """Recover container: docker start if stopped; if running but exec/workdir broken, recreate.

        Returns running|started|recreated.
        """
        insp = await _run_async(
            ["docker", "inspect", "-f", "{{.State.Status}}", self._container_name],
            timeout_sec=15)
        if insp.returncode == 0 and insp.stdout.strip() == "running":
            # Probe default Workdir (/app). OCI chdir failure → unusable despite "running".
            probe = await _run_async(
                ["docker", "exec", self._container_name, "sh", "-lc", "true"],
                timeout_sec=15)
            probe_out = (probe.stdout or "") + (probe.stderr or "")
            if probe.returncode == 0 and not _looks_infra(probe_out):
                self._started = True
                return "running"
            # Workdir missing or exec broken: repair /app from snapshots when possible.
            repair = await _run_async(
                ["docker", "exec", "-w", "/", self._container_name, "sh", "-lc",
                 "if [ -d /app ]; then exit 0; fi; "
                 "if [ -d /tmp/.app_best ]; then rm -rf /app && cp -a /tmp/.app_best /app && exit 0; fi; "
                 "if [ -d /tmp/.app_snap ]; then rm -rf /app && cp -a /tmp/.app_snap /app && exit 0; fi; "
                 "exit 1"],
                timeout_sec=120)
            if repair.returncode == 0:
                probe2 = await _run_async(
                    ["docker", "exec", self._container_name, "sh", "-lc", "true"],
                    timeout_sec=15)
                if probe2.returncode == 0:
                    self._started = True
                    return "started"
            # Fall through to full recreate (image restores /app).
            await self.start()
            return "recreated"
        if insp.returncode == 0:
            r = await self._docker_retry(["docker", "start", self._container_name], 60, 2)
            if r is not None and r.returncode == 0:
                for _ in range(30):
                    i2 = await _run_async(
                        ["docker", "inspect", "-f", "{{.State.Running}}", self._container_name],
                        timeout_sec=15)
                    if i2.returncode == 0 and i2.stdout.strip() == "true":
                        self._started = True
                        return "started"
                    await asyncio.sleep(1)
        await self.start()
        return "recreated"

    async def stop(self):
        """停止并删除本任务容器（不动用户其他镜像/容器）。"""
        try:
            await _run_async(["docker", "rm", "-f", self._container_name], timeout_sec=60)
        except Exception:
            pass
        self._started = False


def _looks_infra(out: str) -> bool:
    """Docker client/daemon/OCI errors → infrastructure fault.

    Narrow on purpose: task output may contain phrases like "is not running"
    (test assertions) without the container being dead.
    """
    markers = (
        "OCI runtime exec failed",
        "Error response from daemon",
        "Cannot connect to the Docker daemon",
        "No such container:",
    )
    return any(m in out for m in markers)


def _looks_tests_infra(out: str) -> bool:
    """exec_tests 的命名空间注入失败（unshare/mount/overlay 被拒或缺失）→ 基础设施故障。

    绝不能把注入失败降级成“测试不过”，否则要么假失败、要么绕过私有命名空间。
    """
    return (
        "VFE_MOUNT_TESTS_FAIL" in out
        or "VFE_OVL_FAIL" in out
        or "VFE_OVL_TMPFS_FAIL" in out
        or "unshare: not found" in out
        or "unshare failed" in out
        or "unshare: cannot" in out
    )


def os_getattr(name: str, default):
    import os
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default
