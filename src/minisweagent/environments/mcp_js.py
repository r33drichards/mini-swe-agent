"""MCP-JS (mcp-v8) environment for mini-SWE-agent.

Executes JavaScript code via the mcp-js HTTP API instead of shell commands.
The mcp-js server provides a V8 JavaScript runtime with optional filesystem
access (policy-gated), making it suitable for code editing tasks.

Two environment classes are provided:

- ``McpJsEnvironment``: Starts mcp-js locally and connects via HTTP.
- ``McpJsDockerEnvironment``: Starts mcp-js inside a Docker container
  alongside the SWE-bench testbed, enabling file manipulation via JS
  filesystem APIs and git diff extraction for submission.
"""

import json
import logging
import os
import shlex
import socket
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

import requests


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_server(url: str, timeout: float = 30.0, interval: float = 0.3) -> None:
    """Poll a URL until it responds or timeout is reached."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            requests.get(url, timeout=2)
            return
        except requests.ConnectionError:
            time.sleep(interval)
    msg = f"mcp-js server at {url} did not become ready within {timeout}s"
    raise TimeoutError(msg)


def _execute_via_http(
    base_url: str,
    code: str,
    *,
    timeout: int = 60,
    execution_timeout_secs: int | None = None,
) -> dict[str, Any]:
    """Submit JS code to the mcp-js HTTP API and wait for the result.

    Returns a dict with ``output`` (str) and ``returncode`` (int).
    """
    payload: dict[str, Any] = {"code": code}
    if execution_timeout_secs is not None:
        payload["execution_timeout_secs"] = execution_timeout_secs

    # 1. Submit execution
    resp = requests.post(f"{base_url}/api/exec", json=payload, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if "error" in data and "execution_id" not in data:
        return {"output": data["error"], "returncode": 1}

    exec_id = data["execution_id"]

    # 2. Poll until terminal state
    poll_interval = 0.1
    deadline = time.monotonic() + timeout
    status = ""
    error_msg = None

    while time.monotonic() < deadline:
        time.sleep(poll_interval)
        poll_interval = min(poll_interval * 1.5, 2.0)  # exponential backoff, capped

        try:
            r = requests.get(f"{base_url}/api/executions/{exec_id}", timeout=5)
            if r.status_code != 200:
                continue
            info = r.json()
            status = info.get("status", "")
            if status in ("completed", "failed", "timed_out", "cancelled"):
                error_msg = info.get("error")
                break
        except requests.RequestException:
            continue

    if not status:
        return {"output": "Execution did not complete within polling timeout", "returncode": 1}

    # 3. Collect console output
    try:
        r = requests.get(
            f"{base_url}/api/executions/{exec_id}/output",
            params={"line_limit": 1000000},
            timeout=10,
        )
        output = r.json().get("data", "") if r.status_code == 200 else ""
    except requests.RequestException:
        output = ""

    if status == "completed":
        return {"output": output, "returncode": 0}

    error_text = error_msg or f"Execution {status}"
    if output:
        return {"output": f"{output}\n{error_text}", "returncode": 1}
    return {"output": error_text, "returncode": 1}


# ── Local environment ────────────────────────────────────────────────────


@dataclass
class McpJsEnvironmentConfig:
    """Configuration for a local mcp-js environment."""

    mcp_js_binary: str = os.getenv("MCP_JS_BINARY", "server")
    """Path to the mcp-js (mcp-v8) binary."""
    port: int = 0
    """HTTP port for mcp-js. 0 = auto-assign."""
    timeout: int = 60
    """Default execution timeout in seconds."""
    execution_timeout_secs: int = 120
    """V8 execution timeout passed to mcp-js per request."""
    policies_json: str = ""
    """Inline JSON or path to policies config for filesystem/fetch access."""
    extra_args: list[str] = field(default_factory=list)
    """Additional CLI arguments for the mcp-js server."""
    env: dict[str, str] = field(default_factory=dict)
    """Extra environment variables (unused by JS runtime but kept for compatibility)."""
    cwd: str = ""
    """Working directory hint (used in templates, not by JS runtime)."""


class McpJsEnvironment:
    """Execute JavaScript code via a local mcp-js server."""

    def __init__(self, *, config_class: type = McpJsEnvironmentConfig, logger: logging.Logger | None = None, **kwargs):
        self.logger = logger or logging.getLogger("minisweagent.environment.mcp_js")
        self.config = config_class(**kwargs)
        self._process: subprocess.Popen | None = None
        self._port = self.config.port or _find_free_port()
        self._base_url = f"http://127.0.0.1:{self._port}"
        self._start_server()

    def _build_cmd(self) -> list[str]:
        cmd = [
            self.config.mcp_js_binary,
            "--stateless",
            "--http-port", str(self._port),
            "--execution-timeout", str(self.config.execution_timeout_secs),
        ]
        if self.config.policies_json:
            cmd.extend(["--policies-json", self.config.policies_json])
        cmd.extend(self.config.extra_args)
        return cmd

    def _start_server(self):
        cmd = self._build_cmd()
        self.logger.info(f"Starting mcp-js server: {shlex.join(cmd)}")
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            _wait_for_server(f"{self._base_url}/api/executions", timeout=15)
        except TimeoutError:
            stderr = self._process.stderr.read().decode() if self._process.stderr else ""
            self.logger.error(f"mcp-js server failed to start. stderr: {stderr}")
            raise
        self.logger.info(f"mcp-js server ready at {self._base_url}")

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute JavaScript code via mcp-js and return the result."""
        return _execute_via_http(
            self._base_url,
            command,
            timeout=timeout or self.config.timeout,
            execution_timeout_secs=self.config.execution_timeout_secs,
        )

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {"environment_type": "mcp_js"}

    def cleanup(self):
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    def __del__(self):
        self.cleanup()


# ── Docker environment (for SWE-bench) ───────────────────────────────────


@dataclass
class McpJsDockerEnvironmentConfig:
    """Configuration for mcp-js running inside a Docker container."""

    image: str = ""
    """Docker image to use (set by SWE-bench runner)."""
    mcp_js_binary: str = os.getenv("MCP_JS_BINARY", "server")
    """Path to the mcp-js binary on the *host*. Will be copied into the container."""
    cwd: str = "/testbed"
    """Working directory inside the container."""
    timeout: int = 60
    """Default execution timeout in seconds."""
    execution_timeout_secs: int = 120
    """V8 execution timeout passed to mcp-js per request."""
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables (used in templates)."""
    forward_env: list[str] = field(default_factory=list)
    """Host env vars to forward."""
    executable: str = os.getenv("MSWEA_DOCKER_EXECUTABLE", "docker")
    """Docker executable."""
    run_args: list[str] = field(default_factory=lambda: ["--rm"])
    """Extra docker run args."""
    container_timeout: str = "2h"
    """Max container lifetime."""
    pull_timeout: int = 120
    """Timeout for pulling docker images."""
    mcp_js_port: int = 8080
    """Port inside the container for mcp-js."""
    server_startup_timeout: float = 30.0
    """How long to wait for mcp-js to become ready."""


class McpJsDockerEnvironment:
    """Execute JavaScript code via mcp-js running inside a Docker container.

    The agent has NO shell/bash access.  All interaction goes through the
    mcp-js V8 runtime which exposes:

    - ``fs.*`` — full filesystem access (read, write, mkdir, stat, …)
    - ``fetch()`` — unrestricted HTTP client

    On submission the environment captures ``git diff`` internally
    (infrastructure-level, not agent-accessible).
    """

    def __init__(
        self,
        *,
        config_class: type = McpJsDockerEnvironmentConfig,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        self.logger = logger or logging.getLogger("minisweagent.environment.mcp_js_docker")
        self.config = config_class(**kwargs)
        self.container_id: str | None = None
        self._host_port = _find_free_port()
        self._base_url = f"http://127.0.0.1:{self._host_port}"
        self._start_container()
        self._setup_mcp_js()

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {"environment_type": "mcp_js_docker"}

    def _start_container(self):
        """Start the Docker container with port mapping."""
        container_name = f"minisweagent-mcpjs-{uuid.uuid4().hex[:8]}"
        cmd = [
            self.config.executable,
            "run", "-d",
            "--name", container_name,
            "-w", self.config.cwd,
            "-p", f"{self._host_port}:{self.config.mcp_js_port}",
            *self.config.run_args,
            self.config.image,
            "sleep", self.config.container_timeout,
        ]
        self.logger.debug(f"Starting container: {shlex.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.config.pull_timeout,
            check=True,
        )
        self.container_id = result.stdout.strip()
        self.logger.info(f"Started container {container_name} ({self.container_id[:12]})")

    def _setup_mcp_js(self):
        """Copy the mcp-js binary and policies into the container, then start the server.

        The server is started with permissive filesystem AND fetch (network)
        policies so the agent can read/write any file and make HTTP requests
        without restrictions.  No shell access is exposed to the agent.
        """
        assert self.container_id

        # Copy mcp-js binary into container
        subprocess.run(
            [self.config.executable, "cp", self.config.mcp_js_binary, f"{self.container_id}:/usr/local/bin/mcp-v8"],
            check=True,
            capture_output=True,
            timeout=30,
        )
        subprocess.run(
            [self.config.executable, "exec", self.container_id, "chmod", "+x", "/usr/local/bin/mcp-v8"],
            check=True,
            capture_output=True,
            timeout=10,
        )

        # Write permissive rego policies inside the container
        # 1. Filesystem: allow all operations on all paths
        # 2. Fetch: allow all HTTP methods to all domains
        rego_files = {
            "/tmp/fs_allow_all.rego": "package mcp.filesystem\ndefault allow = true\n",
            "/tmp/fetch_allow_all.rego": "package mcp.fetch\ndefault allow = true\n",
        }
        for path, content in rego_files.items():
            subprocess.run(
                [self.config.executable, "exec", self.container_id,
                 "bash", "-c", f"cat > {path} << 'REGO'\n{content}REGO"],
                check=True,
                capture_output=True,
                timeout=10,
            )

        policies = json.dumps({
            "filesystem": {
                "mode": "all",
                "policies": [{
                    "url": "file:///tmp/fs_allow_all.rego",
                    "rule": "data.mcp.filesystem.allow",
                }],
            },
            "fetch": {
                "mode": "all",
                "policies": [{
                    "url": "file:///tmp/fetch_allow_all.rego",
                    "rule": "data.mcp.fetch.allow",
                }],
            },
        })

        # Start mcp-js server in the background with full fs + network access
        start_cmd = (
            f"/usr/local/bin/mcp-v8 --stateless "
            f"--http-port {self.config.mcp_js_port} "
            f"--execution-timeout {self.config.execution_timeout_secs} "
            f"--policies-json '{policies}' "
            f">/dev/null 2>&1 &"
        )
        subprocess.run(
            [self.config.executable, "exec", "-d", self.container_id, "bash", "-c", start_cmd],
            check=True,
            capture_output=True,
            timeout=10,
        )

        # Wait for the server to be ready
        try:
            _wait_for_server(
                f"{self._base_url}/api/executions",
                timeout=self.config.server_startup_timeout,
            )
        except TimeoutError:
            self.logger.error("mcp-js server inside container did not become ready")
            raise
        self.logger.info(f"mcp-js server ready inside container at {self._base_url}")

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute JavaScript code via mcp-js HTTP API.

        If the output starts with ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT``,
        a ``git diff`` is captured from the container and appended.
        """
        result = _execute_via_http(
            self._base_url,
            command,
            timeout=timeout or self.config.timeout,
            execution_timeout_secs=self.config.execution_timeout_secs,
        )

        output_text = result.get("output", "")
        if output_text.lstrip().startswith("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"):
            # Infrastructure-level: capture git diff for SWE-bench evaluation.
            # This is NOT exposed to the agent — it runs after the agent signals
            # completion and is used solely to produce the model_patch.
            diff = self._capture_git_diff()
            result["output"] = f"COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n{diff}"

        return result

    def _capture_git_diff(self) -> str:
        """Infrastructure-only: run git add + git diff --cached to capture changes.

        This is called by the harness after the agent signals completion.
        The agent itself has no shell access.
        """
        assert self.container_id
        cmd = [
            self.config.executable, "exec",
            "-w", self.config.cwd,
            self.container_id,
            "bash", "-lc",
            "git add -A && git diff --cached",
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            return result.stdout
        except Exception as e:
            self.logger.error(f"Failed to capture git diff: {e}")
            return f"Error capturing git diff: {e}"

    def cleanup(self):
        if getattr(self, "container_id", None) is not None:
            cmd = (
                f"(timeout 60 {self.config.executable} stop {self.container_id} "
                f"|| {self.config.executable} rm -f {self.container_id}) >/dev/null 2>&1 &"
            )
            subprocess.Popen(cmd, shell=True)

    def __del__(self):
        self.cleanup()
