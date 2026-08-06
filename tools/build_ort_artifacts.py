#!/usr/bin/env python3
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Build and package ONNX Runtime artifacts for Triton.

Two modes:

* Planner (default): resolves Triton's build configuration via build.py,
  writes an ort-build-plan.json, then invokes the builder.  Options before
  ``--`` configure the planner; options after ``--`` are forwarded to
  build.py so the ORT plan uses the same Triton build configuration.

* Builder (``--plan <path>``): reads a plan and runs the ORT build.
  Normally invoked by the planner, but can be called directly to re-build
  from an existing plan.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tarfile


BACKEND_DIR = pathlib.Path(__file__).resolve().parents[1]
WORKSPACE_DIR = BACKEND_DIR.parent


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def git_value(repo_dir, args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), *args], text=True
        ).strip()
    except subprocess.CalledProcessError:
        return None


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


# ---------------------------------------------------------------------------
# Builder mode (reads a plan, builds ORT, packages artifact)
# ---------------------------------------------------------------------------


def load_plan(path):
    with pathlib.Path(path).open() as f:
        return json.load(f)


def run(cmd, cwd=None):
    print("+ {}".format(" ".join(str(c) for c in cmd)))
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_ort_dir(ort_dir):
    required = [
        ort_dir / "include",
        ort_dir / "lib" / "libonnxruntime.so",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("ORT artifact directory is incomplete: {}".format(missing))


def package_artifact(ort_dir, tarball):
    tarball.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(ort_dir, arcname="onnxruntime")


def write_artifact_metadata(ort_dir, plan, backend_source_dir):
    metadata = {
        "schema_version": 1,
        "backend_git_commit": git_value(backend_source_dir, ["rev-parse", "HEAD"]),
        "backend_git_remote": git_value(backend_source_dir, ["remote", "get-url", "origin"]),
        "plan": plan,
    }
    metadata_path = ort_dir / "triton-ort-artifact.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")
    return metadata_path


def build_from_plan(plan_path):
    plan_path = pathlib.Path(plan_path).resolve()
    plan = load_plan(plan_path)
    source_config = plan["source_config"]
    backend_source_dir = pathlib.Path(plan["backend_source_dir"]).resolve()
    artifact_dir = pathlib.Path(plan["artifact_dir"]).resolve()
    build_dir = artifact_dir / "build"
    ort_dir = build_dir / "onnxruntime"
    tarball = artifact_dir / plan["artifact_name"]
    manifest = artifact_dir / "ort-artifact-manifest.json"

    cmake_args = list(source_config["cmake_args"])
    run(
        [
            "cmake",
            "-S",
            str(backend_source_dir),
            "-B",
            str(build_dir),
            *cmake_args,
        ]
    )

    build_cmd = [
        "cmake",
        "--build",
        str(build_dir),
        "--config",
        source_config["build_type"],
        "-t",
        "ort_target",
    ]
    if source_config["build_parallel"]:
        build_cmd.append(f"-j{source_config['build_parallel']}")
    run(build_cmd)

    validate_ort_dir(ort_dir)
    artifact_metadata = write_artifact_metadata(ort_dir, plan, backend_source_dir)
    package_artifact(ort_dir, tarball)

    manifest_data = {
        "artifact": {
            "name": tarball.name,
            "path": str(tarball),
            "sha256": sha256(tarball),
        },
        "backend_git_commit": git_value(backend_source_dir, ["rev-parse", "HEAD"]),
        "backend_git_remote": git_value(backend_source_dir, ["remote", "get-url", "origin"]),
        "embedded_metadata": str(artifact_metadata),
        "plan": plan,
        "plan_path": str(plan_path),
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with manifest.open("w") as f:
        json.dump(manifest_data, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote ORT artifact to {tarball}")
    print(f"Wrote ORT manifest to {manifest}")


# ---------------------------------------------------------------------------
# Planner mode (resolves build.py config, writes plan, invokes builder)
# ---------------------------------------------------------------------------


def load_build_module(server_dir):
    build_py = server_dir / "build.py"
    spec = importlib.util.spec_from_file_location("triton_build", build_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_server_source(server_source_dir):
    if server_source_dir is not None:
        return pathlib.Path(server_source_dir).resolve()
    sibling = WORKSPACE_DIR / "triton-server"
    if sibling.exists():
        return sibling.resolve()
    raise SystemExit(
        "triton-server checkout not found; pass --server-source-dir or place "
        "triton-server next to triton-onnxruntime_backend"
    )


def ensure_onnxruntime_backend(build_args):
    found = False
    for idx, arg in enumerate(build_args):
        if arg == "--backend" and idx + 1 < len(build_args):
            backend = build_args[idx + 1].split(":", 1)[0]
            if backend != "onnxruntime":
                raise SystemExit(
                    "ORT artifact builds only support --backend onnxruntime; "
                    f"got --backend {build_args[idx + 1]}"
                )
            found = True
        if arg.startswith("--backend="):
            backend = arg.split("=", 1)[1].split(":", 1)[0]
            if backend != "onnxruntime":
                raise SystemExit(
                    "ORT artifact builds only support --backend onnxruntime; "
                    f"got {arg}"
                )
            found = True
    if found:
        return build_args
    return build_args + ["--backend", "onnxruntime"]


def normalize_cmake_args(args):
    normalized = []
    for arg in args:
        if arg == "..":
            continue
        if len(arg) >= 2 and arg[0] == '"' and arg[-1] == '"':
            arg = arg[1:-1]
        normalized.append(arg)
    return normalized


def config_hash(config):
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _ort_identity_cmake_args(cmake_args):
    """Extract cmake args that affect the ORT binary identity.

    Only cmake variables that reach gen_ort_dockerfile.py via
    CMakeLists.txt:403/416 (either directly or through _GEN_FLAGS) are
    included.  Triton-generic variables (STATS, METRICS, repo tags, etc.)
    and environment-only variables (TRT_VERSION,
    TRITON_ONNX_TENSORRT_REPO_TAG) are excluded so that non-ORT build
    changes do not invalidate the ORT artifact cache.
    """
    ort_keys = {
        "TRITON_BUILD_CONTAINER",
        "TRITON_ENABLE_ONNXRUNTIME_OPENVINO",
        "TRITON_ENABLE_ONNXRUNTIME_TENSORRT",
    }
    result = []
    for arg in cmake_args:
        if arg.startswith("-D"):
            name = arg[2:].split(":")[0].split("=")[0]
            if name in ort_keys:
                result.append(arg)
    return sorted(result)


def docker_mount_args(paths):
    mounts = []
    seen = []
    for path in paths:
        resolved = pathlib.Path(path).resolve()
        if resolved.is_file():
            resolved = resolved.parent
        if any(
            resolved == existing or resolved.is_relative_to(existing)
            for existing in seen
        ):
            continue
        seen.append(resolved)
        mounts += ["-v", f"{resolved}:{resolved}"]
    return mounts


def docker_env_args(names):
    env_args = []
    for name in names:
        if name in os.environ:
            env_args += ["-e", f"{name}={os.environ[name]}"]
    return env_args


def build_triton_buildbase(build, server_dir, build_config, image):
    if build.FLAGS.no_container_build:
        raise SystemExit(
            "--build-in-triton-build-container cannot be used with --no-container-build"
        )
    if build.target_platform() == "windows":
        raise SystemExit("--build-in-triton-build-container is only supported on Linux")

    pathlib.Path(build.FLAGS.build_dir).mkdir(parents=True, exist_ok=True)
    build.create_build_dockerfiles(
        build_config["script_build_dir"],
        build_config["images"],
        build_config["backends"],
        build_config["repoagents"],
        build_config["caches"],
        build.FLAGS.endpoint,
    )

    dockerfile = pathlib.Path(build.FLAGS.build_dir) / "Dockerfile.buildbase"
    cmd = ["docker", "build", "-t", image, "-f", dockerfile]
    if not build.FLAGS.no_container_pull:
        cmd.append("--pull")
    cmd += [
        "--cache-from=tritonserver_buildbase",
        "--cache-from=tritonserver_buildbase_cache0",
        "--cache-from=tritonserver_buildbase_cache1",
        ".",
    ]
    subprocess.run([str(arg) for arg in cmd], cwd=server_dir, check=True)


def run_builder_in_triton_build_container(
    build,
    server_dir,
    plan_output,
    artifact_dir,
    image,
):
    runargs = [
        "docker",
        "run",
        "--rm",
        "-w",
        str(BACKEND_DIR),
    ]
    runargs += build.docker_runargs()
    runargs += docker_env_args(
        [
            "CUDA_ARCH_LIST",
            "DOCKER_BUILDKIT",
            "TRT_VERSION",
            "CMAKE_TOOLCHAIN_FILE",
            "VCPKG_TARGET_TRIPLET",
            "CCACHE_REMOTE_ONLY",
            "CCACHE_REMOTE_STORAGE",
        ]
    )
    if build.FLAGS.use_user_docker_config and os.path.exists(
        build.FLAGS.use_user_docker_config
    ):
        runargs += [
            "-v",
            "{}:/root/.docker/config.json".format(
                os.path.expanduser(build.FLAGS.use_user_docker_config)
            ),
        ]
    # WORKSPACE_DIR is mounted so host paths (backend source, artifact dir, plan
    # output, and any local org mirror living under it) are visible inside the
    # container. If --github-organization is a remote URL, nested component
    # clones resolve over the network instead.
    mount_paths = [WORKSPACE_DIR, artifact_dir, plan_output.parent]
    runargs += docker_mount_args(mount_paths)
    # Host-owned checkouts are mounted into the container where git runs as root;
    # mark them safe so nested clones don't fail with "dubious ownership".
    inner_cmd = "git config --system --add safe.directory '*' && python3 '{}' --plan '{}'".format(
        BACKEND_DIR / "tools" / "build_ort_artifacts.py", plan_output
    )
    runargs += [image, "sh", "-c", inner_cmd]
    subprocess.run([str(arg) for arg in runargs], check=True)


def resolve_and_build(wrapper_args):
    build_args = wrapper_args.build_args
    if build_args and build_args[0] == "--":
        build_args = build_args[1:]
    build_args = ensure_onnxruntime_backend(build_args)

    server_dir = find_server_source(wrapper_args.server_source_dir)
    build = load_build_module(server_dir)
    build_config = build.resolve_build_config(build.create_arg_parser().parse_args(build_args))

    library_paths = build_config["library_paths"]
    if (
        "onnxruntime" in library_paths
        or getattr(build.FLAGS, "ort_artifacts_dir", None) is not None
    ):
        raise SystemExit(
            "ORT artifact build must build ONNX Runtime; do not pass "
            "--library-paths=onnxruntime:* or --ort-artifacts-dir."
        )

    artifact_dir = wrapper_args.artifact_dir.resolve()
    backend_tag = build_config["backends"]["onnxruntime"]
    cmake_args = normalize_cmake_args(
        build.backend_cmake_args(
            build_config["images"],
            build_config["components"],
            "onnxruntime",
            str(artifact_dir / "install"),
            library_paths,
        )
    )

    cuda_arch_list = build.FLAGS.cuda_arch_list or os.environ.get("CUDA_ARCH_LIST")

    if wrapper_args.ort_docker_build_network:
        cmake_args.append(
            "-DTRITON_ONNXRUNTIME_DOCKER_BUILD_NETWORK:STRING={}".format(
                wrapper_args.ort_docker_build_network
            )
        )
    cmake_args.append(
        "-DTRITON_ONNXRUNTIME_BUILD_AS_ROOT:BOOL={}".format(
            "ON" if wrapper_args.allow_root_ort_build else "OFF"
        )
    )

    source_config = {
        "backend_tag": backend_tag,
        "build_args": build_args,
        "build_parallel": build.FLAGS.build_parallel,
        "build_type": build.FLAGS.build_type,
        "cmake_args": cmake_args,
        "components": build_config["components"],
        "build_environment": (
            "triton-build-container"
            if wrapper_args.build_in_triton_build_container
            else "host"
        ),
        "cuda_arch_list": cuda_arch_list,
        "enable_gpu": build.FLAGS.enable_gpu,
        "github_organization": build.FLAGS.github_organization,
        "images": build_config["images"],
        "min_compute_capability": build.FLAGS.min_compute_capability,
        "ort_build_as_root": wrapper_args.allow_root_ort_build,
        "ort_docker_build_network": wrapper_args.ort_docker_build_network,
        "ort_openvino_version": build.FLAGS.ort_openvino_version,
        "ort_version": build.FLAGS.ort_version,
        "target_machine": build.target_machine(),
        "target_platform": build.target_platform(),
        "triton_buildbase_image": wrapper_args.triton_buildbase_image,
        "triton_container_version": build.FLAGS.triton_container_version,
        "upstream_container_version": build.FLAGS.upstream_container_version,
        "version": build.FLAGS.version,
    }
    ort_identity = {
        "backend_tag": backend_tag,
        "build_environment": (
            "triton-build-container"
            if wrapper_args.build_in_triton_build_container
            else "host"
        ),
        "build_type": build.FLAGS.build_type,
        "cuda_arch_list": cuda_arch_list,
        "enable_gpu": build.FLAGS.enable_gpu,
        "ort_cmake_args": _ort_identity_cmake_args(cmake_args),
        "ort_openvino_version": build.FLAGS.ort_openvino_version,
        "ort_version": build.FLAGS.ort_version,
        "target_machine": build.target_machine(),
        "target_platform": build.target_platform(),
        "triton_buildbase_image": wrapper_args.triton_buildbase_image,
        "triton_container_version": build.FLAGS.triton_container_version,
    }
    config_hash_value = config_hash(ort_identity)
    source_config["config_hash"] = config_hash_value

    plan = {
        "artifact_dir": str(artifact_dir),
        "artifact_name": "onnxruntime-{}-{}-{}.tar.gz".format(
            build.FLAGS.ort_version,
            build.FLAGS.triton_container_version,
            config_hash_value,
        ),
        "backend_source_dir": str(BACKEND_DIR),
        # Hash the tree (content), not the commit: the pipeline re-cherry-picks
        # the overlay branch onto the build branch on every run, which yields a
        # fresh commit SHA for identical content and would invalidate the
        # cached ORT artifact each time (both locally and on Jenkins, where
        # only the docker layer cache hid the churn).
        "backend_git_commit": git_value(
            BACKEND_DIR, ["rev-parse", "HEAD^{tree}"]
        ),
        "server_git_commit": git_value(server_dir, ["rev-parse", "HEAD"]),
        "ort_identity": ort_identity,
        "source_config": source_config,
    }

    plan_output = wrapper_args.plan_output or artifact_dir / "ort-build-plan.json"
    write_json(plan_output.resolve(), plan)
    print(f"Wrote ORT build plan to {plan_output.resolve()}")

    if wrapper_args.plan_only:
        return

    if wrapper_args.build_in_triton_build_container:
        build_triton_buildbase(
            build, server_dir, build_config, wrapper_args.triton_buildbase_image
        )
        run_builder_in_triton_build_container(
            build,
            server_dir,
            plan_output.resolve(),
            artifact_dir,
            wrapper_args.triton_buildbase_image,
        )
    else:
        build_from_plan(plan_output.resolve())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Build and package ONNX Runtime artifacts for Triton."
    )
    # Builder mode: --plan alone skips config resolution and re-builds from a
    # pre-existing plan (used by the planner when re-invoking itself, and by
    # callers that want to re-run a build without re-resolving config).
    parser.add_argument(
        "--plan",
        type=pathlib.Path,
        default=None,
        help="Path to an existing ort-build-plan.json. If set, skip config "
        "resolution and re-build from the plan.",
    )
    # Planner-mode options (ignored in builder mode).
    parser.add_argument(
        "--server-source-dir",
        type=pathlib.Path,
        default=None,
        help="Path to a triton-server checkout. Defaults to a sibling checkout.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=pathlib.Path,
        default=WORKSPACE_DIR / "triton-server" / "build" / "ort-artifacts",
        help="Directory where the ORT plan, manifest and tarball will be written.",
    )
    parser.add_argument(
        "--plan-output",
        type=pathlib.Path,
        default=None,
        help="Path for the generated ORT build plan. Defaults to <artifact-dir>/ort-build-plan.json.",
    )
    parser.add_argument(
        "--ort-docker-build-network",
        default="host",
        help=(
            "Docker network mode for the nested ONNXRuntime docker build. "
            "Defaults to host to avoid DNS issues in docker-in-docker builds. "
            "Use an empty value to keep Docker's default build network."
        ),
    )
    parser.add_argument(
        "--allow-root-ort-build",
        action="store_true",
        help="Allow the ORT Dockerfile to run the ORT build step as root.",
    )
    parser.add_argument(
        "--build-in-triton-build-container",
        action="store_true",
        help=(
            "Build the ORT artifact from inside the Triton buildbase container "
            "generated by build.py, instead of requiring CMake on the host."
        ),
    )
    parser.add_argument(
        "--triton-buildbase-image",
        default="tritonserver_buildbase",
        help="Docker image tag to use for the Triton buildbase container.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write the ORT build plan but do not invoke the builder.",
    )
    parsed, build_args = parser.parse_known_args()

    if parsed.plan is not None:
        build_from_plan(parsed.plan)
        return

    parsed.build_args = build_args
    resolve_and_build(parsed)


if __name__ == "__main__":
    main()
