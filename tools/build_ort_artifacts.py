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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS`` AND ANY
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

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import tarfile


BACKEND_DIR = pathlib.Path(__file__).resolve().parents[1]


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


def git_value(repo_dir, args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), *args], text=True
        ).strip()
    except subprocess.CalledProcessError:
        return None


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


def main():
    parser = argparse.ArgumentParser(
        description="Build and package ONNX Runtime artifacts for Triton."
    )
    parser.add_argument("--plan", required=True, help="Path to ort-build-plan.json.")
    args = parser.parse_args()

    plan_path = pathlib.Path(args.plan).resolve()
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


if __name__ == "__main__":
    main()
