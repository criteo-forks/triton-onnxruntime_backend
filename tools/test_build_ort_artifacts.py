#!/usr/bin/env python3
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import os
import pathlib
import platform
import subprocess
import sys
import tempfile
import unittest


BACKEND_DIR = pathlib.Path(__file__).resolve().parents[1]
BUILDER = BACKEND_DIR / "tools" / "build_ort_artifacts.py"
SERVER_DIR = BACKEND_DIR.parent / "triton-server"


def run(cmd, env=None, check=True, cwd=None):
    result = subprocess.run(
        [str(arg) for arg in cmd],
        cwd=cwd or BACKEND_DIR,
        env=env,
        text=True,
        capture_output=True,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            "command failed with {}\ncmd: {}\nstdout:\n{}\nstderr:\n{}".format(
                result.returncode, " ".join(str(arg) for arg in cmd), result.stdout, result.stderr
            )
        )
    return result


def write_fake_artifact(path, cuda_arch_list):
    (path / "include").mkdir(parents=True)
    (path / "lib").mkdir()
    (path / "lib" / "libonnxruntime.so").touch()
    with (path / "triton-ort-artifact.json").open("w") as f:
        json.dump(
            {
                "plan": {
                    "source_config": {
                        "cuda_arch_list": cuda_arch_list,
                        "target_machine": platform.machine().lower(),
                    },
                },
            },
            f,
        )


class BuildOrtArtifactsTest(unittest.TestCase):
    def test_plan_only_covers_common_parameters(self):
        if not SERVER_DIR.exists():
            self.skipTest("triton-server checkout not found at {}".format(SERVER_DIR))
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            org = tmp / "org"
            org.mkdir()
            for name in ("triton-common", "triton-core", "triton-backend"):
                (org / name).mkdir()
            run(
                [
                    sys.executable,
                    BUILDER,
                    "--server-source-dir",
                    SERVER_DIR,
                    "--artifact-dir",
                    tmp / "ort",
                    "--build-in-triton-build-container",
                    "--plan-only",
                    "--",
                    "--enable-gpu",
                    "--github-organization",
                    str(org),
                    "--cuda-arch-list",
                    "8.6",
                    "--override-backend-cmake-arg",
                    "onnxruntime:TRITON_ENABLE_ONNXRUNTIME_OPENVINO=OFF",
                    "-j",
                    "6",
                ],
            )

            with (tmp / "ort" / "ort-build-plan.json").open() as f:
                plan = json.load(f)
            config = plan["source_config"]
            cmake_args = config["cmake_args"]

            self.assertEqual(config["build_parallel"], 6)
            self.assertEqual(config["build_environment"], "triton-build-container")
            self.assertEqual(config["ort_docker_build_network"], "host")
            self.assertEqual(config["github_organization"], str(org))
            self.assertEqual(config["cuda_arch_list"], "8.6")
            self.assertTrue(config["enable_gpu"])
            self.assertEqual(
                sum("TRITON_CUDA_ARCH_LIST" in arg for arg in cmake_args), 1
            )
            self.assertIn("-DTRITON_ENABLE_ONNXRUNTIME_OPENVINO:BOOL=OFF", cmake_args)
            self.assertIn(
                "-DTRITON_ONNXRUNTIME_DOCKER_BUILD_NETWORK:STRING=host",
                cmake_args,
            )
            self.assertIn("-DTRITON_ONNXRUNTIME_BUILD_AS_ROOT:BOOL=OFF", cmake_args)

            # Cache identity must be content-based, not branch-tag-based: the
            # backend ref name (backend_tag) must not be in the hashed identity,
            # and the ORT-build-driving source hash must be.
            identity = plan["ort_identity"]
            self.assertNotIn("backend_tag", identity)
            self.assertIn("ort_build_source", identity)
            self.assertTrue(identity["ort_build_source"])

            # EP-specific values are gated on the EP toggle. TRT EP defaults ON
            # with --enable-gpu, so trt_version is present (empty = use the
            # parser version shipping with ORT). OpenVINO is OFF here, so its
            # version must be absent.
            self.assertIn("trt_version", identity)
            self.assertEqual(identity["trt_version"], "")
            self.assertIn("onnx_tensorrt_repo_tag", identity)
            self.assertEqual(identity["onnx_tensorrt_repo_tag"], "")
            self.assertNotIn("ort_openvino_version", identity)

    def test_full_build_reuses_matching_artifact_only(self):
        if not SERVER_DIR.exists():
            self.skipTest("triton-server checkout not found at {}".format(SERVER_DIR))
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            artifact = tmp / "onnxruntime"
            write_fake_artifact(artifact, "8.6")

            env = os.environ.copy()
            env["CUDA_ARCH_LIST"] = "8.6"
            result = run(
                [
                    sys.executable,
                    SERVER_DIR / "build.py",
                    "--dryrun",
                    "--no-container-build",
                    "--no-core-build",
                    "--build-dir",
                    tmp / "build",
                    "--install-dir",
                    tmp / "install",
                    "--backend",
                    "onnxruntime",
                    "--ort-artifacts-dir",
                    artifact,
                    "--enable-gpu",
                ],
                env=env,
                cwd=SERVER_DIR,
            )
            self.assertEqual(result.returncode, 0)

            with (tmp / "build" / "cmake_build").open() as f:
                cmake_build = f.read()
            self.assertIn("TRITON_ONNXRUNTIME_ARTIFACTS_PATH", cmake_build)
            self.assertIn(str(artifact), cmake_build)

            env["CUDA_ARCH_LIST"] = "8.0"
            result = run(
                [
                    sys.executable,
                    SERVER_DIR / "build.py",
                    "--dryrun",
                    "--no-container-build",
                    "--no-core-build",
                    "--build-dir",
                    tmp / "mismatch-build",
                    "--install-dir",
                    tmp / "mismatch-install",
                    "--backend",
                    "onnxruntime",
                    "--ort-artifacts-dir",
                    artifact,
                    "--enable-gpu",
                ],
                env=env,
                check=False,
                cwd=SERVER_DIR,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("cuda_arch_list does not match", result.stderr)


if __name__ == "__main__":
    unittest.main()
