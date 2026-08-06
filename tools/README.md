# ORT Artifact Build

This directory contains `build_ort_artifacts.py`, the entrypoint for building
ONNX Runtime artifacts with the same versions and CMake settings that
`build.py` would use for the Triton ONNX Runtime backend.

The script resolves Triton's build configuration (by importing
`triton-server/build.py`), writes an `ort-build-plan.json`, then invokes the
`ort_target` CMake target and packages the resulting `onnxruntime/` payload.

## Prerequisites

- Run commands from the `triton-onnxruntime_backend` checkout.
- Keep `triton-server` checked out next to `triton-onnxruntime_backend`, or
  pass `--server-source-dir`.
- Pass `--github-organization` (forwarded to `build.py`) so the backend's
  nested component clones (common, core, backend) resolve from the same
  source as the top-level Triton build. Accepts a local path (repo mirror)
  or a remote URL (`https://github.com/...`). With a local mirror, all
  clones stay offline; with a remote URL, clones go over the network inside
  the container.
- Use the same branch in both repositories, for example
  `standalone_ort_build`.
- Have Docker available. For rootless Docker, export `DOCKER_HOST`, for
  example:

```bash
export DOCKER_HOST=unix:///run/user/${UID}/docker.sock
```

## Generate a Plan Only

Use this first to inspect the resolved versions, repo tags, container
version, parallelism, CUDA architecture list, and backend CMake arguments
without running the ORT build.

```bash
./tools/build_ort_artifacts.py \
  --server-source-dir ../triton-server \
  --artifact-dir /tmp/triton-ort \
  --plan-only \
  -- \
  --enable-gpu \
  -j 8
```

The plan is written to `/tmp/triton-ort/ort-build-plan.json`.

## Build ORT Artifacts

Remove `--plan-only` to run the ORT-only build.

```bash
./tools/build_ort_artifacts.py \
  --server-source-dir ../triton-server \
  --artifact-dir /tmp/triton-ort \
  -- \
  --enable-gpu \
  -j 8
```

By default, this configures the backend CMake build on the host, so the
host must provide `cmake`. To use the same Triton buildbase container flow
as the top-level container build, add `--build-in-triton-build-container`.
This builds the `tritonserver_buildbase` image from `build.py`'s generated
`Dockerfile.buildbase`, then runs the builder inside that container. The
nested ORT Docker build uses Docker's `host` network mode by default to
avoid DNS issues in docker-in-docker builds; override it with
`--ort-docker-build-network` if needed.

```bash
./tools/build_ort_artifacts.py \
  --build-in-triton-build-container \
  --server-source-dir ../triton-server \
  --artifact-dir /tmp/triton-ort \
  -- \
  --enable-gpu \
  -j 8
```

Expected outputs:

- `/tmp/triton-ort/ort-build-plan.json`
- `/tmp/triton-ort/ort-artifact-manifest.json`
- `/tmp/triton-ort/onnxruntime-<ort-version>-<container-version>-<config-hash>.tar.gz`

The tarball contains `onnxruntime/triton-ort-artifact.json`, which records
the resolved Triton build plan used to create the artifact, including
the `build.py` CUDA architecture list before ONNX Runtime backend
conversion. The backend owns the conversion from that Triton value to ORT
`CMAKE_CUDA_ARCHITECTURES`.

## Reuse the Artifact in a Triton Build

Use the unpacked `onnxruntime/` payload as `--ort-artifacts-dir` in a later
Triton build. The path must contain `include/`, `lib/libonnxruntime.so`,
and `triton-ort-artifact.json`.

```bash
mkdir -p /tmp/triton-ort/reuse
tar -xzf /tmp/triton-ort/onnxruntime-*.tar.gz -C /tmp/triton-ort/reuse

CUDA_ARCH_LIST="8.6" \
python ./build.py \
  --backend onnxruntime \
  --ort-artifacts-dir /tmp/triton-ort/reuse/onnxruntime \
  --enable-gpu \
  -j 8
```

The full Triton build validates the current `build.py` CUDA architecture
list and target machine against the artifact metadata before reusing the
artifact. Reuse fails when either value differs.

## Re-build from an Existing Plan

The planner writes `ort-build-plan.json` and then invokes the builder. To
re-run the builder against an existing plan (e.g. after a failed build),
pass `--plan` directly:

```bash
./tools/build_ort_artifacts.py --plan /tmp/triton-ort/ort-build-plan.json
```

## Common Options

Control nested ORT build parallelism through the normal Triton build option:

```bash
./tools/build_ort_artifacts.py --server-source-dir ../triton-server --artifact-dir /tmp/triton-ort -- -j 16
```

Build CPU-only ORT by omitting `--enable-gpu`:

```bash
./tools/build_ort_artifacts.py --server-source-dir ../triton-server --artifact-dir /tmp/triton-ort -- -j 8
```

Override an ONNX Runtime backend CMake argument through the forwarded
`build.py` options. For example, disable OpenVINO support:

```bash
./tools/build_ort_artifacts.py \
  --server-source-dir ../triton-server \
  --artifact-dir /tmp/triton-ort \
  -- \
  --enable-gpu \
  --override-backend-cmake-arg onnxruntime:TRITON_ENABLE_ONNXRUNTIME_OPENVINO=OFF \
  -j 8
```

Set CUDA architectures explicitly through the forwarded `build.py` option:

```bash
./tools/build_ort_artifacts.py \
  --server-source-dir ../triton-server \
  --artifact-dir /tmp/triton-ort \
  -- \
  --enable-gpu \
  --cuda-arch-list "8.0 8.6 9.0" \
  -j 8
```

If `--cuda-arch-list` is omitted, `build.py` falls back to `CUDA_ARCH_LIST`
from the environment when it is set.

The preferred path runs the ORT clone/build step as a non-root user inside
the generated Dockerfile. If an upstream ORT build assumption requires
root, use the explicit compatibility fallback:

```bash
./tools/build_ort_artifacts.py \
  --allow-root-ort-build \
  --server-source-dir ../triton-server \
  --artifact-dir /tmp/triton-ort \
  -- \
  --enable-gpu \
  -j 8
```
