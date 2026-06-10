# 3dconvert — podman wrappers for 2D→stereo-3D video conversion

Self-contained packaging of two complementary 2D→stereo-3D pipelines on Debian 13
with 2× RTX 3090s, each as a podman image plus a thin host-side CLI:

| Tool | Backend | Approach | Speed | Quality |
|---|---|---|---|---|
| **`./stereocrafter`** | [TencentARC/StereoCrafter](https://github.com/TencentARC/StereoCrafter) | Depth + forward-splatting + **SVD diffusion inpaint** | Slow (~4 s/frame 1080p on 2× 3090) | Strong, handles disocclusion via generative inpainting |
| **`./depth-surge`** | [Tok/depth-surge-3d](https://github.com/Tok/depth-surge-3d) | Depth-Anything + DIBR + classical hole-fill (+ optional Real-ESRGAN, fisheye) | Fast (~1 s/frame 1080p on 1 × 3090) | Lower in disocclusion regions, better for VR (native fisheye) |

Choose **stereocrafter** for highest quality on complex content with
disocclusions (people, foreground objects).
Choose **depth-surge** for fast batch processing, VR headsets (fisheye output),
or long videos where wall-time matters more than perfect inpainting.

---

# stereocrafter — TencentARC/StereoCrafter wrapper

Self-contained 2D → stereo-3D video pipeline using
[TencentARC/StereoCrafter](https://github.com/TencentARC/StereoCrafter), packaged
as a podman image plus a thin host-side CLI.

- **Image**: CUDA 11.8 / Python 3.8 / PyTorch 2.0.1+cu118 / xformers 0.0.20, with
  StereoCrafter + DepthCrafter + Forward-Warp CUDA extension pre-built for sm_86
  (RTX 30-series).
- **Wrapper**: handles GPU selection, weights download (gated SVD repo via your
  HF token), per-video chunking with OOM-retry, multi-GPU parallelism, and
  ffmpeg-based pre-split / post-concat.

---

## Requirements

- Debian 13 (or any Linux with podman ≥ 4 + NVIDIA Container Toolkit / CDI)
- NVIDIA GPU(s) with sm_86 (RTX 30-series). Other CCs would need `TORCH_CUDA_ARCH_LIST`
  changed in the Dockerfile.
- ~30 GB free disk (image 13.6 GB + weights ~24 GB)
- A Hugging Face account with access granted to
  `stabilityai/stable-video-diffusion-img2vid-xt-1-1` (gated; one-click on the repo
  page while signed in)
- HF token at `~/.cache/huggingface/token` (run `huggingface-cli login` on the host
  once) — the wrapper passes it into the container via `--env-file`.

## Quick start

```bash
# one-time setup (~30 min image build + ~20 GB weight download)
./stereocrafter build
./stereocrafter pull-weights

# convert a video — uses all visible GPUs by default
./stereocrafter convert /path/to/clip.mp4
# outputs land in ./output/clip_sbs.mp4 + clip_anaglyph.mp4
```

Check what's installed at any time:

```bash
./stereocrafter status
```

## Subcommands

| Command | What it does |
|---|---|
| `build` | Build the podman image (one-time) |
| `pull-weights` | Download SVD-xt-1-1, DepthCrafter, StereoCrafter weights into `./weights/` |
| `convert <video> [out_dir]` | Full pipeline; auto multi-GPU, auto chunking |
| `splat <video> [out.mp4]` | Stage 1 only (depth + forward splatting) |
| `inpaint <splat.mp4> [out_dir]` | Stage 2 only (stereo inpainting) |
| `shell` | Interactive bash in the container with the same mounts |
| `status` | Image + weights state |

## Environment variables

All are optional. When unset, `convert` auto-detects and picks safe values.

| Variable | Default | Meaning |
|---|---|---|
| `STEREOCRAFTER_ROOT` | dir of script | Project root |
| `STEREOCRAFTER_WEIGHTS` | `$ROOT/weights` | Where weights live |
| `STEREOCRAFTER_OUTPUT` | `$ROOT/output` | Default output dir |
| `STEREOCRAFTER_IMAGE` | `localhost/stereocrafter:latest` | Image tag |
| `STEREOCRAFTER_GPU` | unset (all GPUs) | Pin to one GPU (`0` or `1`). When unset, `convert` splits N ways across all visible GPUs. |
| `STEREOCRAFTER_CHUNK_FRAMES` | auto from free VRAM | Frames per chunk for the inner DepthCrafter loop. Auto-tuned with formula `(0.8×free_MB − 12000) / 4`, clamped to [64, 1024]. |
| `STEREOCRAFTER_TILE_NUM` | auto from free VRAM | Tile count for stage 2 inpaint. Auto: ≥18 GB→2, 10–18 GB→3, <10 GB→4. |
| `STEREOCRAFTER_KEEP_CHUNKS` | `0` | Set to `1` to keep intermediate per-chunk files for debugging |

## Output layout

```
./output/
├── <basename>_sbs.mp4        # side-by-side stereo (3840×H for 1080p in)
└── <basename>_anaglyph.mp4   # red/cyan anaglyph (1920×H)
```

For long inputs, intermediates live in `./output/.parts_<basename>/...` and are
cleaned up on completion (set `STEREOCRAFTER_KEEP_CHUNKS=1` to retain).

## Multi-GPU behaviour

If you have N visible GPUs and the input is ≥ 2× the auto chunk size:

1. ffmpeg splits the input into N approximately-equal parts (forced key-frame +
   segment muxer for codec-param uniformity).
2. N container processes start in parallel, one pinned to each GPU via
   `--device nvidia.com/gpu=<idx>`.
3. Each pipeline runs its own inner chunking + retry loop.
4. The N per-part outputs are concatenated to a single result via
   `ffmpeg -f concat -c copy`.

Set `STEREOCRAFTER_GPU=<idx>` to force single-GPU mode (useful when one GPU is
busy or for benchmarking).

## OOM retry

If a chunk OOMs (matches `OutOfMemoryError|out of memory` in stderr), the
wrapper:

1. Bumps `tile_num` by 1 (lowers stage-2 peak).
2. Splits the failing chunk in half with ffmpeg and processes each half.
3. Recurses up to depth 3, then gives up.

This means transient OOMs from VRAM fragmentation or concurrent desktop
workloads are handled automatically; you don't need to babysit long runs.

## Known limitations

- **Depth is computed independently per chunk and per part**, so very subtle
  disparity drift can occur at boundaries. Imperceptible in most content; if you
  spot a seam, lower `STEREOCRAFTER_CHUNK_FRAMES` (smaller chunks → more
  boundaries but each boundary's drift is smaller) or open an issue.
- The internal depth resolution is fixed at 1024×576 (StereoCrafter's
  `max_res=1024`). Inputs above 1080p see no extra 3D fidelity but pay 4× cost
  in stage 2.
- StereoCrafter was designed for ~1-minute clips; movie-length material on
  consumer GPUs is impractical (see [CHEATSHEET.md](CHEATSHEET.md)).
- Token leaks: the wrapper hides your HF token from `ps` via `--env-file` (mode
  0600), but anyone with access to your home dir can read
  `~/.cache/huggingface/token`. Standard HF caveat.

## Troubleshooting

**`Cannot access gated repo`** — you haven't agreed to the SVD license. Visit
https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1
signed in, click "Agree and access repository". `gated: auto` means access is
instant.

**`short-name did not resolve`** — podman registry config. The Dockerfile uses
`docker.io/nvidia/cuda:...` explicitly; if you change it, keep the registry
prefix.

**Forward-Warp `ModuleNotFoundError`** — your image was built before the
`pip install` fix. Rebuild: `./stereocrafter build`.

**Output dir says "MISSING"** — `./stereocrafter status` shows `[ ]` next to a
weights dir because either `git lfs` didn't fetch real files (you got pointer
stubs) or download was interrupted. Re-run `./stereocrafter pull-weights` (it
skips dirs that look complete).

---

# depth-surge — Tok/depth-surge-3d wrapper

Faster, simpler pipeline. Uses Depth-Anything-V2/V3 for depth and DIBR (depth-image-based
rendering) with classical hole-fill — no diffusion inpainting. Single-pass, frame-by-frame,
no chunking required.

## Quick start

```bash
# one-time setup (~10-15 min image build + 1-2 GB model download)
./depth-surge build
./depth-surge pull-weights         # downloads Depth-Anything-V2-Large by default

# convert
./depth-surge convert /path/to/clip.mp4
# outputs land in ./output/<clip_stem>_<timestamp>/{video}_SBS.mp4
```

## Env vars (override defaults)

```
DEPTHSURGE_VERSION       v2 | v3                       (default: v2)
DEPTHSURGE_MODEL         depth-anything/Depth-Anything-V2-Large (default; switch to V2-Base for speed)
DEPTHSURGE_FORMAT        side_by_side | over_under     (default: side_by_side)
DEPTHSURGE_RESOLUTION    auto | 16x9-1080p | square-4k | custom:WxH ... (default: auto)
DEPTHSURGE_BASELINE      stereo baseline in meters     (default: 0.065 = average IPD)
DEPTHSURGE_HOLE_FILL     fast | advanced               (default: fast)
DEPTHSURGE_UPSCALE       none | x2 | x4 | x4-conservative  (default: none)
DEPTHSURGE_GPU           pin to a specific GPU index   (default: first visible)
DEPTHSURGE_ANAGLYPH      set to 1 to also emit a red/cyan anaglyph mp4 via ffmpeg
DEPTHSURGE_EXTRA         extra args appended to depth_surge_3d.py invocation
```

## When to pick depth-surge over stereocrafter

- **Long videos** — DIBR is much faster; 1 hour of 1080p ≈ 1 h on one 3090 vs ~10 days on StereoCrafter
- **VR headset playback** — native fisheye projection with selectable lens model
- **Real-Time-ish testing** — quick visual preview of stereo conversion at low quality
- **No disocclusion-heavy content** — landscapes, slow pans, static scenes

## When to pick stereocrafter

- **People + objects with parallax** — generative inpainting hides disocclusion artifacts much better
- **Best possible quality** for short clips you'll re-watch

## Files in this repo

| File | Purpose |
|---|---|
| `Dockerfile` | StereoCrafter image recipe (cu118 + py3.8 + StereoCrafter + Forward-Warp) |
| `Dockerfile.depth-surge` | Depth Surge image recipe (cu124 + py3.12 + Depth-Anything) |
| `stereocrafter` | StereoCrafter host CLI wrapper |
| `depth-surge` | Depth Surge host CLI wrapper |
| `README.md` | This file |
| `CHEATSHEET.md` | Per-task time + recommended-settings tables for both tools |
| `.gitignore` | Excludes `weights/`, `input/`, `output/`, `out/` from version control |
