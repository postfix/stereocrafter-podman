# Cheatsheet — estimated time + recommended settings

All estimates are for **2× RTX 3090 (24 GB)** at default `tile_num=2`,
`max_res=1024`, with the wrapper's auto-config picking `chunk_frames` from free
VRAM. Numbers are **wall time** (not GPU-time); both GPUs run in parallel.

Empirical base rate (measured on this hardware):

- **1080p, auto-config**: ~4 s per frame (post-optimisation)
- **Stage 2 inpaint scales with input pixel count**, so 4K is ~4× slower per
  frame than 1080p; 720p is ~2× faster.

## Time-vs-frames at common source resolutions

| Source | Frames/sec | 10 s clip | 1 min clip | 15 min anime | 1 h show | 2 h movie |
|---|---|---|---|---|---|---|
| **4K @ 24 fps** | 24 × 16 s | 4 min | 24 min | ~96 h ≈ 4 d | ~16 d | ~32 d |
| **1080p @ 24 fps** | 24 × 4 s | 1 min | 6 min | ~24 h ≈ 1 d | ~4 d | ~8 d |
| **1080p @ 30 fps** | 30 × 4 s | 1.3 min | 8 min | 30 h | 5 d | 10 d |
| **720p @ 24 fps** | 24 × 1.7 s | 0.7 min | 2.5 min | ~10 h | ~1.7 d | ~3.4 d |

> "1 d" = 1 day, "4 d" = 4 days. These are pessimistic — the auto-config gains
> are not yet measured on a long real run; could be 15–25% faster.

## Decision table — what to run

| Task | Run this | Why |
|---|---|---|
| **Test the install** (~10 s clip) | `./stereocrafter convert sample.mp4` | Auto-detects everything; produces both SBS + anaglyph in a few minutes. |
| **Music video** (3–5 min, 1080p) | `./stereocrafter convert clip.mp4` | Defaults are fine; ~20–35 min wall. |
| **TV episode** (~25 min, 1080p) | `./stereocrafter convert ep.mp4` | ~1.5–2 h wall on 2× 3090. Set `STEREOCRAFTER_KEEP_CHUNKS=1` if you want to inspect intermediates. |
| **Full 15-min anime ep** | same | ~24 h overnight run; check `output/.parts_*/`*.log on slow days. |
| **4K source ≥ 5 min** | Pre-downscale to 1080p, then convert. See below. | 4K inpaint is 4× slower for no extra 3D fidelity (depth is at 1024×576 internally either way). |
| **Single-GPU benchmark** | `STEREOCRAFTER_GPU=0 ./stereocrafter convert ...` | Pins to GPU 0, leaves GPU 1 free. |
| **Two videos at once** | run twice with `STEREOCRAFTER_GPU=0` and `=1` | Each gets a full 24 GB card. |
| **Tight on VRAM** (other apps using GPU 0) | `STEREOCRAFTER_GPU=1 ./stereocrafter convert ...` | Or let auto-config pick smaller chunks. |
| **OOM mid-run** | nothing — wrapper auto-halves the failing chunk up to depth 3 | If it gives up after 3 levels, manually set `STEREOCRAFTER_CHUNK_FRAMES=64` or lower. |

## Pre-downscale recipe (4K → 1080p)

For movie-length 4K material, downscale first — same 3D quality, ~4× faster:

```bash
ffmpeg -i input_4k.mp4 -vf scale=1920:-2 -c:v libx264 -preset slow -crf 18 \
       -c:a copy input_1080p.mp4
./stereocrafter convert input_1080p.mp4
```

Optionally upscale the resulting SBS back to 4K:

```bash
ffmpeg -i output/input_1080p_sbs.mp4 -vf scale=7680:-2 -c:v libx264 \
       -preset slow -crf 18 -c:a copy output/input_4k_sbs.mp4
```

(Or use Real-ESRGAN / Topaz for AI upscale.)

## Quality knobs

| Setting | Effect | When to change |
|---|---|---|
| `STEREOCRAFTER_TILE_NUM=1` | Process inpaint full-frame; ~30% faster stage 2 | Only if free VRAM ≥ 22 GB and chunk_frames ≤ 64 (fits in tile_num=1) |
| `STEREOCRAFTER_TILE_NUM=3` (default for tight VRAM) | Smaller tiles, ~40% slower | Auto-set when free VRAM < 18 GB |
| `STEREOCRAFTER_CHUNK_FRAMES=128` | Smaller chunks; smaller per-chunk VRAM | Manual override when auto-config OOMs on first chunk |
| `STEREOCRAFTER_CHUNK_FRAMES=512+` | Bigger chunks; fewer model reloads | Free VRAM ≥ 20 GB and want every drop of throughput |

## What can't be optimised in the wrapper

These need code changes in StereoCrafter itself:

- **Anime-specific**: line-art handles depth poorly. The model is photo-trained.
  No wrapper-side fix.
- **`max_res > 1024`**: would require larger chunks; defeated by stage 2 cost.
- **fp8 / int8**: not in the upstream model.
- **Replacing inpaint with SVD-Lightning (4-step instead of 25)**: ~6× faster
  stage 2 but quality drop unknown. Would need a separate model fine-tune.

## Hardware speedup options (no wrapper change)

- **Add a 3rd GPU**: linear speedup (the wrapper splits N ways for any N).
- **Replace 3090 with 5090** (~2.5× faster per card on diffusion workloads, more
  VRAM enables larger chunks): ~2.5–3× total speedup.
- **Cloud A100/H100 ×8 for a single job**: split the input 8 ways, ~4× faster
  than 2× 3090 (sub-linear because of model-load overhead per split).
