# Plan: Uncharted: Legacy of Thieves (PPSA05684) Language screen

Status at start: HEAD `5ccbbff`. The videos run at 45-59 fps. The Language screen runs at 7-12 fps and renders black or blinking, with no options shown.

## 1. Frame rate on the Language screen

The GPU command thread is CPU-bound at about 750 draws per frame. The profile (`KYTY_SAMPLE_GPU=1`, run as1) breaks the time down as follows.

| Share | Where | Fix |
|---|---|---|
| ~30% | Shader resource materialization (per draw) | Memoization is in `5ccbbff`; measure it. |
| ~28% | Buffer uploads (`SynchronizeBuffer`/`UploadCopies`/`ObtainBuffer`), including `RecordGpuBaseline` at 14% | 1a. Record baselines only for pages the game polls. |
| ~20% | `DispatchDirect` → `PrepareBda` re-syncs every mapped buffer | 1b. Sync only the ranges the CPU dirtied since the last sync. |
| ~13% | Draw path (`PrepareDrawRenderState`, `RebindBuffers`, `GetGraphicsPrograms`) | 1c. Cache resolved programs and bindings for repeated identical draw state. |

Each step gets one measured run on the Language screen.

## 2. Black and blinking UI

- 2a. Screen-size (2560x1440) textures get null-bound because their descriptors carry impossible depth and layer counts (for example 1569 layers, 3 GB). Decode those descriptors: they are either a PS5 descriptor variant the emulator misreads, which would need a decoder fix, or real corruption, which would need `KYTY_WATCH_ADDR` to find the writer.
- 2b. The UI shader `a874be3684b65f03` samples a table that mixes R8_UINT, RGBA16_UINT and float textures. Check how the integer entries are read, since integer data sampled as float renders black.
- 2c. Confirm the blink pattern: check whether the black frames correspond to frames with null-bound UI targets.

## Rules

- Tell the user before every launch. Use one run per fix.
- Commit each verified step with a single-line message.
