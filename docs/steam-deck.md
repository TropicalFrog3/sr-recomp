# Steam Deck (LCD & OLED) and Linux Distribution Guide

This guide covers building, configuring, and launching **Splatoon Raiders** (`0400380020056000`, v1.0.0) in **sr-recomp** on Linux desktops and Steam Deck (LCD and OLED, Desktop Mode and Steam Game Mode).

Nothing copyrighted from the game ships in this repository or its releases; you must provide your own legal dump (`exefs/` and `romfs/`, or packed `.nup` / `.nx`).

---

## 1. Dependencies & Building on Linux / SteamOS

System packages required to build from source (see [tools/build/README.md](../tools/build/README.md)):

- `uv`, `clang`, `lld`, `ninja`
- `SDL3`, `Opus`, `libvpx`, `zstd`, `xxHash`, Vulkan headers, and Vulkan Memory Allocator (VMA `3.4.0` header in `scratchpad/vma/vk_mem_alloc.h`)
- CPU with AVX2 (x86-64) or ARMv8.2 (Linux AArch64), and a Vulkan 1.3 driver (`RADV` / Mesa on SteamOS / Linux AMD).

Build from `sr-recomp/`:

```sh
# 1. Lift AArch64 ExeFS -> C (using Splatoon Raiders 1.0.0 patch table)
uv run python recomp/gen.py --exefs game/exefs --out build/generated --jobs 8 --patches recomp/patches-sr-100.txt

# 2. Extract and translate SM86 shaders -> SPIR-V
uv run tools/shaders/extract.py --romfs game/romfs
uv run tools/shaders/translate.py

# 3. Build build/mkw (or ninja -C build/obj for incremental rebuilds)
uv run tools/build/build.py
```

---

## 2. Launching on Linux & Steam Deck Game Mode

### Audio & Shared Library Isolation (`build/lib/`)

`build/lib/` contains symlinks for only the runtime libraries needed by `build/mkw`:
- `libSDL3.so.0`
- `libopus.so.0`
- `libvpx.so.6`

Always launch `./build/mkw` with `LD_LIBRARY_PATH` pointing to `build/lib/` (avoid mixing system PipeWire/JACK client libs into the preload path):

```sh
LD_LIBRARY_PATH="$PWD/build/lib" ./build/mkw \
  --exefs "/path/to/game/exefs" \
  --romfs "/path/to/game/romfs" \
  --profile "/path/to/profile" \
  --play
```

On success, SDL opens a `48000 Hz stereo float` device (`prefill 20 ms`) and `nn::audren` initializes REV15 (`48000 Hz x240`, `voices 96`).

### Adding to Steam / Steam Game Mode

1. Copy the built `sr-recomp/` runtime folder (`build/mkw`, `build/lib/`, `build/spirv/`, `build/mkw.ini`, etc.) and your dumped `exefs/` + `romfs/` onto the Steam Deck (internal NVMe recommended).
2. Switch to **Desktop Mode** -> open Steam -> **Games -> Add a Non-Steam Game to My Library...** -> browse to `build/mkw` (or a small launcher script).
3. In Steam **Properties -> Shortcut**, set:
   - **Start In**: `/path/to/sr-recomp`
   - **Launch Options**:
     ```sh
     LD_LIBRARY_PATH=/path/to/sr-recomp/build/lib MKW_DEVICE_PROFILE=deck %command% --exefs "/path/to/dump/exefs" --romfs "/path/to/dump/romfs" --profile "/home/deck/.local/share/sr-recomp/profile" --play
     ```
4. Note: `MKW_DEVICE_PROFILE=auto` (default) also auto-detects Steam Deck hardware via firmware DMI / `SteamDeck=1` (`runtime/core/base/cfg.c`).
5. Return to **Gaming Mode** and launch the shortcut. Ensure any per-game SteamOS frame rate cap is set to **60 Hz** (or unlocked if allowing `MKW_FPS=60` + `MKW_VK_PRESENT=fifo` to pace frames).

---

## 3. Steam Deck Default Profile (`MKW_DEVICE_PROFILE=deck`)

Defined in [runtime/core/base/cfg.c](../runtime/core/base/cfg.c) (`DECK[]`) and asserted by `--probe-cfg` (`runtime/probes/cfg_probe.c`), activating `deck` sets the following defaults whenever `mkw.ini` / env vars do not override them:

| Key | Deck Default | Purpose |
|---|---|---|
| `MKW_FULLSCREEN` | `1` | Fullscreen output in Game Mode / Gamescope |
| `MKW_WINDOW_SIZE` | `1280x800` | Native Steam Deck 16:10 panel size |
| `MKW_PICTURE_FIT` | `keep` (global default) | Preserves 16:9 image with clean letterboxing on 16:10 |
| `MKW_RESOLUTION` | `720` | Patches `Gyml/Default.game__gfx__parameter__ModuleParam.bgyml` (`Layer3DResolution` & `FrameBufferResolution` -> `720`) |
| `MKW_OPERATION_MODE` | `handheld` | Handheld console mode |
| `MKW_RENDER_SCALE` | `85` | Internal 85% scale (`1088x612`) before FSR 1 (`EASU` + `RCAS`) |
| `MKW_DYN_RES` | `1` | Dynamic resolution headroom under GPU load |
| `MKW_UPSCALER` | `fsr` | AMD FidelityFX FSR 1 upscaling |
| `MKW_SHARPNESS` | `75` | FSR RCAS sharpening |
| `MKW_QUALITY` | `deck` | Balanced preset (`ANISO=game`, `TEX_QUALITY=full`, `PARTICLES=medium`, `POST_PROCESS=medium`, `AA=auto`) |
| `MKW_FPS` | `60` | 60 fps gameplay target (`16.67 ms`) |
| `MKW_VK_PRESENT` | `fifo` | Tear-free swapchain VSync for Gamescope |
| `MKW_FRAME_PACING` | `low_latency` | Minimal queued frames for 10–150 ms input-to-photon latency |
| `MKW_HUD_SCALE` | `110` | Readable in-game host overlay & pause menu scaling on 7" / 7.4" screens |
| `MKW_EARLY_READ` / `MKW_SUBMIT_CATCHUP` / `MKW_SPLIT_DRAWS` | `0` | Avoids per-submit stalls & sync bottlenecks with lazy GPU mirror write-back (`MKW_LAZY_WB=1`) |

### Observed Performance & Latency Measurements

Measured on the Linux Vulkan test host (`MKW_DEVICE_PROFILE=deck`, scripted 3D gameplay under `MKW_TRACE=perf`):
- **Lazy GPU Mirror Write-Back (`MKW_LAZY_WB`, default `1`)**: Reduced CPU write-back stalls from `55.721 MiB/frame` (`gpu avg 36.40 ms` / ~10 fps) to `0.014 MiB/frame`, bringing `interval avg` to **16.65–16.68 ms (`60.0 fps`)**, `0` Vulkan validation errors, `0` shader failures, `0` skipped draws.
- **`MKW_RESOLUTION=720` RomFS Overlay**: Reduced `gpu avg` from `3.84–4.35 ms` (`original` `1080`) down to **`2.07–2.73 ms`** (`~38%` GPU savings on top of `FSR` `85%`), while caching the 788-byte overlay so subsequent boots mount RomFS in **0.07 s**.
- **End-to-End Pipeline Latency**: Inside each `16.67 ms` frame, guest simulation submits around `+5–8 ms` after the previous present and Vulkan command recording completes in `~9–11 ms` (`replay avg`) with `FRAME_PACING=low_latency` and `render thread blocked: pacing 0% fence 0%`, keeping display pipeline queuing to 1–2 frames (`~17–34 ms` + hardware USB/IMU/panel latency, within the 10–150 ms target window).

*(Physical Steam Deck LCD and OLED thermal/field verification should be confirmed on-device using [docs/playtest-checklist.md](playtest-checklist.md).)*

---

## 4. Controller Support, Gyro Aiming, Flick Stick & Steam Input

Configured in **F1 / Pause Menu -> Controls** (`runtime/ui/menu/settings_tabs.cpp`, `runtime/input/input.c`, `runtime/input/motion_fusion.c`):

### Controllers, Layouts, Prompts & Remapping
- **Supported Controllers**: Steam Deck integrated controls, Xbox, PlayStation DualShock/DualSense, Nintendo Switch Pro / Joy-Con / Switch 2 BLE (`MKW_SWITCH2`), plus keyboard/mouse.
- **Steam Input Compatibility**: In Steam Game Mode, use a standard **Gamepad** or **Gamepad with Gyro** layout. Both SDL native IMU motion (`MKW_GYRO_MODE=native`) and Steam Input right-stick/mouse translation (`MKW_GYRO_MODE=stick`, automatic fallback under `MKW_GYRO_MODE=auto`) work without conflicting with manual right-stick camera movement. Avoid mapping the *same* gyro signal simultaneously to both raw IMU and Right Stick inside Steam Input (or select `MKW_GYRO_MODE=native` when passing raw IMU through).
- **Face Button Layouts & Prompts**:
  - `MKW_PAD_LAYOUT=auto|nintendo|xbox` (`auto` chooses Nintendo east-A for Nintendo/GuliKit pads and Xbox south-A otherwise)
  - `MKW_BUTTON_ICONS=auto|nintendo|xbox|playstation|keyboard`
- **Sticks & Rumble**:
  - `MKW_STICK_DEADZONE`: `0..40` % (default `8`) with radial rescaling
  - `MKW_RUMBLE_STRENGTH`: `0..300` % (default `130`), `MKW_RUMBLE_CURVE`: `0.2..1.5` (default `0.6`), `MKW_RUMBLE_TRIGGERS`: `0|1`
- **Full Remapping**: Rebind every Switch action in the Controls UI or via `MKW_BIND_<ACTION>` (`pad:<name>`, `key:<name>`, `none`).

### Gyro Aiming, Calibration & Flick Stick
- `MKW_MOTION` (`0|1`, default `1`): Enables six-axis sensor fusion (`nn::hid::SixAxisSensorState`).
- `MKW_GYRO_MODE` (`auto|native|stick`, default `auto`): Switches between native Nintendo six-axis orientation and additive right-stick delta mode (great for mixed Steam Input setups).
- `MKW_GYRO_SENS` (`0.2..5.0`, default `1.0`), `MKW_GYRO_DEADZONE` (`0..20` deg/s scaled, default `1`), `MKW_GYRO_INVERT_X` (`0|1`), `MKW_GYRO_INVERT_Y` (`0|1`).
- **Drift & Calibration**: `MKW_GYRO_CALIBRATE` (`0|1`, default `1`) automatically estimates still bias when the controller rests; use the **Recalibrate gyro** button (`motion_calibrate`) in **Controls** while setting the Deck/pad still for ~2 seconds if drift appears.
- **Optional Flick Stick**: `MKW_FLICK_STICK` (`0|1`, default `0`) and `MKW_FLICK_TIME` (`40..250` ms, default `100`). When enabled (`input_flick_stick`), deflecting the right stick past `r >= 0.80` snaps the camera toward the stick angle over `MKW_FLICK_TIME` ms and smoothly sweeps as the stick rotates around the outer rim; gyro aiming stays active simultaneously without fighting stick snaps.

---

## 5. Graphics & Display Options Reference

Available in **Settings -> Screen** and **Settings -> Graphics** (`runtime/core/base/cfg.c`):

- **Resolution & Scaling**: `MKW_RESOLUTION` (`original|540|720|900|1080|1440|2160`), `MKW_RENDER_SCALE` (`50..200`%), `MKW_DYN_RES` (`0|1`), `MKW_UPSCALER` (`fsr|bilinear`), `MKW_SHARPNESS` (`0..100`%), `MKW_PICTURE_FIT` (`keep|stretch`), `MKW_HUD_SCALE` (`75..150`%), `MKW_ASPECT` (`16:9|window|21:9|32:9`).
- **FPS & VSync / Pacing**: `MKW_FPS` (`auto|30|40|60|120|unlimited`), `MKW_VK_PRESENT` (`mailbox|fifo|immediate`), `MKW_FRAME_PACING` (`auto|low_latency|smooth`), `MKW_FRAMEGEN` (`off|auto|2|3|4`; note frame generation adds latency and is off in `deck` default).
- **Quality & Filters**: `MKW_QUALITY` (`low|medium|deck|console|high|ultra|custom`), `MKW_AA` (`auto|off|fxaa`), `MKW_ANISO` (`game|2|4|8|16`), `MKW_TEX_QUALITY` (`full|half|quarter`), `MKW_PARTICLES` (`low|medium|console|high`), `MKW_POST_PROCESS` (`low|medium|console|high`).
- **Title-Specific Notes for Splatoon Raiders**:
  - `MKW_RESOLUTION` patches loose `Gyml/Default.game__gfx__parameter__ModuleParam.bgyml` in-place (preserving exact 788-byte size).
  - `MKW_SHADOW_SIZE` and `MKW_DETAIL_DISTANCE` remain active for MKW patches/archives (`npatch_sets > 0` / `LODDistanceRatio`) and are safe no-ops in Splatoon Raiders (`npatch_sets == 0`, no `LODDistanceRatio` field in RomFS).

---

## 6. Saves, Crash Recovery & Offline Scope

### Transactional Saves & Recovery
Each profile directory (`--profile DIR`) maintains:
- `account.uid` (profile account ID)
- `save/<uid>/` (last cleanly committed save data)
- `save-tx/work/` (active staging workspace during runtime commits)
- `save-backup/<timestamp>/` (automatic snapshots rotated on commit)

If the process crashes mid-write or before a commit, the next startup inspects `save-tx/` and cleanly rolls back unfinished commits or finalizes committed transactions, verified across 6 simulated crash points by `./build/mkw --probe-fs`.

### Online / Multiplayer Scope (Out of Scope / Offline by Default)
Nintendo Switch Online matchmaking, servers, friend lists, play reports, and local wireless (`ldn`) are **out of scope** for `sr-recomp` and intentionally stubbed in [runtime/sdk/ov/ov_net.c](../runtime/sdk/ov/ov_net.c):
- `nn::nifm`, `nn::friends`, `nn::ldn`, `nn::nsd`, and `nn::prepo` return offline-appropriate status codes/empty friend lists so single-player story & offline gameplay proceed cleanly without hanging on network calls.
- Optional host LAN datagram sockets (`MKW_LAN=1` in [runtime/sdk/ov/ov_socket.c](../runtime/sdk/ov/ov_socket.c)) only permit local subnet sockets (`getaddrinfo` returns unavailable; internet routing is blocked). Single-player start-to-finish completion requires `MKW_LAN=0` (default).

---

## 7. Troubleshooting

| Symptom | Cause & Fix |
|---|---|
| No audio / crash on PipeWire init | Ensure launch uses `LD_LIBRARY_PATH="$PWD/build/lib"` so bundled `libSDL3.so.0`, `libopus.so.0`, and `libvpx.so.6` are picked up. Check `logs/mkw.log` for `audio: SDL device ... opened, 48000 Hz`. |
| Black bars on Steam Deck `1280x800` | Expected with `MKW_PICTURE_FIT=keep` (preserves true `16:9` `1280x720` aspect). Switch `PICTURE_FIT` to `stretch` only if you want vertical stretch. |
| Gyro drifts slowly | Leave Deck/controller resting flat for 2 s with `MKW_GYRO_CALIBRATE=1`, or press **Recalibrate gyro** in Controls. |
| Gyro fights right stick / double camera move | In Steam Game Mode, ensure Steam Input isn't sending right-stick/mouse gyro while `MKW_GYRO_MODE=native` is active, or set `MKW_GYRO_MODE=stick`. |
| First-run hitching | First-time `build/pipeline-cache.bin` and `build/cache/resolution-720` generation happens once; subsequent runs mount cached RomFS in `~0.07 s` and reuse compiled pipelines. |
| Suspected save issue after forced power-off | Check `logs/mkw.log` for `host save recovery:` lines. Previous snapshots remain preserved in `<profile>/save-backup/`. |
| Frame drops below 60 fps in heavy combat on Deck OLED/LCD | Confirm `MKW_DEVICE_PROFILE=deck` is active (`MKW_RESOLUTION=720`, `MKW_RENDER_SCALE=85`, `MKW_DYN_RES=1`). Optionally lower `MKW_RENDER_SCALE=75` or `MKW_TEX_QUALITY=half`. Enable `MKW_PERF_OVERLAY=full` or `MKW_TRACE=perf` to diagnose CPU vs GPU bound. |

