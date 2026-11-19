# Verification & 100% Playtest Checklist (Splatoon Raiders)

Use both the **Automated Smoke & Regression Suite** (run before every release or code change) and the **Manual Playtest Checklist** (run on real Steam Deck LCD, Steam Deck OLED, and Linux desktop hardware before declaring a release complete).

---

## 1. Automated Verification Suite

Run from `sr-recomp/` after building (`PATH="$PWD/scratchpad/bin:$PATH" ninja -C build/obj`):

### A. Static & Tooling Regression Tests
```sh
uv run python3 tools/docs/check.py
uv run python3 tools/git/cfg_rows.py
uv run python3 tools/analisis/muerto.py
uv run pytest tests
```

Expected: `docs/check.py` exits `0`, `cfg_rows.py` exits `0`, and `pytest tests` passes all tests (`51 passed` covering defaults, shader control flow, issue parsing, setup/update paths, Switch 2 BLE protocol, and i18n formatting).

### B. Runtime Self-Tests & Sub-System Probes
Run with isolated libraries (`LD_LIBRARY_PATH="$PWD/build/lib"`):

```sh
for flag in \
  --selftest \
  --probe-cfg \
  --probe-loader \
  --probe-kernel \
  --probe-fs \
  --probe-gfx-abi \
  --probe-nvn \
  --probe-nvn-tex \
  --probe-window \
  --probe-vkmem \
  --probe-vkexec \
  --probe-astc; do
  echo "=== $flag ==="
  LD_LIBRARY_PATH="$PWD/build/lib" ./build/mkw "$flag"
done
```

What key probes verify:
- `--selftest`: Guest dispatch, ABI layout, host overrides.
- `--probe-cfg`: `mkw.ini`, `MKW_DEVICE_PROFILE=deck` defaults (`1280x800`, `60 fps`, `low_latency`, `deck` preset, `720p` internal resolution, `85%` render scale, `DYN_RES=1`, `110%` HUD scale), `MKW_BIND_*` validation, gyro/Flick Stick defaults.
- `--probe-fs`: Full RomFS parity + transactional save create/commit/unmount and crash recovery across 6 simulated failure steps (`3` clean rollbacks, `3` finalized commits, `0` mixed saves, `save-backup` rotation).
- `--probe-vkmem` & `--probe-vkexec`: Vulkan memory mirroring (`MKW_LAZY_WB`) and command buffer replay with `0` validation errors.

### C. Scripted 3D Smoke & Perf Regression Check
Run a timed `--play` pass with `MKW_DEVICE_PROFILE=deck MKW_TRACE=perf` and a scratch copy of your test profile:
- Confirm `RomFS ready: ... 1 overlays` (after first run creates `build/cache/resolution-720`).
- Confirm `audio: SDL device ... 48000 Hz stereo float` and `audren: renderer open 48000 Hz`.
- Confirm normal 3D gameplay windows report `interval avg ~16.67 ms (60.0 fps)`, `validation errors 0`, `shader failures 0`, `draws skipped 0`.
- Confirm clean `fs commit save` and subsequent reload without recovery errors.

---

## 2. Manual Playtest & 100% Completion Checklist

Record Hardware (`Deck LCD` / `Deck OLED` / `Desktop`), Build/Commit, Date, and SteamOS/Mesa version for each sign-off pass.

### A. Fresh Boot, Title & Onboarding
- [ ] Cold launch from Steam Game Mode with a brand-new empty `--profile` dir (`MKW_DEVICE_PROFILE=deck`).
- [ ] Confirm splash screen, title screen audio (music & UI SFX at 48 kHz, zero clipping/crackling), and readable text/UI (`HUD_SCALE=110`).
- [ ] Start **New Game**, clear opening cutscene/intro sequence without softlocks, and reach playable control.
- [ ] Verify initial save commits into `<profile>/save/<uid>/` without errors.

### B. Controllers, Prompts, Rumble & Remapping
Verify across Steam Deck built-in controls plus external controllers:
- [ ] **Steam Deck (LCD & OLED)** via Steam Game Mode (60 Hz)
- [ ] **Xbox Series / One** controller (`MKW_BUTTON_ICONS=auto` -> Xbox prompts, south-A)
- [ ] **PlayStation DualSense / DualShock 4** (`MKW_BUTTON_ICONS=auto` -> PS prompts, Cross-A)
- [ ] **Nintendo Switch Pro / Switch 2 Controller** (`MKW_PAD_LAYOUT=auto` -> Nintendo east-A & Nintendo icons)
- [ ] Analog walk/run tilt range vs deadzone (`MKW_STICK_DEADZONE=8`) feels smooth in all 360° directions.
- [ ] Rumble scales cleanly (`MKW_RUMBLE_STRENGTH`, `MKW_RUMBLE_CURVE`) when shooting ink, taking hits, or swimming.
- [ ] Remap buttons (`Controls` UI -> `MKW_BIND_*`) and confirm persistence after restart.

### C. Gyro Aiming & Flick Stick Quality
- [ ] Confirm smooth gyro pitch/yaw aiming (`MKW_MOTION=1`), zero drift after staying still 2 s (`MKW_GYRO_CALIBRATE=1`), and working manual **Recalibrate gyro** button.
- [ ] Test **Stick + Gyro** together (`MKW_GYRO_MODE=auto|native|stick`): right-stick horizontal turns never cancel or fight gyro micro-aiming.
- [ ] Test **Gyro Invert X/Y** (`MKW_GYRO_INVERT_X`, `MKW_GYRO_INVERT_Y`), **Sensitivity** (`0.5`, `1.0`, `2.5`), and **Deadzone** (`0`, `1`, `5`).
- [ ] Toggle **Flick Stick** (`MKW_FLICK_STICK=1`, `MKW_FLICK_TIME=100`): snap 90°/180° flicks and smooth circular sweeps work cleanly alongside gyro aiming.
- [ ] Confirm end-to-end input responsiveness (`10–150 ms`) with no perceptible buffer lag.

### D. Story Segments, Combat-Heavy Areas & End-Game Content
- [ ] **Story progression**: Play through early, mid, and late story missions/chapters; confirm dialog boxes, objectives, scripted events, and boss encounters advance normally.
- [ ] **Combat-heavy stress test**: Heavy ink coverage, multiple enemies, specials, sub-weapons, and explosions sustain **60 fps (`~16.7 ms`)** on Steam Deck (`MKW_PERF_OVERLAY=fps` / `full`) with zero GPU hangs or missing effects.
- [ ] **Loading & transitions**: Area transitions, fast travel/menus, and shader/texture streaming complete quickly without prolonged freezes or audio dropouts.
- [ ] **Cutscenes & video/audio**: Pre-rendered (`libvpx` / `opus`) and real-time cutscenes stay in A/V sync (`16:9` properly framed on `1280x800`).
- [ ] **100% completion pass**: Complete all story levels, optional collectibles/challenges, upgrades/unlocks, final boss, credits, and post-game 100% completion state/save flag.

### E. Save/Load Integrity & Crash Recovery
- [ ] Manual/auto save -> quit cleanly via menu -> relaunch -> verify **Continue** restores exact checkpoint, inventory, collectibles, and 100% flags.
- [ ] Simulate unclean termination (`SIGKILL` during active gameplay, *after* a committed checkpoint) -> relaunch -> confirm `save-tx` recovery restores the last valid save cleanly with `save-backup/` intact.

### F. Graphics Scaling & Presets Verification
- [ ] Verify `MKW_QUALITY=deck` sustains `60 fps` cleanly and remains sharp via `FSR` (`85%` scale + `720p` internal).
- [ ] Toggle live graphics options (`MKW_UPSCALER`, `MKW_SHARPNESS`, `MKW_AA`, `MKW_POST_PROCESS`, `MKW_FPS` `30/40/60`, `MKW_HUD_SCALE`) via the in-game menu (**F1** / menu shortcut) and verify no crashes or corrupted render targets.
- [ ] Confirm offline stub behavior (`MKW_LAN=0`): any online-related menu or check fails gracefully offline without freezing or stalling gameplay.

