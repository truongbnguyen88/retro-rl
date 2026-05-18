# Environment notes

## stable-retro on macOS Apple Silicon — fixed via source-build patch

**Symptom (out of the box).** `pip install stable-retro` succeeds but
`import retro` fails with:

```
dlopen(.../stable_retro/_retro.cpython-312-darwin.so):
  (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64'))
```

**Root cause.** Two compounding upstream issues:

1. **PyPI wheels are mislabeled.** `stable_retro-1.0.0-cp312-cp312-macosx_11_0_arm64.whl`
   (and `0.9.9`) are tagged `arm64` but contain x86_64 native binaries.
   Verified via `lipo -info`.
2. **Vendored zlib breaks on Apple Silicon.** Multiple libretro cores
   (`pce_fast`, `genesis_plus_gx`, `32x`, `n64`, …) ship copies of an ancient
   zlib whose `zutil.h` does:
   ```c
   #if defined(MACOS) || defined(TARGET_OS_MAC)
   ...
   #      ifndef fdopen
   #        define fdopen(fd,mode) NULL /* No fdopen() */
   ```
   `TARGET_OS_MAC` is defined on modern Darwin/macOS too, not just Classic
   Mac OS. When the system `<stdio.h>` later declares
   `FILE *fdopen(int, const char *)`, the preprocessor substitutes `NULL`
   and the SDK header fails to parse.

**Fix.** [`scripts/install_stable_retro_macos.sh`](../scripts/install_stable_retro_macos.sh)
clones stable-retro from GitHub, patches every `zutil.h` to also exclude
`__APPLE__` from the broken redef, and source-builds + installs.

```bash
brew install cmake pkg-config
source .venv/bin/activate
./scripts/install_stable_retro_macos.sh
```

After this, `import retro` works and the Airstriker smoke test passes.
Verified on macOS 26 (Tahoe) + Apple Silicon + Python 3.12.

**Confirming the install.** From the venv:

```bash
python -c "import retro; print(retro.data.get_romfile_path('Airstriker-Genesis-v0'))"
pytest tests/test_env.py -v   # expect green
```

## ROM acquisition

**Airstriker (Genesis)** — the default target — ships with stable-retro at
`<site-packages>/stable_retro/data/stable/Airstriker-Genesis-v0/rom.md`. It's
freely distributable; nothing to download. The repo's `configs/env.yaml`
points at it out of the box.

**User-supplied ROMs.** If you want to point this codebase at a game whose ROM
you legally own, dump the cartridge with hardware like the Retrode 2 or
INL Retro Dumper, then:

```bash
cp /path/to/your/ROM.ext roms/
python -m retro.import roms/
python -c "import retro; print(retro.data.get_romfile_path('<game-id>'))"
```

Update `configs/env.yaml` with the new `game`, `state`, and `info_keys` (see
the integration's `data.json` at
`<site-packages>/stable_retro/data/stable/<game>/data.json`).

## Reward shaping caveats

The shaping function in
[src/retro_rl/env/reward_shaping.py](../src/retro_rl/env/reward_shaping.py)
reads from the `info` dict produced by stable-retro's integration. Per-game
key names live in `configs/env.yaml` under `info_keys:`. On the first step,
any missing key emits a one-shot WARNING:

```
WARNING retro_rl.env.reward_shaping: info key 'x_pos' not present; corresponding term will be zero.
```

That's expected for Airstriker (vertical scroll, no `x_pos` variable). For
a new game, watch these warnings and fix the mapping in `configs/env.yaml`.

To discover the actual keys, run a few steps via
`scripts/play_random.py --config configs/env.yaml` and print `info.keys()`,
or inspect the integration's `data.json` directly.

## Known minor issue: pyglet teardown on macOS

`env.close()` triggers pyglet's Cocoa event-loop teardown which has a
known `AttributeError` in pyglet 1.5.x (the version stable-retro pins).
Wrapping the call in `try: ... except AttributeError: pass` is harmless —
the OS reaps the subprocess at process exit. This is what
`test_make_env_airstriker_smoke` does.

Will resolve naturally when stable-retro relaxes its pyglet pin to 2.x.
