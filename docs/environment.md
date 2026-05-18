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
pytest tests/test_env.py -v   # expect 17 passed, 1 skipped (Contra ROM-gated)
```

## ROM acquisition

See [README.md](../README.md) for the legality discussion. Short version:

- **Contra (NES)**: dump your own cartridge with a Retrode 2 / INL Retro Dumper.
  No clean alternative.
- **Airstriker (Genesis)**: ships with stable-retro at
  `<site-packages>/stable_retro/data/stable/Airstriker-Genesis-v0/rom.md`,
  freely distributable. Used as the validation stand-in via
  [configs/env-airstriker.yaml](../configs/env-airstriker.yaml).

After placing `Contra.nes` (or any NES ROM matching stable-retro's expected
SHA-1) in `roms/`:

```bash
python -m retro.import roms/
python -c "import retro; print(retro.data.get_romfile_path('Contra-Nes'))"
```

The smoke test [`test_make_env_contra_smoke`](../tests/test_env.py) will pick
it up automatically. Note: in stable-retro 1.x the integration may be named
`Contra-Nes-v0` — verify with `retro.data.list_games()` and update
`configs/env.yaml` if so.

## Reward shaping caveats (to verify post-ROM)

The shaping function in
[src/contra_rl/env/reward_shaping.py](../src/contra_rl/env/reward_shaping.py)
reads from the `info` dict produced by stable-retro's integration. The
expected key names (`score`, `xpos`, `lives`, `stage_clear`) are best-effort
defaults — when you first run training, watch for one-shot warnings like:

```
WARNING contra_rl.env.reward_shaping: info key 'xpos' not present; corresponding term will be zero.
```

That means stable-retro's Contra integration uses a different key name in
its `data.json`. Fix by overriding `info_keys` in `configs/env.yaml`:

```yaml
info_keys:
  score: score_p1     # whatever the integration actually exposes
  x_pos: x
  lives: lives
  stage_clear: cleared
```

To discover the actual keys, run a few steps and print `info.keys()`. The
integration's `data.json` lives at
`<site-packages>/stable_retro/data/stable/<game>/data.json`.

## Known minor issue: pyglet teardown on macOS

`env.close()` triggers pyglet's Cocoa event-loop teardown which has a
known `AttributeError` in pyglet 1.5.x (the version stable-retro pins).
Wrapping the call in `try: ... except AttributeError: pass` is harmless —
the OS reaps the subprocess at process exit. This is what
`test_make_env_airstriker_smoke` does.

Will resolve naturally when stable-retro relaxes its pyglet pin to 2.x.
