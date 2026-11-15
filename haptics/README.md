# HID Touch: no force on launch (fixed 2026-11-14)

## Symptom

The simulator starts with no force feedback on the drill Touch. The drill
moves, buttons and inkwell work, and the haptic monitor shows force output on,
calibration OK, no HD errors. Restarting the simulator sometimes brings force
back. This happened at home and in the lab.

## Cause

Newer HID-type Geomagic Touch (USB `2988:0304`, no `/dev/ttyACM*`) with the
Sep 2024 Touch driver (`Dev_Sep 19 2024`):

- `hdInitDevice()` sometimes leaves the motor amplifiers off. Nothing
  reports it: force output reads on and there is no HD error.
- Whether the motors come on depends on how many times the device has been
  opened **in the same process**. Open 1 is about a coin flip; opens 1-20
  follow a near-identical on/off pattern in every process; from about the
  21st open they always come on.
- The simulator opens the drill once per launch, so every launch was a coin
  flip.
- The older COM/serial Touch (`2988:0302`) works on every open with the same
  driver. The Dec 2025 driver does not fix it (first open still dead, second
  buzzes). USB port/hub, CPU, configs, calibration, power and heat were ruled
  out.

Driver or firmware, or both: not separable from here. Worth reporting to
3D Systems.

## Fix

`ambf_hdPhantom_hid_warmup.patch` (ambf commit `38c2b1c1`, branch
`fivrs/hid-touch-warmup` on the home machine): before the real open,
`chai3d/extras/hdPhantom/hdPhantom.cpp` runs 20 quick
`hdInitDevice`/`hdDisableDevice` cycles (~1.2 s, no servo loop, no force).
`FIVRS_HAPTIC_WARMUP=0` turns it off; another number sets the count.

Measured with a resting (undocked) stylus, a short force pulse and the
encoders, every launch a fresh process:

| | motors on |
|---|---|
| no warm-up, plain OpenHaptics | 14/30, 26/50 |
| no warm-up, through libhdPhantom | 0/10 |
| warm-up, through libhdPhantom | 20/20, then 100/100 |
| warm-up, loaded like the simulator (`setup_haptic_env.sh`, by name) | 50/50 |
| real simulator launch | force |

## Install (each machine)

```
haptics/install_hdphantom_warmup.sh            # ambf next to this repo
haptics/install_hdphantom_warmup.sh /path/to/ambf
```

It patches and builds hdPhantom and installs `~/.local/lib/libhdPhantom.so`.
**That is the copy the launcher loads** (`scripts/setup_haptic_env.sh` puts
`~/.local/lib` first); `/usr/local/lib` is only the fallback, and the script
prints the sudo command for it. Undo: copy the `.bak-<date>` file back.

Check a running simulator:

```
grep libhdPhantom /proc/$(pgrep -x ambf_simulator)/maps
```

## Test tools (`diag/`)

Build with `gcc -O2 -o <name> <name>.c -lHD -lm` (`bridge_test`: `-ldl -lm`).
Only one process can own a Touch: close the simulator, TouchCheckup and the
haptic monitor first.

- `pulse_test <opens> [device|-] [warmups]`: opens the device repeatedly and
  sends a 0.8 N, 100 ms pulse per direction; MOVED / NO MOVE from the
  encoders. The stylus must rest free on the desk: a docked stylus cannot
  move.
- `bridge_test <libhdPhantom.so>`: the same check through libhdPhantom, the
  way CHAI3D uses it.
- `calib_check <seconds> <device...>`: read-only calibration state and
  position.
