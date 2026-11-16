# HID Touch: no force on launch

Warm-up 2026-11-14; motor self-check and automatic re-open 2026-11-15.

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
  driver. USB port/hub, CPU, configs, calibration, power and heat were ruled
  out.
- The Dec 2025 driver (`Dev_Dec 10 2025`) behaves the same (measured
  2026-11-15, table below).
- The HID Touch reports firmware version 0.00 (the COM Touch reports 41.00),
  so its firmware cannot be compared. No firmware updater ships with the
  Linux driver.

Driver or firmware, or both: not separable from here. Worth reporting to
3D Systems with the numbers below.

## Fix, part 1: warm-up

`ambf_hdPhantom_hid_warmup.patch` (ambf commit `38c2b1c1`, branch
`fivrs/hid-touch-warmup` on the home machine): before the real open,
`chai3d/extras/hdPhantom/hdPhantom.cpp` runs 20 quick
`hdInitDevice`/`hdDisableDevice` cycles (~1.2 s, no servo loop, no force).
`FIVRS_HAPTIC_WARMUP=0` turns it off; another number sets the count.

On a bench program this is enough (0 dead in 28+). **In the real simulator it
is not**: launches with the warm-up were still dead 3-7 times in 40.

## Fix, part 2: motor self-check and re-open

`ambf_hdPhantom_selfcheck.patch` (applies on top of part 1; adds
`fivrs_selfcheck.h`). Inside the real servo loop, per device, ~0.6 s after
the servo starts:

- waits for the stylus to be still for 0.25 s (< 0.3 mm);
- taps it with +0.8 N up for 0.10 s (added to what CHAI3D commands, cut at
  100 mm/s);
- moved > 0.1 mm = **ok**; no move = **dead**.

A dead device is closed and re-opened by a helper thread (servo paused,
scheduler stopped, `hdDisableDevice` + `hdInitDevice`, scheduler restarted)
and checked again, up to 3 times. A stylus docked in the inkwell cannot move:
the check waits until it is lifted. A stylus that is never still is tapped
anyway after 20 s and reported **held**, not dead.

Results: `/tmp/fivrs-rtx/haptic_selfcheck.json`
(`"status": "ok" | "recovering" | "pending" | "dead"`), stderr, and
`~/.local/share/fivrs/haptic_health.log` (`[hdPhantom]` lines). The
launcher's Study tab launch status shows a **Force feedback (Touch motors)**
row and pops up "No force feedback" if re-opening did not help.

Settings (environment):

| variable | default | |
|---|---|---|
| `FIVRS_HAPTIC_SELFCHECK` | on | `0` turns the check off |
| `FIVRS_HAPTIC_SELFCHECK_N` | 0.8 | tap force, N (0.1-1.5) |
| `FIVRS_HAPTIC_RECOVER` | 3 | re-open tries; `0` = only report |
| `FIVRS_HAPTIC_SELFCHECK_RECHECK` | none | e.g. `10,20`: tap again at those seconds (testing) |
| `FIVRS_HAPTIC_SELFCHECK_FILE` | `/tmp/fivrs-rtx/haptic_selfcheck.json` | |
| `FIVRS_HAPTIC_HEALTH_LOG` | `~/.local/share/fivrs/haptic_health.log` | `""` = off |

## Measurements (home machine, both Touches connected, styluses free on the desk)

Drill Touch dead at launch. The COM suction Touch passed every check in every
run.

| run | driver | warm-up | re-open | dead |
|---|---|---|---|---|
| bench (libhdPhantom, no AMBF) | Sep 2024 | off | off | 6/10 |
| bench | Dec 2025 | off | off | 11/20 |
| bench | Sep 2024 | on | off | 0/28 |
| bench | Sep 2024 | off | on | 0/12 final (10 recovered) |
| real `ambf_simulator` launch | Sep 2024 | on | off | 7/40 |
| real launch | Dec 2025 | on | off | 5/40 |
| real launch | Sep 2024 | off | on | 0/15 final (4 recovered) |
| real launch | Sep 2024 | on | on | **0/40 final** (3 recovered, 1 re-open each) |

- Every dead launch was dead at the first check (~2.4 s after load). In 72
  launches re-checked at 10 s and 20 s, none that passed was dead later.
- A re-open always fixed it on the first try.
- An earlier re-open crashed AMBF (the haptic monitor feed read the device
  while the scheduler was stopped). Fixed by pausing the servo
  (`servoStarted = false`) 300 ms before stopping the scheduler; 0 crashes in
  55 launches since.

Launched with both styluses docked (4 launches): both report **docked** and
the check waits, as designed. Not yet seen: lifting a docked stylus out
during a launch and the tap that follows.

## Install (each machine)

```
haptics/install_hdphantom_warmup.sh            # ambf next to this repo
haptics/install_hdphantom_warmup.sh /path/to/ambf
```

It applies both patches, builds hdPhantom and installs
`~/.local/lib/libhdPhantom.so`. **That is the copy the launcher loads**
(`scripts/setup_haptic_env.sh` puts `~/.local/lib` first); `/usr/local/lib`
is only the fallback, and the script prints the sudo command for it. Undo:
copy the `.bak-<date>` file back.

Check a running simulator:

```
grep libhdPhantom /proc/$(pgrep -x ambf_simulator)/maps
cat /tmp/fivrs-rtx/haptic_selfcheck.json
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
