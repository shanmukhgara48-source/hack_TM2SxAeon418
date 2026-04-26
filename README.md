# BOOMSCAN

**Autonomous Earth-Observation Imaging Planner**
*Team 404 — Lost in Space Track*
Gara Shanmukh Sai · Rishika Reddy · Taniskha Reddy · Shaswath Rekham

BOOMSCAN is a deterministic onboard imaging planner for small Earth-observation
satellites. Given a TLE, an AOI polygon, a spacecraft configuration, and a
12-minute imaging window, it produces a valid attitude trajectory and shutter
schedule that maximizes AOI coverage while respecting the smear, off-nadir, and
reaction-wheel hard gates.

---

## Executive Summary

Commercial EO operators monetize valid ground coverage, not commanded images. A
frame that violates the smear gate, exceeds the off-nadir limit, or saturates a
reaction wheel produces no sellable area and wastes an irrecoverable pass.
BOOMSCAN addresses this scheduling problem with a compact numerical planner
that selects *where* and *when* to image, then emits stop-and-stare attitude
commands engineered to keep each 120 ms exposure stable.

**One-sentence strategy:** SGP4-driven greedy AOI mosaic, followed by
stop-and-stare frame execution under explicit smear, wheel-momentum, and
off-nadir gates.

**Local mock-harness result:**

```text
S_total       = 1.1583   (weighted across the three cases)
Frames kept   = 72 / 72  (zero rejections)
Q_smear       = 1.0000   on all three cases
```

> Scores reported here are from the supplied local mock simulator. Basilisk was
> not available on our machine, so the controller-dynamics leaderboard score is
> not locally verified. See [§4 Honesty Note](#4-honesty-note) for the specific
> ways the mock and real harnesses can diverge.

---

## 1. Problem

During a LEO pass, the satellite, AOI footprint, and feasible pointing geometry
all change continuously. The customer is a constellation tasking or
mission-operations engineer who needs reliable coverage without violating
spacecraft limits. They would pay for BOOMSCAN because it increases usable AOI
coverage from existing satellite passes — improving revenue per revisit without
adding spacecraft, downlink, or human replanning cost.

The problem has three hard gates that disqualify a frame regardless of its
geometric appeal:

| Gate                | Limit            | Failure mode                           |
| ------------------- | ---------------- | -------------------------------------- |
| Smear during shutter | ≤ 0.05 °/s     | Body rotation blurs the integration    |
| Reaction-wheel momentum | ≤ 30 mNms   | Wheel saturates, attitude control lost |
| Off-nadir angle     | ≤ 60 °           | Frame is rejected as out of envelope   |

A naive nadir-greedy planner can hit the geometric sweet spot and still lose
every frame to the smear gate — this is the trap BOOMSCAN is built to avoid.

---

## 2. Solution

BOOMSCAN ships as a single Python file, [`plan_imaging.py`](plan_imaging.py),
exporting the required `plan_imaging(...)` entry point. It is **not** an ML
system: there is no dataset, training loop, fine-tuning recipe, or model
artifact. The planner is deterministic orbital geometry and constraint-aware
scheduling.

### 2.1 Pipeline

```text
TLE + AOI + pass window
        │
        ▼
  SGP4 propagation @ 50 Hz
        │
        ▼
  AOI grid + visibility / off-nadir screening
        │
        ▼
  Greedy coverage mosaic selection
        │
        ▼
  Slew-aware local ordering
        │
        ▼
  Wheel-momentum & off-nadir validation
        │
        ▼
  Stop-and-stare attitude construction
        │
        ▼
  50 Hz attitude samples + 120 ms shutter windows
```

### 2.2 Key design choice: stop-and-stare

For every accepted frame, BOOMSCAN holds the same body-to-inertial quaternion
*before* shutter open, *throughout* the 120 ms integration, and *briefly after*
shutter close. The grader's SLERP between identical waypoints yields ω ≈ 0
during exposure, which keeps the smear metric at its ceiling
(`Q_smear = 1.0`). Slews between frames are unconstrained by the smear gate and
are sized only against the wheel-momentum budget.

### 2.3 Margin policy

The planner does not plan exactly at the spec limits — it leaves explicit
margin for real controller dynamics:

| Limit       | Spec        | Plan target      | Margin              |
| ----------- | ----------- | ---------------- | ------------------- |
| Off-nadir   | 60 °        | 58 °             | 2 ° below spec      |
| Per-wheel H | 30 mNms     | 25 mNms eff.     | 5 mNms below spec   |
| Smear       | 0.05 °/s    | ~0 °/s by design | full budget unused  |

---

## 3. Measurement

Evaluation was performed with the organizer harness in `--mock` mode:

```bash
python run_evaluation.py \
  --submission /Users/shanmukhsai/Desktop/Boomscan/plan_imaging.py \
  --all --mock
```

The structural-stub baseline returns a valid no-imaging schedule and therefore
scores zero. BOOMSCAN improves from zero to full scored imaging across all
three geometries.

### 3.1 Headline scores

| Submission       | Case 1 | Case 2 | Case 3 | Weighted `S_total` |
| ---------------- | -----: | -----: | -----: | -----------------: |
| Structural stub  | 0.0000 | 0.0000 | 0.0000 |             0.0000 |
| **BOOMSCAN**     | **1.0194** | **1.1158** | **1.2824** |     **1.1583** |

### 3.2 Score components

| Case | Geometry            | Coverage `C` | `η_E`  | `η_T`  | `Q_smear` | Frames kept |
| ---- | ------------------- | -----------: | -----: | -----: | --------: | ----------: |
| 1    | Direct overpass     |       0.8451 | 0.4666 | 0.8953 |    1.0000 |     33 / 33 |
| 2    | ~30° offset         |       0.9105 | 0.5338 | 0.9203 |    1.0000 |     32 / 32 |
| 3    | ~60° offset (hard)  |       0.9649 | 0.9297 | 0.9664 |    1.0000 |       7 / 7 |

### 3.3 Rejection counts

| Reject reason     | Case 1 | Case 2 | Case 3 |
| ----------------- | -----: | -----: | -----: |
| Wheel saturation  |      0 |      0 |      0 |
| Smear exceeded    |      0 |      0 |      0 |
| Off-nadir         |      0 |      0 |      0 |
| Ray miss          |      0 |      0 |      0 |

---

## 4. Honesty Note

We report mock-harness numbers verbatim and flag the places where they may
diverge from a real-harness run. Judges should weight the limitations here
when interpreting Case 3 in particular.

- **Coverage metric in the mock vs real harness.** The local mock harness
  credits a cell as covered using a centroid-in-FOV test. A stricter
  polygon-intersection coverage metric in the real harness would lower `C` —
  particularly on Case 3, where 7 frames covering 96.5 % of cells is plausible
  under centroid coverage but is unlikely to survive exact intersection.
- **Controller dynamics.** The mock simulator perfectly tracks the commanded
  attitude. Basilisk introduces lag and overshoot, which is most likely to
  affect edge-of-envelope Case 3 frames near the 60° boundary. Our 2° off-nadir
  and 5 mNms wheel margins are the buffer against this, but they are not a
  substitute for a Basilisk run.
- **Case 3 frame count.** Seven kept frames is small by design: the planner
  refuses targets that would cross the off-nadir margin or break the wheel
  budget. The high `C` reflects mock-harness scoring, not a claim that 7 frames
  cover 96 % of the AOI under any reasonable real-world coverage metric.

We have therefore made no claim about the Basilisk leaderboard score and
report only what the local mock measured.

---

## 5. Case 3 Strategy (60° offset)

Case 3 is weighted 40 % of `S_total` and contains AOI corners that are
physically unreachable inside the off-nadir envelope. Treating it identically
to Cases 1–2 is a known failure mode.

BOOMSCAN's Case 3 adaptations:

1. **Reachable-anchor planning.** The visibility window is anchored on the AOI
   cell with the *minimum* off-nadir angle over the pass, not on the centroid.
   This keeps planning inside the reachable cone when the centroid sits at the
   limit.
2. **Tightened imaging window.** The pass window is trimmed to the interval
   where any portion of the AOI is below the 58 ° margin.
3. **Accepted partial coverage.** Frames that would require pointing past 58 °
   or that would break the wheel budget on the approach slew are dropped
   rather than scheduled and lost to a hard-gate rejection.
4. **Frame parsimony.** Case 3 keeps only 7 frames in the mock run. Each frame
   is wheel-validated end-to-end before commit.

---

## 6. Compute Footprint

BOOMSCAN is intentionally lightweight and auditable. It is suitable for ground
tasking and plausible for onboard pass planning on smallsat-class avionics.

| Dimension              | Implementation                                  |
| ---------------------- | ----------------------------------------------- |
| Form factor            | Single Python file                              |
| Required dependencies  | `numpy`, `sgp4`                                 |
| Optional dependency    | `shapely` (pure-numpy fallback if absent)       |
| Model / data footprint | None                                            |
| Runtime behavior       | Deterministic, CPU-only                         |
| External access        | None during planning                            |
| Planner budget         | Below the 120 s/case contest budget locally     |
| Memory profile         | 50 Hz pass arrays, AOI grid, attitude trajectory |

The design avoids stochastic search and ML inference, so behavior is
reproducible and easy to validate. The dominant work is SGP4 propagation, AOI
visibility scoring, and quaternion trajectory construction.

---

## 7. Limitations and Future Work

The primary remaining risk is the gap between mock attitude tracking and real
Basilisk controller dynamics; Case 3 is the most important real-simulation
validation target.

Known limitations:

- Greedy mosaic selection is not globally optimal; a beam search or small
  ILP over candidate frames would likely improve `S_total` on Cases 1–2.
- Slew effort is managed through heuristics rather than a full optimal-control
  solve.
- The pure-numpy area fallback is robust but less precise than exact polygon
  clipping.
- The planner does not model closed-loop ACS overshoot before scheduling
  edge-of-envelope frames — it relies on margin instead.

With another day we would add a closed-loop ACS margin model, replace greedy
selection with a beam search over candidate frames, and tighten the pure-numpy
area intersection so coverage scoring no longer depends on optional geometry
libraries.

---

## 8. Repository Layout

```text
Boomscan/
├── plan_imaging.py        # Submission entry point (single-file planner)
├── test_plan.py           # Local unit tests
├── PRESENTATION_2MIN.md   # 2-minute demo script
├── README.md              # This file
└── Lost-In-Space/         # Organizer harness + cases
```

---

## 9. Status

| Field                          | Value             |
| ------------------------------ | ----------------- |
| Local mock `S_total`           | **1.1583**        |
| Frame rejections (mock)        | **0 / 72**        |
| `Q_smear` (all cases)          | **1.0000**        |
| Basilisk leaderboard score     | *not locally verified* |
| Submission file                | [`plan_imaging.py`](plan_imaging.py) |
| Demo script                    | [`PRESENTATION_2MIN.md`](PRESENTATION_2MIN.md) |

For questions or bug reports, please open an issue or contact the team.
