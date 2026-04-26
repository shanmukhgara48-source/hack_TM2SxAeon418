# Team 404

**Gara Shanmukh Sai · Rishika Reddy · Taniskha Reddy · Shaswath Rekham**

---

## 1. Problem

Earth-observation smallsat operators — Planet, Satellogic, Iceye, and the long tail of sub-100 kg constellations — uplink ground-planned imaging schedules tens of minutes to hours before each pass. When pass geometry shifts, an AOI is added late, or a reaction wheel approaches saturation, the pre-uploaded schedule cannot adapt and frames are lost. Each missed AOI is unbillable revenue (commercial EO buyers pay per imaged-km²) and consumes an irrecoverable revisit window. The customer is the constellation's tasking engineer; the pain is the gap between fixed pre-pass plans and the dynamic geometry of a real LEO pass over a constrained ACS. **PUSHBROOM** solves onboard mosaic-and-slew tasking: given a TLE, an AOI polygon, a 12-minute pass window, and spacecraft parameters, decide which frames to capture and at what attitude — deterministic, single-pass, under two seconds, no human in the loop. **They would pay because every additional 1% of AOI coverage at the same satellite count is direct margin, and on extreme-geometry passes the difference between the right plan and a naive plan is the entire revenue from that pass.**

## 2. What We Built

PUSHBROOM is a single-file deterministic planner — `plan_imaging.py` — exposing the `plan_imaging(...)` entry point required by the grader. **No ML, no dataset, no fine-tuning**: a numerical optimizer specialized for the Lost-in-Space scoring metric `S_orbit = C·(1 + 0.25·η_E + 0.10·η_T)·Q_smear`.

```
   TLE                                AOI polygon
    │                                      │
    ▼                                      ▼
  SGP4 (TEME) ─► 50 Hz state          144-cell AOI grid
    │                                      │
    └──────────────┬───────────────────────┘
                   ▼
        per-cell visibility windows (off-nadir vs ground track)
                   │
                   ▼
        ┌────────────────────────────────────┐
        │  AUTO-PROFILE FRONT-END            │
        │   min off-nadir < 20°  → C-lite    │
        │   20°–50°              → default   │
        │   > 50°                → Recon r¼  │
        └────────────────────────────────────┘
                   │
                   ▼
   Greedy mosaic   →   Revisit-aware gain   →   2-opt slew reorder
                   │
                   ▼
   Constraint validator   (60° hard gate · wheels ≤ 25 mNms · √2 pyramid)
                   │
                   ▼
   Identical-q triplet at t_settle / t_open / t_close
   → grader SLERP yields ω = 0 over the 120 ms integration
                   │
                   ▼
        36 001-sample attitude trajectory + shutter list
```

The auto-profile front-end picks one of three knob triples by the min-off-nadir-at-midpass bracket. The greedy mosaic scores each candidate frame by area-clipped new-coverage gain minus a slew-distance penalty (slew angle dominates ΔH, which dominates η_E). Revisit-aware bonus values re-imaging at low off-nadir where geometry pays. Bounded 2-opt local search reorders selected frames against the slot grid to minimize cumulative inter-frame slew. Constraint validator filters every event against a 5 mNms-margin wheel envelope and the 60° hard gate. Identical-q waypoint triplets make the grader's SLERP yield ω = 0 throughout the 120 ms integration — verified at 0.000°/s, not asserted.

## 3. Measurement & Results

All numbers below come from `python test_my_submission.py plan_imaging.py` against the official mock harness in `Lost-In-Space/teams_kit/`. Same validator, scorer, gate logic, and configs as the organizers' Basilisk grader. Section 7.4 guarantees only `numpy / scipy / sgp4`; we measure both code paths and lead with the no-shapely number as the realistic grader score.

| Approach | case1 | case2 | case3 | **S_total** |
|---|---|---|---|---|
| Structural stub (no shutters, scores 0 by design) | 0.000 | 0.000 | 0.000 | **0.0000** |
| Internal v5 baseline (greedy only) | — | — | — | **1.1167** |
| **PUSHBROOM v8b — no shapely (grader-realistic)** | **1.0128** | **1.1124** | **1.2824** | **1.1555** |
| PUSHBROOM v8b — with shapely (upper bound) | 1.0194 | 1.1158 | 1.2824 | 1.1583 |

> **Claim:** S_total = **1.1555** on a `numpy / scipy / sgp4`-only environment, +0.0028 if shapely is present.
> **Breaks if:** real Basilisk controller overshoot on case-3 frames exceeds **0.76°** against the 60° hard gate. We have not measured the true overshoot distribution.

Verified empirically across all three cases: max body rate during integration = **0.000°/s**; max off-nadir = **49.0° / 52.0° / 59.24°**; **zero frame discards** under the mock harness; planning runtime **0.5–1.4 s/case** against a 120 s budget. The 0 → 1.12 step is the value of any planner; the 1.12 → 1.155 step is the value of 2-opt slew minimization, revisit-aware scoring, the auto-profile bracket, and the case-3 overshoot margin.

## 4. Orbital Compute Story

| Metric | Value | Notes |
|---|---|---|
| Code size | ~50 KB single `.py` | numpy + sgp4 only; shapely optional and `try/except`-guarded |
| Peak RAM | < 30 MB | numpy arrays for 36 001-sample attitude grid + AOI grid |
| Latency (Apple M-class) | 0.5–1.4 s/case | measured via `time.perf_counter` |
| Latency (RPi CM4-class) | ~5–10 s/case (est.) | 3–5× slowdown vs M-class; well under 120 s budget |
| Latency (rad-hard LEON3 / RAD750, ~200 MHz) | ~60–100 s/case (est.) | within budget; survivability over speed |
| Power | ~5 W active | typical smallsat compute envelope |
| GPU / accelerator | none | CPU-only, deterministic, no model weights ship |

This submission fits a modern smallsat avionics computer (Snapdragon-class or Raspberry Pi CM4-class) with substantial margin, and a rad-hard CPU within the 120 s budget. No retraining, no inference pipeline, no model versioning — the planner is plain numerical Python that ships as one file.

## 5. What Doesn't Work Yet

Each gap below is framed as the next-question we'd carry into a real engagement.

| # | What's broken / unproven | Next question |
|---|---|---|
| 1 | **Case-3 overshoot margin is 0.76°.** We chose r=0.25 over r=0.0 to buy 0.25° of margin. Sensitivity: r=0.0 → S=1.31, r=0.25 → 1.28, r=0.5 → 1.21, r≥0.625 → 0. | What is the true Basilisk controller overshoot distribution, and is r=0.5 (1.0° margin, ΔS_total ≈ −0.012) the correct operating point? |
| 2 | **Between-frame ω is bounded only by the wheel-momentum gate**, not by an explicit rate model under ACS lag. Inside integration ω=0 is structurally guaranteed (measured). | Would an explicit between-frame peak-ω feasibility check — modeling actual controller response, not just the wheel envelope — buy Basilisk robustness without rejecting frames the mock harness accepts? |
| 3 | **Auto-profile uses a single discriminator** (min off-nadir at midpass). | Does adding ground-track angle or AOI swath-crossing duration as a second discriminator unlock the case-2 ridge that all named profiles currently regress? |
| 4 | **Pure-numpy fallback loses ~0.003 in S_total** vs the shapely path. | Can vectorized polygon intersection in pure numpy reach < 1% accuracy loss without exceeding the latency budget? |
| 5 | **Greedy + 2-opt is a local optimum** — case-1's η_E = 0.467 is the lowest of the three. | Does global ILP or beam search over the slot grid recover η_E within the 120 s budget? |
