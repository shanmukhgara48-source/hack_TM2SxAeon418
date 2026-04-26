"""
Quick test runner for plan_imaging.py
Run: python test_plan.py
"""

import sys
import plan_imaging

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def tle2_cs(line):
    return sum(int(c) if c.isdigit() else (1 if c == '-' else 0) for c in line) % 10


def make_tles(satnum, inc, raan, argp, ma, mm, epoch_day, revnum=1):
    l1 = f'1 {satnum:05d}U 23001A   {epoch_day}  .00000000  00000-0  00000-0 0  9991'
    body = f'2 {satnum:05d} {inc:8.4f} {raan:8.4f} 0001000 {argp:8.4f} {ma:8.4f} {mm:11.8f}{revnum:05d}'
    return l1, body + str(tle2_cs(body))


SC = {
    'fov_deg': 2.0,
    'shutter_duration_s': 0.120,
    'max_off_nadir_deg': 60.0,
    'max_wheel_momentum_mNms': 30.0,
}

failures = 0


import re

def _score(r):
    m = re.search(r'S_claim=([0-9.]+)', r.get('notes', ''))
    return float(m.group(1)) if m else 0.0


def check(label, result, min_obj=None):
    global failures
    r = result
    ok = True
    reasons = []
    obj = _score(r)

    if len(r['attitude']) != 36001:
        ok = False; reasons.append(f"att={len(r['attitude'])} (want 36001)")
    for s in r['shutter']:
        if abs(s['duration'] - 0.120) > 1e-9:
            ok = False; reasons.append(f"duration={s['duration']}")
            break
    ts = [s['t_start'] for s in r['shutter']]
    if ts != sorted(ts):
        ok = False; reasons.append("shutter not sorted")
    if min_obj is not None and obj < min_obj:
        ok = False; reasons.append(f"obj={obj:.3f} < {min_obj}")

    status = PASS if ok else FAIL
    print(f"  [{status}] {label}")
    print(f"         obj={obj:.3f}  frames={len(r['shutter'])}  att={len(r['attitude'])}")
    if not ok:
        print(f"         REASONS: {', '.join(reasons)}")
        failures += 1


# ── Valid cases ────────────────────────────────────────────────────────────────
print("\n=== Valid inputs ===")

l1, l2 = make_tles(99001, 51.64, 135.0, 90.0, 30.0, 15.50, '23001.42')
r1 = plan_imaging.plan_imaging(
    l1, l2,
    [{'lat_deg': 44, 'lon_deg': 8}, {'lat_deg': 47, 'lon_deg': 8},
     {'lat_deg': 47, 'lon_deg': 12}, {'lat_deg': 44, 'lon_deg': 12}],
    '2023-01-01T10:00:00Z', '2023-01-01T10:12:00Z', SC)
check("Case 1 — Northern Italy (expect ~0.93, 76 frames)", r1, min_obj=0.5)

l1, l2 = make_tles(99002, 97.4, 100.0, 90.0, 0.0, 15.19, '23001.50')
r2 = plan_imaging.plan_imaging(
    l1, l2,
    [{'lat_deg': 43, 'lon_deg': 1}, {'lat_deg': 45, 'lon_deg': 1},
     {'lat_deg': 45, 'lon_deg': 5}, {'lat_deg': 43, 'lon_deg': 5}],
    '2023-01-01T12:00:00Z', '2023-01-01T12:12:00Z', SC)
check("Case 2 — Southern France (expect ~1.07, 44 frames)", r2, min_obj=0.5)

l1, l2 = make_tles(99003, 97.4, 125.0, 90.0, 130.0, 15.19, '23001.60')
r3 = plan_imaging.plan_imaging(
    l1, l2,
    [{'lat_deg': 48.8, 'lon_deg': 2.2}, {'lat_deg': 49.0, 'lon_deg': 2.2},
     {'lat_deg': 49.0, 'lon_deg': 2.5}, {'lat_deg': 48.8, 'lon_deg': 2.5}],
    '2023-01-01T14:00:00Z', '2023-01-01T14:12:00Z', SC)
check("Case 3 — Paris AOI  (expect ~1.31, 4 frames)", r3, min_obj=0.5)

# ── Crash-proofing cases ───────────────────────────────────────────────────────
print("\n=== Crash-proofing (any valid structure = pass) ===")

l1_good, l2_good = make_tles(99001, 51.64, 135.0, 90.0, 30.0, 15.50, '23001.42')
AOI = [{'lat_deg': 44, 'lon_deg': 8}, {'lat_deg': 47, 'lon_deg': 12}]

check("Empty AOI []",
    plan_imaging.plan_imaging(l1_good, l2_good, [], '2023-01-01T10:00:00Z', '2023-01-01T10:12:00Z', SC))

check("None AOI",
    plan_imaging.plan_imaging(l1_good, l2_good, None, '2023-01-01T10:00:00Z', '2023-01-01T10:12:00Z', SC))

check("AOI with one garbage vertex",
    plan_imaging.plan_imaging(l1_good, l2_good,
        [{'lat_deg': 44, 'lon_deg': 8}, 'GARBAGE', {'lat_deg': 47, 'lon_deg': 12}],
        '2023-01-01T10:00:00Z', '2023-01-01T10:12:00Z', SC))

check("Non-numeric lat/lon",
    plan_imaging.plan_imaging(l1_good, l2_good,
        [{'lat_deg': 'x', 'lon_deg': 8}, {'lat_deg': 47, 'lon_deg': 12}],
        '2023-01-01T10:00:00Z', '2023-01-01T10:12:00Z', SC))

check("Garbage TLE strings",
    plan_imaging.plan_imaging('BAD LINE 1', 'BAD LINE 2', AOI,
        '2023-01-01T10:00:00Z', '2023-01-01T10:12:00Z', SC))

check("Invalid timestamps",
    plan_imaging.plan_imaging(l1_good, l2_good, AOI, 'not-a-date', 'also-bad', SC))

check("Degenerate TLE (zero mean motion)",
    plan_imaging.plan_imaging(
        '1 99002U 23001A   23001.50  .00000000  00000-0  00000-0 0  9994',
        '2 99002   0.0000   0.0000 0000000   0.0000   0.0000  0.00000000    07',
        AOI, '2023-01-01T10:00:00Z', '2023-01-01T10:12:00Z', SC))

# ── Summary ────────────────────────────────────────────────────────────────────
print()
if failures == 0:
    print("\033[92mAll tests passed.\033[0m")
else:
    print(f"\033[91m{failures} test(s) FAILED.\033[0m")
    sys.exit(1)
