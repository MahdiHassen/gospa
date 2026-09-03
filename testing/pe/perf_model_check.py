#!/usr/bin/env python3
"""
perf_model_check.py -- standalone cycle-accurate REPLAY PREDICTOR for the PE
perf CSV written by testing/pe/test_pe.py (test_perf).

For every CSV row it:
  1. regenerates the identical (activation, kernels) pair from the exact
     per-(case, seed) RNG recipe used by test_perf,
  2. routes it through the real functional front end (sw/functional.py) to get
     the union-WSP-gated FIFO-B stream one V2 PE sees,
  3. replays the pe_fetch.sv microarchitecture beat-by-beat (IDLE/KEEP/BANK/
     SKIP per lane, per-lane SRAM banks, curr_slot+1 prefetch, cold-start SKIP
     after arm) to predict:
         consumed  (= len(fifo_b): every entry is eventually admitted)
         macs      (= per-lane WSP hits summed over admitted beats)
         stalled   (= whole-PE SKIP stall beats; b_ready low)
     and compares them against the measured RTL counters.

Usage:
    source .venv/bin/activate   # from the repo root
    python perf_model_check.py <csv-path> [more csvs...]

No cocotb needed; test_pe.py is NOT imported (it needs cocotb at import time).
The few helpers it shares are copied here with a note.
"""

import csv
import os
import random
import re
import sys

# sw/functional.py provides the routing chain (dense_to_csr -> ... -> fifo_b).
_SW_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sw"))
sys.path.insert(0, _SW_DIR)
import functional as fm  # noqa: E402

fm._VERBOSE = False


# ---------------------------------------------------------------------------
# Stream regeneration -- these helpers MIRROR test_pe.py exactly (copied, not
# imported, because test_pe.py imports cocotb at module load).
# ---------------------------------------------------------------------------
def rand_matrix(R, C, density, rng, lo=-9, hi=9):
    """Mirrors test_pe.py:rand_matrix. Draw order matters: per cell the
    ternary evaluates rng.random() first and calls rng.randint only when the
    density test passes."""
    return [[(rng.randint(lo, hi) if rng.random() < density else 0)
             for _ in range(C)] for _ in range(R)]


def _fill_nonzero(mat):
    """Mirrors test_pe.py:test_perf._fill_nonzero (density-1.0 rows must have
    no structural zeros; randint returns 0 ~1/19 of the time)."""
    return [[(v if v != 0 else 1) for v in row] for row in mat]


def regen_case(H, F, num_mults, da, dw, seed):
    """Mirrors the per-(case, seed) draw inside test_pe.py:test_perf."""
    rng = random.Random(0xBEEF ^ (int(da * 100) << 20)
                        ^ (int(dw * 100) << 8) ^ seed)
    act = rand_matrix(H, H, da, rng)
    if da >= 1.0:
        act = _fill_nonzero(act)
    kernels = []
    for _ in range(num_mults):
        ker = rand_matrix(F, F, dw, rng)
        if dw >= 1.0:
            ker = _fill_nonzero(ker)
        if all(v == 0 for r in ker for v in r):
            ker[0][0] = 3        # all-zero fallback, same cell/value as the test
        kernels.append(ker)
    return act, kernels


def route_v2_one_pe(act, kernels, F, H, S):
    """Mirrors test_pe.py:route_v2_one_pe -- APU Stage 1 + Stage 2 for one V2
    PE holding `kernels`, FIFO-B gated by the UNION of the lane WSPs."""
    per_lane_wsp = []
    per_lane_sw = []
    for ker in kernels:
        w, s = fm.kernel_to_sparse(ker)
        per_lane_wsp.append(w)
        per_lane_sw.append(s)
    union = fm.wsp_union(per_lane_wsp)

    values, col_idx, row_ptr = fm.dense_to_csr(act)
    pos = fm.csr_to_positional(values, col_idx, row_ptr)
    pairs = []
    for (axy, x, y) in pos:
        a, px, py, cx, cy = fm.axy_to_pcid(axy, x, y, S)
        pairs.extend(fm.pcid_to_cid_pid(a, px, py, cx, cy, F, H, S))
    pairs = fm.zero_act_filter(pairs)
    fifo_a = fm.route_to_fifo_a(pairs, F)
    fifo_b = fm.broadcast_to_fifo_b(fifo_a, [union])[0]
    return fifo_b, per_lane_sw, per_lane_wsp


# ---------------------------------------------------------------------------
# pe_fetch.sv replay
# ---------------------------------------------------------------------------
class Lane:
    """Architectural state of one pe_fetch lane + its pe_mem SRAM bank.

    slots are PID-ordered: slot i holds the lane's i-th nonzero weight, so
    target_slot(b_pid) = popcount(wsp below b_pid) = index of b_pid in `pids`.
    The bank read register (rd_val/rd_pid) holds the last slot read; we track
    it as (bank_valid, bank_slot) exactly like the RTL does.
    """
    __slots__ = ("pids", "pid_set", "slot_of", "n",
                 "have_curr", "curr_pid", "curr_slot",
                 "bank_valid", "bank_slot")

    def __init__(self, pids):
        self.pids = pids                       # sorted (kernel_to_sparse is PID order)
        self.pid_set = set(pids)
        self.slot_of = {p: i for i, p in enumerate(pids)}
        self.n = len(pids)                     # popcount(wsp_q)
        # rst_n state; wload_done (arm) clears only have_curr/bank_valid.
        self.have_curr = False
        self.curr_pid = 0
        self.curr_slot = 0
        self.bank_valid = False
        self.bank_slot = 0


IDLE, KEEP, BANK, SKIP = "IDLE", "KEEP", "BANK", "SKIP"


def replay_pe_fetch(fifo_b, per_lane_pids):
    """Beat-by-beat replay of pe_fetch.sv over a back-to-back FIFO-B stream
    (the cocotb driver never deasserts b_valid between entries, so every clk
    in the streaming phase is an offered beat carrying the head entry).

    Returns (consumed, macs, stalled)."""
    lanes = [Lane(p) for p in per_lane_pids]
    consumed = macs = stalled = 0

    for (_axy, _cid, pid) in fifo_b:
        guard = 0
        while True:  # one iteration per offered beat of this entry
            guard += 1
            if guard > len(lanes) + 2:
                raise RuntimeError("replay livelock: entry never admitted")

            # ---- combinational action-eval (all lanes, old state) ----------
            actions = []
            for ln in lanes:
                wsp_hit = pid in ln.pid_set                    # b_valid && wsp_q[b_pid]
                keep = wsp_hit and ln.have_curr and pid == ln.curr_pid
                bank_hit = (wsp_hit and not keep and ln.bank_valid
                            and ln.pids[ln.bank_slot] == pid)  # rd_pid == b_pid
                if not wsp_hit:
                    actions.append(IDLE)
                elif keep:
                    actions.append(KEEP)
                elif bank_hit:
                    actions.append(BANK)
                else:
                    actions.append(SKIP)

            consume = not any(a == SKIP for a in actions)      # b_ready

            # ---- read-port drive + sequential update -----------------------
            for ln, a in zip(lanes, actions):
                rd_slot = None
                if a == SKIP:
                    rd_slot = ln.slot_of[pid]                  # target_slot: direct jump
                else:
                    # want_prefetch: running && have_curr && action not BANK/SKIP
                    #                && next_exists && !bank_has_next
                    nxt = ln.curr_slot + 1
                    next_exists = nxt < ln.n
                    bank_has_next = ln.bank_valid and ln.bank_slot == nxt
                    if (ln.have_curr and a in (IDLE, KEEP)
                            and next_exists and not bank_has_next):
                        rd_slot = nxt

                # consumed BANK beat: promote bank -> Curr (uses OLD bank_slot;
                # a BANK lane never fires rd_en this beat, so order is safe).
                if consume and a == BANK:
                    ln.curr_pid = pid                          # rd_pid
                    ln.curr_slot = ln.bank_slot
                    ln.have_curr = True

                # bank register tracks whatever rd_en just fetched.
                if rd_slot is not None:
                    ln.bank_valid = True
                    ln.bank_slot = rd_slot

            if consume:
                consumed += 1
                macs += sum(1 for a in actions if a in (KEEP, BANK))
                break
            stalled += 1  # whole-PE stall beat; all SKIP lanes fetched above

    return consumed, macs, stalled


# ---------------------------------------------------------------------------
# CSV check
# ---------------------------------------------------------------------------
_CFG_RE = re.compile(r"H(\d+)F(\d+)S(\d+)M(\d+)")


def check_csv(path):
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)

    hdr = (f"{'case':<14} {'seed':>4} | {'consumed':>14} {'macs':>14} "
           f"{'stalled':>14} | ok")
    print(f"\n=== {path} ({len(rows)} rows) ===")
    print(hdr)
    print("-" * len(hdr))

    n = 0
    exact = {"consumed": 0, "macs": 0, "stalled": 0}
    mismatches = []
    for row in rows:
        m = _CFG_RE.fullmatch(row["config"])
        if not m:
            raise ValueError(f"unparseable config {row['config']!r}")
        H, F, S, M = (int(g) for g in m.groups())
        da = float(row["act_density"])
        dw = float(row["wgt_density"])
        seed = int(row["seed"])

        act, kernels = regen_case(H, F, M, da, dw, seed)
        fifo_b, per_lane_sw, _wsp = route_v2_one_pe(act, kernels, F, H, S)
        per_lane_pids = [[p for (p, _v) in sw] for sw in per_lane_sw]

        p_con, p_mac, p_stl = replay_pe_fetch(fifo_b, per_lane_pids)
        m_con = int(row["consumed"])
        m_mac = int(row["macs"])
        m_stl = int(row["stalled"])

        n += 1
        oks = []
        for name, pred, meas in (("consumed", p_con, m_con),
                                 ("macs", p_mac, m_mac),
                                 ("stalled", p_stl, m_stl)):
            if pred == meas:
                exact[name] += 1
                oks.append(True)
            else:
                oks.append(False)
                mismatches.append((row["case"], seed, name, pred, meas))
        ok = "OK" if all(oks) else "MISMATCH"
        print(f"{row['case']:<14} {seed:>4} | "
              f"{p_con:>6}/{m_con:<6}  {p_mac:>6}/{m_mac:<6}  "
              f"{p_stl:>6}/{m_stl:<6}  | {ok}")

    return n, exact, mismatches


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    total_n = 0
    total_exact = {"consumed": 0, "macs": 0, "stalled": 0}
    all_mismatches = []
    for path in argv[1:]:
        n, exact, mm = check_csv(path)
        total_n += n
        for k in total_exact:
            total_exact[k] += exact[k]
        all_mismatches.extend(mm)

    print("\n=== SUMMARY ===")
    print(f"rows checked          : {total_n}")
    for k in ("consumed", "macs", "stalled"):
        print(f"exact {k:<9} matches: {total_exact[k]}/{total_n}")
    if all_mismatches:
        print(f"\n{len(all_mismatches)} counter mismatches:")
        for (case, seed, name, pred, meas) in all_mismatches:
            print(f"  {case}/seed{seed}: {name} predicted={pred} "
                  f"measured={meas} (delta {pred - meas:+d})")
        return 1
    print("\nALL COUNTERS MATCH EXACTLY.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
