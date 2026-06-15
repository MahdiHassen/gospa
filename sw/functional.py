"""
goSPA functional model.

Reference: GoSPA: An Energy-efficient High-performance Globally Optimized
Sparse Convolutional Neural Network Accelerator (ISCA 2021).

Pipeline:
  APU Stage 1 (ID Generation)  -> per-element (Axy, X, Y) then (CID, PID)
  APU Stage 2 (PE Assignment)  -> route element to PE
  PE          (Processing)     -> MAC with reordered reuse

This file builds the model bottom-up. Each substage is its own function with
prints so we can checkpoint functional correctness as we go.
"""

import contextlib
from collections import namedtuple


# Global verbosity flag. Pipeline functions guard their hot-loop prints with
# `if _VERBOSE:` so large inputs (e.g. 224x224 activations) don't pay the
# f-string formatting cost. Use `_quiet()` to silence within a block.
_VERBOSE = True


@contextlib.contextmanager
def _quiet():
    global _VERBOSE
    orig = _VERBOSE
    _VERBOSE = False
    try:
        yield
    finally:
        _VERBOSE = orig


# ---------------------------------------------------------------------------
# APU Stage 1
# ---------------------------------------------------------------------------

def csr_to_positional(values, col_idx, row_ptr):
    """
    Decode a CSR-encoded matrix into a stream of positional tuples (Axy, X, Y).

    CSR inputs:
        values   : non-zero values, in row-major scan order
        col_idx  : column index for each non-zero, same length as `values`
        row_ptr  : length (num_rows + 1); row r occupies values[row_ptr[r]:row_ptr[r+1]]

    Output:
        list of (Axy, X, Y) tuples, where (per the paper's a_xy convention)
            Axy = the non-zero value
            X   = its row    index in the dense matrix
            Y   = its column index in the dense matrix
    """
    if _VERBOSE:
        print("[csr_to_positional] decoding CSR -> positional stream")
        print(f"  values  = {values}")
        print(f"  col_idx = {col_idx}")
        print(f"  row_ptr = {row_ptr}")

    num_rows = len(row_ptr) - 1
    stream = []
    for row in range(num_rows):
        start = row_ptr[row]
        end = row_ptr[row + 1]
        for k in range(start, end):
            axy = values[k]
            col = col_idx[k]
            stream.append((axy, row, col))
            if _VERBOSE: print(f"  emit (Axy={axy}, X={row}, Y={col})")

    if _VERBOSE: print(f"[csr_to_positional] emitted {len(stream)} elements")
    return stream


def route_to_fifo_a(stream, F):
    """
    Final APU Stage 1 step: bin each (Axy, CID, PID) into FIFO-A[PID].

    There are F*F FIFO-A's, one per possible PID in [0, F^2 - 1]. The PID is
    implicit in the FIFO index, so only (Axy, CID) is stored in each entry.
    """
    num_fifos = F * F
    fifos = [[] for _ in range(num_fifos)]
    if _VERBOSE:
        print(f"[route_to_fifo_a] {num_fifos} FIFOs (F={F}), routing {len(stream)} elements")
    for (axy, cid, pid) in stream:
        fifos[pid].append((axy, cid))
        if _VERBOSE: print(f"  push (Axy={axy}, CID={cid}) -> FIFO-A[{pid}]")
    if _VERBOSE:
        for pid, fifo in enumerate(fifos):
            print(f"  FIFO-A[{pid}] = {fifo}")
    return fifos


def zero_act_filter(stream):
    """
    Drop any (Axy, CID, PID) whose Axy == 0. Sits after ID generation.

    Even though CSR-encoded inputs only carry non-zeros, this stage is kept
    as its own block so the model mirrors the RTL pipeline (zero_act.sv) and
    handles dense feeds or post-decode zeros.
    """
    if _VERBOSE: print(f"[zero_act_filter] in: {len(stream)} elements")
    kept = []
    for entry in stream:
        axy = entry[0]
        if axy == 0:
            if _VERBOSE: print(f"  drop {entry} (Axy==0)")
        else:
            if _VERBOSE: print(f"  keep {entry}")
            kept.append(entry)
    if _VERBOSE: print(f"[zero_act_filter] out: {len(kept)} elements")
    return kept


def pcid_to_cid_pid(axy, px, py, cx, cy, F, H, S):
    """
    Generate (CID, PID) pairs for one activation a_xy, per Fig. 6.

    G = floor(F / S),  E = floor((H - F) / S) + 1.
    For each (m, n) in [0, G):
        if 0 <= Cx-m < E and 0 <= Cy-n < E
           and Px + mS < F and Py + nS < F:
            CID = (Cx - m) * E + (Cy - n)
            PID = (Px + mS) * F + (Py + nS)

    One a_xy can match multiple weights, so multiple (CID, PID) pairs may
    come out. Axy is carried through with each pair.
    """
    # Paper writes G = floor(F/S), but that under-counts when F % S != 0
    # (e.g. F=3, S=2 -> floor=1 but we need m=1 to reach kernel row 2).
    # The in_f gate below still filters invalid (m, n), so widening to
    # ceil(F/S) is safe.
    G = (F + S - 1) // S
    E = (H - F) // S + 1
    if _VERBOSE:
        print(f"[pcid_to_cid_pid] Axy={axy} Px={px} Py={py} Cx={cx} Cy={cy} "
              f"(F={F}, H={H}, S={S} -> G={G}, E={E})")

    pairs = []
    for m in range(G):
        for n in range(G):
            cx_off = cx - m
            cy_off = cy - n
            in_e   = (0 <= cx_off < E) and (0 <= cy_off < E)
            in_f   = (px + m * S < F)  and (py + n * S < F)
            if in_e and in_f:
                cid = cx_off * E + cy_off
                pid = (px + m * S) * F + (py + n * S)
                pairs.append((axy, cid, pid))
                if _VERBOSE: print(f"  match m={m}, n={n} -> CID={cid}, PID={pid}")

    if _VERBOSE and not pairs:
        print("  (no matches)")
    return pairs


def axy_to_pcid(axy, x, y, stride):
    """
    Decompose one activation coord (X, Y) into (Px, Py, Cx, Cy) for stride S.

    Per Fig. 5 of the paper:
        Px = x mod S      Cx = floor(x / S)
        Py = y mod S      Cy = floor(y / S)

    Axy passes through unchanged so the downstream stage still has the value.
    """
    px = x % stride
    py = y % stride
    cx = x // stride
    cy = y // stride
    if _VERBOSE:
        print(f"[axy_to_pcid] (Axy={axy}, X={x}, Y={y}, S={stride}) -> "
              f"Px={px}, Py={py}, Cx={cx}, Cy={cy}")
    return axy, px, py, cx, cy


# ---------------------------------------------------------------------------
# APU Stage 2
# ---------------------------------------------------------------------------

def broadcast_to_fifo_b(fifo_a, wsps):
    """
    Drain each FIFO-A in PID order and broadcast its contents into FIFO-B
    of every PE k for which WSP_k[PID] == 1. Skip otherwise.

        fifo_a : list of F^2 FIFOs from Stage 1; fifo_a[p] holds (Axy, CID)
                 entries with PID = p.
        wsps   : list of WSP bit-arrays, one per PE. Each WSP has length F^2
                 indexed by PID. WSP_k[p] == 1 means PE k has a non-zero
                 weight at the kernel position with PID p.

    Returns one FIFO-B per PE (same order as `wsps`). Each FIFO-B entry
    keeps (Axy, CID, PID) so the PE knows which weight to multiply against.
    """
    num_pes = len(wsps)
    fifo_b = [[] for _ in range(num_pes)]
    if _VERBOSE:
        print(f"[broadcast_to_fifo_b] {len(fifo_a)} FIFO-A's -> {num_pes} FIFO-B's")
    for pid, entries in enumerate(fifo_a):
        if _VERBOSE: print(f"  draining FIFO-A[{pid}] ({len(entries)} entries)")
        if not entries:
            if _VERBOSE: print(f"    (empty, nothing to broadcast)")
            continue
        for k, wsp in enumerate(wsps):
            if wsp[pid] == 1:
                for (axy, cid) in entries:
                    fifo_b[k].append((axy, cid, pid))
                    if _VERBOSE: print(f"    broadcast (Axy={axy}, CID={cid}, PID={pid}) -> FIFO-B[{k}]")
            else:
                if _VERBOSE: print(f"    PE#{k} WSP[{pid}]=0, skip")
    if _VERBOSE:
        for k, fb in enumerate(fifo_b):
            print(f"  FIFO-B[{k}] = {fb}")
    return fifo_b


# ---------------------------------------------------------------------------
# PE
# ---------------------------------------------------------------------------

def pe_process(fifo_b, sparse_weights, pe_id=0, num_mults=4):
    """
    Cycle-accurate PE with `num_mults` multiplier lanes.

    Dispatch rule per cycle: peek FIFO-B head, take up to `num_mults` entries
    from the head that ALL share the head's PID. As soon as a differing PID
    is encountered, the batch is cut short for this cycle (the differing
    entry waits for the next cycle, even if a multiplier slot is free).

    State: Curr_Wgt (+Curr_Wgt_PID), Next_Wgt (+Next_Wgt_PID).
        KEEP   : batch PID == Curr_Wgt_PID. Reuse Curr_Wgt.
        UPDATE : batch PID changed. Promote Next -> Curr, pull a fresh Next
                 from sparse weight storage.

    Each lane k owns its own CID-indexed accumulator ACCUM_k. This avoids
    write-port contention on the accum file (the lanes can target different
    CIDs in parallel). At readout we sum across lanes for the final result.

        fifo_b         : ordered (Axy, CID, PID) stream from Stage 2.
        sparse_weights : [(PID, weight), ...] in PID order.
        num_mults      : multiplier lanes per PE (default 4).
    """
    if _VERBOSE:
        print(f"[PE#{pe_id}] sparse_weights={sparse_weights}, "
              f"stream_len={len(fifo_b)}, mults={num_mults}")
    if not sparse_weights or not fifo_b:
        if _VERBOSE: print(f"[PE#{pe_id}] nothing to compute")
        return {}

    curr_pid, curr_wgt = sparse_weights[0]
    next_pid, next_wgt = (sparse_weights[1] if len(sparse_weights) >= 2
                         else (None, None))
    cursor = 2
    if _VERBOSE:
        print(f"[PE#{pe_id}] init Curr=(PID={curr_pid}, w={curr_wgt}) "
              f"Next=(PID={next_pid}, w={next_wgt})")

    accums = [{} for _ in range(num_mults)]
    i = 0
    cycle = 0
    while i < len(fifo_b):
        cycle += 1
        head_pid = fifo_b[i][2]

        j = i
        while (j < len(fifo_b)
               and j - i < num_mults
               and fifo_b[j][2] == head_pid):
            j += 1
        batch = fifo_b[i:j]

        if head_pid == curr_pid:
            action = "KEEP  "
        else:
            action = "UPDATE"
            curr_pid, curr_wgt = next_pid, next_wgt
            if cursor < len(sparse_weights):
                next_pid, next_wgt = sparse_weights[cursor]
                cursor += 1
            else:
                next_pid, next_wgt = None, None

        if _VERBOSE:
            print(f"  cyc#{cycle} {action} | batch={len(batch)} PID={head_pid} "
                  f"| Curr=(PID={curr_pid}, w={curr_wgt}) Next=(PID={next_pid}, w={next_wgt})")
        for lane, entry in enumerate(batch):
            axy, cid = entry[0], entry[1]
            prod = axy * curr_wgt
            accums[lane][cid] = accums[lane].get(cid, 0) + prod
            if _VERBOSE:
                print(f"    lane#{lane}: ACT={axy} CID={cid} | {axy}*{curr_wgt}={prod} "
                      f"-> ACCUM_{lane}[{cid}]={accums[lane][cid]}")

        i = j

    combined = {}
    for a in accums:
        for cid, val in a.items():
            combined[cid] = combined.get(cid, 0) + val
    if _VERBOSE:
        print(f"[PE#{pe_id}] per-lane ACCUMs:")
        for lane, a in enumerate(accums):
            print(f"  lane#{lane} = {dict(sorted(a.items()))}")
        print(f"[PE#{pe_id}] combined ACCUM = {dict(sorted(combined.items()))}")
    return combined


# ---------------------------------------------------------------------------
# PE v2 -- multi-WSP interpretation
# ---------------------------------------------------------------------------

def wsp_union(wsps):
    """OR all WSPs together (used to gate Stage 2 broadcast for the v2 PE)."""
    if not wsps:
        return []
    F2 = len(wsps[0])
    return [1 if any(w[p] for w in wsps) else 0 for p in range(F2)]


def pe_process_v2(fifo_b, wsps, sparse_weights_list, pe_id=0):
    """
    Multi-WSP PE. The PE holds K kernels (one per multiplier lane); FIFO-B
    is gated upstream by the UNION of all `wsps`. Per cycle:
        - pop one activation (ACT, CID, PID) from FIFO-B
        - for each lane k, if wsps[k][PID] == 1:
              maybe UPDATE lane k's Curr/Next from its private weight stream
              ACCUM_k[CID] += ACT * curr_wgt_k
          else:
              lane idles this cycle.

    Each lane represents a distinct output channel held in this PE, so we
    return a list of K ACCUM dicts (NOT summed).
    """
    K = len(wsps)
    if _VERBOSE: print(f"[PE-v2#{pe_id}] lanes={K}, stream_len={len(fifo_b)}")

    curr = [(sw[0] if sw else (None, None)) for sw in sparse_weights_list]
    nxt  = [(sw[1] if len(sw) >= 2 else (None, None)) for sw in sparse_weights_list]
    cursors = [2] * K
    accums = [{} for _ in range(K)]

    if _VERBOSE:
        for lane in range(K):
            print(f"  lane#{lane} init Curr=(PID={curr[lane][0]}, w={curr[lane][1]}) "
                  f"Next=(PID={nxt[lane][0]}, w={nxt[lane][1]}) wsp={wsps[lane]}")

    for cycle, (axy, cid, pid) in enumerate(fifo_b, start=1):
        if _VERBOSE: print(f"  cyc#{cycle} ACT={axy} CID={cid} PID={pid}")
        for lane in range(K):
            if wsps[lane][pid] != 1:
                if _VERBOSE: print(f"    lane#{lane} IDLE  (WSP_{lane}[{pid}]=0)")
                continue

            if curr[lane][0] != pid:
                action = "UPDATE"
                curr[lane] = nxt[lane]
                sw = sparse_weights_list[lane]
                if cursors[lane] < len(sw):
                    nxt[lane] = sw[cursors[lane]]
                    cursors[lane] += 1
                else:
                    nxt[lane] = (None, None)
            else:
                action = "KEEP  "

            wgt = curr[lane][1]
            prod = axy * wgt
            accums[lane][cid] = accums[lane].get(cid, 0) + prod
            if _VERBOSE:
                print(f"    lane#{lane} {action} | Curr=(PID={curr[lane][0]}, w={wgt}) "
                      f"Next=(PID={nxt[lane][0]}, w={nxt[lane][1]}) "
                      f"| {axy}*{wgt}={prod} -> ACCUM_{lane}[{cid}]={accums[lane][cid]}")

    if _VERBOSE:
        print(f"[PE-v2#{pe_id}] per-lane ACCUMs:")
        for lane, a in enumerate(accums):
            print(f"  lane#{lane} = {dict(sorted(a.items()))}")
    return accums


# ---------------------------------------------------------------------------
# Full pipeline & reference
# ---------------------------------------------------------------------------

def dense_to_csr(matrix):
    """Convert a dense 2D list into CSR (values, col_idx, row_ptr)."""
    values, col_idx, row_ptr = [], [], [0]
    for row in matrix:
        for j, v in enumerate(row):
            if v != 0:
                values.append(v)
                col_idx.append(j)
        row_ptr.append(len(values))
    return values, col_idx, row_ptr


def kernel_to_sparse(kernel):
    """
    Dense FxF kernel -> (wsp, sparse_weights) using PID = i*F + j.
        wsp            : length-F^2 0/1 array indexed by PID
        sparse_weights : [(PID, weight), ...] in PID order, non-zero only
    """
    F = len(kernel)
    wsp = [0] * (F * F)
    sparse_weights = []
    for i in range(F):
        for j in range(F):
            pid = i * F + j
            w = kernel[i][j]
            if w != 0:
                wsp[pid] = 1
                sparse_weights.append((pid, w))
    return wsp, sparse_weights


def accum_to_matrix(accum, E):
    """ACCUM[CID] -> ExE output, with CID = out_row * E + out_col."""
    out = [[0] * E for _ in range(E)]
    for cid, val in accum.items():
        out[cid // E][cid % E] = val
    return out


def conv2d_reference(activation, kernel, stride):
    """Plain dense 2D convolution. Single channel in, single channel out."""
    H = len(activation)
    F = len(kernel)
    E = (H - F) // stride + 1
    out = [[0] * E for _ in range(E)]
    for i in range(E):
        for j in range(E):
            s = 0
            for m in range(F):
                for n in range(F):
                    s += kernel[m][n] * activation[i * stride + m][j * stride + n]
            out[i][j] = s
    return out


def gospa_conv2d_v2(activation, kernels, stride):
    """
    Drive the multi-WSP PE (v2). All `kernels` are packed into one PE, one
    per multiplier lane. Returns a list of ExE output matrices, one per kernel.
    """
    H = len(activation)
    F = len(kernels[0])
    E = (H - F) // stride + 1

    values, col_idx, row_ptr = dense_to_csr(activation)
    wsps, sparse_weights_list = [], []
    for k in kernels:
        wsp, sw = kernel_to_sparse(k)
        wsps.append(wsp)
        sparse_weights_list.append(sw)
    union = wsp_union(wsps)

    positional = csr_to_positional(values, col_idx, row_ptr)
    decoded = [axy_to_pcid(axy, x, y, stride) for (axy, x, y) in positional]
    pairs = []
    for (axy, px, py, cx, cy) in decoded:
        pairs.extend(pcid_to_cid_pid(axy, px, py, cx, cy, F=F, H=H, S=stride))
    filtered = zero_act_filter(pairs)
    fifo_a = route_to_fifo_a(filtered, F=F)
    fifo_b_list = broadcast_to_fifo_b(fifo_a, [union])

    accums = pe_process_v2(fifo_b_list[0], wsps, sparse_weights_list, pe_id=0)
    return [accum_to_matrix(a, E) for a in accums]


def gospa_conv2d(activation, kernel, stride):
    """
    Drive a dense (activation, kernel, stride) through the full goSPA pipeline
    (Stage 1 -> Stage 2 -> single PE) and return the ExE output matrix.
    """
    H = len(activation)
    F = len(kernel)
    E = (H - F) // stride + 1

    values, col_idx, row_ptr = dense_to_csr(activation)
    wsp, sparse_weights = kernel_to_sparse(kernel)

    positional = csr_to_positional(values, col_idx, row_ptr)
    decoded = [axy_to_pcid(axy, x, y, stride) for (axy, x, y) in positional]
    pairs = []
    for (axy, px, py, cx, cy) in decoded:
        pairs.extend(pcid_to_cid_pid(axy, px, py, cx, cy, F=F, H=H, S=stride))
    filtered = zero_act_filter(pairs)
    fifo_a = route_to_fifo_a(filtered, F=F)
    fifo_b_list = broadcast_to_fifo_b(fifo_a, [wsp])
    accum = pe_process(fifo_b_list[0], sparse_weights, pe_id=0)

    return accum_to_matrix(accum, E)


# ---------------------------------------------------------------------------
# goSPA accelerator (multi-PE)
# ---------------------------------------------------------------------------

# Lightweight container for the routed structures produced by the front end.
# Returned by `goSPA_route` and consumed both by `goSPA_run` (to run the PEs)
# and by the performance model (to instrument timing over the real FIFO-B
# streams -- see sw/perf_pe.py and sw/perf_model.py).
Routing = namedtuple("Routing", [
    "fifo_b_list",  # per-PE [(Axy, CID, PID), ...], PID-ordered (Stage 2 out)
    "pe_chunks",    # per-PE list of the output-channel ids that PE handles
    "wsps",         # per output-channel WSP bit-array (length F^2)
    "sw_lists",     # per output-channel sparse weight stream [(PID, w), ...]
    "n_nz",         # non-zero activations decoded (Stage 1 input count)
    "n_pairs",      # (CID,PID) pairs emitted == sum_pid len(FIFO_A[pid])
    "E",            # output map dimension (E x E)
])


def goSPA_route(activation, kernels, stride,
                num_pes=8, num_mults=4, interpretation="v1"):
    """
    Run only the goSPA front end (APU Stage 1 + Stage 2) and return the routed
    data structures, WITHOUT running the PEs. This is the shared routing core
    of `goSPA_run`; the performance model calls it to count cycles over the
    *real* routed FIFO-B streams rather than re-implementing the dataflow.

    Mapping policy matches `goSPA_run`:
      "v1": one kernel per PE   (gating WSP = that kernel's WSP).
      "v2": up to `num_mults` kernels per PE (gating WSP = union of the chunk).

    Pure and silent: no prints, no PE compute. Returns a `Routing`.
    """
    H = len(activation)
    F = len(kernels[0])
    E = (H - F) // stride + 1
    K = len(kernels)

    cap = num_pes if interpretation == "v1" else num_pes * num_mults
    if K > cap:
        raise ValueError(f"{interpretation}: capacity is {cap} kernels, got {K}")

    values, col_idx, row_ptr = dense_to_csr(activation)
    wsps, sw_lists = [], []
    for k in kernels:
        w, sw = kernel_to_sparse(k)
        wsps.append(w)
        sw_lists.append(sw)

    if interpretation == "v1":
        pe_chunks = [[i] for i in range(K)]
        pe_wsps = [wsps[c[0]] for c in pe_chunks]
    else:
        pe_chunks = []
        for pe in range(num_pes):
            chunk = list(range(pe * num_mults, min((pe + 1) * num_mults, K)))
            if chunk:
                pe_chunks.append(chunk)
        pe_wsps = [wsp_union([wsps[k] for k in c]) for c in pe_chunks]

    with _quiet():
        positional = csr_to_positional(values, col_idx, row_ptr)
        decoded = [axy_to_pcid(axy, x, y, stride) for (axy, x, y) in positional]
        pairs = []
        for (axy, px, py, cx, cy) in decoded:
            pairs.extend(pcid_to_cid_pid(axy, px, py, cx, cy, F=F, H=H, S=stride))
        filtered = zero_act_filter(pairs)
        fifo_a = route_to_fifo_a(filtered, F=F)
        fifo_b_list = broadcast_to_fifo_b(fifo_a, pe_wsps)

    return Routing(
        fifo_b_list=fifo_b_list,
        pe_chunks=pe_chunks,
        wsps=wsps,
        sw_lists=sw_lists,
        n_nz=len(positional),
        n_pairs=len(filtered),
        E=E,
    )


def goSPA_run(activation, kernels, stride,
              num_pes=8, num_mults=4, interpretation="v1",
              initial_outputs=None, verbose=False):
    """
    Run a full conv layer on a goSPA accelerator with `num_pes` PEs of
    `num_mults` multipliers each.

    Mapping policies:
      "v1": one kernel per PE; needs len(kernels) <= num_pes.
            Each PE uses pe_process (same-PID batching across `num_mults`).
      "v2": up to `num_mults` kernels per PE (one per lane); needs
            len(kernels) <= num_pes * num_mults.
            Each PE uses pe_process_v2 (one activation/cycle, lanes idle when
            their WSP[PID]=0).

    `initial_outputs`: optional list of ExE matrices, one per kernel. If
    provided, each PE accumulator is treated as if it already held these
    values (models retaining the PE's CID-indexed accumulator across
    multiple input-channel passes). Defaults to None (start at zero).

    Returns one ExE output map per kernel, in the same order as `kernels`.
    """
    H = len(activation)
    F = len(kernels[0])
    K = len(kernels)

    routing = goSPA_route(activation, kernels, stride,
                          num_pes=num_pes, num_mults=num_mults,
                          interpretation=interpretation)
    E = routing.E

    print(f"[goSPA] {num_pes} PEs x {num_mults} mults | mode={interpretation} "
          f"| H={H}, F={F}, S={stride} -> E={E} | {K} output channels")

    outputs = [None] * K
    print(f"  non-zero activations after Stage 1 filter = {routing.n_pairs}")
    for pe, chunk in enumerate(routing.pe_chunks):
        fb = routing.fifo_b_list[pe]
        detail = (verbose and interpretation == "v2" and pe == 1)
        if detail:
            print(f"  PE#{pe}: kernels={chunk}")
        else:
            print(f"  PE#{pe}: kernels={chunk}, |FIFO-B|={len(fb)}")
        with _quiet():
            if interpretation == "v1":
                k = chunk[0]
                accum = pe_process(fb, routing.sw_lists[k],
                                   pe_id=pe, num_mults=num_mults)
                outputs[k] = accum_to_matrix(accum, E)
            else:
                accums = pe_process_v2(fb,
                                       [routing.wsps[k] for k in chunk],
                                       [routing.sw_lists[k] for k in chunk],
                                       pe_id=pe)
                for lane, k in enumerate(chunk):
                    outputs[k] = accum_to_matrix(accums[lane], E)

        if detail:
            for lane, k in enumerate(chunk):
                # cumulative ACCUM = this-channel partials + any prior channels
                if initial_outputs is not None:
                    cumul = [[outputs[k][i][j] + initial_outputs[k][i][j]
                              for j in range(E)] for i in range(E)]
                else:
                    cumul = outputs[k]
                kw = max(2, max(len(str(v)) for row in kernels[k] for v in row))
                aw = max(2, max(len(str(v)) for row in cumul for v in row))
                print(f"    lane#{lane} (out-channel {k}):")
                print(f"      kernel ({F}x{F}):")
                for row in kernels[k]:
                    print("        " + " ".join(f"{v:>{kw}d}" for v in row))
                print(f"      ACCUM file ({E}x{E}, CID = row*{E} + col):")
                for row in cumul:
                    print("        " + " ".join(f"{v:>{aw}d}" for v in row))

    if initial_outputs is not None:
        for oc in range(K):
            for i in range(E):
                for j in range(E):
                    outputs[oc][i][j] += initial_outputs[oc][i][j]

    return outputs


def goSPA_multichannel(activations, kernels_full, stride,
                       num_pes=8, num_mults=4, interpretation="v1",
                       verbose=False):
    """
    Multi-input-channel conv on goSPA. The PE accumulators are kept across
    input-channel iterations (modeled here by threading `initial_outputs`).

        activations  : list of C_in 2D activation maps (each H x H).
        kernels_full : list of C_out kernels; each entry is a list of C_in
                       FxF slices, so kernels_full[oc][ic] is the slice for
                       output channel oc, input channel ic.

    Returns C_out output maps (ExE each), with sums over all input channels.
    """
    C_in = len(activations)
    C_out = len(kernels_full)
    if any(len(k) != C_in for k in kernels_full):
        raise ValueError("each kernel must have a slice per input channel")

    outputs = None
    for ic in range(C_in):
        kernels_ic = [kernels_full[oc][ic] for oc in range(C_out)]
        print(f"--- input channel {ic+1}/{C_in} ---")
        outputs = goSPA_run(activations[ic], kernels_ic, stride,
                            num_pes=num_pes, num_mults=num_mults,
                            interpretation=interpretation,
                            initial_outputs=outputs, verbose=verbose)
    return outputs


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Dense reference:
    #   [[5, 0, 0, 0],
    #    [0, 8, 0, 0],
    #    [0, 0, 3, 0],
    #    [0, 6, 0, 0]]
    values  = [5, 8, 3, 6]
    col_idx = [0, 1, 2, 1]
    row_ptr = [0, 1, 2, 3, 4]

    print("=== APU Stage 1: csr_to_positional ===")
    stream = csr_to_positional(values, col_idx, row_ptr)
    print(f"result: {stream}")

    print("\n=== APU Stage 1: axy_to_pcid (S=2) ===")
    decoded = [axy_to_pcid(axy, x, y, stride=2) for (axy, x, y) in stream]
    print(f"result: {decoded}")

    # Pick a config where Fig. 6's constraints actually fire. Using the
    # paper's running example: F=2, H=4, S=2 -> G=1, E=2.
    print("\n=== APU Stage 1: pcid_to_cid_pid (F=2, H=4, S=2) ===")
    all_pairs = []
    for (axy, px, py, cx, cy) in decoded:
        all_pairs.extend(pcid_to_cid_pid(axy, px, py, cx, cy, F=2, H=4, S=2))
    print(f"result: {all_pairs}")

    print("\n=== APU Stage 1: zero_act_filter ===")
    polluted = all_pairs
    filtered = zero_act_filter(polluted)
    print(f"result: {filtered}")

    print("\n=== APU Stage 1: route_to_fifo_a (F=2) ===")
    fifo_a = route_to_fifo_a(filtered, F=2)
    print(f"result: {fifo_a}")
    print("here no new inputs are accepted until all the fifos are empty\n")

    # APU Stage 2: PE Assignment via WSP-gated broadcast.
    # Example WSPs from the paper's Fig. 7:
    #   PE#1 kernel [[1,0],[3,0]]  -> WSP1 = [1,0,1,0]
    #   PE#2 kernel [[0,2],[-1,0]] -> WSP2 = [0,1,1,0]
    print("=== APU Stage 2: broadcast_to_fifo_b ===")
    wsps = [
        [1, 0, 1, 0],
        [0, 1, 1, 1],
    ]
    fifo_b = broadcast_to_fifo_b(fifo_a, wsps)
    print(f"result: {fifo_b}")

    # PE: replay the paper's Fig. 13 example end-to-end.
    #   kernel [[0,1],[0,2]]  -> WSP=0101, sparse weights [(1,1),(3,2)]
    #   FIFO-B stream (cycle 1..5): see ACT/CID/PID rows in Fig. 13.
    # Expected output (CID-wise accumulation): {0:-4, 1:2, 2:4, 3:-6}
    print("\n=== PE: Fig. 13 replay ===")
    fig13_weights = [(1, 1), (3, 2)]
    fig13_stream = [
        (2,  1, 1),
        (-2, 2, 1),
        (-2, 0, 3),
        (3,  2, 3),
        (-3, 3, 3),
    ]
    pe_process(fig13_stream, fig13_weights, pe_id="fig13")

    # ---- Full 2D conv: goSPA pipeline vs. dense reference ----
    import contextlib, io

    cases = [
        ("Fig. 13 setup (3x3 * 2x2, S=1)",
         [[1, 0, 2], [0, -2, 0], [-1, 3, -3]],
         [[0, 1], [0, 2]],
         1),
        ("Dense 4x4 * 2x2, S=1",
         [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
         [[1, 0], [0, 1]],
         1),
        ("Strided 4x4 * 2x2, S=2",
         [[1, 0, 2, 0], [0, 3, 0, 4], [5, 0, 6, 0], [0, 7, 0, 8]],
         [[1, -1], [2, 0]],
         2),
    ]

    print("\n=== Verification: goSPA full pipeline vs. reference conv2d ===")
    all_ok = True
    for name, activation, kernel, stride in cases:
        ref = conv2d_reference(activation, kernel, stride)
        with contextlib.redirect_stdout(io.StringIO()):
            gospa = gospa_conv2d(activation, kernel, stride)
        ok = (ref == gospa)
        all_ok = all_ok and ok
        print(f"  {name}")
        print(f"    reference = {ref}")
        print(f"    goSPA     = {gospa}")
        print(f"    MATCH     = {ok}")
    print(f"\noverall: {'PASS' if all_ok else 'FAIL'}")

    # ---- Multi-WSP PE (v2): one PE holds several kernels ----
    print("\n=== Verification v2: multi-WSP PE vs. per-kernel reference ===")
    v2_cases = [
        ("3x3 activation, 2 kernels packed (Fig.13 + Fig.7 PE#1)",
         [[1, 0, 2], [0, -2, 0], [-1, 3, -3]],
         [[[0, 1], [0, 2]],
          [[1, 0], [3, 0]]],
         1),
        ("4x4 activation, 3 kernels packed",
         [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
         [[[1, 0], [0, 1]],
          [[0, 2], [-1, 0]],
          [[1, 1], [1, 1]]],
         1),
    ]

    all_ok_v2 = True
    for name, activation, kernels, stride in v2_cases:
        refs = [conv2d_reference(activation, k, stride) for k in kernels]
        with contextlib.redirect_stdout(io.StringIO()):
            gospas = gospa_conv2d_v2(activation, kernels, stride)
        ok = (refs == gospas)
        all_ok_v2 = all_ok_v2 and ok
        print(f"  {name}")
        for i, (r, g) in enumerate(zip(refs, gospas)):
            print(f"    kernel#{i} reference = {r}")
            print(f"    kernel#{i} goSPA-v2  = {g}")
        print(f"    MATCH = {ok}")
    print(f"\nv2 overall: {'PASS' if all_ok_v2 else 'FAIL'}")

    # ---- Map MobileNetV2's first conv (red channel) onto goSPA ----
    # First conv shape: (out=32, in=3, kH=3, kW=3), stride=2.
    # We treat one input channel (red, in_ch=0) per filter as a 3x3 kernel
    # and run all 32 filter-red-channels through the accelerator.
    print("\n=== goSPA: MobileNetV2 first conv (red channel) ===")
    import os, sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "testing", "ref"))
    from mobilenet import get_first_conv  # noqa: E402

    _, conv0 = get_first_conv()
    int_w = conv0.weight().int_repr().numpy()  # (32, 3, 3, 3) int8
    red_kernels = [[[int(v) for v in row] for row in int_w[f, 0]]
                   for f in range(int_w.shape[0])]
    print(f"  loaded {len(red_kernels)} red-channel 3x3 kernels from MobileNetV2 first conv")

    # Realistic MobileNetV2 input: 224x224 single-channel (red) INT8 activation.
    # We zero-pad to 226x226 to mirror the conv's padding=1, so with F=3, S=2
    # we get E = (226-3)/2 + 1 = 112  -- matches the network's spec.
    import random
    rng = random.Random(0)
    H_RAW = 96
    SPARSITY = 0.5  # post-quantization activations: very rough placeholder
    raw_red = [[0 if rng.random() < SPARSITY else rng.randint(-128, 127)
                for _ in range(H_RAW)] for _ in range(H_RAW)]
    # Manual zero-padding (padding=1 on all sides) -> 226x226.
    PAD = 1
    H_PAD = H_RAW + 2 * PAD
    activation_red = [[0] * H_PAD]
    for row in raw_red:
        activation_red.append([0] + list(row) + [0])
    activation_red.append([0] * H_PAD)

    NUM_PES, NUM_MULTS, STRIDE = 8, 4, 2

    # Independent ground truth via PyTorch's conv2d (cross-correlation, no flip
    # -- same convention as our model).
    import torch
    import torch.nn.functional as Fnn

    def torch_conv_ref(activation, kernels, stride):
        act_t = torch.tensor(activation, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        ker_t = torch.tensor(kernels, dtype=torch.float32).unsqueeze(1)
        out_t = Fnn.conv2d(act_t, ker_t, stride=stride, padding=0).squeeze(0)
        return [out_t[c].int().tolist() for c in range(out_t.shape[0])]

    print(f"\n-- v2 mapping: 32 filters' red channel across "
          f"{NUM_PES} PEs x {NUM_MULTS} lanes --")
    out_v2 = goSPA_run(activation_red, red_kernels, stride=STRIDE,
                      num_pes=NUM_PES, num_mults=NUM_MULTS, interpretation="v2")
    refs_v2 = [conv2d_reference(activation_red, k, STRIDE) for k in red_kernels]
    torch_v2 = torch_conv_ref(activation_red, red_kernels, STRIDE)
    ok_v2_ref = (out_v2 == refs_v2)
    ok_v2_torch = (out_v2 == torch_v2)
    print(f"  v2 vs Python ref : {ok_v2_ref}")
    print(f"  v2 vs PyTorch    : {ok_v2_torch}")

    print(f"\nMobileNet goSPA mapping (vs PyTorch): "
          f"v2={'PASS' if ok_v2_torch else 'FAIL'}")

    # ---- Full 3-input-channel conv: accumulators persist across input channels ----
    print("\n=== goSPA: MobileNetV2 first conv, all 3 input channels (RGB) ===")
    # Build 3 padded input channels (G and B with different seeds so they differ).
    def _padded_channel(seed):
        r = random.Random(seed)
        raw = [[0 if r.random() < SPARSITY else r.randint(-128, 127)
                for _ in range(H_RAW)] for _ in range(H_RAW)]
        pad = [[0] * H_PAD]
        for row in raw:
            pad.append([0] + list(row) + [0])
        pad.append([0] * H_PAD)
        return pad

    activations_rgb = [activation_red,           # ic=0 reuses the red channel above
                       _padded_channel(seed=1),  # ic=1
                       _padded_channel(seed=2)]  # ic=2

    # Full 4D kernel: (32 output channels, 3 input channels, 3, 3)
    full_kernels = [
        [[[int(v) for v in row] for row in int_w[oc, ic]]
         for ic in range(int_w.shape[1])]
        for oc in range(int_w.shape[0])
    ]

    # v2: 8 PEs x 4 lanes -> 32 output channels (all filters), summed over 3 inputs
    print("\n-- v2: 3 input channels x 32 output channels --")
    out_v2_mc = goSPA_multichannel(activations_rgb, full_kernels, STRIDE,
                                   num_pes=NUM_PES, num_mults=NUM_MULTS,
                                   interpretation="v2")
    act_3ch = torch.tensor(activations_rgb, dtype=torch.float32).unsqueeze(0)  # (1,3,H,H)
    ker_v2_mc = torch.tensor(full_kernels, dtype=torch.float32)                # (32,3,3,3)
    torch_v2_mc_t = Fnn.conv2d(act_3ch, ker_v2_mc, stride=STRIDE, padding=0).squeeze(0)
    torch_v2_mc = [torch_v2_mc_t[c].int().tolist() for c in range(torch_v2_mc_t.shape[0])]
    ok_v2_mc = (out_v2_mc == torch_v2_mc)
    print(f"  v2 multi-channel vs PyTorch: {ok_v2_mc}")

    print(f"\nMobileNet RGB conv on goSPA (vs PyTorch): "
          f"v2={'PASS' if ok_v2_mc else 'FAIL'}")

    # ---- End-to-end against MobileNetV2's first_conv module ----
    # The module is QuantizedConvReLU2d, so the full pipeline is:
    #   acc   = sum((q_a - z_a) * q_w)
    #   floaty = acc * s_a * s_w + bias
    #   relu  = max(floaty, 0)
    #   q_y   = round(relu / s_y) + z_y, clamped to quint8
    print("\n=== End-to-end vs first_conv module (with bias/ReLU/requant) ===")
    s_a, z_a = 0.01, 0
    rng2 = random.Random(7)
    q_a_int_np = [[[rng2.randint(0, 80) if rng2.random() > 0.5 else 0
                    for _ in range(H_RAW)] for _ in range(H_RAW)]
                  for _ in range(3)]
    q_a_int = torch.tensor(q_a_int_np, dtype=torch.uint8).unsqueeze(0)  # (1,3,H,H)
    q_a = torch._make_per_tensor_quantized_tensor(q_a_int, s_a, z_a)
    with torch.no_grad():
        q_y = conv0(q_a)
    y_actual = q_y.int_repr()[0]  # (32, E, E) uint8

    # Pad with z_a (= 0 here) and feed each channel into goSPA
    padded = torch.zeros(1, 3, H_PAD, H_PAD, dtype=torch.int32)
    padded[:, :, PAD:-PAD, PAD:-PAD] = q_a_int.int() - z_a
    activations_3ch_q = [padded[0, c].tolist() for c in range(3)]
    full_kernels_int = [[[[int(v) for v in row] for row in int_w[oc, ic]]
                         for ic in range(int_w.shape[1])]
                        for oc in range(int_w.shape[0])]

    acc_lists = goSPA_multichannel(activations_3ch_q, full_kernels_int,
                                   stride=STRIDE, num_pes=NUM_PES,
                                   num_mults=NUM_MULTS, interpretation="v2")
    acc_t = torch.tensor(acc_lists, dtype=torch.int32)  # (32, E, E)

    s_w = conv0.weight().q_scale()
    s_y = conv0.scale
    z_y = conv0.zero_point
    bias = conv0.bias()  # float32 or None

    # PyTorch's QConv pre-quantizes bias to int32 (units of s_a*s_w) and adds
    # it to the int accumulator before the float-rescale step. Mirror that.
    s_aw = s_a * s_w
    if bias is not None:
        bias_int = torch.round(bias / s_aw).to(torch.int32)
        acc_t = acc_t + bias_int[:, None, None]
    y_float = acc_t.float() * s_aw
    y_float = torch.clamp(y_float, min=0.0)  # ReLU
    y_quant = torch.quantize_per_tensor(y_float, s_y, z_y, torch.quint8)
    y_pred = y_quant.int_repr()

    exact = bool((y_pred == y_actual).all().item())
    n_match = int((y_pred == y_actual).sum().item())
    n_total = int(y_actual.numel())
    diff = (y_pred.int() - y_actual.int()).abs()
    print(f"  match {n_match}/{n_total} = {n_match/n_total:.4%}")
    print(f"  exact match against first_conv output: {exact}")
    print(f"  diff histogram: 0={int((diff==0).sum())} 1={int((diff==1).sum())} "
          f">=2={int((diff>=2).sum())}, max diff={int(diff.max())}")
