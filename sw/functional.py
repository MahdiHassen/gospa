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
            print(f"  emit (Axy={axy}, X={row}, Y={col})")

    print(f"[csr_to_positional] emitted {len(stream)} elements")
    return stream


def route_to_fifo_a(stream, F):
    """
    Final APU Stage 1 step: bin each (Axy, CID, PID) into FIFO-A[PID].

    There are F*F FIFO-A's, one per possible PID in [0, F^2 - 1]. The PID is
    implicit in the FIFO index, so only (Axy, CID) is stored in each entry.
    """
    num_fifos = F * F
    fifos = [[] for _ in range(num_fifos)]
    print(f"[route_to_fifo_a] {num_fifos} FIFOs (F={F}), routing {len(stream)} elements")
    for (axy, cid, pid) in stream:
        fifos[pid].append((axy, cid))
        print(f"  push (Axy={axy}, CID={cid}) -> FIFO-A[{pid}]")
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
    print(f"[zero_act_filter] in: {len(stream)} elements")
    kept = []
    for entry in stream:
        axy = entry[0]
        if axy == 0:
            print(f"  drop {entry} (Axy==0)")
        else:
            print(f"  keep {entry}")
            kept.append(entry)
    print(f"[zero_act_filter] out: {len(kept)} elements")
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
    G = F // S
    E = (H - F) // S + 1
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
                print(f"  match m={m}, n={n} -> CID={cid}, PID={pid}")

    if not pairs:
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
    print(f"[broadcast_to_fifo_b] {len(fifo_a)} FIFO-A's -> {num_pes} FIFO-B's")
    for pid, entries in enumerate(fifo_a):
        print(f"  draining FIFO-A[{pid}] ({len(entries)} entries)")
        if not entries:
            print(f"    (empty, nothing to broadcast)")
            continue
        for k, wsp in enumerate(wsps):
            if wsp[pid] == 1:
                for (axy, cid) in entries:
                    fifo_b[k].append((axy, cid, pid))
                    print(f"    broadcast (Axy={axy}, CID={cid}, PID={pid}) -> FIFO-B[{k}]")
            else:
                print(f"    PE#{k} WSP[{pid}]=0, skip")
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
    print(f"[PE#{pe_id}] sparse_weights={sparse_weights}, "
          f"stream_len={len(fifo_b)}, mults={num_mults}")
    if not sparse_weights or not fifo_b:
        print(f"[PE#{pe_id}] nothing to compute")
        return {}

    curr_pid, curr_wgt = sparse_weights[0]
    next_pid, next_wgt = (sparse_weights[1] if len(sparse_weights) >= 2
                         else (None, None))
    cursor = 2
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

        print(f"  cyc#{cycle} {action} | batch={len(batch)} PID={head_pid} "
              f"| Curr=(PID={curr_pid}, w={curr_wgt}) Next=(PID={next_pid}, w={next_wgt})")
        for lane, entry in enumerate(batch):
            axy, cid = entry[0], entry[1]
            prod = axy * curr_wgt
            accums[lane][cid] = accums[lane].get(cid, 0) + prod
            print(f"    lane#{lane}: ACT={axy} CID={cid} | {axy}*{curr_wgt}={prod} "
                  f"-> ACCUM_{lane}[{cid}]={accums[lane][cid]}")

        i = j

    combined = {}
    for a in accums:
        for cid, val in a.items():
            combined[cid] = combined.get(cid, 0) + val
    print(f"[PE#{pe_id}] per-lane ACCUMs:")
    for lane, a in enumerate(accums):
        print(f"  lane#{lane} = {dict(sorted(a.items()))}")
    print(f"[PE#{pe_id}] combined ACCUM = {dict(sorted(combined.items()))}")
    return combined


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
