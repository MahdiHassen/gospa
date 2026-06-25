"""
Cocotb testbench for routing (APU Stage 2).

Models FIFO-A behaviourally (presents lane heads + empty/almost_empty, pops on
a_pop) and FIFO-B as ready-controllable sinks, then checks two things:
  1. Per-PE FIFO-B contents against sw/functional.py `broadcast_to_fifo_b`.
  2. Timing the model can't express: backpressure, no-idle-bubble, framing,
     empty lanes, and the MSB-first WSP orientation.

Shape (N_PID/N_PE/ACT_WIDTH/CID_WIDTH) comes from the Makefile via -P/-G and the env.
"""

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles

# Functional golden model. sw/ is placed on PYTHONPATH by the Makefile.
import functional
functional._VERBOSE = False


N_PID = int(os.environ.get("N_PID", "9"))
N_PE  = int(os.environ.get("N_PE", "4"))
ACT_WIDTH = int(os.environ.get("ACT_WIDTH", "8"))
CID_WIDTH = int(os.environ.get("CID_WIDTH", "6"))
PID_WIDTH = 1 if N_PID < 2 else (N_PID - 1).bit_length()

ACT_MASK = (1 << ACT_WIDTH) - 1
CID_MASK = (1 << CID_WIDTH) - 1

CLK_NS = 10
MAX_CYCLES = 20000


def pack_wsp(wsps):
    """list[N_PE][N_PID] (index = PID) -> RTL bus. MSB-first: PID p -> bit
    k*N_PID + (N_PID-1-p), matching wsp[k][N_PID-1-pid] in the RTL."""
    val = 0
    for k in range(N_PE):
        for p in range(N_PID):
            if wsps[k][p]:
                val |= 1 << (k * N_PID + (N_PID - 1 - p))
    return val


def slice_elem(packed, idx, width):
    return (packed >> (idx * width)) & ((1 << width) - 1)


def rand_lanes(max_depth=4):
    return [
        [(random.randint(0, ACT_MASK), random.randint(0, CID_MASK))
         for _ in range(random.randint(0, max_depth))]
        for _ in range(N_PID)
    ]


def rand_wsps():
    return [[random.randint(0, 1) for _ in range(N_PID)] for _ in range(N_PE)]


def expected_fifo_b(lanes, wsps):
    """Golden per-PE FIFO-B contents, masked to the DUT port widths."""
    fb = functional.broadcast_to_fifo_b(lanes, wsps)
    return [[(a & ACT_MASK, c & CID_MASK, p) for (a, c, p) in fbk] for fbk in fb]


async def reset_dut(dut):
    dut.start.value = 0
    dut.wsp.value = 0
    dut.a_act.value = 0
    dut.a_cid.value = 0
    dut.a_empty.value = (1 << N_PID) - 1
    dut.a_almost_empty.value = 0
    dut.b_ready.value = (1 << N_PE) - 1
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


def drive_heads(dut, queues):
    """Present each lane's head + empty/almost_empty from the queues."""
    a_act = a_cid = a_empty = a_ae = 0
    for j, q in enumerate(queues):
        if q:
            a_act |= (q[0][0] & ACT_MASK) << (j * ACT_WIDTH)
            a_cid |= (q[0][1] & CID_MASK) << (j * CID_WIDTH)
            if len(q) == 1:
                a_ae |= 1 << j
        else:
            a_empty |= 1 << j
    dut.a_act.value = a_act
    dut.a_cid.value = a_cid
    dut.a_empty.value = a_empty
    dut.a_almost_empty.value = a_ae


async def run_pass(dut, lanes, wsp_bus, ready_prob=1.0):
    """Drive one start..done pass. Returns (scoreboard, stats), where
    scoreboard[k] is the in-order (Axy, CID, PID) stream pushed to FIFO-B#k and
    stats counts busy cycles, bubbles (busy cycles with no pop), and pops."""
    queues = [list(l) for l in lanes]
    sb = [[] for _ in range(N_PE)]
    busy_cycles = bubbles = pops = 0

    dut.wsp.value = wsp_bus
    drive_heads(dut, queues)
    dut.b_ready.value = (1 << N_PE) - 1
    dut.start.value = 1

    started = False
    for _ in range(MAX_CYCLES):
        if ready_prob >= 1.0:
            dut.b_ready.value = (1 << N_PE) - 1
        else:
            rv = 0
            for k in range(N_PE):
                if random.random() < ready_prob:
                    rv |= 1 << k
            dut.b_ready.value = rv

        await FallingEdge(dut.clk)               # outputs settled for this cycle
        busy = int(dut.busy.value)
        done = int(dut.done.value)
        a_pop = int(dut.a_pop.value)
        b_push = int(dut.b_push.value)

        if busy:
            busy_cycles += 1
            if a_pop == 0:
                bubbles += 1
            started = True
            dut.start.value = 0

        if b_push:
            ba = int(dut.b_act.value)
            bc = int(dut.b_cid.value)
            bp = int(dut.b_pid.value)
            for k in range(N_PE):
                if (b_push >> k) & 1:
                    sb[k].append((slice_elem(ba, k, ACT_WIDTH),
                                  slice_elem(bc, k, CID_WIDTH),
                                  slice_elem(bp, k, PID_WIDTH)))

        await RisingEdge(dut.clk)                # DUT commits pops here
        if a_pop:
            for j in range(N_PID):
                if (a_pop >> j) & 1:
                    queues[j].pop(0)
                    pops += 1
            drive_heads(dut, queues)

        if started and done:
            break
    else:
        raise cocotb.result.TestFailure("run_pass did not reach done in time")

    assert all(len(q) == 0 for q in queues), "FIFO-A lanes not fully drained"
    return sb, dict(busy=busy_cycles, bubbles=bubbles, pops=pops)


async def start_clock(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())


@cocotb.test()
async def directed_small(dut):
    """Small deterministic case checked against the functional model."""
    await start_clock(dut)
    await reset_dut(dut)

    lanes = [[] for _ in range(N_PID)]
    for j in range(min(3, N_PID)):
        lanes[j] = [((j * 10 + i) & ACT_MASK, (i + 1) & CID_MASK) for i in range(j + 1)]
    wsps = [[1 if (p % (k + 1) == 0) else 0 for p in range(N_PID)] for k in range(N_PE)]

    sb, _ = await run_pass(dut, lanes, pack_wsp(wsps))
    exp = expected_fifo_b(lanes, wsps)
    for k in range(N_PE):
        assert sb[k] == exp[k], f"PE{k}: got {sb[k]} exp {exp[k]}"


@cocotb.test()
async def random_full_throughput(dut):
    """Random shapes, sinks always ready: contents match the model, and no idle
    bubble when every lane is non-empty (a_almost_empty fix)."""
    await start_clock(dut)
    await reset_dut(dut)
    random.seed(1)

    for it in range(40):
        lanes = rand_lanes()
        wsps = rand_wsps()
        sb, stats = await run_pass(dut, lanes, pack_wsp(wsps))
        exp = expected_fifo_b(lanes, wsps)
        for k in range(N_PE):
            assert sb[k] == exp[k], f"iter{it} PE{k}: got {sb[k]} exp {exp[k]}"
        total_entries = sum(len(l) for l in lanes)
        assert stats["pops"] == total_entries, \
            f"iter{it}: popped {stats['pops']} != {total_entries} entries"
        if all(len(l) > 0 for l in lanes):
            assert stats["bubbles"] == 0, f"iter{it}: {stats['bubbles']} idle bubble(s)"


@cocotb.test()
async def backpressure(dut):
    """Random sink backpressure: all-or-nothing multicast must not drop or
    duplicate -- per-PE contents identical to the model, just slower."""
    await start_clock(dut)
    await reset_dut(dut)
    random.seed(2)

    for it in range(30):
        lanes = rand_lanes()
        wsps = rand_wsps()
        sb, stats = await run_pass(dut, lanes, pack_wsp(wsps), ready_prob=0.5)
        exp = expected_fifo_b(lanes, wsps)
        for k in range(N_PE):
            assert sb[k] == exp[k], \
                f"iter{it} PE{k} under backpressure: got {sb[k]} exp {exp[k]}"
        total_entries = sum(len(l) for l in lanes)
        assert stats["pops"] == total_entries, \
            f"iter{it}: popped {stats['pops']} != {total_entries}"


@cocotb.test()
async def wsp_orientation(dut):
    """Bus-level MSB=PID0 check, independent of pack_wsp."""
    await start_clock(dut)
    await reset_dut(dut)
    if N_PE < 1:
        return

    lanes = [[((j + 1) & ACT_MASK, (j + 1) & CID_MASK)] for j in range(N_PID)]

    # PE0's MSB selects PID 0
    sb, _ = await run_pass(dut, lanes, 1 << (N_PID - 1))
    assert sb[0] == [(lanes[0][0][0], lanes[0][0][1], 0)], \
        f"MSB should select PID 0; got {sb[0]}"
    for k in range(1, N_PE):
        assert sb[k] == [], f"PE{k} selected nothing but got {sb[k]}"

    # PE0's LSB selects PID N_PID-1
    await reset_dut(dut)
    last = N_PID - 1
    sb, _ = await run_pass(dut, lanes, 1 << 0)
    assert sb[0] == [(lanes[last][0][0], lanes[last][0][1], last)], \
        f"LSB should select PID {last}; got {sb[0]}"


@cocotb.test()
async def empty_lanes(dut):
    """All lanes empty: pass still completes and pushes nothing."""
    await start_clock(dut)
    await reset_dut(dut)

    lanes = [[] for _ in range(N_PID)]
    wsps = [[1] * N_PID for _ in range(N_PE)]
    sb, stats = await run_pass(dut, lanes, pack_wsp(wsps))
    for k in range(N_PE):
        assert sb[k] == [], f"PE{k} got data from empty lanes: {sb[k]}"
    assert stats["pops"] == 0
