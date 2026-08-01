// =============================================================================
// pe_array.sv -- GoSPA PE Array (V2 "act" dataflow: NUM_PE one-kernel PEs)
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// NUM_PE PEs, one output channel each. Each PE takes an M-wide FIFO-B beat (M
// activations sharing one PID, distinct CIDs) and exports its WSP to the routing
// module. Total output channels = NUM_PE.
//
//   apu.fifob_*[p]  <->  pe[p]   (M-wide beat)      pe[p].wsp  ->  routing module
//
// Weight loading
//   - Fill is addressed by wfill_pe; each cycle appends one {wfill_pid,
//     wfill_val} to that PE's bank (PID order). Slot and WSP are derived.
//   - Pulse wload_done; every PE arms simultaneously (a bare wload_done re-arms).
//
// Drain: pulse drain_start; each PE drains its accumulator to a single CID
// stream (out_* is NUM_PE wide).
// =============================================================================

`default_nettype none

module pe_array #(
    parameter int NUM_PE     = 8,
    parameter int NUM_MULTS  = 4,
    parameter int NUM_PID    = 9,
    parameter int NUM_CID    = 36,
    parameter int DW_COLW    = 0,    // >0: 2-D window drains (depthwise mosaic)
    parameter int DRAIN_W    = 1,    // accumulator values drained per PE per cycle
    parameter int DATA_WIDTH = 16,
    parameter int ACC_WIDTH  = 32,

    // -- Derived --------------------------------------------------------------
    localparam int PID_WIDTH   = (NUM_PID < 2) ? 1 : $clog2(NUM_PID),
    localparam int CID_WIDTH   = (NUM_CID < 2) ? 1 : $clog2(NUM_CID)
)(
    input  logic                                                clk,
    input  logic                                                rst_n,

    // -- Weight SRAM fill: PE-PARALLEL, one PID per cycle. Lane k writes
    //    {wfill_pid, wfill_val[k]} into PE k's shadow bank when wfill_we[k]
    //    is set, so a whole array row loads in one cycle. ---------------------
    input  logic [NUM_PE-1:0]                                   wfill_we,
    input  logic [PID_WIDTH-1:0]                                wfill_pid,
    input  logic signed [NUM_PE-1:0][DATA_WIDTH-1:0]            wfill_val,
    input  logic                                                wload_done,

    // -- WSP export to the routing module (one per PE) -----------------------
    output logic [NUM_PE-1:0][NUM_PID-1:0]                      wsp,
    output logic [NUM_PE-1:0]                                   wsp_valid,

    // -- FIFO-B input beats (one M-wide beat per PE) -------------------------
    input  logic [NUM_PE-1:0]                                   fifob_valid,
    input  logic [NUM_PE-1:0][PID_WIDTH-1:0]                    fifob_pid,
    input  logic [NUM_PE-1:0][NUM_MULTS-1:0]                    fifob_lane_valid,
    input  logic [NUM_PE-1:0][NUM_MULTS-1:0][DATA_WIDTH-1:0]    fifob_act,
    input  logic [NUM_PE-1:0][NUM_MULTS-1:0][CID_WIDTH-1:0]     fifob_cid,
    output logic [NUM_PE-1:0]                                   fifob_ready,

    // -- Drain / per-PE output streams ---------------------------------------
    input  logic                                               drain_start,
    input  logic [NUM_PE-1:0][CID_WIDTH-1:0]                   drain_r0,
    input  logic [NUM_PE-1:0][CID_WIDTH-1:0]                   drain_c0,
    input  logic [CID_WIDTH-1:0]                               drain_rlen,
    input  logic [CID_WIDTH-1:0]                               drain_clen,
    output logic                                               drain_busy,
    output logic                                               drain_done,
    output logic [NUM_PE-1:0]                                  out_valid,
    output logic [NUM_PE-1:0][DRAIN_W-1:0]                     out_lane_valid,
    output logic [NUM_PE-1:0][DRAIN_W-1:0][CID_WIDTH-1:0]      out_cid,
    output logic [NUM_PE-1:0][DRAIN_W-1:0][ACC_WIDTH-1:0]      out_acc,
    input  logic [NUM_PE-1:0]                                  out_ready
);

    // -------------------------------------------------------------------------
    // Signal declarations
    // -------------------------------------------------------------------------
    logic [NUM_PE-1:0] pe_busy;
    logic              busy_q;

    // -------------------------------------------------------------------------
    // PEs
    // -------------------------------------------------------------------------
    genvar p;
    /* verilator lint_off PINCONNECTEMPTY */
    generate
        for (p = 0; p < NUM_PE; p++) begin : g_pe
            pe #(
                .NUM_MULTS(NUM_MULTS), .NUM_PID(NUM_PID), .NUM_CID(NUM_CID),
                .DW_COLW(DW_COLW), .DRAIN_W(DRAIN_W),
                .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)
            ) u_pe (
                .clk          (clk),
                .rst_n        (rst_n),
                // Weight fill, per-PE lane; arm broadcast to all.
                .wfill_we     (wfill_we[p]),
                .wfill_pid    (wfill_pid),
                .wfill_val    (wfill_val[p]),
                .wload_done   (wload_done),
                // WSP export.
                .wsp          (wsp[p]),
                .wsp_valid    (wsp_valid[p]),
                // FIFO-B M-wide beat.
                .b_valid      (fifob_valid[p]),
                .b_pid        (fifob_pid[p]),
                .b_lane_valid (fifob_lane_valid[p]),
                .b_act        (fifob_act[p]),
                .b_cid        (fifob_cid[p]),
                .b_ready      (fifob_ready[p]),
                // Drain / output.
                .drain_start  (drain_start),
                .drain_r0     (drain_r0[p]),
                .drain_c0     (drain_c0[p]),
                .drain_rlen   (drain_rlen),
                .drain_clen   (drain_clen),
                .drain_busy   (pe_busy[p]),
                .drain_done   (),
                .out_valid    (out_valid[p]),
                .out_lane_valid(out_lane_valid[p]),
                .out_cid      (out_cid[p]),
                .out_acc      (out_acc[p]),
                .out_ready    (out_ready[p])
            );
        end
    endgenerate
    /* verilator lint_on PINCONNECTEMPTY */

    // -------------------------------------------------------------------------
    // Array-level drain status (any PE busy / falling edge done).
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n)
            busy_q <= 1'b0;
        else
            busy_q <= drain_busy;
    end

    // -------------------------------------------------------------------------
    // Combinational assigns
    // -------------------------------------------------------------------------
    assign drain_busy = |pe_busy;
    assign drain_done = busy_q && !drain_busy;

endmodule

`default_nettype wire
