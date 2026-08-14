// =============================================================================
// pe_array.sv -- GoSPA PE Array (V2: N_PE PEs, each with N_MULTS lanes)
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// N_PE PEs, each holding N_MULTS output channels (V2). Total output
// channels = N_PE * N_MULTS. Each PE is wired directly onto its FIFO-B
// read port from apu_stage2.
//
//   apu.fifob_rd_valid/data/ready[k]  <->  pe[k]   (per-PE lanes
//                                                   k.0 .. k.{N_MULTS-1})
//
// Weight loading (V2)
//   - Weight SRAM fill is addressed by {wfill_pe, wfill_lane, wfill_slot};
//     each cycle writes one {wfill_pid, wfill_val} into PE wfill_pe's
//     internal SRAM at the addressed lane/slot.
//   - Per-(PE,lane) WSP register file: write via {wsp_pe, wsp_lane, wsp_data}.
//   - Per-(PE,lane) weight count: drive {wload_count[pe][lane]} and pulse
//     wload_done; every PE in the array arms simultaneously.
//
// Drain: pulse drain_start; each PE drains its N_MULTS accumulators in
// parallel (output bus is N_PE*N_MULTS wide).
// =============================================================================

`default_nettype none

module pe_array #(
    parameter int N_PE    = 8,
    parameter int N_MULTS = 4,
    parameter int N_PID   = 9,
    parameter int N_CID   = 36,
    parameter int DATA_W  = 16,
    parameter int ACC_W   = 32,

    // -- Derived --------------------------------------------------------------
    localparam int PID_W      = (N_PID   < 2) ? 1 : $clog2(N_PID),
    localparam int CID_W      = (N_CID   < 2) ? 1 : $clog2(N_CID),
    localparam int FIFOB_W    = DATA_W + PID_W + CID_W,
    localparam int PESEL_W    = (N_PE    < 2) ? 1 : $clog2(N_PE),
    localparam int LANE_W     = (N_MULTS < 2) ? 1 : $clog2(N_MULTS),
    localparam int WPTR_W     = $clog2(N_PID + 1)
)(
    input  wire  logic                                  clk,
    input  wire  logic                                  rst_n,

    // -- Weight SRAM fill (PE + lane + slot addressed) ----------------------
    input  wire  logic                                  wfill_we,
    input  wire  logic [PESEL_W-1:0]                    wfill_pe,
    input  wire  logic [LANE_W-1:0]                     wfill_lane,
    input  wire  logic [WPTR_W-1:0]                     wfill_slot,
    input  wire  logic [PID_W-1:0]                      wfill_pid,
    input  wire  logic signed [DATA_W-1:0]              wfill_val,

    // -- Per-(PE,lane) WSP write ---------------------------------------------
    input  wire  logic                                  wsp_we,
    input  wire  logic [PESEL_W-1:0]                    wsp_pe,
    input  wire  logic [LANE_W-1:0]                     wsp_lane,
    input  wire  logic [N_PID-1:0]                      wsp_data,

    // -- Per-(PE,lane) weight count, latched on `wload_done` -----------------
    input  wire  logic [N_PE-1:0][N_MULTS-1:0][WPTR_W-1:0] wload_count,
    input  wire  logic                                     wload_done,

    // -- FIFO-B input streams (one per PE; matches apu.fifob_rd_*) ------------
    input  wire  logic [N_PE-1:0]                       fifob_valid,
    input  wire  logic [N_PE-1:0][FIFOB_W-1:0]          fifob_data,
    output logic [N_PE-1:0]                              fifob_ready,

    // -- Drain / per-(PE,lane) output streams --------------------------------
    input  wire  logic                                  drain_start,
    output logic                                        drain_busy,
    output logic                                        drain_done,
    output logic [N_PE-1:0][N_MULTS-1:0]                 out_valid,
    output logic [N_PE-1:0][N_MULTS-1:0][CID_W-1:0]      out_cid,
    output logic [N_PE-1:0][N_MULTS-1:0][ACC_W-1:0]      out_acc,
    input  wire  logic [N_PE-1:0][N_MULTS-1:0]          out_ready
);

    logic [N_PE-1:0] pe_busy;

    genvar p;
    /* verilator lint_off PINCONNECTEMPTY */
    generate
        for (p = 0; p < N_PE; p++) begin : g_pe
            // Unpack this PE's FIFO-B payload: {act, pid, cid}.
            logic signed [DATA_W-1:0] p_act;
            logic [PID_W-1:0]         p_pid;
            logic [CID_W-1:0]         p_cid;
            assign p_act = fifob_data[p][FIFOB_W-1 -: DATA_W];
            assign p_pid = fifob_data[p][CID_W +: PID_W];
            assign p_cid = fifob_data[p][CID_W-1:0];

            pe #(
                .N_MULTS(N_MULTS), .N_PID(N_PID), .N_CID(N_CID),
                .DATA_W(DATA_W), .ACC_W(ACC_W)
            ) u_pe (
                .clk        (clk),
                .rst_n      (rst_n),
                // Weight SRAM fill, gated per PE; arm broadcast to all.
                .wfill_we   (wfill_we && (wfill_pe == PESEL_W'(p))),
                .wfill_lane (wfill_lane),
                .wfill_slot (wfill_slot),
                .wfill_pid  (wfill_pid),
                .wfill_val  (wfill_val),
                // WSP write, gated per PE.
                .wsp_we     (wsp_we && (wsp_pe == PESEL_W'(p))),
                .wsp_lane   (wsp_lane),
                .wsp_data   (wsp_data),
                // Arm.
                .wload_done (wload_done),
                .wload_count(wload_count[p]),
                // FIFO-B stream.
                .b_valid    (fifob_valid[p]),
                .b_act      (p_act),
                .b_pid      (p_pid),
                .b_cid      (p_cid),
                .b_ready    (fifob_ready[p]),
                // Drain / output.
                .drain_start(drain_start),
                .drain_busy (pe_busy[p]),
                .drain_done (),
                .out_valid  (out_valid[p]),
                .out_ready  (out_ready[p]),
                .out_cid    (out_cid[p]),
                .out_acc    (out_acc[p])
            );
        end
    endgenerate
    /* verilator lint_on PINCONNECTEMPTY */

    // Array-level drain status (any PE busy / falling edge done).
    assign drain_busy = |pe_busy;
    logic busy_q;
    always_ff @(posedge clk) begin
        if (!rst_n) busy_q <= 1'b0;
        else        busy_q <= drain_busy;
    end
    assign drain_done = busy_q && !drain_busy;

endmodule

`default_nettype wire
