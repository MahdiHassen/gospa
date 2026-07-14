// =============================================================================
// pe_array.sv -- GoSPA PE Array (V2: NUM_PE PEs, each with NUM_MULTS lanes)
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// NUM_PE PEs, each holding NUM_MULTS output channels (V2). Total output
// channels = NUM_PE * NUM_MULTS. Each PE is wired directly onto its FIFO-B
// read port from apu_stage2.
//
//   apu.fifob_rd_valid/data/ready[k]  <->  pe[k]   (per-PE lanes
//                                                   k.0 .. k.{NUM_MULTS-1})
//
// Weight loading (V2)
//   - Weight SRAM fill is addressed by {wfill_pe, wfill_lane}; each cycle
//     appends one {wfill_pid, wfill_val} to PE wfill_pe's lane. The PE derives
//     the SRAM slot, per-lane weight count, and per-lane WSP from this stream.
//   - Pulse wload_done; every PE in the array arms simultaneously. A bare
//     wload_done re-arms the same weights (see pe.sv).
//
// Drain: pulse drain_start; each PE drains its NUM_MULTS accumulators in
// parallel (output bus is NUM_PE*NUM_MULTS wide).
// =============================================================================

`default_nettype none

module pe_array #(
    parameter int NUM_PE     = 8,
    parameter int NUM_MULTS  = 4,
    parameter int NUM_PID    = 9,
    parameter int NUM_CID    = 36,
    parameter int DATA_WIDTH = 16,
    parameter int ACC_WIDTH  = 32,

    // -- Derived --------------------------------------------------------------
    localparam int PID_WIDTH   = (NUM_PID  < 2) ? 1 : $clog2(NUM_PID),
    localparam int CID_WIDTH   = (NUM_CID    < 2) ? 1 : $clog2(NUM_CID),
    localparam int FIFOB_WIDTH = DATA_WIDTH + PID_WIDTH + CID_WIDTH,
    localparam int PESEL_WIDTH = (NUM_PE   < 2) ? 1 : $clog2(NUM_PE),
    localparam int LANE_WIDTH  = (NUM_MULTS < 2) ? 1 : $clog2(NUM_MULTS)
)(
    input  logic                                          clk,
    input  logic                                          rst_n,

    // -- Weight SRAM fill (PE + lane addressed; slot/count/WSP derived) ------
    input  logic                                          wfill_we,
    input  logic [PESEL_WIDTH-1:0]                        wfill_pe,
    input  logic [LANE_WIDTH-1:0]                         wfill_lane,
    input  logic [PID_WIDTH-1:0]                          wfill_pid,
    input  logic signed [DATA_WIDTH-1:0]                  wfill_val,

    // -- Arm (every PE arms simultaneously) ----------------------------------
    input  logic                                          wload_done,

    // -- FIFO-B input streams (one per PE; matches apu.fifob_rd_*) ------------
    input  logic [NUM_PE-1:0]                               fifob_valid,
    input  logic [NUM_PE-1:0][FIFOB_WIDTH-1:0]              fifob_data,
    output logic [NUM_PE-1:0]                               fifob_ready,

    // -- Drain / per-(PE,lane) output streams --------------------------------
    input  logic                                          drain_start,
    output logic                                          drain_busy,
    output logic                                          drain_done,
    output logic [NUM_PE-1:0][NUM_MULTS-1:0]                out_valid,
    output logic [NUM_PE-1:0][NUM_MULTS-1:0][CID_WIDTH-1:0] out_cid,
    output logic [NUM_PE-1:0][NUM_MULTS-1:0][ACC_WIDTH-1:0] out_acc,
    input  logic [NUM_PE-1:0][NUM_MULTS-1:0]                out_ready
);

    logic [NUM_PE-1:0] pe_busy;

    genvar p;
    /* verilator lint_off PINCONNECTEMPTY */
    generate
        for (p = 0; p < NUM_PE; p++) begin : g_pe
            // Unpack this PE's FIFO-B payload: {act, pid, cid}.
            logic signed [DATA_WIDTH-1:0] p_act;
            logic [PID_WIDTH-1:0]         p_pid;
            logic [CID_WIDTH-1:0]         p_cid;

            pe #(
                .NUM_MULTS(NUM_MULTS), .NUM_PID(NUM_PID), .NUM_CID(NUM_CID),
                .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)
            ) u_pe (
                .clk        (clk),
                .rst_n      (rst_n),
                // Weight SRAM fill, gated per PE; arm broadcast to all.
                .wfill_we   (wfill_we && (wfill_pe == PESEL_WIDTH'(p))),
                .wfill_lane (wfill_lane),
                .wfill_pid  (wfill_pid),
                .wfill_val  (wfill_val),
                // Arm.
                .wload_done (wload_done),
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

            assign p_act = fifob_data[p][FIFOB_WIDTH-1 -: DATA_WIDTH];
            assign p_pid = fifob_data[p][CID_WIDTH +: PID_WIDTH];
            assign p_cid = fifob_data[p][CID_WIDTH-1:0];
        end
    endgenerate
    /* verilator lint_on PINCONNECTEMPTY */

    // Array-level drain status (any PE busy / falling edge done).

    logic busy_q;
    always_ff @(posedge clk) begin
        if (!rst_n) 
            busy_q <= 1'b0;
        else        
            busy_q <= drain_busy;
    end

    assign drain_done = busy_q && !drain_busy;
    assign drain_busy = |pe_busy;

endmodule

`default_nettype wire
