// =============================================================================
// pe_array.sv -- GoSPA PE Array (N_PE processing elements)
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// Instantiates N_PE `pe` units, one per output channel, wired directly onto the
// APU's FIFO-B read ports. Each PE owns the kernel for its output channel and
// produces that channel's E x E map.
//
//   apu.fifob_rd_valid/data/ready[k]  <->  pe[k]
//
// FIFO-B payload matches apu_stage2.sv: {a_xy[MSB], pid, cid[LSB]}, with
//   FIFOB_W = DATA_W + PID_W + CID_W.
//
// Weight preload: drive (wload_pe, wload_pid, wload_val) with wload_en to append
// a weight to one PE's store (PID order); pulse wload_done ONCE after every PE
// is loaded to arm the whole array. Drain: pulse drain_start to stream every
// PE's accumulators out in parallel (one E x E channel per PE).
// =============================================================================

`default_nettype none

module pe_array #(
    parameter int N_PE   = 8,    // # PEs = # output channels packed this pass
    parameter int N_PID  = 9,    // kernel positions = F*F
    parameter int N_CID  = 36,   // output positions = E*E
    parameter int DATA_W = 16,
    parameter int ACC_W  = 32,

    // -- Derived --------------------------------------------------------------
    localparam int PID_W   = (N_PID < 2) ? 1 : $clog2(N_PID),
    localparam int CID_W   = (N_CID < 2) ? 1 : $clog2(N_CID),
    localparam int FIFOB_W = DATA_W + PID_W + CID_W,
    localparam int PESEL_W = (N_PE < 2) ? 1 : $clog2(N_PE)
)(
    input  wire  logic                          clk,
    input  wire  logic                          rst_n,

    // -- Weight preload (routed to PE `wload_pe`; wload_done arms all) --------
    input  wire  logic                          wload_en,
    input  wire  logic [PESEL_W-1:0]            wload_pe,
    input  wire  logic [PID_W-1:0]              wload_pid,
    input  wire  logic signed [DATA_W-1:0]      wload_val,
    input  wire  logic                          wload_done,

    // -- FIFO-B input streams (one per PE; matches apu.fifob_rd_*) ------------
    input  wire  logic [N_PE-1:0]               fifob_valid,
    input  wire  logic [N_PE-1:0][FIFOB_W-1:0]  fifob_data,   // {act, pid, cid}
    output logic [N_PE-1:0]                      fifob_ready,

    // -- Drain / output (broadcast start; per-PE output streams) -------------
    input  wire  logic                          drain_start,
    output logic                                drain_busy,   // any PE draining
    output logic                                drain_done,   // all PEs finished
    output logic [N_PE-1:0]                      out_valid,
    output logic [N_PE-1:0][CID_W-1:0]           out_cid,
    output logic [N_PE-1:0][ACC_W-1:0]           out_acc,
    input  wire  logic [N_PE-1:0]               out_ready
);

    logic [N_PE-1:0] pe_busy;

    genvar k;
    /* verilator lint_off PINCONNECTEMPTY */
    generate
        for (k = 0; k < N_PE; k++) begin : g_pe
            // Unpack this PE's FIFO-B payload: {act, pid, cid}.
            logic signed [DATA_W-1:0] k_act;
            logic [PID_W-1:0]         k_pid;
            logic [CID_W-1:0]         k_cid;
            assign k_act = fifob_data[k][FIFOB_W-1 -: DATA_W];
            assign k_pid = fifob_data[k][CID_W +: PID_W];
            assign k_cid = fifob_data[k][CID_W-1:0];

            pe #(
                .N_PID(N_PID), .N_CID(N_CID), .DATA_W(DATA_W), .ACC_W(ACC_W)
            ) u_pe (
                .clk        (clk),
                .rst_n      (rst_n),
                // weight load: enable only the addressed PE; arm is broadcast
                .wload_en   (wload_en && (wload_pe == PESEL_W'(k))),
                .wload_pid  (wload_pid),
                .wload_val  (wload_val),
                .wload_done (wload_done),
                // FIFO-B stream
                .b_valid    (fifob_valid[k]),
                .b_act      (k_act),
                .b_pid      (k_pid),
                .b_cid      (k_cid),
                .b_ready    (fifob_ready[k]),
                // drain / output
                .drain_start(drain_start),
                .drain_busy (pe_busy[k]),
                .drain_done (),                // array uses the busy falling edge
                .out_valid  (out_valid[k]),
                .out_ready  (out_ready[k]),
                .out_cid    (out_cid[k]),
                .out_acc    (out_acc[k])
            );
        end
    endgenerate
    /* verilator lint_on PINCONNECTEMPTY */

    // Array-level drain status: busy while any PE drains; done on the falling
    // edge (last PE finished). All PEs drain N_CID beats, so with a common
    // out_ready they finish together, but the edge detect is robust either way.
    assign drain_busy = |pe_busy;

    logic busy_q;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) busy_q <= 1'b0;
        else        busy_q <= drain_busy;
    end
    assign drain_done = busy_q && !drain_busy;

endmodule

`default_nettype wire
