// =============================================================================
// pe_lane.sv -- one PE multiplier lane: pipelined multiply + CID-indexed bank
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// One of the M multipliers under a PE's single kernel (V2 "act" dataflow). All
// lanes share the one selected weight (mac_w) but each takes its own activation
// and CID from the M-wide FIFO-B beat:
//   - Multiply: the custom deeply-pipelined structural multiplier mult_pipe
//     (rtl/common/arith) replaces the bare `b_act * mac_w`. This moves the
//     multiply off the critical path (higher Fmax, no DSP). The beat's CID rides
//     the multiplier's aux passthrough so it stays aligned to the product; the
//     accumulate happens LAT cycles later when out_valid asserts. mac_busy = the
//     multiplier's in-flight status, so the PE holds the drain off until the
//     pipeline flushes; mac_fire pulses once per accumulate for the perf tally.
//   - Accumulate: single-cycle read-modify-write into this lane's own
//     CID-indexed bank. Per-lane banks make M CID retirements/cycle
//     conflict-free; the PE sums the banks per CID at drain. Banks clear only on
//     reset, so partial sums persist across a re-arm (input channels chain into
//     one drain).
// =============================================================================

`default_nettype none

module pe_lane #(
    parameter int NUM_CID    = 36,   // # output positions = accumulator depth
    parameter int DRAIN_W    = 1,    // parallel drain read ports
    parameter int DATA_WIDTH = 16,
    parameter int ACC_WIDTH  = 32,

    localparam int CID_WIDTH  = (NUM_CID < 2) ? 1 : $clog2(NUM_CID),
    localparam int PROD_WIDTH = 2 * DATA_WIDTH
)(
    input  logic                          clk,
    input  logic                          rst_n,

    // -- Stage-1 multiply inputs ---------------------------------------------
    input  logic signed [DATA_WIDTH-1:0]  b_act,          // this lane's activation
    input  logic [CID_WIDTH-1:0]          b_cid,          // this lane's output-pixel id
    input  logic [DATA_WIDTH-1:0]         mac_w,          // shared selected weight (2's-comp)
    input  logic                          mac_go,         // fire this lane this beat
    output logic                          mac_busy,       // MAC pipeline in flight (drain-flush gate)
    output logic                          mac_fire,       // 1-cycle pulse per accumulate (perf tally)

    // -- Stage-2 drain read ports (driven by the PE). The bank is a flop
    //    array, so DRAIN_W parallel combinational reads are free. -------------
    input  logic [DRAIN_W-1:0][CID_WIDTH-1:0] drain_idx,  // bank indices to read
    output logic [DRAIN_W-1:0][ACC_WIDTH-1:0] drain_val   // acc[drain_idx[i]]
);

    // -------------------------------------------------------------------------
    // Signal declarations
    // -------------------------------------------------------------------------
    logic signed [PROD_WIDTH-1:0]  prod_p;      // pipelined product
    logic [CID_WIDTH-1:0]          cid_p;       // CID aligned to prod_p (aux passthrough)
    logic                          prod_valid;  // accumulate this beat
    logic                          mult_busy;   // any multiply in flight
    logic signed [ACC_WIDTH-1:0]   acc [0:NUM_CID-1];

    // -------------------------------------------------------------------------
    // Multiply: custom deeply-pipelined structural multiplier. The CID rides the
    // aux passthrough so it emerges aligned with the product (LAT cycles later).
    // -------------------------------------------------------------------------
    mult_pipe #(
        .A_W(DATA_WIDTH), .B_W(DATA_WIDTH), .AUX_W(CID_WIDTH)
    ) u_mult (
        .clk      (clk),
        .rst_n    (rst_n),
        .in_valid (mac_go),
        .a        (b_act),
        .b        ($signed(mac_w)),
        .aux_in   (b_cid),
        .out_valid(prod_valid),
        .p        (prod_p),
        .aux_out  (cid_p),
        .busy     (mult_busy)
    );

    // -------------------------------------------------------------------------
    // CID-indexed accumulate on the pipelined product. The PE gates mac_go off
    // and waits for mac_busy == 0 before draining, so no product lands mid-drain
    // and the bank only ever needs this one write port.
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int i = 0; i < NUM_CID; i++)
                acc[i] <= '0;
        end else if (prod_valid) begin
            acc[cid_p] <= acc[cid_p] + ACC_WIDTH'(prod_p);
        end
    end

    // -------------------------------------------------------------------------
    // Combinational assigns.
    // -------------------------------------------------------------------------
    assign mac_busy = mult_busy;    // hold drain until the pipeline flushes
    assign mac_fire = prod_valid;   // 1-cycle pulse per accumulate
    always_comb
        for (int i = 0; i < DRAIN_W; i++)
            drain_val[i] = ACC_WIDTH'(acc[drain_idx[i]]);

endmodule

`default_nettype wire
