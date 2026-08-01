// =============================================================================
// pe_lane.sv -- one PE multiplier lane: bare multiply + CID-indexed bank
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// One of the M multipliers under a PE's single kernel (V2 "act" dataflow). All
// lanes share the one selected weight (mac_w) but each takes its own activation
// and CID from the M-wide FIFO-B beat:
//   - Bare (non-pipelined) baseline: b_act * mac_w is combinational, so the
//     product and its CID are ready the same cycle the beat fires. mac_busy is
//     therefore always 0 (nothing in flight); the drain never has to wait for a
//     multiplier to flush. (The deep structural pipeline mult_pipe/mac_pipe in
//     rtl/common/arith is bypassed here to bring up the raw design first; it
//     trades combinational delay -- lower Fmax -- for zero extra latency and
//     does not change cycle counts / throughput.)
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
    logic signed [PROD_WIDTH-1:0]  prod_c;       // combinational product
    logic signed [ACC_WIDTH-1:0]   acc [0:NUM_CID-1];

    // -------------------------------------------------------------------------
    // Multiply: bare combinational product (no pipeline). Product + CID are
    // valid the same cycle the beat fires, so accumulate happens in that cycle.
    // -------------------------------------------------------------------------
    assign prod_c = b_act * $signed(mac_w);

    // -------------------------------------------------------------------------
    // CID-indexed accumulate. The PE gates mac_go off before draining, so no
    // product lands mid-drain; the bank only ever needs this one write port.
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int i = 0; i < NUM_CID; i++)
                acc[i] <= '0;
        end else if (mac_go) begin
            acc[b_cid] <= acc[b_cid] + ACC_WIDTH'(prod_c);
        end
    end

    // -------------------------------------------------------------------------
    // Combinational assigns. No pipeline -> nothing ever in flight.
    // -------------------------------------------------------------------------
    assign mac_busy  = 1'b0;
    assign mac_fire  = mac_go;
    always_comb
        for (int i = 0; i < DRAIN_W; i++)
            drain_val[i] = ACC_WIDTH'(acc[drain_idx[i]]);

endmodule

`default_nettype wire
