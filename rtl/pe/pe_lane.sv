// =============================================================================
// pe_lane.sv -- one PE lane: pipelined multiply + CID-indexed accumulator
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// One output-channel datapath:
//   - Stage 1: b_act * weight via the deep, structural, custom multiplier
//     `mult_pipe` (rtl/common/arith). The CID rides the multiplier's aux
//     passthrough so it stays aligned with the product at ANY multiplier depth
//     (no manual latency matching); the multiplier's `out_valid` is the
//     aligned accumulate-enable, and `busy` (any result in flight) is exported
//     as mac_busy so the PE holds the drain off until the MAC pipeline flushes.
//   - Stage 2: single-cycle read-modify-write accumulate into a CID-indexed
//     bank, then an in-order drain (one CID per accepted beat). Accumulators
//     clear only on reset, so partial sums persist across a re-arm (chaining
//     input channels into one drain).
//
// Result is independent of multiplier depth (accumulation commutes); the deeper
// pipeline only shifts timing, and the drain flush (mac_busy) absorbs it.
// =============================================================================

`default_nettype none

module pe_lane #(
    parameter int NUM_CID    = 36,   // # output positions = accumulator banks
    parameter int DATA_WIDTH = 16,
    parameter int ACC_WIDTH  = 32,

    localparam int CID_WIDTH  = (NUM_CID < 2) ? 1 : $clog2(NUM_CID),
    localparam int PROD_WIDTH = 2 * DATA_WIDTH
)(
    input  logic                     clk,
    input  logic                     rst_n,

    // -- Stage-1 multiply inputs ---------------------------------------------
    input  logic signed [DATA_WIDTH-1:0] b_act,          // shared activation
    input  logic [CID_WIDTH-1:0]         b_cid,          // shared output-pixel id
    input  logic [DATA_WIDTH-1:0]        mac_w,          // this lane's weight (2's-comp)
    input  logic                         mac_en,         // KEEP/UPDATE this beat
    input  logic                         consume,        // beat accepted (b_ready)
    input  logic                         drain_busy_any, // any lane draining -> no new MAC
    output logic                         mac_busy,       // MAC pipeline in flight (drain-flush gate)
    output logic                         mac_fire,       // 1-cycle pulse per accumulate (perf tally)

    // -- Stage-2 drain -------------------------------------------------------
    input  logic                         drain_pulse,
    output logic                         drain_busy,
    input  logic                         out_ready,
    output logic                         out_valid,
    output logic [CID_WIDTH-1:0]         out_cid,
    output logic [ACC_WIDTH-1:0]         out_acc
);

    // ----- Stage 1: deep structural multiplier (custom, no `*`/`+`) -----
    logic                         mac_go;
    assign mac_go = consume && mac_en && !drain_busy_any;

    logic                         prod_valid;   // aligned accumulate-enable
    logic signed [PROD_WIDTH-1:0] prod_p;       // aligned product
    logic        [CID_WIDTH-1:0]  cid_al;       // aligned CID (aux passthrough)
    logic                         mult_busy;    // any product in flight

    mult_pipe #(
        .A_W(DATA_WIDTH), .B_W(DATA_WIDTH), .AUX_W(CID_WIDTH)
    ) u_mult (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_valid  (mac_go),
        .a         (b_act),
        .b         ($signed(mac_w)),
        .aux_in    (b_cid),
        .out_valid (prod_valid),
        .p         (prod_p),
        .aux_out   (cid_al),
        .busy      (mult_busy)
    );

    assign mac_busy = mult_busy;    // multi-cycle: hold drain until the pipeline flushes
    assign mac_fire = prod_valid;   // 1-cycle: exactly one pulse per accumulate

    // ----- Stage 2: CID-indexed accumulator + in-order drain -----
    logic signed [ACC_WIDTH-1:0] acc [0:NUM_CID-1];
    logic                        draining;
    logic [CID_WIDTH-1:0]        drain_idx;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            draining  <= 1'b0;
            drain_idx <= '0;
            for (int i = 0; i < NUM_CID; i++) acc[i] <= '0;
        end else if (!draining) begin
            // Accumulate phase (aligned product -> its CID bank).
            if (drain_pulse) begin
                draining  <= 1'b1;
                drain_idx <= '0;
            end else if (prod_valid) begin
                acc[cid_al] <= acc[cid_al] + ACC_WIDTH'(prod_p);
            end
        end else if (out_ready) begin
            // Drain phase: one bank per accepted beat.
            if (drain_idx == CID_WIDTH'(NUM_CID-1))
                draining  <= 1'b0;
            else
                drain_idx <= drain_idx + CID_WIDTH'(1);
        end
    end

    assign drain_busy = draining;
    assign out_valid  = draining;
    assign out_cid    = drain_idx;
    assign out_acc    = acc[drain_idx];

endmodule

`default_nettype wire
