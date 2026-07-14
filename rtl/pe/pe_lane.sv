// =============================================================================
// pe_lane.sv -- one PE lane: pipelined multiply + CID-indexed accumulator
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// One output-channel datapath, formerly split across an inline multiply stage
// in pe.sv and a separate pe_acc:
//   - Stage 1: b_act * weight, registered (the DSP output register). This
//     breaks the long  FIFO-B -> mac_w mux -> 16x16 multiply -> 32b RMW  path.
//     CID and the accumulate-enable are delayed one cycle to stay aligned with
//     the registered product; mac_busy exports that enable so the PE can hold
//     the drain off until every lane's MAC pipeline has flushed.
//   - Stage 2: single-cycle read-modify-write accumulate into a CID-indexed
//     bank, then an in-order drain (one CID per accepted beat). Accumulators
//     clear only on reset, so partial sums persist across a re-arm (chaining
//     input channels into one drain).
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
    output logic                         mac_busy,       // registered product valid (flush probe)

    // -- Stage-2 drain -------------------------------------------------------
    input  logic                         drain_pulse,
    output logic                         drain_busy,
    input  logic                         out_ready,
    output logic                         out_valid,
    output logic [CID_WIDTH-1:0]         out_cid,
    output logic [ACC_WIDTH-1:0]         out_acc
);
    logic signed [PROD_WIDTH-1:0] prod_k;   // combinational product
    logic signed [PROD_WIDTH-1:0] prod_q;   // registered product (DSP output register)
    logic        [CID_WIDTH-1:0]  cid_q;    // CID aligned to prod_q
    logic                         mac_en_q; // stage-2 accumulate enable

    logic signed [ACC_WIDTH-1:0] acc [0:NUM_CID-1];
    logic                        draining;
    logic [CID_WIDTH-1:0]        drain_idx;

    // ----- Stage 1: multiply (combinational) -> product register -----
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            prod_q   <= '0;
            cid_q    <= '0;
            mac_en_q <= 1'b0;
        end else begin
            prod_q   <= prod_k;
            cid_q    <= b_cid;
            mac_en_q <= consume && mac_en && !drain_busy_any;
        end
    end

    assign prod_k   = b_act * $signed(mac_w);
    assign mac_busy = mac_en_q;   

    // ----- Stage 2: CID-indexed accumulator + in-order drain -----
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            draining  <= 1'b0;
            drain_idx <= '0;
            for (int i = 0; i < NUM_CID; i++) acc[i] <= '0;
        end else if (!draining) begin
            // Accumulate phase (registered product -> its CID bank).
            if (drain_pulse) begin
                draining  <= 1'b1;
                drain_idx <= '0;
            end else if (mac_en_q) begin
                acc[cid_q] <= acc[cid_q] + ACC_WIDTH'(prod_q);
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
