// =============================================================================
// mult_pipe.sv -- Deeply-pipelined signed multiplier (structural, no functions)
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// Reusable, architecture-independent (FPGA/ASIC) signed multiplier built WITHOUT
// the `*`/`+` operators and WITHOUT SystemVerilog functions:
//   * `*` -> explicit signed partial-product generation (always_comb).
//   * `+` -> structural ripple-carry adder module `rca_add` (per tree node).
//
// Deep pipeline (one register per stage):
//   stage 0 : register inputs (a, b)
//   stage 1 : register the partial products
//   stage 2..L+1 : one register per binary adder-tree level
//   => LATENCY = ceil(log2(B_W)) + 2 cycles,  throughput = 1 result / cycle.
//
// Passthrough + status for datapath integration:
//   * aux_in -> aux_out, delayed by exactly LATENCY (carry the CID / tag along,
//     so the consumer needs no manual latency matching at any depth).
//   * out_valid = in_valid delayed by LATENCY.
//   * busy = 1 while ANY result is in flight (OR of the valid pipeline) -- lets a
//     consumer hold a drain until the multiplier has fully flushed.
//
// Signed handling: the top partial product is the two's-complement negation
// ~(a<<msb)+1 (rca_add, cin=1). Tree math is exact mod 2^P_W; the product fits
// P_W bits. Assumes B_W >= 2. Sync reset.
// =============================================================================

`default_nettype none

module mult_pipe #(
    parameter  int A_W   = 16,             // multiplicand width (signed)
    parameter  int B_W   = 16,             // multiplier width   (signed)
    parameter  int AUX_W = 1,              // passthrough tag width (>=1)

    localparam int P_W = A_W + B_W,        // product width
    localparam int NP  = 1 << ($clog2(B_W)),   // #partial products, padded to 2^k
    localparam int L   = $clog2(NP),       // adder-tree depth
    localparam int LAT = L + 2             // total pipeline latency
)(
    input  wire logic                  clk,
    input  wire logic                  rst_n,
    input  wire logic                  in_valid,
    input  wire logic signed [A_W-1:0] a,
    input  wire logic signed [B_W-1:0] b,
    input  wire logic [AUX_W-1:0]      aux_in,
    output logic                       out_valid,
    output logic signed [P_W-1:0]      p,
    output logic [AUX_W-1:0]           aux_out,
    output logic                       busy
);

    /* verilator lint_off PINCONNECTEMPTY */   // rca_add cout is intentionally unused

    // ---- stage 0: register inputs -----------------------------------------
    logic signed [A_W-1:0] a_q;
    logic signed [B_W-1:0] b_q;
    always_ff @(posedge clk) begin
        if (!rst_n) begin a_q <= '0; b_q <= '0; end
        else        begin a_q <= a;  b_q <= b;  end
    end

    // ---- signed partial products (combinational, from registered inputs) ---
    logic signed [P_W-1:0] a_ext;
    assign a_ext = {{(P_W-A_W){a_q[A_W-1]}}, a_q};   // explicit sign-extension

    logic [P_W-1:0] neg_msb;
    rca_add #(.W(P_W)) u_neg (
        .x(~(a_ext <<< (B_W-1))), .y('0), .cin(1'b1), .s(neg_msb), .cout());

    logic [P_W-1:0] pp [0:NP-1];
    always_comb begin
        for (int i = 0; i < NP; i++) pp[i] = '0;
        for (int i = 0; i < B_W; i++) begin
            if (b_q[i]) begin
                if (i == B_W-1) pp[i] = neg_msb;
                else            pp[i] = a_ext <<< i;
            end
        end
    end

    // ---- stage 1: register the partial products ---------------------------
    logic [P_W-1:0] pp_q [0:NP-1];
    always_ff @(posedge clk) begin
        if (!rst_n) for (int j = 0; j < NP; j++) pp_q[j] <= '0;
        else        for (int j = 0; j < NP; j++) pp_q[j] <= pp[j];
    end

    // ---- stages 2..L+1: registered adder tree -----------------------------
    logic [P_W-1:0] sum_c [1:L][0:NP-1];
    logic [P_W-1:0] treg  [1:L][0:NP-1];

    genvar gl, gj;
    generate
        for (gl = 1; gl <= L; gl++) begin : g_lvl
            for (gj = 0; gj < (NP >> gl); gj++) begin : g_node
                if (gl == 1) begin : g_from_pp
                    rca_add #(.W(P_W)) u_add (
                        .x(pp_q[2*gj]), .y(pp_q[2*gj+1]), .cin(1'b0),
                        .s(sum_c[gl][gj]), .cout());
                end else begin : g_from_treg
                    rca_add #(.W(P_W)) u_add (
                        .x(treg[gl-1][2*gj]), .y(treg[gl-1][2*gj+1]), .cin(1'b0),
                        .s(sum_c[gl][gj]), .cout());
                end
            end
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int l = 1; l <= L; l++)
                for (int j = 0; j < NP; j++) treg[l][j] <= '0;
        end else begin
            for (int l = 1; l <= L; l++)
                for (int j = 0; j < (NP >> l); j++) treg[l][j] <= sum_c[l][j];
        end
    end

    // ---- valid + aux pipeline (depth LAT, tracks the product) --------------
    logic [LAT-1:0]   vpipe;
    logic [AUX_W-1:0] auxp [1:LAT];
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            vpipe <= '0;
            for (int l = 1; l <= LAT; l++) auxp[l] <= '0;
        end else begin
            vpipe   <= {vpipe[LAT-2:0], in_valid};
            auxp[1] <= aux_in;
            for (int l = 2; l <= LAT; l++) auxp[l] <= auxp[l-1];
        end
    end

    /* verilator lint_on PINCONNECTEMPTY */

    assign p         = treg[L][0];
    assign out_valid = vpipe[LAT-1];
    assign aux_out   = auxp[LAT];
    assign busy      = |vpipe;

endmodule

`default_nettype wire
