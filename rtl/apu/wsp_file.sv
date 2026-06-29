// =============================================================================
// wsp_file.sv -- Per-PE WSP register file
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// Small on-chip storage for the Weight Sparsity Patterns that routing.sv uses
// to gate FIFO-A -> FIFO-B multicasts. One WSP per PE (the V2 interpretation
// pre-computes the union of all lanes' WSPs per PE in software, then loads
// that union here).
//
// Storage: a flop array, not an SRAM, because routing.sv needs all N_PE
// WSPs visible in parallel every cycle (one bit per PE per current PID).
// A true SRAM would need N_PE read ports which is more area than the
// N_PE x N_PID flops we use here. For the typical configuration
// (N_PE=8, N_PID=F*F=9) the file is only 72 bits.
//
// Interface:
//   - Synchronous write port: one PE's full WSP per cycle (wsp_we + waddr + wdata).
//     The host can rewrite any subset of PEs at any time the routing isn't
//     in the middle of a pass (typical use: between s2_start pulses, or
//     right after reset).
//   - Combinational read of the whole bank: drives apu_stage2.wsp.
//
// Reset clears every entry to 0 (no broadcasts will fire until a WSP is
// written, which fails safe).
// =============================================================================

`default_nettype none

module wsp_file #(
    parameter int N_PE  = 8,
    parameter int N_PID = 9,

    localparam int PE_IDX_W = (N_PE < 2) ? 1 : $clog2(N_PE)
)(
    input  wire  logic                          clk,
    input  wire  logic                          rst_n,

    // -- Write port (one PE's full WSP per cycle) ----------------------------
    input  wire  logic                          wsp_we,
    input  wire  logic [PE_IDX_W-1:0]           wsp_waddr,   // which PE
    input  wire  logic [N_PID-1:0]              wsp_wdata,   // new WSP value

    // -- Parallel read (consumed by routing.sv) ------------------------------
    output logic [N_PE-1:0][N_PID-1:0]          wsp
);

    logic [N_PID-1:0] wsp_q [0:N_PE-1];

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int i = 0; i < N_PE; i++) wsp_q[i] <= '0;
        end else if (wsp_we) begin
            wsp_q[wsp_waddr] <= wsp_wdata;
        end
    end

    always_comb begin
        for (int k = 0; k < N_PE; k++) wsp[k] = wsp_q[k];
    end

endmodule

`default_nettype wire
