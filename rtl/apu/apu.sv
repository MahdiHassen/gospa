// =============================================================================
// apu.sv -- APU Top Level (Stage 1 + Stage 2)
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// CSR streams --> apu_stage1 (FIFO-A bank) --> apu_stage2 (FIFO-B bank) --> PEs
//
// Two-phase pass:
//   1) Drive CSR until row_ptr / entry handshakes finish (FIFO-A fills).
//   2) Pulse `s2_start`; routing drains FIFO-A in PID order, WSP-multicasting
//      into the per-PE FIFO-Bs. `s2_done` pulses on the last lane drain.
//
// Backpressure is end-to-end: a full FIFO-B stalls routing, which freezes the
// FIFO-A drain, which can in turn freeze Stage 1's CSR ingest if the writer
// was filling FIFO-A concurrently.
// =============================================================================

`default_nettype none

module apu #(
    parameter int H      = 32,   // activation map (H x H)
    parameter int F      = 3,    // kernel size (F x F)
    parameter int S      = 1,    // stride
    parameter int N_PE   = 8,    // number of PEs / FIFO-Bs
    parameter int DATA_W = 16,
    parameter int FIFO_D = 64,

    // -- Derived widths ------------------------------------------------------
    localparam int E       = (H - F)/S + 1,
    localparam int N_PID   = F*F,
    localparam int IDX_W   = (H     < 2) ? 1 : $clog2(H),
    localparam int CID_W   = (E*E   < 2) ? 1 : $clog2(E*E),
    localparam int PID_W   = (N_PID < 2) ? 1 : $clog2(N_PID),
    localparam int FIFOA_W = DATA_W + CID_W,
    localparam int FIFOB_W = DATA_W + PID_W + CID_W,
    localparam int CNT_W   = $clog2(FIFO_D) + 1
)(
    input  wire  logic                              clk,
    input  wire  logic                              rst_n,

    // -- CSR input (Stage 1 producer) ----------------------------------------
    input  wire  logic                              row_ptr_valid,
    input  wire  logic [$clog2(H*H):0]              row_ptr_data,
    output logic                                    row_ptr_ready,

    input  wire  logic                              entry_valid,
    input  wire  logic [DATA_W-1:0]                 entry_value,
    input  wire  logic [IDX_W-1:0]                  entry_col,
    output logic                                    entry_ready,

    // -- Stage 2 framing -----------------------------------------------------
    input  wire  logic                              s2_start,
    output logic                                    s2_busy,
    output logic                                    s2_done,

    // -- WSP register file (one per PE, MSB-first by PID) --------------------
    input  wire  logic [N_PE-1:0][N_PID-1:0]        wsp,

    // -- FIFO-B read ports (consumed by the PE array) ------------------------
    output logic [N_PE-1:0]                         fifob_rd_valid,
    output logic [N_PE-1:0][FIFOB_W-1:0]            fifob_rd_data,
    input  wire  logic [N_PE-1:0]                   fifob_rd_ready
);

    // -------------------------------------------------------------------------
    // FIFO-A bank crosses the Stage 1 / Stage 2 boundary
    // -------------------------------------------------------------------------
    logic [N_PID-1:0]               fa_rd_valid;
    logic [N_PID-1:0][FIFOA_W-1:0]  fa_rd_data;
    logic [N_PID-1:0]               fa_rd_ready;
    logic [N_PID-1:0][CNT_W-1:0]    fa_count;

    apu_stage1 #(
        .H(H), .F(F), .S(S), .DATA_W(DATA_W), .FIFO_D(FIFO_D)
    ) u_stage1 (
        .clk           (clk),
        .rst_n         (rst_n),
        .row_ptr_valid (row_ptr_valid),
        .row_ptr_data  (row_ptr_data),
        .row_ptr_ready (row_ptr_ready),
        .entry_valid   (entry_valid),
        .entry_value   (entry_value),
        .entry_col     (entry_col),
        .entry_ready   (entry_ready),
        .fifoa_rd_valid(fa_rd_valid),
        .fifoa_rd_data (fa_rd_data),
        .fifoa_rd_ready(fa_rd_ready),
        .fifoa_count   (fa_count)
    );

    apu_stage2 #(
        .H(H), .F(F), .S(S),
        .N_PE(N_PE), .DATA_W(DATA_W), .FIFO_D(FIFO_D)
    ) u_stage2 (
        .clk           (clk),
        .rst_n         (rst_n),
        .start         (s2_start),
        .busy          (s2_busy),
        .done          (s2_done),
        .wsp           (wsp),
        .fifoa_rd_valid(fa_rd_valid),
        .fifoa_rd_data (fa_rd_data),
        .fifoa_rd_ready(fa_rd_ready),
        .fifoa_count   (fa_count),
        .fifob_rd_valid(fifob_rd_valid),
        .fifob_rd_data (fifob_rd_data),
        .fifob_rd_ready(fifob_rd_ready)
    );

endmodule

`default_nettype wire
