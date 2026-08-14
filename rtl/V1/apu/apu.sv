// =============================================================================
// apu.sv -- APU Top Level (CSR Activation SRAM -> Stage 1 -> Stage 2)
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// Pipeline (~1 non-zero per cycle steady-state, modulo SRAM read latency):
//
//   {entries SRAM, row_ptr SRAM}  <-- fill from DRAM (TB model)
//             |
//             v
//   [act_sram_scanner]  --(val, x=row, y=col)-->  [apu_stage1]
//                                                      | F^2 FIFO-A slots
//                                                      v
//                                                 [apu_stage2]  --> N_PE FIFO-Bs
//
// Per-input-channel sequence:
//   1) Fill the row_ptr SRAM (N_ROWS+1 pointers) and the entry SRAM
//      (N_NZ_MAX {value, col} words) via the two fill ports.
//   2) Pulse scan_start with n_rows + base_row. scan_busy=1 while the FSM
//      walks the CSR; scan_done pulses when the last tuple is accepted.
//   3) Pulse s2_start; routing drains FIFO-A and multicasts WSP-gated
//      entries into per-PE FIFO-Bs. s2_done pulses on the last lane drain.
//
// Backpressure end-to-end (FIFO-B full -> routing stall -> FIFO-A drain
// stall -> Stage 1 stall -> scanner stall via out_ready).
// =============================================================================

`default_nettype none

module apu #(
    parameter int H        = 32,   // activation map (H x H)
    parameter int F        = 3,    // kernel size (F x F)
    parameter int S        = 1,    // stride
    parameter int N_PE     = 8,    // number of PEs / FIFO-Bs
    parameter int N_ROWS   = 32,   // rows scanner can hold (rptr SRAM depth - 1)
    parameter int N_NZ_MAX = 1024, // max non-zeros stored in entry SRAM
    parameter int DATA_W   = 16,
    parameter int FIFO_D   = 64,

    // -- Derived widths ------------------------------------------------------
    localparam int E         = (H - F)/S + 1,
    localparam int N_PID     = F*F,
    localparam int IDX_W     = (H     < 2) ? 1 : $clog2(H),
    localparam int CID_W     = (E*E   < 2) ? 1 : $clog2(E*E),
    localparam int PID_W     = (N_PID < 2) ? 1 : $clog2(N_PID),
    localparam int FIFOA_W   = DATA_W + CID_W,
    localparam int FIFOB_W   = DATA_W + PID_W + CID_W,
    localparam int CNT_W     = $clog2(FIFO_D) + 1,
    localparam int PTR_W     = (N_NZ_MAX + 1 < 2) ? 1 : $clog2(N_NZ_MAX + 1),
    localparam int ENT_AW    = (N_NZ_MAX < 2)     ? 1 : $clog2(N_NZ_MAX),
    localparam int RPTR_AW   = (N_ROWS + 1 < 2)   ? 1 : $clog2(N_ROWS + 1),
    localparam int N_CNT_W   = (N_ROWS + 1 < 2)   ? 1 : $clog2(N_ROWS + 1),
    localparam int PE_IDX_W  = (N_PE  < 2)        ? 1 : $clog2(N_PE)
)(
    input  wire  logic                              clk,
    input  wire  logic                              rst_n,

    // -- Entry SRAM fill (one {value, col} per cycle) ------------------------
    input  wire  logic                              fill_entry_we,
    input  wire  logic [ENT_AW-1:0]                 fill_entry_addr,
    input  wire  logic [DATA_W-1:0]                 fill_entry_value,
    input  wire  logic [IDX_W-1:0]                  fill_entry_col,

    // -- Row-pointer SRAM fill (one pointer per cycle) -----------------------
    input  wire  logic                              fill_rptr_we,
    input  wire  logic [RPTR_AW-1:0]                fill_rptr_addr,
    input  wire  logic [PTR_W-1:0]                  fill_rptr_data,

    // -- Scan control --------------------------------------------------------
    input  wire  logic                              scan_start,
    input  wire  logic [N_CNT_W-1:0]                scan_n_rows,
    input  wire  logic [IDX_W-1:0]                  scan_base_row,
    output logic                                    scan_busy,
    output logic                                    scan_done,

    // -- Stage 2 framing -----------------------------------------------------
    input  wire  logic                              s2_start,
    output logic                                    s2_busy,
    output logic                                    s2_done,

    // -- WSP register file write port (one PE's full WSP per cycle) ----------
    //    MSB-first by PID -- wsp_wdata[N_PID-1] = bit for PID 0, etc.
    input  wire  logic                              wsp_we,
    input  wire  logic [PE_IDX_W-1:0]               wsp_waddr,
    input  wire  logic [N_PID-1:0]                  wsp_wdata,

    // -- FIFO-B read ports (consumed by the PE array) ------------------------
    output logic [N_PE-1:0]                         fifob_rd_valid,
    output logic [N_PE-1:0][FIFOB_W-1:0]            fifob_rd_data,
    input  wire  logic [N_PE-1:0]                   fifob_rd_ready
);

    // -------------------------------------------------------------------------
    // Activation SRAM (CSR) + scan FSM
    // -------------------------------------------------------------------------
    logic                scan_valid;
    logic [DATA_W-1:0]   scan_value;
    logic [IDX_W-1:0]    scan_x, scan_y;
    logic                scan_ready;

    act_sram_scanner #(
        .H(H), .N_ROWS(N_ROWS), .N_NZ_MAX(N_NZ_MAX), .DATA_W(DATA_W)
    ) u_scanner (
        .clk               (clk),
        .rst_n             (rst_n),
        .fill_entry_we     (fill_entry_we),
        .fill_entry_addr   (fill_entry_addr),
        .fill_entry_value  (fill_entry_value),
        .fill_entry_col    (fill_entry_col),
        .fill_rptr_we      (fill_rptr_we),
        .fill_rptr_addr    (fill_rptr_addr),
        .fill_rptr_data    (fill_rptr_data),
        .scan_start        (scan_start),
        .scan_n_rows       (scan_n_rows),
        .scan_base_row     (scan_base_row),
        .scan_busy         (scan_busy),
        .scan_done         (scan_done),
        .out_valid         (scan_valid),
        .out_value         (scan_value),
        .out_x             (scan_x),
        .out_y             (scan_y),
        .out_ready         (scan_ready)
    );

    // -------------------------------------------------------------------------
    // Stage 1 / FIFO-A bank
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
        .in_valid      (scan_valid),
        .in_value      (scan_value),
        .in_x          (scan_x),
        .in_y          (scan_y),
        .in_ready      (scan_ready),
        .fifoa_rd_valid(fa_rd_valid),
        .fifoa_rd_data (fa_rd_data),
        .fifoa_rd_ready(fa_rd_ready),
        .fifoa_count   (fa_count)
    );

    // -------------------------------------------------------------------------
    // WSP register file: small flop-based store, one full WSP per PE.
    // Host pre-computes the V2 union of per-lane WSPs in software and writes
    // it here; routing.sv reads all N_PE values in parallel each cycle.
    // -------------------------------------------------------------------------
    logic [N_PE-1:0][N_PID-1:0] wsp_q;

    wsp_file #(.N_PE(N_PE), .N_PID(N_PID)) u_wsp_file (
        .clk      (clk),
        .rst_n    (rst_n),
        .wsp_we   (wsp_we),
        .wsp_waddr(wsp_waddr),
        .wsp_wdata(wsp_wdata),
        .wsp      (wsp_q)
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
        .wsp           (wsp_q),
        .fifoa_rd_valid(fa_rd_valid),
        .fifoa_rd_data (fa_rd_data),
        .fifoa_rd_ready(fa_rd_ready),
        .fifoa_count   (fa_count),
        .fifob_rd_valid(fifob_rd_valid),
        .fifob_rd_data (fifob_rd_data),
        .fifob_rd_ready(fifob_rd_ready)
    );

    // -- Optional VCD waveform dump (enable with `+define+DUMP_VCD`) ---------
`ifdef DUMP_VCD
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, apu);
    end
`endif

endmodule

`default_nettype wire
