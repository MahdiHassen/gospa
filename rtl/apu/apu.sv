// =============================================================================
// apu.sv -- APU Top Level (CSR Activation SRAM -> Stage 1 -> Stage 2)
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// Per-input-channel sequence:
//   1) Fill the row_ptr SRAM (N_ROWS+1 pointers) and the entry SRAM
//      (N_NZ_MAX {value, col} words) via the two fill ports.
//   2) Pulse scan_start with n_rows + base_row. scan_busy=1 while the FSM
//      walks the CSR; scan_done pulses when the last tuple is accepted.
//   3) Pulse s2_start; routing drains FIFO-A and multicasts WSP-gated
//      entries into per-PE FIFO-Bs. s2_done pulses on the last lane drain.
//
// =============================================================================

`default_nettype none

module apu #(
    parameter int H         = 32,   // activation map (H x H)
    parameter int F         = 3,    // kernel size (F x F)
    parameter int S         = 1,    // stride
    parameter int N_PE      = 8,    // number of PEs / FIFO-Bs
    parameter int NUM_MULTS = 4,    // activations per beat = FIFO-A read width
    parameter int N_ROWS    = 32,   // rows scanner can hold (rptr SRAM depth - 1)
    parameter int N_NZ_MAX  = 1024, // max non-zeros stored in entry SRAM
    parameter int DATA_W    = 16,
    parameter int FIFO_D    = 64,
    parameter int STAGE1_BATCH = 1, // activations scanned/enumerated per cycle

    localparam int E         = (H - F)/S + 1,
    localparam int N_PID     = F*F,
    localparam int IDX_W     = (H     < 2) ? 1 : $clog2(H),
    localparam int CID_W     = (E*E   < 2) ? 1 : $clog2(E*E),
    localparam int PID_W     = (N_PID < 2) ? 1 : $clog2(N_PID),
    localparam int FIFOA_W   = DATA_W + CID_W,
    localparam int CNT_W     = $clog2(FIFO_D) + 1,
    localparam int PTR_W     = (N_NZ_MAX + 1 < 2) ? 1 : $clog2(N_NZ_MAX + 1),
    localparam int ENT_AW    = (N_NZ_MAX < 2)     ? 1 : $clog2(N_NZ_MAX),
    localparam int RPTR_AW   = (N_ROWS + 1 < 2)   ? 1 : $clog2(N_ROWS + 1),
    localparam int N_CNT_W   = (N_ROWS + 1 < 2)   ? 1 : $clog2(N_ROWS + 1)
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

    // -- Per-PE WSP, driven straight from the PE array (pe.wsp exports) -------
    //    LSB-first by PID: wsp[k][p] = 1 means PE k has a weight at PID p.
    input  wire  logic [N_PE-1:0][N_PID-1:0]        wsp,

    // -- FIFO-B read ports (one M-wide beat per PE; consumed by the PE array) -
    output logic [N_PE-1:0]                              fifob_rd_valid,
    output logic [N_PE-1:0][PID_W-1:0]                   fifob_rd_pid,
    output logic [N_PE-1:0][NUM_MULTS-1:0]               fifob_rd_lane_valid,
    output logic [N_PE-1:0][NUM_MULTS-1:0][DATA_W-1:0]   fifob_rd_act,
    output logic [N_PE-1:0][NUM_MULTS-1:0][CID_W-1:0]    fifob_rd_cid,
    input  wire  logic [N_PE-1:0]                        fifob_rd_ready
);

    // -------------------------------------------------------------------------
    // Activation SRAM (CSR) + scan FSM. STAGE1_BATCH lanes/cycle.
    // -------------------------------------------------------------------------
    logic                            scan_valid;
    logic [STAGE1_BATCH-1:0]         scan_lane_valid;
    logic [STAGE1_BATCH-1:0][DATA_W-1:0] scan_value;
    logic [STAGE1_BATCH-1:0][IDX_W-1:0]  scan_x, scan_y;
    logic                            scan_ready;

    act_sram_scanner #(
        .H(H), .N_ROWS(N_ROWS), .N_NZ_MAX(N_NZ_MAX), .DATA_W(DATA_W),
        .STAGE1_BATCH(STAGE1_BATCH)
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
        .out_lane_valid    (scan_lane_valid),
        .out_value         (scan_value),
        .out_x             (scan_x),
        .out_y             (scan_y),
        .out_ready         (scan_ready)
    );

    // -------------------------------------------------------------------------
    // Stage 1 / FIFO-A bank
    // -------------------------------------------------------------------------
    logic [N_PID-1:0][NUM_MULTS-1:0]              fa_rd_valid;
    logic [N_PID-1:0][NUM_MULTS-1:0][FIFOA_W-1:0] fa_rd_data;
    logic [N_PID-1:0][NUM_MULTS-1:0]              fa_rd_ready;
    logic [N_PID-1:0][CNT_W-1:0]                  fa_count;

    apu_stage1 #(
        .H(H), .F(F), .S(S), .NUM_MULTS(NUM_MULTS), .DATA_W(DATA_W),
        .FIFO_D(FIFO_D), .STAGE1_BATCH(STAGE1_BATCH)
    ) u_stage1 (
        .clk           (clk),
        .rst_n         (rst_n),
        .in_valid      (scan_valid),
        .in_lane_valid (scan_lane_valid),
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
    // WSP comes straight from the PE array (each PE derives its own WSP from
    // its weight bank and exports it); routing.sv reads all N_PE in parallel.
    // -------------------------------------------------------------------------
    apu_stage2 #(
        .H(H), .F(F), .S(S),
        .N_PE(N_PE), .NUM_MULTS(NUM_MULTS), .DATA_W(DATA_W), .FIFO_D(FIFO_D)
    ) u_stage2 (
        .clk                (clk),
        .rst_n              (rst_n),
        .start              (s2_start),
        .busy               (s2_busy),
        .done               (s2_done),
        .wsp                (wsp),
        .fifoa_rd_valid     (fa_rd_valid),
        .fifoa_rd_data      (fa_rd_data),
        .fifoa_rd_ready     (fa_rd_ready),
        .fifoa_count        (fa_count),
        .fifob_rd_valid     (fifob_rd_valid),
        .fifob_rd_pid       (fifob_rd_pid),
        .fifob_rd_lane_valid(fifob_rd_lane_valid),
        .fifob_rd_act       (fifob_rd_act),
        .fifob_rd_cid       (fifob_rd_cid),
        .fifob_rd_ready     (fifob_rd_ready)
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
