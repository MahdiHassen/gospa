// =============================================================================
// gospa.sv -- goSPA Accelerator Top Level
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// Stitches the APU (activation SRAM + Stage 1 + Stage 2) to the PE Array
// (N_PE V2 PEs of N_MULTS lanes each), exposing one combined host interface.
//
//   CSR/scan/WSP control     ───►  apu (act SRAM + FIFO-A + routing + FIFO-B)
//                                                │
//                                                │  N_PE FIFO-B streams
//                                                ▼
//   PE weight/WSP/count/arm ─►   pe_array  ──► per-(PE,lane) E×E output map
//                                                │
//   drain control            ───►                ▼
//                                          drain busy / done / per-lane streams
//
// Channel-pass usage (one input channel):
//   1) Load PE weights + per-lane WSPs + per-lane counts; pulse pe_wload_done.
//   2) Load APU per-PE union WSP (apu_wsp_we / waddr / wdata).
//   3) Load the activation SRAM in CSR form (fill_entry_* and fill_rptr_*).
//   4) Pulse scan_start; wait for scan_done (FIFO-A fills behind apu_stage1).
//   5) Pulse s2_start; wait for s2_done (FIFO-A drained into per-PE FIFO-B).
//   6) The PE array consumes FIFO-B in parallel; per-lane accumulators
//      hold partial sums.
//   7) For multi-input-channel convs, re-load weights/WSPs/activation for
//      the next channel and repeat steps 4-6 WITHOUT pulsing drain_start
//      (accumulators persist).
//   8) After the last input channel, pulse drain_start; collect per-(PE, lane)
//      out_valid/cid/acc streams. drain_done pulses on the last beat.
//
// =============================================================================

`default_nettype none

module gospa #(
    // -- Layer geometry ------------------------------------------------------
    parameter int H        = 32,    // activation map (H x H, square)
    parameter int F        = 3,     // kernel size (F x F)
    parameter int S        = 1,     // stride

    // -- Array geometry ------------------------------------------------------
    parameter int N_PE     = 8,     // # PEs / FIFO-B ports
    parameter int N_MULTS  = 4,     // multiplier lanes per PE (V2 = output channels)

    // -- Memories ------------------------------------------------------------
    parameter int N_ROWS   = 32,    // activation SRAM depth (rows)
    parameter int N_NZ_MAX = 1024,  // activation SRAM depth (non-zeros)
    parameter int FIFO_D   = 64,    // FIFO-A / FIFO-B depth (power of 2)

    // -- Datapath widths -----------------------------------------------------
    parameter int DATA_W   = 16,    // activation / weight width
    parameter int ACC_W    = 32,    // accumulator width

    // -- Derived widths (informational; mirror the children) -----------------
    localparam int E          = (H - F)/S + 1,
    localparam int N_PID      = F*F,
    localparam int N_CID      = E*E,
    localparam int IDX_W      = (H     < 2) ? 1 : $clog2(H),
    localparam int CID_W      = (N_CID < 2) ? 1 : $clog2(N_CID),
    localparam int PID_W      = (N_PID < 2) ? 1 : $clog2(N_PID),
    localparam int FIFOB_W    = DATA_W + PID_W + CID_W,
    localparam int PESEL_W    = (N_PE  < 2) ? 1 : $clog2(N_PE),
    localparam int LANE_W     = (N_MULTS < 2) ? 1 : $clog2(N_MULTS),
    localparam int WPTR_W     = $clog2(N_PID + 1),
    localparam int WSRAM_AW   = (N_PID < 2) ? 1 : $clog2(N_PID),
    localparam int PTR_W      = (N_NZ_MAX + 1 < 2) ? 1 : $clog2(N_NZ_MAX + 1),
    localparam int ENT_AW     = (N_NZ_MAX < 2)     ? 1 : $clog2(N_NZ_MAX),
    localparam int RPTR_AW    = (N_ROWS + 1 < 2)   ? 1 : $clog2(N_ROWS + 1),
    localparam int N_CNT_W    = (N_ROWS + 1 < 2)   ? 1 : $clog2(N_ROWS + 1)
)(
    input  wire  logic                                  clk,
    input  wire  logic                                  rst_n,

    // ------------------------------------------------------------------------
    // Activation SRAM fill (CSR-encoded): entry SRAM + row_ptr flop array
    // ------------------------------------------------------------------------
    input  wire  logic                                  fill_entry_we,
    input  wire  logic [ENT_AW-1:0]                     fill_entry_addr,
    input  wire  logic [DATA_W-1:0]                     fill_entry_value,
    input  wire  logic [IDX_W-1:0]                      fill_entry_col,

    input  wire  logic                                  fill_rptr_we,
    input  wire  logic [RPTR_AW-1:0]                    fill_rptr_addr,
    input  wire  logic [PTR_W-1:0]                      fill_rptr_data,

    // ------------------------------------------------------------------------
    // APU per-PE WSP file (one PE per cycle; pre-computed UNION of lane WSPs)
    // ------------------------------------------------------------------------
    input  wire  logic                                  apu_wsp_we,
    input  wire  logic [PESEL_W-1:0]                    apu_wsp_waddr,
    input  wire  logic [N_PID-1:0]                      apu_wsp_wdata,

    // ------------------------------------------------------------------------
    // Scan control (kicks the APU front end after the activation SRAM is filled)
    // ------------------------------------------------------------------------
    input  wire  logic                                  scan_start,
    input  wire  logic [N_CNT_W-1:0]                    scan_n_rows,
    input  wire  logic [IDX_W-1:0]                      scan_base_row,
    output logic                                        scan_busy,
    output logic                                        scan_done,

    // ------------------------------------------------------------------------
    // Stage 2 framing (kicks routing after FIFO-A is filled)
    // ------------------------------------------------------------------------
    input  wire  logic                                  s2_start,
    output logic                                        s2_busy,
    output logic                                        s2_done,

    // ------------------------------------------------------------------------
    // PE weight SRAM fill (per-PE, per-lane, per-slot)
    // ------------------------------------------------------------------------
    input  wire  logic                                  pe_wfill_we,
    input  wire  logic [PESEL_W-1:0]                    pe_wfill_pe,
    input  wire  logic [LANE_W-1:0]                     pe_wfill_lane,
    input  wire  logic [WSRAM_AW-1:0]                   pe_wfill_slot,
    input  wire  logic [PID_W-1:0]                      pe_wfill_pid,
    input  wire  logic signed [DATA_W-1:0]              pe_wfill_val,

    // ------------------------------------------------------------------------
    // Per-(PE, lane) WSP write
    // ------------------------------------------------------------------------
    input  wire  logic                                  pe_wsp_we,
    input  wire  logic [PESEL_W-1:0]                    pe_wsp_pe,
    input  wire  logic [LANE_W-1:0]                     pe_wsp_lane,
    input  wire  logic [N_PID-1:0]                      pe_wsp_data,

    // ------------------------------------------------------------------------
    // Per-(PE, lane) valid-weight count + array-wide arm
    // ------------------------------------------------------------------------
    input  wire  logic [N_PE-1:0][N_MULTS-1:0][WPTR_W-1:0] pe_wload_count,
    input  wire  logic                                     pe_wload_done,

    // ------------------------------------------------------------------------
    // Drain control + per-(PE, lane) output streams
    // ------------------------------------------------------------------------
    input  wire  logic                                  drain_start,
    output logic                                        drain_busy,
    output logic                                        drain_done,

    output logic [N_PE-1:0][N_MULTS-1:0]                out_valid,
    output logic [N_PE-1:0][N_MULTS-1:0][CID_W-1:0]     out_cid,
    output logic [N_PE-1:0][N_MULTS-1:0][ACC_W-1:0]     out_acc,
    input  wire  logic [N_PE-1:0][N_MULTS-1:0]          out_ready
);

    // -------------------------------------------------------------------------
    // FIFO-B bus between APU and PE Array
    // -------------------------------------------------------------------------
    logic [N_PE-1:0]               fifob_valid;
    logic [N_PE-1:0][FIFOB_W-1:0]  fifob_data;
    logic [N_PE-1:0]               fifob_ready;

    // -------------------------------------------------------------------------
    // APU instance
    // -------------------------------------------------------------------------
    apu #(
        .H(H), .F(F), .S(S),
        .N_PE(N_PE), .N_ROWS(N_ROWS), .N_NZ_MAX(N_NZ_MAX),
        .DATA_W(DATA_W), .FIFO_D(FIFO_D)
    ) u_apu (
        .clk             (clk),
        .rst_n           (rst_n),

        .fill_entry_we   (fill_entry_we),
        .fill_entry_addr (fill_entry_addr),
        .fill_entry_value(fill_entry_value),
        .fill_entry_col  (fill_entry_col),

        .fill_rptr_we    (fill_rptr_we),
        .fill_rptr_addr  (fill_rptr_addr),
        .fill_rptr_data  (fill_rptr_data),

        .scan_start      (scan_start),
        .scan_n_rows     (scan_n_rows),
        .scan_base_row   (scan_base_row),
        .scan_busy       (scan_busy),
        .scan_done       (scan_done),

        .s2_start        (s2_start),
        .s2_busy         (s2_busy),
        .s2_done         (s2_done),

        .wsp_we          (apu_wsp_we),
        .wsp_waddr       (apu_wsp_waddr),
        .wsp_wdata       (apu_wsp_wdata),

        .fifob_rd_valid  (fifob_valid),
        .fifob_rd_data   (fifob_data),
        .fifob_rd_ready  (fifob_ready)
    );

    // -------------------------------------------------------------------------
    // PE Array instance
    // -------------------------------------------------------------------------
    pe_array #(
        .N_PE(N_PE), .N_MULTS(N_MULTS),
        .N_PID(N_PID), .N_CID(N_CID),
        .DATA_W(DATA_W), .ACC_W(ACC_W)
    ) u_pe_array (
        .clk         (clk),
        .rst_n       (rst_n),

        .wfill_we    (pe_wfill_we),
        .wfill_pe    (pe_wfill_pe),
        .wfill_lane  (pe_wfill_lane),
        .wfill_slot  (pe_wfill_slot),
        .wfill_pid   (pe_wfill_pid),
        .wfill_val   (pe_wfill_val),

        .wsp_we      (pe_wsp_we),
        .wsp_pe      (pe_wsp_pe),
        .wsp_lane    (pe_wsp_lane),
        .wsp_data    (pe_wsp_data),

        .wload_count (pe_wload_count),
        .wload_done  (pe_wload_done),

        .fifob_valid (fifob_valid),
        .fifob_data  (fifob_data),
        .fifob_ready (fifob_ready),

        .drain_start (drain_start),
        .drain_busy  (drain_busy),
        .drain_done  (drain_done),
        .out_valid   (out_valid),
        .out_cid     (out_cid),
        .out_acc     (out_acc),
        .out_ready   (out_ready)
    );

    // -------------------------------------------------------------------------
    // Optional VCD waveform dump (enable with `+define+DUMP_VCD`)
    // -------------------------------------------------------------------------
`ifdef DUMP_VCD
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, gospa);
    end
`endif

endmodule

`default_nettype wire
