// =============================================================================
// gospa.sv -- goSPA Accelerator Top Level (V2 "act" dataflow)
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// Stitches the APU (activation SRAM + Stage 1 + FIFO-A + routing + FIFO-B) to
// the PE Array (N_PE one-kernel PEs of N_MULTS lanes each), exposing one host
// interface.
//
//   CSR / scan / s2 control  ---->  apu  (act SRAM + FIFO-A + routing + FIFO-B)
//                                     |  ^                                |
//               per-PE WSP  <---------+  |  N_PE M-wide FIFO-B beats       v
//                    |                    |                            pe_array
//                    +-- pe_array.wsp ----+  (each PE derives its own WSP from
//                                            its weight bank and exports it;
//                                            the router gates FIFO-B on it)
//
// The WSP the router needs is NOT a separately loaded host file anymore: each
// PE derives it from its weight bank (pe_mem.wsp_q) and drives it straight into
// the APU router. Loading PE weights is therefore all it takes to set up routing.
//
// Channel-pass usage (one input channel):
//   1) Load PE weights (pe_wfill_*, in PID order); pulse pe_wload_done. This
//      also arms each PE's WSP, which flows to the router.
//   2) Load the activation SRAM in CSR form (fill_entry_* and fill_rptr_*).
//   3) Pulse scan_start; wait scan_done (FIFO-A fills behind apu_stage1).
//   4) Pulse s2_start; wait s2_done (FIFO-A drained into per-PE FIFO-B beats).
//   5) The PE array consumes FIFO-B in parallel; per-lane accumulators hold
//      partial sums.
//   6) For multi-input-channel convs, re-load weights/activation for the next
//      channel and repeat steps 3-5 WITHOUT pulsing drain_start (accumulators
//      persist).
//   7) After the last input channel, pulse drain_start; collect per-PE
//      out_valid/cid/acc streams. drain_done pulses on the last beat.
// =============================================================================

`default_nettype none

module gospa #(
    // -- Layer geometry ------------------------------------------------------
    parameter int H        = 32,    // activation map (H x H, square)
    parameter int F        = 3,     // kernel size (F x F)
    parameter int S        = 1,     // stride

    // -- Array geometry ------------------------------------------------------
    parameter int N_PE     = 8,     // # PEs / FIFO-B ports / output channels
    parameter int N_MULTS  = 4,     // multiplier lanes per PE = activations/beat
    parameter int STAGE1_BATCH = 1, // activations scanned/enumerated per cycle

    // -- Memories ------------------------------------------------------------
    parameter int N_ROWS   = 32,    // activation SRAM depth (rows)
    parameter int N_NZ_MAX = 1024,  // activation SRAM depth (non-zeros)
    parameter int FIFO_D   = 64,    // FIFO-A / FIFO-B depth (power of 2)

    // -- Datapath widths -----------------------------------------------------
    parameter int DATA_W   = 16,    // activation / weight width
    parameter int ACC_W    = 32,    // accumulator width

    // -- Derived widths (mirror the children) --------------------------------
    localparam int E          = (H - F)/S + 1,
    localparam int N_PID      = F*F,
    localparam int N_CID      = E*E,
    localparam int IDX_W      = (H     < 2) ? 1 : $clog2(H),
    localparam int CID_W      = (N_CID < 2) ? 1 : $clog2(N_CID),
    localparam int PID_W      = (N_PID < 2) ? 1 : $clog2(N_PID),
    localparam int PESEL_W    = (N_PE  < 2) ? 1 : $clog2(N_PE),
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
    // PE weight fill (per-PE, one weight/cycle in PID order; WSP is derived).
    // Arming (pe_wload_done) is what makes each PE's WSP visible to the router.
    // ------------------------------------------------------------------------
    input  wire  logic                                  pe_wfill_we,
    input  wire  logic [PESEL_W-1:0]                    pe_wfill_pe,
    input  wire  logic [PID_W-1:0]                      pe_wfill_pid,
    input  wire  logic signed [DATA_W-1:0]              pe_wfill_val,
    input  wire  logic                                  pe_wload_done,

    // ------------------------------------------------------------------------
    // Drain control + per-PE output streams (one CID stream per output channel)
    // ------------------------------------------------------------------------
    input  wire  logic                                  drain_start,
    output logic                                        drain_busy,
    output logic                                        drain_done,

    output logic [N_PE-1:0]                             out_valid,
    output logic [N_PE-1:0][CID_W-1:0]                  out_cid,
    output logic [N_PE-1:0][ACC_W-1:0]                  out_acc,
    input  wire  logic [N_PE-1:0]                       out_ready
);

    // -------------------------------------------------------------------------
    // Per-PE WSP: PE array derives it from its weights and feeds the APU router.
    // -------------------------------------------------------------------------
    logic [N_PE-1:0][N_PID-1:0]                   pe_wsp;

    // -------------------------------------------------------------------------
    // FIFO-B beat bus between APU and PE Array (one M-wide beat per PE).
    // -------------------------------------------------------------------------
    logic [N_PE-1:0]                              fifob_valid;
    logic [N_PE-1:0][PID_W-1:0]                   fifob_pid;
    logic [N_PE-1:0][N_MULTS-1:0]                 fifob_lane_valid;
    logic [N_PE-1:0][N_MULTS-1:0][DATA_W-1:0]     fifob_act;
    logic [N_PE-1:0][N_MULTS-1:0][CID_W-1:0]      fifob_cid;
    logic [N_PE-1:0]                              fifob_ready;

    // -------------------------------------------------------------------------
    // APU instance
    // -------------------------------------------------------------------------
    apu #(
        .H(H), .F(F), .S(S),
        .N_PE(N_PE), .NUM_MULTS(N_MULTS), .N_ROWS(N_ROWS), .N_NZ_MAX(N_NZ_MAX),
        .DATA_W(DATA_W), .FIFO_D(FIFO_D), .STAGE1_BATCH(STAGE1_BATCH)
    ) u_apu (
        .clk                (clk),
        .rst_n              (rst_n),

        .fill_entry_we      (fill_entry_we),
        .fill_entry_addr    (fill_entry_addr),
        .fill_entry_value   (fill_entry_value),
        .fill_entry_col     (fill_entry_col),

        .fill_rptr_we       (fill_rptr_we),
        .fill_rptr_addr     (fill_rptr_addr),
        .fill_rptr_data     (fill_rptr_data),

        .scan_start         (scan_start),
        .scan_n_rows        (scan_n_rows),
        .scan_base_row      (scan_base_row),
        .scan_busy          (scan_busy),
        .scan_done          (scan_done),

        .s2_start           (s2_start),
        .s2_busy            (s2_busy),
        .s2_done            (s2_done),

        .wsp                (pe_wsp),

        .fifob_rd_valid     (fifob_valid),
        .fifob_rd_pid       (fifob_pid),
        .fifob_rd_lane_valid(fifob_lane_valid),
        .fifob_rd_act       (fifob_act),
        .fifob_rd_cid       (fifob_cid),
        .fifob_rd_ready     (fifob_ready)
    );

    // -------------------------------------------------------------------------
    // PE Array instance
    // -------------------------------------------------------------------------
    /* verilator lint_off PINCONNECTEMPTY */
    pe_array #(
        .NUM_PE(N_PE), .NUM_MULTS(N_MULTS),
        .NUM_PID(N_PID), .NUM_CID(N_CID),
        .DATA_WIDTH(DATA_W), .ACC_WIDTH(ACC_W)
    ) u_pe_array (
        .clk              (clk),
        .rst_n            (rst_n),

        .wfill_we         (pe_wfill_we),
        .wfill_pe         (pe_wfill_pe),
        .wfill_pid        (pe_wfill_pid),
        .wfill_val        (pe_wfill_val),
        .wload_done       (pe_wload_done),

        .wsp              (pe_wsp),
        .wsp_valid        (),

        .fifob_valid      (fifob_valid),
        .fifob_pid        (fifob_pid),
        .fifob_lane_valid (fifob_lane_valid),
        .fifob_act        (fifob_act),
        .fifob_cid        (fifob_cid),
        .fifob_ready      (fifob_ready),

        .drain_start      (drain_start),
        .drain_busy       (drain_busy),
        .drain_done       (drain_done),
        .out_valid        (out_valid),
        .out_cid          (out_cid),
        .out_acc          (out_acc),
        .out_ready        (out_ready)
    );
    /* verilator lint_on PINCONNECTEMPTY */

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
