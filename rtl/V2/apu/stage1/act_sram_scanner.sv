// =============================================================================
// act_sram_scanner.sv -- Activation SRAM (CSR) + wide CSR-to-coordinate scan
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// One on-chip activation SRAM holds the current input channel in CSR layout.
// A small FSM walks the channel and emits up to STAGE1_BATCH (val, x=row, y=col)
// tuples PER CYCLE into the APU front end -- widening Stage 1 so the scanner
// stops being the throughput bottleneck (1 activation/cycle was the floor).
//
// Banked entry SRAM: NB = max(FILL_W, STAGE1_BATCH) banks, entry g -> bank
// g % NB at local address g / NB. Both movers use the striping:
//   - Fill: FILL_W entries/cycle (contiguous, lane 0 at fill_entry_addr, base
//     FILL_W-aligned) land in FILL_W distinct banks -- one write port each.
//   - Scan: one aligned batch [b*W, b*W+W) per cycle; lane i comes from bank
//     (b%RGRP)*W + i of bank row b/RGRP (all banks share the read address).
//     The last batch is masked by out_lane_valid when total % W != 0.
//
// Row (x): entries are stored row-major but rows are variable length, so a
// batch can straddle row boundaries. Each lane's row is recovered from the
// row_ptr flop array: x_i = base_row + #{ r : row_ptr[r+1] <= g_i }.
//
// row_ptr lives in flops (small, read combinationally; filled FILL_W/cycle).
// out_x is the row, out_y the column.
// =============================================================================

`default_nettype none

module act_sram_scanner #(
    parameter int H           = 32,    // activation map width (full)
    parameter int N_ROWS      = 32,    // rows the scanner can hold
    parameter int N_NZ_MAX    = 1024,  // max non-zeros stored in the activation SRAM
    parameter int DATA_W      = 16,    // bits per activation value
    parameter int STAGE1_BATCH = 1,    // activations emitted per cycle (>=1)
    parameter int FILL_W      = 1,     // CSR entries / row-ptrs written per cycle

    // -- Derived widths --------------------------------------------------------
    localparam int IDX_W     = (H            < 2) ? 1 : $clog2(H),
    localparam int PTR_W     = (N_NZ_MAX + 1 < 2) ? 1 : $clog2(N_NZ_MAX + 1),
    localparam int ENT_W     = DATA_W + IDX_W,
    localparam int ENT_AW    = (N_NZ_MAX     < 2) ? 1 : $clog2(N_NZ_MAX),
    localparam int RPTR_DEPTH = N_ROWS + 1,
    localparam int RPTR_AW   = (RPTR_DEPTH   < 2) ? 1 : $clog2(RPTR_DEPTH),
    localparam int N_CNT_W   = (N_ROWS + 1   < 2) ? 1 : $clog2(N_ROWS + 1),
    localparam int W         = STAGE1_BATCH,
    localparam int NB        = (FILL_W > W) ? FILL_W : W,  // physical banks
    localparam int RGRP      = NB / W,        // W-wide read subgroups per bank row
    localparam int BANK_DEPTH = (N_NZ_MAX + NB - 1) / NB,
    localparam int BANK_AW   = (BANK_DEPTH   < 2) ? 1 : $clog2(BANK_DEPTH)
)(
    input  wire  logic                    clk,
    input  wire  logic                    rst_n,

    // -- Fill: activation SRAM, FILL_W CSR entries/cycle. Lane j writes global
    //    address fill_entry_addr + j; we is a contiguous prefix and the base
    //    address must be FILL_W-aligned (lanes then map 1:1 onto banks). ------
    input  wire  logic [FILL_W-1:0]               fill_entry_we,
    input  wire  logic [ENT_AW-1:0]               fill_entry_addr,
    input  wire  logic [FILL_W-1:0][DATA_W-1:0]   fill_entry_value,
    input  wire  logic [FILL_W-1:0][IDX_W-1:0]    fill_entry_col,

    // -- Fill: row_ptr flop array, FILL_W pointers/cycle (lane j -> addr + j) -
    input  wire  logic [FILL_W-1:0]               fill_rptr_we,
    input  wire  logic [RPTR_AW-1:0]              fill_rptr_addr,
    input  wire  logic [FILL_W-1:0][PTR_W-1:0]    fill_rptr_data,

    // -- Scan control --------------------------------------------------------
    input  wire  logic                    scan_start,
    input  wire  logic [N_CNT_W-1:0]      scan_n_rows,    // rows to scan, 1..N_ROWS
    input  wire  logic [IDX_W-1:0]        scan_base_row,  // global row of slot 0
    output logic                          scan_busy,
    output logic                          scan_done,      // 1-cycle pulse at end

    // -- Scanner output: up to W lanes/cycle, all-or-nothing accept ----------
    output logic                          out_valid,      // a batch is presented
    output logic [W-1:0]                  out_lane_valid, // per-lane real entry (thermo)
    output logic [W-1:0][DATA_W-1:0]      out_value,
    output logic [W-1:0][IDX_W-1:0]       out_x,          // row per lane
    output logic [W-1:0][IDX_W-1:0]       out_y,          // col per lane
    input  wire  logic                    out_ready       // consumer takes the batch
);

    // -------------------------------------------------------------------------
    // Signal declarations
    // -------------------------------------------------------------------------
    logic [PTR_W-1:0]  row_ptr_q [0:RPTR_DEPTH-1];

    logic [BANK_AW-1:0] bank_rd_addr;                   // shared read addr = bank row
    logic               bank_rd_en;
    logic [ENT_W-1:0]   bank_rdata [0:NB-1];

    // Fill write steering: lane j -> bank (addr + j) % NB, local addr / NB.
    logic [NB-1:0]      bank_we;
    logic [ENT_W-1:0]   bank_wdata [0:NB-1];
    logic [BANK_AW-1:0] fill_local;

    typedef enum logic [1:0] {S_IDLE, S_ISSUE, S_STREAM, S_DRAIN} state_t;
    state_t state;

    logic [PTR_W-1:0]   cur_b;        // current batch index (global/W)
    logic [PTR_W-1:0]   total_q;      // #non-zeros to scan = row_ptr[n_rows]
    logic [N_CNT_W-1:0] n_rows_q;
    logic [IDX_W-1:0]   base_row_q;

    logic accept, last_batch;
    logic [PTR_W-1:0] next_b;

    // -------------------------------------------------------------------------
    // row_ptr flop array (CSR metadata; combinational reads)
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int i = 0; i < RPTR_DEPTH; i++) row_ptr_q[i] <= '0;
        end else begin
            for (int j = 0; j < FILL_W; j++)
                if (fill_rptr_we[j] && (int'(fill_rptr_addr) + j < RPTR_DEPTH))
                    row_ptr_q[int'(fill_rptr_addr) + j] <= fill_rptr_data[j];
        end
    end

    // -------------------------------------------------------------------------
    // Entry SRAM, split into NB banks. Bank k holds global entries g with
    // g % NB == k, at local address g / NB. A FILL_W-aligned fill beat lands
    // in FILL_W distinct banks at one shared local address; all banks read the
    // same local address (= bank row) so one aligned batch is W parallel words.
    // -------------------------------------------------------------------------
    assign fill_local = BANK_AW'(int'(fill_entry_addr) / NB);

    always_comb begin
        for (int k = 0; k < NB; k++) begin
            bank_we[k]    = 1'b0;
            bank_wdata[k] = '0;
        end
        for (int j = 0; j < FILL_W; j++) begin
            if (fill_entry_we[j]) begin
                bank_we[(int'(fill_entry_addr) + j) % NB]    = 1'b1;
                bank_wdata[(int'(fill_entry_addr) + j) % NB] = {fill_entry_value[j], fill_entry_col[j]};
            end
        end
    end

    genvar b;
    /* verilator lint_off PINCONNECTEMPTY */
    generate
        for (b = 0; b < NB; b++) begin : g_bank
            sram #(
                .DATA_WIDTH    (ENT_W),
                .ADDR_WIDTH    (BANK_AW),
                .USE_DUAL_PORT (1'b1),
                .OUTPUT_REG    (1'b0)
            ) u_bank (
                .clk         (clk),
                .rst_n       (rst_n),
                .a_en        (bank_we[b]),
                .a_we        (bank_we[b]),
                .a_addr      (fill_local),
                .a_wdata     (bank_wdata[b]),
                .a_rdata     (),
                .a_rdata_vld (),
                .b_en        (bank_rd_en),
                .b_addr      (bank_rd_addr),
                .b_rdata     (bank_rdata[b]),
                .b_rdata_vld ()
            );
        end
    endgenerate
    /* verilator lint_on PINCONNECTEMPTY */

    // -------------------------------------------------------------------------
    // Control
    // -------------------------------------------------------------------------
    assign next_b      = cur_b + PTR_W'(1);
    assign accept      = (state == S_STREAM) && out_ready;
    // Last batch once the NEXT batch would start at or beyond total.
    assign last_batch  = (next_b * PTR_W'(W)) >= total_q;

    // Bank read address: issue batch 0 in S_ISSUE, pre-issue batch b+1 on
    // accept. Batch b lives in bank row b / RGRP (consecutive batches share a
    // row when NB > W; re-reading the same row is harmless).
    always_comb begin
        bank_rd_en   = 1'b0;
        bank_rd_addr = BANK_AW'(int'(cur_b) / RGRP);
        unique case (state)
            S_ISSUE: begin
                bank_rd_en   = 1'b1;
                bank_rd_addr = BANK_AW'(int'(cur_b) / RGRP);
            end
            S_STREAM: begin
                if (accept && !last_batch) begin
                    bank_rd_en   = 1'b1;
                    bank_rd_addr = BANK_AW'(int'(next_b) / RGRP);
                end
            end
            default: ;
        endcase
    end

    // -------------------------------------------------------------------------
    // Output lanes: lane i = global g = cur_b*W + i. value/col from bank i;
    // row recovered by counting row_ptr boundaries at or below g.
    // -------------------------------------------------------------------------
    always_comb begin
        for (int i = 0; i < W; i++) begin
            logic [PTR_W-1:0] g;
            logic [IDX_W-1:0] xo;
            int sub;   // which W-wide subgroup of the bank row cur_b sits in
            sub = (RGRP < 2) ? 0 : (int'(cur_b) % RGRP);
            g  = cur_b * PTR_W'(W) + PTR_W'(i);
            xo = '0;
            for (int r = 0; r < N_ROWS; r++) begin
                if ((r < int'(n_rows_q)) && (row_ptr_q[r + 1] <= g))
                    xo = xo + IDX_W'(1);
            end
            out_lane_valid[i] = (state == S_STREAM) && (g < total_q);
            out_value[i]      = bank_rdata[sub * W + i][ENT_W-1 -: DATA_W];
            out_y[i]          = bank_rdata[sub * W + i][IDX_W-1:0];
            out_x[i]          = base_row_q + xo;
        end
    end

    assign out_valid = (state == S_STREAM);
    assign scan_busy = (state != S_IDLE);
    assign scan_done = (state == S_DRAIN);

    // -------------------------------------------------------------------------
    // Sequencer
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            cur_b      <= '0;
            total_q    <= '0;
            n_rows_q   <= '0;
            base_row_q <= '0;
        end else begin
            unique case (state)
                S_IDLE: begin
                    if (scan_start) begin
                        cur_b      <= '0;
                        n_rows_q   <= scan_n_rows;
                        base_row_q <= scan_base_row;
                        total_q    <= row_ptr_q[scan_n_rows];
                        if (row_ptr_q[scan_n_rows] == '0)
                            state <= S_DRAIN;            // empty channel
                        else
                            state <= S_ISSUE;
                    end
                end

                S_ISSUE: begin
                    // batch 0 read issued combinationally above; data next cycle
                    state <= S_STREAM;
                end

                S_STREAM: begin
                    if (accept) begin
                        if (last_batch)
                            state <= S_DRAIN;
                        else
                            cur_b <= next_b;             // next batch pre-issued above
                    end
                end

                S_DRAIN: state <= S_IDLE;

                default: state <= S_IDLE;
            endcase
        end
    end

    // -------------------------------------------------------------------------
    // Config / interface guards
    // -------------------------------------------------------------------------
    initial begin
        if (NB % W != 0)
            $fatal(1, "act_sram_scanner: FILL_W (%0d) must be a multiple of STAGE1_BATCH (%0d) when larger",
                   FILL_W, W);
    end

`ifndef SYNTHESIS
    // A fill beat must start FILL_W-aligned so lanes map 1:1 onto banks.
    always_comb begin
        assert #0 (!(|fill_entry_we) || (int'(fill_entry_addr) % FILL_W == 0))
            else $error("act_sram_scanner: fill_entry_addr %0d not FILL_W=%0d aligned",
                        fill_entry_addr, FILL_W);
    end
`endif

endmodule

`default_nettype wire
