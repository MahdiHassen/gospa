// =============================================================================
// act_sram_scanner.sv -- Activation SRAM (CSR) + CSR-to-coordinate scan
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// One on-chip activation SRAM holds the current input channel in CSR layout.
// A small FSM ("CSR to Coordinate") walks the channel and emits
// (a_xy, x=row, y=col) tuples into the APU front end -- mirroring the
// pre-processing block in the goSPA paper's PE diagram.
//
// Storage
//   Activation SRAM (sram.sv instance) -- depth N_NZ_MAX, width DATA_W+IDX_W
//     each word = {value, col_idx}; written row-major in CSR scan order.
//   row_ptr  (in-module flop array)    -- N_ROWS+1 small registers
//     row_ptr[r] = index into the SRAM where row r begins.
//
// row_ptr lives in flops, not a second SRAM, because the table is small
// (a few tens of entries) and is read combinationally by the FSM. Externally
// the scanner exposes one fill port per region (entry SRAM and row_ptr
// table), but there's only one true SRAM instance.
//
// Output convention: out_x is the row, out_y is the column. Matches
// csr_decode.sv so downstream zero_act / position_encode / idgen don't
// change. zero_act is functionally redundant here (CSR never stores
// zeros) but is left in the chain in case a future dense-bypass path is
// added.
// =============================================================================

`default_nettype none

module act_sram_scanner #(
    parameter int H        = 32,    // activation map width (full)
    parameter int N_ROWS   = 32,    // rows the scanner can hold
    parameter int N_NZ_MAX = 1024,  // max non-zeros stored in the activation SRAM
    parameter int DATA_W   = 16,    // bits per activation value

    // -- Derived widths --------------------------------------------------------
    localparam int IDX_W   = (H            < 2) ? 1 : $clog2(H),
    localparam int PTR_W   = (N_NZ_MAX + 1 < 2) ? 1 : $clog2(N_NZ_MAX + 1),
    localparam int ENT_W   = DATA_W + IDX_W,
    localparam int ENT_AW  = (N_NZ_MAX     < 2) ? 1 : $clog2(N_NZ_MAX),
    localparam int RPTR_DEPTH = N_ROWS + 1,
    localparam int RPTR_AW = (RPTR_DEPTH   < 2) ? 1 : $clog2(RPTR_DEPTH),
    localparam int N_CNT_W = (N_ROWS + 1   < 2) ? 1 : $clog2(N_ROWS + 1)
)(
    input  wire  logic                    clk,
    input  wire  logic                    rst_n,

    // -- Fill: activation SRAM (one CSR entry per write) ---------------------
    input  wire  logic                    fill_entry_we,
    input  wire  logic [ENT_AW-1:0]       fill_entry_addr,
    input  wire  logic [DATA_W-1:0]       fill_entry_value,
    input  wire  logic [IDX_W-1:0]        fill_entry_col,

    // -- Fill: row_ptr flop array (one pointer per write) --------------------
    input  wire  logic                    fill_rptr_we,
    input  wire  logic [RPTR_AW-1:0]      fill_rptr_addr,
    input  wire  logic [PTR_W-1:0]        fill_rptr_data,

    // -- Scan control --------------------------------------------------------
    input  wire  logic                    scan_start,
    input  wire  logic [N_CNT_W-1:0]      scan_n_rows,    // rows to scan, 1..N_ROWS
    input  wire  logic [IDX_W-1:0]        scan_base_row,  // global row of slot 0
    output logic                          scan_busy,
    output logic                          scan_done,      // 1-cycle pulse at end

    // -- Scanner output handshake (val, x=row, y=col) ------------------------
    output logic                          out_valid,
    output logic [DATA_W-1:0]             out_value,
    output logic [IDX_W-1:0]              out_x,
    output logic [IDX_W-1:0]              out_y,
    input  wire  logic                    out_ready
);

    // -------------------------------------------------------------------------
    // The single activation SRAM. Port A = fill (write), Port B = scan (read).
    // Each word packs {value, col_idx}; row index is derived externally via
    // the row_ptr flop array below.
    // -------------------------------------------------------------------------
    logic              ent_rd_en;
    logic [ENT_AW-1:0] ent_rd_addr;
    logic [ENT_W-1:0]  ent_rd_data;

    /* verilator lint_off PINCONNECTEMPTY */
    sram #(
        .DATA_WIDTH    (ENT_W),
        .ADDR_WIDTH    (ENT_AW),
        .USE_DUAL_PORT (1'b1),
        .OUTPUT_REG    (1'b0)
    ) u_act_sram (
        .clk         (clk),
        .rst_n       (rst_n),
        .a_en        (fill_entry_we),
        .a_we        (fill_entry_we),
        .a_addr      (fill_entry_addr),
        .a_wdata     ({fill_entry_value, fill_entry_col}),
        .a_rdata     (),
        .a_rdata_vld (),
        .b_en        (ent_rd_en),
        .b_addr      (ent_rd_addr),
        .b_rdata     (ent_rd_data),
        .b_rdata_vld ()
    );
    /* verilator lint_on PINCONNECTEMPTY */

    // -------------------------------------------------------------------------
    // row_ptr flop array (small CSR metadata; read combinationally by FSM).
    // -------------------------------------------------------------------------
    logic [PTR_W-1:0] row_ptr_q [0:RPTR_DEPTH-1];

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int i = 0; i < RPTR_DEPTH; i++) row_ptr_q[i] <= '0;
        end else if (fill_rptr_we) begin
            row_ptr_q[fill_rptr_addr] <= fill_rptr_data;
        end
    end

    // -------------------------------------------------------------------------
    // Scan FSM
    //   IDLE    : waiting for scan_start.
    //   ISSUE   : issue activation-SRAM read for entry_idx (no in-flight emit
    //             allowed if we'd overwrite stalled data; gated by out_ready).
    //   STREAM  : entry_rdata holds (val, col); emit (val, cur_row, col).
    //   DRAIN   : last entry accepted -> pulse scan_done, back to IDLE.
    //
    // Row pointers come straight from the row_ptr flop array (no extra read
    // latency). When the current row is exhausted we advance cur_row and
    // re-evaluate; empty rows skip without firing an entry read.
    // -------------------------------------------------------------------------
    typedef enum logic [1:0] {S_IDLE, S_ISSUE, S_STREAM, S_DRAIN} state_t;
    state_t state;

    logic [IDX_W-1:0]    cur_row;
    logic [PTR_W-1:0]    entry_idx;          // next entry to fetch
    logic [PTR_W-1:0]    row_end_q;          // row_ptr[cur_row+1] cached
    logic [IDX_W-1:0]    row_dly;            // row tag of in-flight entry
    logic [N_CNT_W-1:0]  n_rows_q;
    logic [IDX_W-1:0]    base_row_q;
    logic                in_flight;          // entry rdata is a pending emit

    // Combinational pointers off the row_ptr flop array. We index with
    // RPTR_AW bits because cur_row + 1 can equal H (= row_ptr terminator),
    // which doesn't fit in IDX_W bits.
    logic [PTR_W-1:0] row_start_now, row_end_now;
    logic [RPTR_AW-1:0] cur_row_idx;
    assign cur_row_idx   = {{(RPTR_AW - IDX_W){1'b0}}, cur_row};
    assign row_start_now = row_ptr_q[cur_row_idx];
    assign row_end_now   = row_ptr_q[cur_row_idx + RPTR_AW'(1)];

    logic last_in_row, last_row, accept;
    assign accept      = in_flight && out_ready;
    assign last_in_row = (entry_idx + PTR_W'(1) == row_end_q);
    assign last_row    = (cur_row == IDX_W'(n_rows_q - 1));

    // Look-ahead helper: when accepting the current entry, what's the next
    // entry_idx, and does it overflow the current row?
    logic [PTR_W-1:0] next_entry_idx;
    assign next_entry_idx = entry_idx + PTR_W'(1);

    // -- Outputs --------------------------------------------------------------
    assign out_valid = (state == S_STREAM) && in_flight;
    assign out_value = ent_rd_data[ENT_W-1 -: DATA_W];
    assign out_y     = ent_rd_data[IDX_W-1:0];
    assign out_x     = base_row_q + row_dly;

    assign scan_busy = (state != S_IDLE);
    assign scan_done = (state == S_DRAIN);

    // -- SRAM enable / address (single combinational driver) -----------------
    always_comb begin
        ent_rd_en   = 1'b0;
        ent_rd_addr = entry_idx[ENT_AW-1:0];
        unique case (state)
            // Issuing the first entry of a (non-empty) row.
            S_ISSUE: begin
                if (row_end_now != row_start_now) begin
                    ent_rd_en   = 1'b1;
                    ent_rd_addr = row_start_now[ENT_AW-1:0];
                end
            end
            // Streaming: pre-issue the NEXT entry on accept, unless that
            // would be off the end of the row.
            S_STREAM: begin
                if (accept && !last_in_row) begin
                    ent_rd_en   = 1'b1;
                    ent_rd_addr = next_entry_idx[ENT_AW-1:0];
                end
            end
            default: ;
        endcase
    end

    // -- Sequencer ------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            cur_row    <= '0;
            entry_idx  <= '0;
            row_end_q  <= '0;
            row_dly    <= '0;
            in_flight  <= 1'b0;
            n_rows_q   <= '0;
            base_row_q <= '0;
        end else begin
            unique case (state)
                S_IDLE: begin
                    if (scan_start) begin
                        cur_row    <= '0;
                        n_rows_q   <= scan_n_rows;
                        base_row_q <= scan_base_row;
                        in_flight  <= 1'b0;
                        state      <= S_ISSUE;
                    end
                end

                S_ISSUE: begin
                    // We have row_start / row_end available from the row_ptr
                    // flop array combinationally.
                    if (row_end_now == row_start_now) begin
                        // Empty row -- skip without firing a read.
                        if (last_row) begin
                            state <= S_DRAIN;
                        end else begin
                            cur_row <= cur_row + IDX_W'(1);
                        end
                    end else begin
                        // Issue (combinational override above presents the
                        // address); data appears next cycle.
                        entry_idx <= row_start_now;
                        row_end_q <= row_end_now;
                        row_dly   <= cur_row;
                        in_flight <= 1'b1;
                        state     <= S_STREAM;
                    end
                end

                S_STREAM: begin
                    if (accept) begin
                        if (last_in_row) begin
                            // Row finished -- advance row.
                            in_flight <= 1'b0;
                            if (last_row) begin
                                state <= S_DRAIN;
                            end else begin
                                cur_row <= cur_row + IDX_W'(1);
                                state   <= S_ISSUE;
                            end
                        end else begin
                            // Pre-issued next entry above; tag and stay here.
                            entry_idx <= next_entry_idx;
                            row_dly   <= cur_row;
                            in_flight <= 1'b1;
                        end
                    end
                end

                S_DRAIN: begin
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
