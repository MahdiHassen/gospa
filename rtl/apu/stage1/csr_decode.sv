// =============================================================================
// csr_decode.sv -- CSR Format Decoder
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// Converts a CSR-format sparse activation matrix into a flat stream of
// (value, x, y) coordinate tuples, one entry per clock cycle.
//
// CSR format recap:
//   row_ptr[r]   -- index into values[] where row r starts
//   row_ptr[r+1] -- index where row r ends (exclusive)
//   values[k]    -- non-zero value at position k
//   col_idx[k]   -- column (y coordinate) of values[k]
//   row          -- implicit from which row_ptr interval k falls in (= x)
//
// This module accepts two independent input streams:
//   1. row_ptr stream  -- one entry per row (H+1 entries total for H rows)
//   2. entry stream    -- (value, col) pairs in row-major order
//
// It outputs one (value, x, y) tuple per cycle, including zero values.
// zero_act.sv downstream filters out zeros before IDGen.
//
// Backpressure: out_ready stalls entry consumption; row_ptr is consumed
// one cycle ahead of the entries it governs.
//
// Constraints:
//   - H must be a power of 2 for clean $clog2 widths (e.g. 32, 64)
//   - row_ptr values must be monotonically non-decreasing
//   - Exactly H+1 row_ptr entries expected per matrix
// =============================================================================

module csr_decode #(
    parameter int H      = 32,   // activation map size (H x H, square)
    parameter int DATA_W = 16    // activation value width in bits
)(
    input  logic                      clk,
    input  logic                      rst_n,

    // -- row_ptr stream (H+1 entries per matrix) ---------------------------
    input  logic                      row_ptr_valid,
    input  logic [$clog2(H*H):0]      row_ptr_data,   // up to H*H non-zeros
    output logic                      row_ptr_ready,

    // -- entry stream (value + column index, one per non-zero) -------------
    input  logic                      entry_valid,
    input  logic [DATA_W-1:0]         entry_value,
    input  logic [$clog2(H)-1:0]      entry_col,      // y coordinate
    output logic                      entry_ready,

    // -- output stream (value, x, y) ---------------------------------------
    output logic                      out_valid,
    output logic [DATA_W-1:0]         out_value,
    output logic [$clog2(H)-1:0]      out_x,
    output logic [$clog2(H)-1:0]      out_y,
    input  logic                      out_ready
);

    // -------------------------------------------------------------------------
    // State machine
    // -------------------------------------------------------------------------
    typedef enum logic [1:0] {
        LOAD_ROW_START,   // waiting for row_ptr[r]   (first ptr of new row)
        LOAD_ROW_END,     // waiting for row_ptr[r+1] (end ptr of same row)
        STREAM            // emitting entries for current row
    } state_t;

    state_t state;

    // -------------------------------------------------------------------------
    // Registers
    // -------------------------------------------------------------------------
    logic [$clog2(H)-1:0]      row_ctr;        // current row = x coordinate
    logic [$clog2(H*H):0]      row_start;      // row_ptr[r]
    logic [$clog2(H*H):0]      row_end;        // row_ptr[r+1]
    logic [$clog2(H*H):0]      entry_ctr;      // how many entries emitted this row

    // -------------------------------------------------------------------------
    // Derived: number of entries left in current row
    // -------------------------------------------------------------------------
    logic [$clog2(H*H):0] entries_in_row;
    assign entries_in_row = row_end - row_start;

    // -------------------------------------------------------------------------
    // Output -- combinational from entry stream when in STREAM state
    // -------------------------------------------------------------------------
    assign out_valid  = (state == STREAM) && entry_valid;
    assign out_value  = entry_value;
    assign out_x      = row_ctr;
    assign out_y      = entry_col;

    // Consume an entry only when we are streaming AND downstream accepts
    logic do_emit;
    assign do_emit = out_valid && out_ready;

    // entry_ready: accept entries only in STREAM state and when downstream ready
    assign entry_ready = (state == STREAM) && out_ready;

    // row_ptr_ready: accept row_ptr only when we need one
    assign row_ptr_ready = (state == LOAD_ROW_START) || (state == LOAD_ROW_END);

    // -------------------------------------------------------------------------
    // State machine
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state     <= LOAD_ROW_START;
            row_ctr   <= '0;
            row_start <= '0;
            row_end   <= '0;
            entry_ctr <= '0;
        end else begin
            unique case (state)

                // -- Consume row_ptr[r] ---------------------------------------
                LOAD_ROW_START: begin
                    if (row_ptr_valid) begin
                        row_start <= row_ptr_data;
                        state     <= LOAD_ROW_END;
                    end
                end

                // -- Consume row_ptr[r+1] -------------------------------------
                LOAD_ROW_END: begin
                    if (row_ptr_valid) begin
                        row_end   <= row_ptr_data;
                        entry_ctr <= '0;
                        // Empty row: skip directly to next row
                        if (row_ptr_data == row_start) begin
                            row_ctr   <= row_ctr + 1'b1;
                            row_start <= row_ptr_data;
                            state     <= LOAD_ROW_END;
                        end else begin
                            state <= STREAM;
                        end
                    end
                end

                // -- Emit entries for current row -----------------------------
                STREAM: begin
                    if (do_emit) begin
                        entry_ctr <= entry_ctr + 1'b1;
                        // Last entry in this row?
                        if (entry_ctr == entries_in_row - 1) begin
                            row_ctr   <= row_ctr + 1'b1;
                            row_start <= row_end;
                            state     <= LOAD_ROW_END;
                        end
                    end
                end

                default: state <= LOAD_ROW_START;
            endcase
        end
    end

endmodule
