// =============================================================================
// fifo.sv -- Parameterized Synchronous FIFO
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// In GoSPA this module is instantiated as:
//
//   FIFO-A (one per PID slot, F^2 total):
//     DATA_WIDTH = ACT_W + CID_W   (e.g. 16+6 = 22 bits)
//     DEPTH      = 64
//     Payload    = { act_value[ACT_W-1:0], cid[CID_W-1:0] }
//     PID is implicit - it is the array index, never stored inside.
//
//   FIFO-B (one per PE, N_PE total):
//     DATA_WIDTH = ACT_W + PID_W + CID_W   (e.g. 16+4+6 = 26 bits)
//     DEPTH      = 64
//     Payload    = { act_value[ACT_W-1:0], pid[PID_W-1:0], cid[CID_W-1:0] }
//
// Interface: AXI-Stream style valid/ready handshake on both ports.
//
// Constraints:
//   - DEPTH must be a power of 2 (pointer wrap-around relies on natural overflow).
//   - Simultaneous push and pop are fully supported at any occupancy.
//   - Overflow  (push when full) : wr_ready is de-asserted; push is dropped.
//   - Underflow (pop when empty) : rd_valid is de-asserted; rd_data is stale.
//   - Asynchronous read: rd_data updates immediately when rd_ptr changes.
// =============================================================================

`default_nettype none

module fifo #(
    parameter int DATA_WIDTH = 22,   // payload width in bits
    parameter int DEPTH      = 64    // number of entries; MUST be a power of 2
) (
    input  wire  logic                    clk,
    input  wire  logic                    rst_n,    // active-low synchronous reset

    // ---- Write / Producer port -----------------------------------------------
    input  wire  logic                    wr_valid,  // producer presents valid data
    input  wire  logic [DATA_WIDTH-1:0]   wr_data,
    output logic                          wr_ready,  // FIFO can accept (not full)

    // ---- Read / Consumer port ------------------------------------------------
    output logic                          rd_valid,  // FIFO has data (not empty)
    output logic [DATA_WIDTH-1:0]         rd_data,
    input  wire  logic                    rd_ready,  // consumer accepts this cycle

    // ---- Status --------------------------------------------------------------
    output logic                          full,
    output logic                          empty,
    output logic [$clog2(DEPTH):0]        count      // occupancy: 0 .. DEPTH
);

    // -------------------------------------------------------------------------
    // Derived widths
    // -------------------------------------------------------------------------
    localparam int PTR_W = $clog2(DEPTH);   // pointer width   (e.g. 6 for DEPTH=64)
    localparam int CNT_W = PTR_W + 1;       // counter width - one extra bit so we
                                             // can represent DEPTH without aliasing

    // -------------------------------------------------------------------------
    // Storage
    // -------------------------------------------------------------------------
    logic [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    // -------------------------------------------------------------------------
    // Pointers and occupancy counter
    // -------------------------------------------------------------------------
    logic [PTR_W-1:0] wr_ptr;
    logic [PTR_W-1:0] rd_ptr;
    logic [CNT_W-1:0] cnt;

    // -------------------------------------------------------------------------
    // Handshake qualifiers (transaction occurs only when both sides agree)
    // -------------------------------------------------------------------------
    logic do_push, do_pop;
    assign do_push = wr_valid & wr_ready;
    assign do_pop  = rd_valid & rd_ready;

    // -------------------------------------------------------------------------
    // Write pointer - wraps naturally at DEPTH (power-of-2 overflow)
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) wr_ptr <= '0;
        else if (do_push) wr_ptr <= wr_ptr + 1'b1;
    end

    // -------------------------------------------------------------------------
    // Read pointer
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) rd_ptr <= '0;
        else if (do_pop) rd_ptr <= rd_ptr + 1'b1;
    end

    // -------------------------------------------------------------------------
    // Occupancy counter
    //   Simultaneous push + pop -> no change (+/-1 cancel out)
    //   Push only              -> +1
    //   Pop  only              -> -1
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) cnt <= '0;
        else unique case ({do_push, do_pop})
            2'b10:   cnt <= cnt + 1'b1;
            2'b01:   cnt <= cnt - 1'b1;
            default: cnt <= cnt;
        endcase
    end

    // -------------------------------------------------------------------------
    // Memory - synchronous write, asynchronous read
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (do_push) mem[wr_ptr] <= wr_data;
    end

    // -------------------------------------------------------------------------
    // Outputs - all combinatorial from cnt/ptrs
    // -------------------------------------------------------------------------
    assign full     = (cnt == CNT_W'(DEPTH));
    assign empty    = (cnt == '0);
    assign count    = cnt;

    assign wr_ready = ~full;
    assign rd_valid = ~empty;
    assign rd_data  = mem[rd_ptr];    // stale when empty; consumer must check rd_valid

endmodule

`default_nettype wire
