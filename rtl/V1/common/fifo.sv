// =============================================================================
// fifo.sv -- Parameterized Synchronous FIFO (block-RAM friendly)
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
// Interface: AXI-Stream style valid/ready handshake on both ports, with
// FIRST-WORD-FALL-THROUGH read semantics -- whenever rd_valid is high, rd_data
// already holds the head, and asserting rd_ready pops it that same cycle. The
// downstream consumers depend on this (routing.sv advances FIFO-A lanes on
// rd_valid going low, and the PE MACs combinationally off rd_data).
//
// --- Storage: synchronous write + REGISTERED read (maps to block RAM) --------
// The bulk storage is a simple dual-port RAM with a synchronous write port and
// a registered (1-cycle latency) read port -- the template FPGA synthesis maps
// to a block RAM. An older revision used an asynchronous (combinational) read
// (`rd_data = mem[rd_ptr]`), which forced LUT-RAM / flop arrays and never hit a
// BRAM; at DEPTH=2048 that was a large area/timing problem.
//
// To keep the FWFT interface on top of a 1-cycle RAM, a small register-based
// "show-ahead" output buffer (OBUF_D entries) sits in front of the RAM. It is
// kept primed by issuing RAM reads whenever the RAM holds data and the buffer
// (plus any in-flight read) has room, so the head is continuously presented and
// rd_valid never glitches low while data remains. OBUF_D=4 (>=3) is enough to
// hide the 2-stage read pipeline (RAM output reg -> buffer load) without a
// bubble under back-to-back pops -- see the analysis in the read path below.
//
// Latency note vs. the old async-read FIFO: a word written into an EMPTY FIFO
// becomes visible on rd_data a couple of cycles later (RAM read + buffer load)
// instead of next cycle. `count` still reflects total occupancy immediately, so
// almost_empty (count==1) stays exact; consumers only ever act on a word when
// rd_valid is high, so the extra fill latency is transparent to them.
//
// Constraints:
//   - DEPTH must be a power of 2 (pointer wrap-around relies on natural overflow).
//   - Simultaneous push and pop are fully supported at any occupancy.
//   - Overflow  (push when full) : wr_ready is de-asserted; push is dropped.
//   - Underflow (pop when empty) : rd_valid is de-asserted; rd_data is stale.
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
    localparam int PTR_W   = $clog2(DEPTH);   // RAM pointer width (e.g. 6 for DEPTH=64)
    localparam int CNT_W   = PTR_W + 1;       // occupancy width (represent 0..DEPTH)

    // Output show-ahead buffer: register FIFO that restores fall-through on top
    // of the 1-cycle RAM read. Depth 4 (power of 2) >= 3 hides the read pipeline
    // so the head stays continuously valid under back-to-back pops.
    localparam int OBUF_D   = 4;
    localparam int OBPTR_W  = $clog2(OBUF_D); // 2
    localparam int OBCNT_W  = OBPTR_W + 1;    // represent 0..OBUF_D

    // -------------------------------------------------------------------------
    // Deep storage: synchronous write, registered read => infers block RAM.
    // -------------------------------------------------------------------------
    logic [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    logic [DATA_WIDTH-1:0] mem_q;             // BRAM output register (1-cycle read)

    logic [PTR_W-1:0]      wr_ptr;            // RAM write pointer
    logic [PTR_W-1:0]      rd_ptr;            // RAM read-issue pointer
    logic [CNT_W-1:0]      cnt;               // TOTAL occupancy (RAM + in-flight + obuf)
    logic [CNT_W-1:0]      ram_cnt;           // words resident in RAM, not yet read-issued
    logic                  rd_en_q;           // a RAM read was issued last cycle (mem_q valid now)

    // -------------------------------------------------------------------------
    // Show-ahead output buffer: small register FIFO fed by the RAM read.
    // -------------------------------------------------------------------------
    logic [DATA_WIDTH-1:0] obuf [0:OBUF_D-1];
    logic [OBPTR_W-1:0]    ob_head, ob_tail;
    logic [OBCNT_W-1:0]    ob_cnt;

    // -------------------------------------------------------------------------
    // Handshake qualifiers
    // -------------------------------------------------------------------------
    logic do_push, do_pop, mem_rd_en;

    assign wr_ready = (cnt != CNT_W'(DEPTH));
    assign full     = (cnt == CNT_W'(DEPTH));
    assign empty    = (cnt == '0);
    assign count    = cnt;

    assign rd_valid = (ob_cnt != '0);         // head present iff output buffer non-empty
    assign rd_data  = obuf[ob_head];

    assign do_push  = wr_valid & wr_ready;
    assign do_pop   = rd_valid & rd_ready;

    // Issue a RAM read when the RAM holds data AND the read-side path
    // (output buffer + the at-most-one in-flight read) still has room. With the
    // gate `rsv < OBUF_D`, the committed read-side count rsv = ob_cnt + in-flight
    // is bounded by OBUF_D, so the buffer never overflows and -- because
    // OBUF_D >= 3 -- never underflows mid-stream either.
    logic [OBCNT_W-1:0] rsv;
    assign rsv       = ob_cnt + OBCNT_W'(rd_en_q);
    assign mem_rd_en = (ram_cnt != '0) && (rsv < OBCNT_W'(OBUF_D));

    // -------------------------------------------------------------------------
    // Next-state occupancy counters (split +1/-1 so widths stay clean)
    // -------------------------------------------------------------------------
    logic [CNT_W-1:0]  cnt_n, ram_cnt_n;
    logic [OBCNT_W-1:0] ob_cnt_n;

    always_comb begin
        cnt_n = cnt;
        if      (do_push && !do_pop) cnt_n = cnt + CNT_W'(1);
        else if (!do_push && do_pop) cnt_n = cnt - CNT_W'(1);

        ram_cnt_n = ram_cnt;
        if      (do_push && !mem_rd_en) ram_cnt_n = ram_cnt + CNT_W'(1);
        else if (!do_push && mem_rd_en) ram_cnt_n = ram_cnt - CNT_W'(1);

        ob_cnt_n = ob_cnt;
        if      (rd_en_q && !do_pop) ob_cnt_n = ob_cnt + OBCNT_W'(1);
        else if (!rd_en_q && do_pop) ob_cnt_n = ob_cnt - OBCNT_W'(1);
    end

    // -------------------------------------------------------------------------
    // RAM write port (synchronous)
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (do_push) mem[wr_ptr] <= wr_data;
    end

    // -------------------------------------------------------------------------
    // RAM read port (registered output -> block-RAM output register)
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (mem_rd_en) mem_q <= mem[rd_ptr];
    end

    // -------------------------------------------------------------------------
    // Pointers, counters, and the show-ahead output buffer
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            wr_ptr  <= '0;
            rd_ptr  <= '0;
            cnt     <= '0;
            ram_cnt <= '0;
            rd_en_q <= 1'b0;
            ob_head <= '0;
            ob_tail <= '0;
            ob_cnt  <= '0;
        end else begin
            if (do_push)   wr_ptr <= wr_ptr + PTR_W'(1);
            if (mem_rd_en) rd_ptr <= rd_ptr + PTR_W'(1);

            cnt     <= cnt_n;
            ram_cnt <= ram_cnt_n;
            ob_cnt  <= ob_cnt_n;

            // A RAM read issued this cycle lands in mem_q next cycle.
            rd_en_q <= mem_rd_en;

            // Output buffer: append the just-landed RAM word at the tail,
            // pop the head when the consumer accepts. Both may occur together.
            if (rd_en_q) begin
                obuf[ob_tail] <= mem_q;
                ob_tail       <= ob_tail + OBPTR_W'(1);
            end
            if (do_pop) ob_head <= ob_head + OBPTR_W'(1);
        end
    end

`ifndef SYNTHESIS
    // -------------------------------------------------------------------------
    // Simulation sanity checks (stripped for synthesis)
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst_n) begin
            assert (ob_cnt <= OBCNT_W'(OBUF_D))
                else $error("fifo: output buffer overflow ob_cnt=%0d", ob_cnt);
            assert (!(do_pop && ob_cnt == '0))
                else $error("fifo: pop while output buffer empty");
            assert (cnt <= CNT_W'(DEPTH))
                else $error("fifo: occupancy overflow cnt=%0d", cnt);
        end
    end
`endif

endmodule

`default_nettype wire
