// =============================================================================
// pe_mem.sv -- PE weight memory: single PID-sorted bank + fill bookkeeping
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// One kernel per PE (V2 "act" dataflow), so the PE holds a single weight bank
// shared by all M multipliers -- no per-lane banks.
//   - Port A: one weight per cycle {wfill_val, wfill_pid} appended in PID order;
//     slot and WSP are DERIVED from this stream (no separate slot/WSP ports).
//   - fill_restart: a bare wload_done re-arms the same weights; the first wfill
//     after an arm starts a fresh session (prior count/WSP cleared).
//   - Port B: the single read port (Next look-ahead), 1-cycle latency, unpacked
//     into rd_val / rd_pid.
//
// Slot s holds the s-th nonzero weight in PID order; wsp_q[p]=1 marks a nonzero.
// =============================================================================

`default_nettype none

module pe_mem #(
    parameter int NUM_PID    = 9,
    parameter int DATA_WIDTH = 16,

    // -- Derived --------------------------------------------------------------
    localparam int PID_WIDTH   = (NUM_PID < 2) ? 1 : $clog2(NUM_PID),
    localparam int RPTR_WIDTH  = $clog2(NUM_PID + 1),
    localparam int SLOT_WIDTH  = (NUM_PID < 2) ? 1 : $clog2(NUM_PID),
    localparam int WSRAM_DW    = DATA_WIDTH + PID_WIDTH
)(
    input  logic                          clk,
    input  logic                          rst_n,

    // -- Weight fill (Port A) -- one weight per cycle ------------------------
    input  logic                          wfill_we,
    input  logic [PID_WIDTH-1:0]          wfill_pid,
    input  logic signed [DATA_WIDTH-1:0]  wfill_val,
    input  logic                          wload_done,

    // -- Single read port (Port B) -- 1-cycle latency ------------------------
    input  logic                          rd_en,
    input  logic [SLOT_WIDTH-1:0]         rd_slot,
    output logic [DATA_WIDTH-1:0]         rd_val,
    output logic [PID_WIDTH-1:0]          rd_pid,

    // -- Derived state exported to pe_fetch / router -------------------------
    output logic [NUM_PID-1:0]            wsp_q        // weight sparsity pattern
);

    // -------------------------------------------------------------------------
    // Signal declarations
    // -------------------------------------------------------------------------
    logic                    fill_restart;   // first write of a new session
    logic [RPTR_WIDTH-1:0]   wfill_cnt;      // # weights written = next slot
    logic [SLOT_WIDTH-1:0]   wfill_slot_eff;
    logic                    bank_we;
    logic [WSRAM_DW-1:0]     bank_rdata;

    // -------------------------------------------------------------------------
    // Fill bookkeeping: count and WSP are derived from the fill stream.
    // fill_restart marks the first write of a new session (clears prior state).
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            wfill_cnt    <= '0;
            fill_restart <= 1'b0;
            wsp_q        <= '0;
        end else begin
            if (wload_done)
                fill_restart <= 1'b1;

            if (wfill_we) begin
                if (fill_restart) begin
                    wfill_cnt    <= RPTR_WIDTH'(1);
                    wsp_q        <= (NUM_PID'(1) << wfill_pid);
                    fill_restart <= 1'b0;
                end else begin
                    wfill_cnt        <= wfill_cnt + RPTR_WIDTH'(1);
                    wsp_q[wfill_pid] <= 1'b1;
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // Weight SRAM: Port A = fill, Port B = the Next look-ahead read port.
    // -------------------------------------------------------------------------
    /* verilator lint_off PINCONNECTEMPTY */
    sram #(
        .DATA_WIDTH    (WSRAM_DW),
        .ADDR_WIDTH    (SLOT_WIDTH),
        .USE_DUAL_PORT (1'b1),
        .OUTPUT_REG    (1'b0)
    ) u_wsram (
        .clk         (clk),
        .rst_n       (rst_n),
        .a_en        (bank_we),
        .a_we        (bank_we),
        .a_addr      (wfill_slot_eff),
        .a_wdata     ({wfill_val, wfill_pid}),
        .a_rdata     (),
        .a_rdata_vld (),
        .b_en        (rd_en),
        .b_addr      (rd_slot),
        .b_rdata     (bank_rdata),
        .b_rdata_vld ()
    );
    /* verilator lint_on PINCONNECTEMPTY */

    // -------------------------------------------------------------------------
    // Combinational assigns
    // -------------------------------------------------------------------------
    assign bank_we        = wfill_we;
    assign wfill_slot_eff = fill_restart ? SLOT_WIDTH'(0) : SLOT_WIDTH'(wfill_cnt);
    assign rd_pid         = bank_rdata[PID_WIDTH-1:0];
    assign rd_val         = bank_rdata[WSRAM_DW-1 -: DATA_WIDTH];

endmodule

`default_nettype wire
