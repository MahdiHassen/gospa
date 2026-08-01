// =============================================================================
// pe_mem.sv -- PE weight memory: DOUBLE-BUFFERED PID-sorted banks
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// One kernel per PE (V2 "act" dataflow); the physical SRAM holds TWO banks so
// the next round's kernel streams in WHILE the current one is being used:
//   - Port A (fill): one weight per cycle {wfill_val, wfill_pid} in PID order,
//     always written into the INACTIVE bank; slot and WSP are derived from the
//     stream into wsp_fill_q. Filling never disturbs the live bank or the
//     exported WSP, so it is safe during routing/MACs.
//   - wload_done: 1-cycle BANK SWAP -- the just-filled bank becomes active and
//     its WSP is published (wsp_q -> router). Always fill a session before
//     arming; a bare re-arm would publish the other bank's stale pattern.
//   - Port B: the single read port (Next look-ahead) into the ACTIVE bank,
//     1-cycle latency, unpacked into rd_val / rd_pid.
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
    logic                    active_q;       // bank the fetch/router read
    logic [NUM_PID-1:0]      wsp_fill_q;     // WSP of the session being filled

    // -------------------------------------------------------------------------
    // Fill bookkeeping: count and WSP are derived from the fill stream, into
    // the shadow (inactive) bank's state. wload_done swaps banks and publishes
    // the freshly filled WSP; fill_restart marks the next session's first write.
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            wfill_cnt    <= '0;
            fill_restart <= 1'b0;
            wsp_q        <= '0;
            wsp_fill_q   <= '0;
            active_q     <= 1'b0;
        end else begin
            if (wload_done) begin
                fill_restart <= 1'b1;
                active_q     <= ~active_q;
                wsp_q        <= wsp_fill_q;
            end

            if (wfill_we) begin
                if (fill_restart) begin
                    wfill_cnt    <= RPTR_WIDTH'(1);
                    wsp_fill_q   <= (NUM_PID'(1) << wfill_pid);
                    fill_restart <= 1'b0;
                end else begin
                    wfill_cnt             <= wfill_cnt + RPTR_WIDTH'(1);
                    wsp_fill_q[wfill_pid] <= 1'b1;
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // Weight SRAM, 2 banks deep: Port A = fill (inactive bank), Port B = the
    // Next look-ahead read port (active bank).
    // -------------------------------------------------------------------------
    /* verilator lint_off PINCONNECTEMPTY */
    sram #(
        .DATA_WIDTH    (WSRAM_DW),
        .ADDR_WIDTH    (SLOT_WIDTH + 1),
        .USE_DUAL_PORT (1'b1),
        .OUTPUT_REG    (1'b0)
    ) u_wsram (
        .clk         (clk),
        .rst_n       (rst_n),
        .a_en        (bank_we),
        .a_we        (bank_we),
        .a_addr      ({~active_q, wfill_slot_eff}),
        .a_wdata     ({wfill_val, wfill_pid}),
        .a_rdata     (),
        .a_rdata_vld (),
        .b_en        (rd_en),
        .b_addr      ({active_q, rd_slot}),
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
