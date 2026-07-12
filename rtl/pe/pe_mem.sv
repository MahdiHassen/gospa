// =============================================================================
// pe_mem.sv -- PE weight memory: per-lane banked SRAM + fill bookkeeping
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// Owns the per-PE weight memory and everything on the "load" side:
//   - Port A: one weight per cycle {wfill_val, wfill_pid} appended to lane's
//     slots. slot, per-lane count (wfill_cnt) and WSP (wsp_q) are all DERIVED
//     from this stream; the host drives no separate slot / count / WSP ports.
//   - fill_restart: a bare wload_done re-arms the same weights; the first
//     wfill after an arm starts a fresh session (prior counts/WSP cleared).
//   - Banked read side: one SRAM bank per lane, so every lane has its own
//     read port and can fetch a weight independently every cycle (no arbiter).
//     1-cycle latency; read data is unpacked into rd_val[k] / rd_pid[k].
//
// Within a bank, slot s holds the lane's s-th weight in PID order.
// =============================================================================

`default_nettype none

module pe_mem #(
    parameter int NUM_MULTS  = 4,
    parameter int NUM_PID    = 9,
    parameter int DATA_WIDTH = 16,

    // -- Derived --------------------------------------------------------------
    localparam int PID_WIDTH   = (NUM_PID   < 2) ? 1 : $clog2(NUM_PID),
    localparam int LANE_WIDTH  = (NUM_MULTS < 2) ? 1 : $clog2(NUM_MULTS),
    localparam int RPTR_WIDTH  = $clog2(NUM_PID + 1),
    localparam int SLOT_WIDTH  = (NUM_PID < 2) ? 1 : $clog2(NUM_PID),
    localparam int WSRAM_DW    = DATA_WIDTH + PID_WIDTH
)(
    input  logic                                  clk,
    input  logic                                  rst_n,

    // -- Weight fill (Port A) -- one weight per cycle -------------------------
    input  logic                                  wfill_we,
    input  logic [LANE_WIDTH-1:0]                 wfill_lane,
    input  logic [PID_WIDTH-1:0]                  wfill_pid,
    input  logic signed [DATA_WIDTH-1:0]          wfill_val,
    input  logic                                  wload_done,

    // -- Per-lane read ports (Port B per bank) -- 1-cycle latency ------------
    input  logic [NUM_MULTS-1:0]                  rd_en,
    input  logic [NUM_MULTS-1:0][SLOT_WIDTH-1:0]  rd_slot,
    output logic [NUM_MULTS-1:0][DATA_WIDTH-1:0]  rd_val,
    output logic [NUM_MULTS-1:0][PID_WIDTH-1:0]   rd_pid,

    // -- Derived per-lane state exported to pe_fetch / action-eval -----------
    output logic [NUM_MULTS-1:0][NUM_PID-1:0]     wsp_q        // weight sparsity pattern
);

    // -------------------------------------------------------------------------
    // Fill pointer = # weights written this session = next slot to write.
    // fill_restart marks the first write of a new session, which clears the
    // prior count/WSP. The count is internal now (pe_fetch derives its live
    // weight count from popcount(wsp_q)).
    // -------------------------------------------------------------------------
    logic                   fill_restart;
    logic [SLOT_WIDTH-1:0]  wfill_slot_eff;
    logic [NUM_MULTS-1:0][RPTR_WIDTH-1:0]  wfill_cnt;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            wfill_cnt    <= '0;
            fill_restart <= 1'b0;
            for (int k = 0; k < NUM_MULTS; k++)
                wsp_q[k] <= '0;
        end else begin
            if (wload_done)
                fill_restart <= 1'b1;

            if (wfill_we) begin
                if (fill_restart) begin
                    for (int k = 0; k < NUM_MULTS; k++) begin
                        wfill_cnt[k] <= (k == int'(wfill_lane)) ? RPTR_WIDTH'(1) : '0;
                        wsp_q[k]     <= (k == int'(wfill_lane)) ? (NUM_PID'(1) << wfill_pid) : '0;
                    end
                    fill_restart <= 1'b0;
                end else begin
                    wfill_cnt[wfill_lane]        <= wfill_cnt[wfill_lane] + RPTR_WIDTH'(1);
                    wsp_q[wfill_lane][wfill_pid] <= 1'b1;
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // Per-lane weight SRAM banks. Port A = fill (only the addressed lane),
    // Port B = that lane's independent read port.
    // -------------------------------------------------------------------------
    genvar k;
    generate
        for (k = 0; k < NUM_MULTS; k++) begin : g_bank
            logic                  bank_we;
            logic [WSRAM_DW-1:0]   bank_rdata;

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
                .b_en        (rd_en[k]),
                .b_addr      (rd_slot[k]),
                .b_rdata     (bank_rdata),
                .b_rdata_vld ()
            );
            /* verilator lint_on PINCONNECTEMPTY */

            assign bank_we   = wfill_we && (wfill_lane == LANE_WIDTH'(k));
            assign rd_pid[k] = bank_rdata[PID_WIDTH-1:0];
            assign rd_val[k] = bank_rdata[WSRAM_DW-1 -: DATA_WIDTH];
        end
    endgenerate

    assign wfill_slot_eff = fill_restart ? SLOT_WIDTH'(0)
                                         : SLOT_WIDTH'(wfill_cnt[wfill_lane]);

endmodule

`default_nettype wire
