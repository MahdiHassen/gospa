// =============================================================================
// pe_wmem.sv -- PE weight memory: shared on-chip SRAM + fill bookkeeping
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// Owns the per-PE weight SRAM and everything on the "load" side:
//   - Port A: one weight per cycle {wfill_val, wfill_pid} appended to lane's
//     slots. slot, per-lane count (wfill_cnt) and WSP (wsp_q) are all DERIVED
//     from this stream; the host drives no separate slot / count / WSP ports.
//   - fill_restart: a bare wload_done re-arms the same weights; the first
//     wfill after an arm starts a fresh session (prior counts/WSP cleared).
//   - Port B: single read port for the window (warm preload + run refill).
//     1-cycle latency; read data is unpacked into rd_val / rd_pid.
//
// Address layout (both ports): addr = lane * NUM_PID + slot.
// =============================================================================

`default_nettype none

module pe_wmem #(
    parameter int NUM_MULTS  = 4,
    parameter int NUM_PID    = 9,
    parameter int DATA_WIDTH = 16,

    // -- Derived --------------------------------------------------------------
    localparam int PID_WIDTH   = (NUM_PID   < 2) ? 1 : $clog2(NUM_PID),
    localparam int LANE_WIDTH  = (NUM_MULTS < 2) ? 1 : $clog2(NUM_MULTS),
    localparam int RPTR_WIDTH  = $clog2(NUM_PID + 1),
    localparam int WSRAM_DEPTH = NUM_MULTS * NUM_PID,
    localparam int WSRAM_AW    = (WSRAM_DEPTH < 2) ? 1 : $clog2(WSRAM_DEPTH),
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

    // -- Read port (Port B) -- driven by the window, 1-cycle latency ---------
    input  logic                                  rd_en,
    input  logic [WSRAM_AW-1:0]                   rd_addr,
    output logic [DATA_WIDTH-1:0]                 rd_val,
    output logic [PID_WIDTH-1:0]                  rd_pid,

    // -- Derived per-lane state exported to the window / action-eval ---------
    output logic [NUM_MULTS-1:0][RPTR_WIDTH-1:0]  wfill_cnt,   // live weight count
    output logic [NUM_MULTS-1:0][NUM_PID-1:0]     wsp_q        // weight sparsity pattern
);

    // -------------------------------------------------------------------------
    // Fill pointer = # weights written this session = next slot to write; also
    // the arm-time weight count. fill_restart marks the first write of a new
    // session, which clears the prior counts/WSP.
    // -------------------------------------------------------------------------
    logic                   fill_restart;
    logic [RPTR_WIDTH-1:0]  wfill_slot_eff;
    logic [WSRAM_AW-1:0]    wfill_addr;
    logic [WSRAM_DW-1:0]    w_rd_data;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            wfill_cnt    <= '0; fill_restart <= 1'b0;

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
    // Shared weight SRAM (Port A = fill, Port B = window scanner/refill).
    // -------------------------------------------------------------------------
    /* verilator lint_off PINCONNECTEMPTY */
    sram #(
        .DATA_WIDTH    (WSRAM_DW),
        .ADDR_WIDTH    (WSRAM_AW),
        .USE_DUAL_PORT (1'b1),
        .OUTPUT_REG    (1'b0)
    ) u_wsram (
        .clk         (clk),
        .rst_n       (rst_n),
        .a_en        (wfill_we),
        .a_we        (wfill_we),
        .a_addr      (wfill_addr),
        .a_wdata     ({wfill_val, wfill_pid}),
        .a_rdata     (),
        .a_rdata_vld (),
        .b_en        (rd_en),
        .b_addr      (rd_addr),
        .b_rdata     (w_rd_data),
        .b_rdata_vld ()
    );
    /* verilator lint_on PINCONNECTEMPTY */

    assign wfill_slot_eff = fill_restart ? RPTR_WIDTH'(0) : wfill_cnt[wfill_lane];
    assign wfill_addr = WSRAM_AW'(wfill_lane) * WSRAM_AW'(NUM_PID) + WSRAM_AW'(wfill_slot_eff);

    // Unpack read data (raw bits; $signed() applied at the multiplier).
    assign rd_pid = w_rd_data[PID_WIDTH-1:0];
    assign rd_val = w_rd_data[WSRAM_DW-1 -: DATA_WIDTH];

endmodule

`default_nettype wire
