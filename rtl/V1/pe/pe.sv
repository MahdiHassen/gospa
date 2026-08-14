// =============================================================================
// pe.sv -- GoSPA Processing Element (V2: multiple kernels per PE)
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// V2 PE structure
//   - One shared on-chip weight SRAM (sram.sv) per PE, depth N_MULTS*N_PID,
//     width DATA_W+PID_W. Each word holds {value, pid}. Lane k's kernel
//     lives at SRAM addresses [k*N_PID, (k+1)*N_PID).
//   - Per-lane Curr/Next register files (the multiplier's working window).
//   - Per-lane WSP register (N_PID bits) telling the lane which incoming
//     PIDs belong to its kernel; non-WSP PIDs are IDLE (no MAC, no slide).
//   - Per-lane CID-indexed accumulator (one pe_acc.sv instance per lane).
//
// Round-robin refill arbiter
//   The shared SRAM has one read port, so at most one lane can fetch its
//   new Next weight per cycle. While any lane needs a refill (slide), the
//   PE de-asserts b_ready -- the activation isn't consumed until every
//   non-IDLE / non-RETIRED lane is in KEEP or UPDATE state with valid
//   Curr/Next. This is conservative; KEEP-heavy workloads still hit one
//   activation per cycle.
//
// Per-lane action per FIFO-B activation (b_pid arrives monotone, gated
// upstream by the union WSP):
//   IDLE    : wsp[k][PID] == 0                  -> no MAC, no slide
//   RETIRED : have_curr[k] == 0 (out of weights)-> no MAC, no slide
//   KEEP    : PID == curr_pid[k]                -> MAC, no slide
//   UPDATE  : PID == next_pid[k], have_next[k]  -> MAC, slide (refill Next)
//   SLIDE   : PID neither -- need to advance    -> no MAC, slide (refill Next)
//
// Load sequence (host -> PE)
//   1) For each lane k: stream {wfill_we, wfill_lane=k, wfill_slot=s,
//      wfill_pid, wfill_val} for s=0..count_k-1.
//   2) For each lane k: pulse {wsp_we, wsp_lane=k, wsp_data=N_PID bits}.
//   3) Drive wload_count[k] per lane, then pulse wload_done. The FSM
//      issues 2*N_MULTS SRAM reads to pre-load every lane's Curr and Next,
//      then enters S_RUN.
// =============================================================================

`default_nettype none

module pe #(
    parameter int N_MULTS = 4,    // multiplier lanes per PE (output channels held here)
    parameter int N_PID   = 9,    // # kernel positions = F*F (max weights/lane)
    parameter int N_CID   = 36,   // # output positions = E*E (accumulator banks)
    parameter int DATA_W  = 16,
    parameter int ACC_W   = 32,

    // -- Derived --------------------------------------------------------------
    localparam int PID_W       = (N_PID   < 2) ? 1 : $clog2(N_PID),
    localparam int CID_W       = (N_CID   < 2) ? 1 : $clog2(N_CID),
    localparam int LANE_W      = (N_MULTS < 2) ? 1 : $clog2(N_MULTS),
    localparam int WPTR_W      = $clog2(N_PID + 1),
    localparam int WSRAM_DEPTH = N_MULTS * N_PID,
    localparam int WSRAM_AW    = (WSRAM_DEPTH < 2) ? 1 : $clog2(WSRAM_DEPTH),
    localparam int WSRAM_DW    = DATA_W + PID_W,
    localparam int PROD_W      = 2 * DATA_W,
    localparam int WARM_MAX    = 2 * N_MULTS,
    localparam int WARM_W      = (WARM_MAX < 2) ? 1 : $clog2(WARM_MAX + 1)
)(
    input  wire  logic                            clk,
    input  wire  logic                            rst_n,

    // -- Weight SRAM fill -- one (lane, slot) write per cycle ---------------
    input  wire  logic                            wfill_we,
    input  wire  logic [LANE_W-1:0]               wfill_lane,
    input  wire  logic [WPTR_W-1:0]               wfill_slot,
    input  wire  logic [PID_W-1:0]                wfill_pid,
    input  wire  logic signed [DATA_W-1:0]        wfill_val,

    // -- Per-lane WSP write (one lane per cycle) -----------------------------
    input  wire  logic                            wsp_we,
    input  wire  logic [LANE_W-1:0]               wsp_lane,
    input  wire  logic [N_PID-1:0]                wsp_data,

    // -- Arm: per-lane # valid weights latched, Curr/Next pre-loaded --------
    input  wire  logic                            wload_done,
    input  wire  logic [N_MULTS-1:0][WPTR_W-1:0]  wload_count,

    // -- FIFO-B input stream (act, pid, cid), PID monotone --------------------
    input  wire  logic                            b_valid,
    input  wire  logic signed [DATA_W-1:0]        b_act,
    input  wire  logic [PID_W-1:0]                b_pid,
    input  wire  logic [CID_W-1:0]                b_cid,
    output logic                                  b_ready,

    // -- Drain / per-lane output (one N_CID-beat stream per lane) ------------
    input  wire  logic                            drain_start,
    output logic                                  drain_busy,
    output logic                                  drain_done,
    output logic [N_MULTS-1:0]                    out_valid,
    output logic [N_MULTS-1:0][CID_W-1:0]         out_cid,
    output logic [N_MULTS-1:0][ACC_W-1:0]         out_acc,
    input  wire  logic [N_MULTS-1:0]              out_ready
);

    // -------------------------------------------------------------------------
    // Shared weight SRAM (Port A = fill, Port B = scanner/refill arbiter)
    // Address layout: addr = lane * N_PID + slot.
    // -------------------------------------------------------------------------
    logic                w_rd_en;
    logic [WSRAM_AW-1:0] w_rd_addr;
    logic [WSRAM_DW-1:0] w_rd_data;
    logic [WSRAM_AW-1:0] wfill_addr;
    assign wfill_addr = WSRAM_AW'(wfill_lane) * WSRAM_AW'(N_PID) + WSRAM_AW'(wfill_slot);

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
        .b_en        (w_rd_en),
        .b_addr      (w_rd_addr),
        .b_rdata     (w_rd_data),
        .b_rdata_vld ()
    );
    /* verilator lint_on PINCONNECTEMPTY */

    // SRAM unpack helpers (raw bits; $signed() applied at the multiplier).
    logic [PID_W-1:0]  sram_pid;
    logic [DATA_W-1:0] sram_val;
    assign sram_pid = w_rd_data[PID_W-1:0];
    assign sram_val = w_rd_data[WSRAM_DW-1 -: DATA_W];

    // -------------------------------------------------------------------------
    // Per-lane state (Curr/Next window, wptr, n_weights, WSP, refill flag)
    // -------------------------------------------------------------------------
    logic [N_MULTS-1:0][PID_W-1:0]         curr_pid;
    logic [N_MULTS-1:0][DATA_W-1:0]        curr_val;       // 2's-complement, $signed() at use
    logic [N_MULTS-1:0][PID_W-1:0]         next_pid;
    logic [N_MULTS-1:0][DATA_W-1:0]        next_val;
    logic [N_MULTS-1:0]                    have_curr, have_next;
    logic [N_MULTS-1:0][WPTR_W-1:0]        wptr;          // next SRAM slot to read
    logic [N_MULTS-1:0][WPTR_W-1:0]        n_weights;
    logic [N_MULTS-1:0][N_PID-1:0]         wsp_q;
    logic [N_MULTS-1:0]                    refill_in_flight;

    // -------------------------------------------------------------------------
    // FSM
    // -------------------------------------------------------------------------
    typedef enum logic [1:0] {S_LOAD, S_WARM, S_RUN} state_t;
    state_t state;

    // Warm-up sequencer: 2*N_MULTS sequential SRAM reads.
    // warm_addr_idx is the index whose read is being ISSUED this cycle.
    // warm_cap_idx  is the index whose data is being CAPTURED this cycle.
    //   idx in [0, N_MULTS)            -> Curr for lane idx          (slot 0)
    //   idx in [N_MULTS, 2*N_MULTS)    -> Next for lane idx-N_MULTS  (slot 1)
    logic [WARM_W-1:0] warm_addr_idx;
    logic [WARM_W-1:0] warm_cap_idx;
    logic              warm_cap_valid;

    function automatic logic [LANE_W-1:0] warm_lane_of(input logic [WARM_W-1:0] idx);
        if (idx < WARM_W'(N_MULTS)) return LANE_W'(idx);
        else                         return LANE_W'(idx - WARM_W'(N_MULTS));
    endfunction
    function automatic logic warm_is_next(input logic [WARM_W-1:0] idx);
        return (idx >= WARM_W'(N_MULTS));
    endfunction
    function automatic logic [WSRAM_AW-1:0] warm_sram_addr(input logic [WARM_W-1:0] idx);
        // Curr -> slot 0, Next -> slot 1.
        logic [LANE_W-1:0] lane;
        lane = warm_lane_of(idx);
        return WSRAM_AW'(lane) * WSRAM_AW'(N_PID)
             + (warm_is_next(idx) ? WSRAM_AW'(1) : WSRAM_AW'(0));
    endfunction

    // -------------------------------------------------------------------------
    // Per-lane action evaluation (combinational, valid in S_RUN)
    //   IDLE/RETIRED: no MAC, no slide
    //   KEEP        : PID == curr_pid; MAC with curr; no slide
    //   UPDATE      : PID == next_pid; MAC with next; slide afterwards
    //   SLIDE       : neither; no MAC; slide afterwards (stalls b_ready)
    // -------------------------------------------------------------------------
    logic [N_MULTS-1:0]                want_slide;     // UPDATE or SLIDE
    logic [N_MULTS-1:0]                want_skip_only; // SLIDE only (forces stall)
    logic [N_MULTS-1:0]                mac_en;         // KEEP or UPDATE
    logic [N_MULTS-1:0][DATA_W-1:0]    mac_w;

    always_comb begin
        for (int k = 0; k < N_MULTS; k++) begin
            logic act, is_keep, is_update;
            act       = (state == S_RUN) && b_valid && have_curr[k] && wsp_q[k][b_pid];
            is_keep   = act && (b_pid == curr_pid[k]);
            is_update = act && have_next[k] && (b_pid == next_pid[k]) && !is_keep;
            mac_en[k]         = is_keep || is_update;
            mac_w[k]          = is_keep ? curr_val[k] : next_val[k];
            want_slide[k]     = act && !is_keep;                 // UPDATE or SLIDE
            want_skip_only[k] = act && !is_keep && !is_update;   // SLIDE only
        end
    end

    // -------------------------------------------------------------------------
    // Refill arbiter (priority, lowest index first). Only one lane slides
    // per cycle; the SLIDE/UPDATE for that lane fires; others wait.
    //
    // A lane is eligible if it wants to slide AND no refill from its own
    // previous slide is in flight (the SRAM rdata for that lane is still
    // pending; if we issued another read this cycle we'd lose it).
    // -------------------------------------------------------------------------
    logic [N_MULTS-1:0] eligible;
    logic [LANE_W-1:0]  arb_lane;
    logic               arb_valid;       // a lane is being served this cycle
    logic [LANE_W-1:0]  capture_lane_q;  // which lane the rdata next cycle belongs to
    logic               capture_valid_q; // capture pending (next cycle has rdata)

    always_comb begin
        for (int k = 0; k < N_MULTS; k++) begin
            eligible[k] = want_slide[k] && !refill_in_flight[k];
        end

        arb_lane  = '0;
        arb_valid = 1'b0;
        for (int k = 0; k < N_MULTS; k++) begin
            if (!arb_valid && eligible[k]) begin
                arb_lane  = LANE_W'(k);
                arb_valid = 1'b1;
            end
        end
    end

    // Consume condition: every non-IDLE / non-RETIRED lane must be in KEEP
    // or UPDATE with a stable window. Equivalently: nobody wants SKIP and
    // no refill is in flight that we haven't captured yet.
    logic any_skip, any_in_flight;
    assign any_skip      = |want_skip_only;
    assign any_in_flight = |refill_in_flight;
    assign b_ready       = (state == S_RUN) && b_valid && !any_skip && !any_in_flight;
    logic consume;
    assign consume       = b_ready;   // alias

    // -------------------------------------------------------------------------
    // SRAM Port B driver (single source). Three sources of reads:
    //   1) S_WARM: sequential per-lane Curr/Next preload.
    //   2) S_RUN, refill arbiter:  the picked lane's slide.
    // -------------------------------------------------------------------------
    always_comb begin
        w_rd_en   = 1'b0;
        w_rd_addr = '0;
        unique case (state)
            S_LOAD: begin
                if (wload_done) begin
                    w_rd_en   = 1'b1;
                    w_rd_addr = '0;            // first warm read is slot 0 of lane 0
                end
            end
            S_WARM: begin
                if (warm_addr_idx < WARM_W'(WARM_MAX)) begin
                    w_rd_en   = 1'b1;
                    w_rd_addr = warm_sram_addr(warm_addr_idx);
                end
            end
            S_RUN: begin
                if (wload_done) begin
                    // Re-arm path: a fresh wload_done in S_RUN restarts the
                    // warm sequence with the new SRAM contents (host has
                    // overwritten weights via Port A). Issue slot 0 now.
                    w_rd_en   = 1'b1;
                    w_rd_addr = '0;
                end else if (arb_valid) begin
                    w_rd_en   = 1'b1;
                    w_rd_addr = WSRAM_AW'(arb_lane) * WSRAM_AW'(N_PID)
                              + WSRAM_AW'(wptr[arb_lane]);
                end
            end
            default: ;
        endcase
    end

    // -------------------------------------------------------------------------
    // Sequencer
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state              <= S_LOAD;
            warm_addr_idx      <= '0;
            warm_cap_idx       <= '0;
            warm_cap_valid     <= 1'b0;
            refill_in_flight   <= '0;
            capture_lane_q     <= '0;
            capture_valid_q    <= 1'b0;
            for (int k = 0; k < N_MULTS; k++) begin
                curr_pid[k]  <= '0;
                curr_val[k]  <= '0;
                next_pid[k]  <= '0;
                next_val[k]  <= '0;
                have_curr[k] <= 1'b0;
                have_next[k] <= 1'b0;
                wptr[k]      <= '0;
                n_weights[k] <= '0;
                wsp_q[k]     <= '0;
            end
        end else begin
            // --- WSP write (effective any state) -------------------------
            if (wsp_we) wsp_q[wsp_lane] <= wsp_data;

            unique case (state)
                // ---------------------------------------------------------
                S_LOAD: begin
                    if (wload_done) begin
                        // Latch per-lane counts; reset window state.
                        for (int k = 0; k < N_MULTS; k++) begin
                            n_weights[k] <= wload_count[k];
                            have_curr[k] <= 1'b0;
                            have_next[k] <= 1'b0;
                            wptr[k]      <= WPTR_W'(2);   // slot 0 -> Curr, slot 1 -> Next
                        end
                        refill_in_flight <= '0;
                        warm_addr_idx    <= WARM_W'(1);   // we just issued idx=0 above
                        warm_cap_idx     <= '0;           // first capture next cycle
                        warm_cap_valid   <= 1'b1;
                        state            <= S_WARM;
                    end
                end

                // ---------------------------------------------------------
                S_WARM: begin
                    // Capture this cycle's rdata if it represents a real load.
                    if (warm_cap_valid) begin
                        logic [LANE_W-1:0] lane;
                        lane = warm_lane_of(warm_cap_idx);
                        if (warm_is_next(warm_cap_idx)) begin
                            next_pid[lane] <= sram_pid;
                            next_val[lane] <= sram_val;
                            have_next[lane] <= (n_weights[lane] > WPTR_W'(1));
                        end else begin
                            curr_pid[lane] <= sram_pid;
                            curr_val[lane] <= sram_val;
                            have_curr[lane] <= (n_weights[lane] > WPTR_W'(0));
                        end
                    end

                    // Sequence the next issue / capture indices.
                    if (warm_addr_idx < WARM_W'(WARM_MAX)) begin
                        warm_addr_idx  <= warm_addr_idx + WARM_W'(1);
                        warm_cap_idx   <= warm_cap_idx  + WARM_W'(1);
                        warm_cap_valid <= 1'b1;
                    end else begin
                        // Last read already issued previous cycle; one more
                        // capture this cycle covers the final lane's Next.
                        warm_cap_idx   <= warm_cap_idx + WARM_W'(1);
                        warm_cap_valid <= (warm_cap_idx + WARM_W'(1) < WARM_W'(WARM_MAX));
                        if (warm_cap_idx + WARM_W'(1) >= WARM_W'(WARM_MAX)) begin
                            state <= S_RUN;
                        end
                    end
                end

                // ---------------------------------------------------------
                S_RUN: begin
                    if (wload_done) begin
                        // Re-arm: host has rewritten the weight SRAM (and
                        // optionally the WSPs); kick the warm sequence again
                        // to refresh Curr/Next. pe_acc accumulators are NOT
                        // touched (clear is tied 0) so partial sums persist
                        // across the re-arm -- this is what makes back-to-back
                        // input channels work end-to-end on one drain.
                        for (int k = 0; k < N_MULTS; k++) begin
                            n_weights[k] <= wload_count[k];
                            have_curr[k] <= 1'b0;
                            have_next[k] <= 1'b0;
                            wptr[k]      <= WPTR_W'(2);
                        end
                        refill_in_flight <= '0;
                        capture_valid_q  <= 1'b0;
                        warm_addr_idx    <= WARM_W'(1);
                        warm_cap_idx     <= '0;
                        warm_cap_valid   <= 1'b1;
                        state            <= S_WARM;
                    end else begin
                    // 1) Capture any pending refill from previous cycle's
                    //    arbiter-issued read.
                    if (capture_valid_q) begin
                        logic [LANE_W-1:0] lane;
                        lane = capture_lane_q;
                        next_pid[lane]  <= sram_pid;
                        next_val[lane]  <= sram_val;
                        // have_next set iff there was a weight to load
                        // (wptr was incremented when issuing; if pre-issue
                        // wptr exceeded n_weights, we shouldn't have issued).
                        have_next[lane] <= 1'b1;
                        refill_in_flight[lane] <= 1'b0;
                    end

                    // 2) Service the arbiter-selected lane this cycle.
                    if (arb_valid) begin
                        logic [LANE_W-1:0] lane;
                        lane = arb_lane;
                        // Slide: Curr <- Next; pending refill brings new Next.
                        curr_pid[lane]  <= next_pid[lane];
                        curr_val[lane]  <= next_val[lane];
                        have_curr[lane] <= have_next[lane];
                        have_next[lane] <= 1'b0;
                        // Issue refill if there are more weights to load.
                        if (wptr[lane] < n_weights[lane]) begin
                            refill_in_flight[lane] <= 1'b1;
                            capture_lane_q         <= lane;
                            capture_valid_q        <= 1'b1;
                            wptr[lane]             <= wptr[lane] + WPTR_W'(1);
                        end else begin
                            // No more weights; lane's have_next stays 0,
                            // and may eventually retire on a future slide.
                            capture_valid_q <= 1'b0;
                        end
                    end else begin
                        // No arbiter activity this cycle.
                        capture_valid_q <= 1'b0;
                    end
                    end  // not wload_done
                end

                default: state <= S_LOAD;
            endcase
        end
    end

    // -------------------------------------------------------------------------
    // Per-lane multiplier + accumulator. We pipe out_valid/cid/acc through an
    // internal wire so both the module output port AND drain_busy can read it
    // cleanly (one driver -> two readers).
    //
    // The multiply is PIPELINED: the product is registered (stage 1) before it
    // is accumulated (stage 2 inside pe_acc). This breaks the long
    //   FIFO-B -> mac_w mux -> 16x16 multiply -> 32b accumulate-RMW
    // combinational path -- the product register is exactly the DSP block's
    // output register on an FPGA. The CID and accumulate-enable are delayed one
    // cycle so they stay aligned with the registered product.
    //
    // No accumulator hazard is introduced: the accumulate remains a single-cycle
    // read-modify-write, so two back-to-back same-CID products land on
    // consecutive cycles and each reads the prior cycle's committed value.
    // To avoid dropping the trailing product, the accumulator drain is held off
    // (acc_drain_pulse below) until the MAC pipeline has flushed.
    // -------------------------------------------------------------------------
    logic [N_MULTS-1:0]              pe_out_valid_w;
    logic [N_MULTS-1:0][CID_W-1:0]   pe_out_cid_w;
    logic [N_MULTS-1:0][ACC_W-1:0]   pe_out_acc_w;
    logic [N_MULTS-1:0]              pe_drain_busy_w;

    // Per-lane stage-2 accumulate-enable (registered product valid). Held at
    // module scope so the drain can wait for the MAC pipeline to flush.
    logic [N_MULTS-1:0]              mac_en_q;

    // Defer the accumulator drain until every lane's in-flight product has been
    // accumulated (mac_en_q all 0) and no new product is being formed this
    // cycle (!consume). Otherwise the trailing pipelined product would be
    // dropped when pe_acc switches into drain-readout mode. drain_start is
    // latched so a single-cycle pulse is not lost while we wait for the flush.
    logic                            drain_req_q;
    logic                            acc_drain_pulse;
    assign acc_drain_pulse = (drain_start || drain_req_q)
                          && (mac_en_q == '0) && !consume;

    always_ff @(posedge clk) begin
        if      (!rst_n)            drain_req_q <= 1'b0;
        else if (acc_drain_pulse)   drain_req_q <= 1'b0;
        else if (drain_start)       drain_req_q <= 1'b1;
    end

    genvar k;
    /* verilator lint_off PINCONNECTEMPTY */
    generate
        for (k = 0; k < N_MULTS; k++) begin : g_lane
            // ----- Stage 1: multiply (combinational) -> product register -----
            logic signed [PROD_W-1:0] prod_k;     // combinational product
            logic signed [PROD_W-1:0] prod_q;     // registered product (DSP out reg)
            logic        [CID_W-1:0]  cid_q;       // CID aligned to prod_q

            assign prod_k = b_act * $signed(mac_w[k]);

            always_ff @(posedge clk) begin
                if (!rst_n) begin
                    prod_q      <= '0;
                    cid_q       <= '0;
                    mac_en_q[k] <= 1'b0;
                end else begin
                    prod_q      <= prod_k;
                    cid_q       <= b_cid;
                    mac_en_q[k] <= consume && mac_en[k] && !drain_busy;
                end
            end

            // ----- Stage 2: accumulate the registered product -----
            pe_acc #(
                .N_CID(N_CID), .ACC_WIDTH(ACC_W), .PROD_WIDTH(PROD_W)
            ) u_acc (
                .clk        (clk),
                .rst_n      (rst_n),
                .clear      (1'b0),
                .add_en     (mac_en_q[k]),
                .add_cid    (cid_q),
                .add_data   (prod_q),
                .drain_start(acc_drain_pulse),
                .drain_busy (pe_drain_busy_w[k]),
                .drain_done (),
                .out_valid  (pe_out_valid_w[k]),
                .out_ready  (out_ready[k]),
                .out_cid    (pe_out_cid_w[k]),
                .out_acc    (pe_out_acc_w[k])
            );
        end
    endgenerate
    /* verilator lint_on PINCONNECTEMPTY */

    assign out_valid  = pe_out_valid_w;
    assign out_cid    = pe_out_cid_w;
    assign out_acc    = pe_out_acc_w;
    assign drain_busy = |pe_drain_busy_w;

    logic busy_q;
    always_ff @(posedge clk) begin
        if (!rst_n) busy_q <= 1'b0;
        else        busy_q <= drain_busy;
    end
    assign drain_done = busy_q && !drain_busy;

endmodule

`default_nettype wire
