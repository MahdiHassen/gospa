// =============================================================================
// pe_window.sv -- PE Curr/Next weight window + action-eval + refill FSM
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// Owns the multiplier's working weight window and everything that decides,
// per incoming activation, what each lane does -- so it is self-contained
// about why it stalls:
//   - FSM: S_LOAD (idle/arm) -> S_WARM (preload Curr/Next) -> S_RUN.
//   - Warm sequencer: 2*NUM_MULTS sequential reads pre-load every lane's Curr
//     (slot 0) and Next (slot 1). Issue/capture pointers are {lane, phase}
//     pairs, so no index decoding is needed.
//   - Action-eval: per lane, IDLE/RETIRED/KEEP/UPDATE/SLIDE from b_pid + WSP +
//     the window. Exports the MAC controls (mac_en/mac_w); keeps want_slide /
//     want_skip_only internal to drive the arbiter and b_ready.
//   - Refill arbiter: the shared SRAM has one read port, so at most one lane
//     fetches its new Next per cycle. While any non-IDLE/non-RETIRED lane still
//     needs to slide, b_ready is de-asserted so the activation isn't consumed.
//
// A bare wload_done re-arms from freshly rewritten SRAM contents without
// touching the accumulators upstream.
// =============================================================================

`default_nettype none

module pe_window #(
    parameter int NUM_MULTS  = 4,
    parameter int NUM_PID    = 9,
    parameter int DATA_WIDTH = 16,

    localparam int PID_WIDTH   = (NUM_PID   < 2) ? 1 : $clog2(NUM_PID),
    localparam int LANE_WIDTH  = (NUM_MULTS < 2) ? 1 : $clog2(NUM_MULTS),
    localparam int RPTR_WIDTH  = $clog2(NUM_PID + 1),
    localparam int WSRAM_DEPTH = NUM_MULTS * NUM_PID,
    localparam int WSRAM_AW    = (WSRAM_DEPTH < 2) ? 1 : $clog2(WSRAM_DEPTH)
)(
    input  logic                                  clk,
    input  logic                                  rst_n,

    // -- Arm: Curr/Next pre-loaded from the filled weights -------------------
    input  logic                                  wload_done,

    // -- FIFO-B handshake / activation PID (action-eval side) ----------------
    input  logic                                  b_valid,
    input  logic [PID_WIDTH-1:0]                  b_pid,
    output logic                                  b_ready,

    // -- Per-lane weight sparsity pattern + arm-time count (pe_wmem) ----------
    input  logic [NUM_MULTS-1:0][NUM_PID-1:0]     wsp_q,
    input  logic [NUM_MULTS-1:0][RPTR_WIDTH-1:0]  wfill_cnt,

    // -- Weight SRAM read port (pe_wmem) -------------------------------------
    input  logic [DATA_WIDTH-1:0]                 rd_val,         // Port B read data
    input  logic [PID_WIDTH-1:0]                  rd_pid,
    output logic                                  rd_en,
    output logic [WSRAM_AW-1:0]                   rd_addr,

    // -- Per-lane MAC controls exported to the datapath ----------------------
    output logic [NUM_MULTS-1:0]                  mac_en,         // KEEP or UPDATE
    output logic [NUM_MULTS-1:0][DATA_WIDTH-1:0]  mac_w
);

    // -------------------------------------------------------------------------
    // Per-lane window state (Curr/Next window + refill bookkeeping)
    // -------------------------------------------------------------------------
    logic [NUM_MULTS-1:0][PID_WIDTH-1:0]   curr_pid;
    logic [NUM_MULTS-1:0][DATA_WIDTH-1:0]  curr_val;   // 2's-complement, $signed() at use
    logic [NUM_MULTS-1:0][PID_WIDTH-1:0]   next_pid;
    logic [NUM_MULTS-1:0][DATA_WIDTH-1:0]  next_val;
    logic [NUM_MULTS-1:0]                  have_curr, have_next;
    logic [NUM_MULTS-1:0][RPTR_WIDTH-1:0]  rptr;          // next SRAM slot to read
    logic [NUM_MULTS-1:0][RPTR_WIDTH-1:0]  n_weights;
    logic [NUM_MULTS-1:0]                  refill_in_flight;

    // -------------------------------------------------------------------------
    // FSM
    // -------------------------------------------------------------------------
    typedef enum logic [1:0] {S_LOAD, S_WARM, S_RUN} state_t;
    state_t state;
    logic   running;


    // Warm-up sequencer: 2*NUM_MULTS sequential SRAM reads, as {lane, phase} pairs
    // (phase 0 = Curr @ slot 0, phase 1 = Next @ slot 1), lane-major order. The
    // issue pointer leads the capture pointer by one cycle (SRAM read latency).
    logic [LANE_WIDTH-1:0] warm_ilane;   // lane of the read being ISSUED
    logic                  warm_iphase;
    logic                  warm_issuing;  // reads remain to issue
    logic [LANE_WIDTH-1:0] warm_clane;    // lane of the read being CAPTURED
    logic                  warm_cphase;

    // -------------------------------------------------------------------------
    // Per-lane action evaluation (combinational, valid in S_RUN)
    //   IDLE/RETIRED: no MAC, no slide
    //   KEEP        : PID == curr_pid; MAC with curr; no slide
    //   UPDATE      : PID == next_pid; MAC with next; slide afterwards
    //   SLIDE       : neither; no MAC; slide afterwards (stalls b_ready)
    // -------------------------------------------------------------------------
    logic [NUM_MULTS-1:0] want_slide;     // UPDATE or SLIDE
    logic [NUM_MULTS-1:0] want_skip_only; // SLIDE only (forces stall)

    always_comb begin
        for (int k = 0; k < NUM_MULTS; k++) begin
            logic act, is_keep, is_update;
            act       = running && b_valid && have_curr[k] && wsp_q[k][b_pid];
            is_keep   = act && (b_pid == curr_pid[k]);
            is_update = act && have_next[k] && (b_pid == next_pid[k]) && !is_keep;
            mac_en[k]         = is_keep || is_update;
            mac_w[k]          = is_keep ? curr_val[k] : next_val[k];
            want_slide[k]     = act && !is_keep;                 // UPDATE or SLIDE
            want_skip_only[k] = act && !is_keep && !is_update;   // SLIDE only
        end
    end

    // -------------------------------------------------------------------------
    // Refill arbiter (priority, lowest index first). Only one lane slides per
    // cycle. A lane is eligible if it wants to slide AND no refill from its own
    // previous slide is in flight (its SRAM rdata is still pending).
    // -------------------------------------------------------------------------
    logic [NUM_MULTS-1:0]  eligible;
    logic [LANE_WIDTH-1:0] arb_lane;
    logic                  arb_valid;       // a lane is being served this cycle
    logic [LANE_WIDTH-1:0] capture_lane_q;  // which lane the rdata next cycle belongs to
    logic                  capture_valid_q; // capture pending (next cycle has rdata)

    always_comb begin
        for (int k = 0; k < NUM_MULTS; k++) begin
            eligible[k] = want_slide[k] && !refill_in_flight[k];
        end

        arb_lane  = '0;
        arb_valid = 1'b0;
        for (int k = 0; k < NUM_MULTS; k++) begin
            if (!arb_valid && eligible[k]) begin
                arb_lane  = LANE_WIDTH'(k);
                arb_valid = 1'b1;
            end
        end
    end

    // Consume condition: every non-IDLE / non-RETIRED lane must be in KEEP or
    // UPDATE with a stable window. Equivalently: nobody wants SKIP and no refill
    // is in flight that we haven't captured yet.
    logic any_skip, any_in_flight;


    // -------------------------------------------------------------------------
    // SRAM Port B driver (single source). Reads come from:
    //   1) S_WARM: sequential per-lane Curr/Next preload ({lane, phase}).
    //   2) S_RUN, refill arbiter: the picked lane's slide.
    //   3) S_LOAD / re-arm: kick the warm sequence with slot 0 of lane 0.
    // -------------------------------------------------------------------------
    always_comb begin
        rd_en   = 1'b0;
        rd_addr = '0;
        unique case (state)
            S_LOAD: begin
                if (wload_done) begin
                    rd_en   = 1'b1;
                    rd_addr = '0;            // first warm read is slot 0 of lane 0
                end
            end
            S_WARM: begin
                if (warm_issuing) begin
                    rd_en   = 1'b1;
                    rd_addr = WSRAM_AW'(warm_ilane) * WSRAM_AW'(NUM_PID)
                            + (warm_iphase ? WSRAM_AW'(1) : WSRAM_AW'(0));
                end
            end
            S_RUN: begin
                if (wload_done) begin
                    // Re-arm path: a fresh wload_done in S_RUN restarts the warm
                    // sequence with new SRAM contents (host overwrote via Port A).
                    rd_en   = 1'b1;
                    rd_addr = '0;
                end else if (arb_valid) begin
                    rd_en   = 1'b1;
                    rd_addr = WSRAM_AW'(arb_lane) * WSRAM_AW'(NUM_PID)
                            + WSRAM_AW'(rptr[arb_lane]);
                end
            end
        endcase
    end

    // -------------------------------------------------------------------------
    // Sequencer
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state              <= S_LOAD;
            warm_ilane         <= '0;
            warm_iphase        <= 1'b0;
            warm_issuing       <= 1'b0;
            warm_clane         <= '0;
            warm_cphase        <= 1'b0;
            refill_in_flight   <= '0;
            capture_lane_q     <= '0;
            capture_valid_q    <= 1'b0;
            for (int k = 0; k < NUM_MULTS; k++) begin
                curr_pid[k]  <= '0;
                curr_val[k]  <= '0;
                next_pid[k]  <= '0;
                next_val[k]  <= '0;
                have_curr[k] <= 1'b0;
                have_next[k] <= 1'b0;
                rptr[k]      <= '0;
                n_weights[k] <= '0;
            end
        end else begin
            unique case (state)
                // ---------------------------------------------------------
                S_LOAD: begin
                    if (wload_done) begin
                        // Latch per-lane counts (derived from fill); reset window.
                        for (int k = 0; k < NUM_MULTS; k++) begin
                            n_weights[k] <= wfill_cnt[k];
                            have_curr[k] <= 1'b0;
                            have_next[k] <= 1'b0;
                            rptr[k]      <= RPTR_WIDTH'(2);   // slot 0 -> Curr, slot 1 -> Next
                        end
                        refill_in_flight <= '0;
                        // Just issued (lane 0, Curr); next issue is (lane 0, Next),
                        // first capture is (lane 0, Curr).
                        warm_ilane   <= '0;
                        warm_iphase  <= 1'b1;
                        warm_issuing <= 1'b1;
                        warm_clane   <= '0;
                        warm_cphase  <= 1'b0;
                        state        <= S_WARM;
                    end
                end

                // ---------------------------------------------------------
                S_WARM: begin
                    // 1) Capture this cycle's rdata (contiguous, always valid in
                    //    S_WARM) into the current {lane, phase} target.
                    if (warm_cphase) begin
                        next_pid[warm_clane]  <= rd_pid;
                        next_val[warm_clane]  <= rd_val;
                        have_next[warm_clane] <= (n_weights[warm_clane] > RPTR_WIDTH'(1));
                    end else begin
                        curr_pid[warm_clane]  <= rd_pid;
                        curr_val[warm_clane]  <= rd_val;
                        have_curr[warm_clane] <= (n_weights[warm_clane] > RPTR_WIDTH'(0));
                    end

                    // 2) Advance the issue pointer; stop after the final read.
                    if (warm_issuing) begin
                        if (warm_ilane == LANE_WIDTH'(NUM_MULTS-1) && warm_iphase)
                            warm_issuing <= 1'b0;
                        else if (!warm_iphase)
                            warm_iphase  <= 1'b1;
                        else begin
                            warm_iphase  <= 1'b0;
                            warm_ilane   <= warm_ilane + LANE_WIDTH'(1);
                        end
                    end

                    // 3) Advance the capture pointer; finish once the final read
                    //    (last lane's Next) has landed.
                    if (warm_clane == LANE_WIDTH'(NUM_MULTS-1) && warm_cphase)
                        state <= S_RUN;
                    else if (!warm_cphase)
                        warm_cphase <= 1'b1;
                    else begin
                        warm_cphase <= 1'b0;
                        warm_clane  <= warm_clane + LANE_WIDTH'(1);
                    end
                end

                // ---------------------------------------------------------
                S_RUN: begin
                    if (wload_done) begin
                        // Re-arm: host has rewritten the weight SRAM (and
                        // optionally the WSPs); kick the warm sequence again to
                        // refresh Curr/Next. Accumulators upstream are NOT
                        // touched, so partial sums persist across the re-arm --
                        // this is what makes back-to-back input channels work
                        // end-to-end on one drain.
                        for (int k = 0; k < NUM_MULTS; k++) begin
                            n_weights[k] <= wfill_cnt[k];
                            have_curr[k] <= 1'b0;
                            have_next[k] <= 1'b0;
                            rptr[k]      <= RPTR_WIDTH'(2);
                        end
                        refill_in_flight <= '0;
                        capture_valid_q  <= 1'b0;
                        warm_ilane       <= '0;
                        warm_iphase      <= 1'b1;
                        warm_issuing     <= 1'b1;
                        warm_clane       <= '0;
                        warm_cphase      <= 1'b0;
                        state            <= S_WARM;
                    end else begin
                    // 1) Capture any pending refill from previous cycle's
                    //    arbiter-issued read.
                    if (capture_valid_q) begin
                        logic [LANE_WIDTH-1:0] lane;
                        lane = capture_lane_q;
                        next_pid[lane]  <= rd_pid;
                        next_val[lane]  <= rd_val;
                        have_next[lane] <= 1'b1;
                        refill_in_flight[lane] <= 1'b0;
                    end

                    // 2) Service the arbiter-selected lane this cycle.
                    if (arb_valid) begin
                        logic [LANE_WIDTH-1:0] lane;
                        lane = arb_lane;
                        // Slide: Curr <- Next; pending refill brings new Next.
                        curr_pid[lane]  <= next_pid[lane];
                        curr_val[lane]  <= next_val[lane];
                        have_curr[lane] <= have_next[lane];
                        have_next[lane] <= 1'b0;
                        // Issue refill if there are more weights to load.
                        if (rptr[lane] < n_weights[lane]) begin
                            refill_in_flight[lane] <= 1'b1;
                            capture_lane_q         <= lane;
                            capture_valid_q        <= 1'b1;
                            rptr[lane]             <= rptr[lane] + RPTR_WIDTH'(1);
                        end else begin
                            // No more weights; lane's have_next stays 0, and may
                            // eventually retire on a future slide.
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

    assign running = (state == S_RUN);
    assign any_skip      = |want_skip_only;
    assign any_in_flight = |refill_in_flight;
    assign b_ready       = (state == S_RUN) && b_valid && !any_skip && !any_in_flight;

endmodule

`default_nettype wire
