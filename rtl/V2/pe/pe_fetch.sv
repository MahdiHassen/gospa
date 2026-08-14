// =============================================================================
// pe_fetch.sv -- PE weight fetch: Curr/Next selector + direct-index refill
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// One kernel per PE, so a SINGLE selected weight (sel_w) fans out to all M
// multipliers each beat. Weights sit PID-sorted in one bank; the slot for b_pid
// is popcount(wsp below b_pid). A registered Curr weight is held across a
// same-PID run; the bank read register is "Next" (the look-ahead slot). Per beat:
//   IDLE : wsp miss                            -> no MAC, port free to prefetch
//   KEEP : b_pid == curr_pid                   -> MAC Curr
//   BANK : b_pid == Next (bank read register)  -> MAC Next, promote to Curr AND
//                                                 re-issue slot+1 prefetch (so a
//                                                 PID change costs a single cycle)
//   SKIP : hit but neither                     -> fetch target slot, stall 1 beat
// b_ready drops only for SKIP. A cold lane (or a re-arm) SKIPs its first hit.
// =============================================================================

`default_nettype none

module pe_fetch #(
    parameter int NUM_PID    = 9,
    parameter int DATA_WIDTH = 16,

    localparam int PID_WIDTH   = (NUM_PID < 2) ? 1 : $clog2(NUM_PID),
    localparam int RPTR_WIDTH  = $clog2(NUM_PID + 1),
    localparam int SLOT_WIDTH  = (NUM_PID < 2) ? 1 : $clog2(NUM_PID)
)(
    input  logic                          clk,
    input  logic                          rst_n,

    // -- Arm / re-arm --------------------------------------------------------
    input  logic                          wload_done,

    // -- FIFO-B beat handshake / activation PID (shared by the M lanes) ------
    input  logic                          b_valid,
    input  logic [PID_WIDTH-1:0]          b_pid,
    output logic                          b_ready,

    // -- Weight sparsity pattern (pe_mem) ------------------------------------
    input  logic [NUM_PID-1:0]            wsp_q,

    // -- Weight bank read port (pe_mem) --------------------------------------
    input  logic [DATA_WIDTH-1:0]         rd_val,
    input  logic [PID_WIDTH-1:0]          rd_pid,
    output logic                          rd_en,
    output logic [SLOT_WIDTH-1:0]         rd_slot,

    // -- Selected weight exported to the M multipliers -----------------------
    output logic                          mac_en,     // KEEP or BANK this beat
    output logic [DATA_WIDTH-1:0]         mac_w       // the one weight all lanes use
);

    // -------------------------------------------------------------------------
    // Signal declarations
    // -------------------------------------------------------------------------
    // Held Curr weight, plus a valid/slot tag for the bank read register (Next).
    logic [PID_WIDTH-1:0]    curr_pid;
    logic [DATA_WIDTH-1:0]   curr_val;   // 2's-complement, $signed() at use
    logic [SLOT_WIDTH-1:0]   curr_slot;
    logic                    have_curr;
    logic                    bank_valid; // rd_val/rd_pid hold a real read
    logic [SLOT_WIDTH-1:0]   bank_slot;  // slot the bank register holds
    logic                    running;

    typedef enum logic [1:0] {A_IDLE, A_KEEP, A_BANK, A_SKIP} action_t;
    action_t action;

    logic                    wsp_hit, keep, bank_hit;
    logic                    need_fetch;    // SKIP -> stall this beat
    logic                    want_prefetch; // idle port -> fetch look-ahead
    logic                    bank_next_exists, next_exists, bank_has_next;
    logic [RPTR_WIDTH-1:0]   n_weights;     // popcount(wsp): live weight count
    logic [SLOT_WIDTH-1:0]   target_slot;   // bank slot holding b_pid's weight
    logic [NUM_PID-1:0]      below_mask;
    logic                    consume;

    // -------------------------------------------------------------------------
    // Action eval + look-ahead: decides the beat's action and the read port use.
    // -------------------------------------------------------------------------
    always_comb begin
        below_mask  = (NUM_PID'(1) << b_pid) - NUM_PID'(1);
        n_weights   = RPTR_WIDTH'($countones(wsp_q));
        target_slot = SLOT_WIDTH'($countones(wsp_q & below_mask));

        wsp_hit  = running && b_valid && wsp_q[b_pid];
        keep     = wsp_hit && have_curr && (b_pid == curr_pid);
        bank_hit = wsp_hit && !keep && bank_valid && (rd_pid == b_pid);

        if (!wsp_hit)
            action = A_IDLE;
        else if (keep)
            action = A_KEEP;
        else if (bank_hit)
            action = A_BANK;
        else
            action = A_SKIP;

        // Prefetch bounds: is there a weight after the one we would hold next?
        next_exists      = (RPTR_WIDTH'(curr_slot) + RPTR_WIDTH'(1)) < n_weights;
        bank_next_exists = (RPTR_WIDTH'(bank_slot) + RPTR_WIDTH'(1)) < n_weights;
        bank_has_next    = bank_valid
                           && (bank_slot == SLOT_WIDTH'(curr_slot + SLOT_WIDTH'(1)));

        need_fetch    = (action == A_SKIP);
        // Idle/KEEP port free -> prefetch curr_slot+1 for the next PID change.
        want_prefetch = running && have_curr && (action != A_BANK)
                        && (action != A_SKIP) && next_exists && !bank_has_next;

        mac_en = (action == A_KEEP) || (action == A_BANK);
        mac_w  = (action == A_KEEP) ? curr_val : rd_val;
    end

    // -------------------------------------------------------------------------
    // Read-port driver: SKIP jumps to target_slot; a BANK beat re-issues the
    // slot+1 prefetch (the bank data is consumed this cycle, freeing the port);
    // otherwise an idle port prefetches the look-ahead weight.
    // -------------------------------------------------------------------------
    always_comb begin
        rd_en   = 1'b0;
        rd_slot = '0;
        if (running) begin
            if (action == A_SKIP) begin
                rd_en   = 1'b1;
                rd_slot = target_slot;
            end else if ((action == A_BANK) && bank_next_exists) begin
                rd_en   = 1'b1;
                rd_slot = SLOT_WIDTH'(bank_slot + SLOT_WIDTH'(1));
            end else if (want_prefetch) begin
                rd_en   = 1'b1;
                rd_slot = SLOT_WIDTH'(curr_slot + SLOT_WIDTH'(1));
            end
        end
    end

    // -------------------------------------------------------------------------
    // Sequencer
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            running    <= 1'b0;
            curr_pid   <= '0;
            curr_val   <= '0;
            curr_slot  <= '0;
            have_curr  <= 1'b0;
            bank_valid <= 1'b0;
            bank_slot  <= '0;
        end else if (wload_done) begin
            // Arm / re-arm: run from the (freshly loaded) bank with an empty
            // window -- the first hit SKIPs and fetches.
            running    <= 1'b1;
            have_curr  <= 1'b0;
            bank_valid <= 1'b0;
        end else if (running) begin
            // Track what the bank read register will hold next cycle.
            if (rd_en) begin
                bank_valid <= 1'b1;
                bank_slot  <= rd_slot;
            end

            // On an accepted BANK beat, promote the bank weight to Curr.
            if (consume && (action == A_BANK)) begin
                curr_pid  <= rd_pid;
                curr_val  <= rd_val;
                curr_slot <= bank_slot;
                have_curr <= 1'b1;
            end
        end
    end

    // -------------------------------------------------------------------------
    // Combinational assigns
    // -------------------------------------------------------------------------
    assign consume = b_ready;
    assign b_ready = running && b_valid && !need_fetch;

endmodule

`default_nettype wire
