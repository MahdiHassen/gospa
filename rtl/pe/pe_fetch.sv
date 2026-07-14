// =============================================================================
// pe_fetch.sv -- PE per-lane weight fetch: action-eval + direct-index refill
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// Decides, per incoming activation, what each lane does -- and is self-contained
// about why it stalls. Weights sit in each lane's bank in PID order, so the slot
// for b_pid is just popcount(wsp below b_pid); a lane jumps straight to it.
//
// Each lane keeps a registered Curr weight (held across a same-PID run) and uses
// its bank's read-data register as the look-ahead slot -- the SRAM holds its last
// read until the next one, so no separate Next copy is needed. Per lane, one of:
//   IDLE : wsp miss                              -> no MAC, port free to prefetch
//   KEEP : b_pid == curr_pid                     -> MAC Curr
//   BANK : b_pid is in the bank read register    -> MAC bank data, promote to Curr
//   SKIP : hit but neither of the above          -> fetch target slot, stall
// b_ready drops only for SKIP lanes; a BANK hit is MAC'd the same cycle, so any
// jump costs a single stall cycle regardless of distance. A cold lane (or a
// re-arm) simply SKIPs its first hit -- no warm-up sequence.
//
// A bare wload_done re-arms from freshly rewritten SRAM contents without touching
// the accumulators upstream (partial sums chain across input channels).
// =============================================================================

`default_nettype none

module pe_fetch #(
    parameter int NUM_MULTS  = 4,
    parameter int NUM_PID    = 9,
    parameter int DATA_WIDTH = 16,

    localparam int PID_WIDTH   = (NUM_PID   < 2) ? 1 : $clog2(NUM_PID),
    localparam int RPTR_WIDTH  = $clog2(NUM_PID + 1),
    localparam int SLOT_WIDTH  = (NUM_PID < 2) ? 1 : $clog2(NUM_PID)
)(
    input  logic                                  clk,
    input  logic                                  rst_n,

    // -- Arm / re-arm --------------------------------------------------------
    input  logic                                  wload_done,

    // -- FIFO-B handshake / activation PID -----------------------------------
    input  logic                                  b_valid,
    input  logic [PID_WIDTH-1:0]                  b_pid,
    output logic                                  b_ready,

    // -- Per-lane weight sparsity pattern (pe_wmem) --------------------------
    input  logic [NUM_MULTS-1:0][NUM_PID-1:0]     wsp_q,

    // -- Per-lane weight bank read ports (pe_wmem) ---------------------------
    input  logic [NUM_MULTS-1:0][DATA_WIDTH-1:0]  rd_val,
    input  logic [NUM_MULTS-1:0][PID_WIDTH-1:0]   rd_pid,
    output logic [NUM_MULTS-1:0]                  rd_en,
    output logic [NUM_MULTS-1:0][SLOT_WIDTH-1:0]  rd_slot,

    // -- Per-lane MAC controls exported to the datapath ----------------------
    output logic [NUM_MULTS-1:0]                  mac_en,         // KEEP or BANK
    output logic [NUM_MULTS-1:0][DATA_WIDTH-1:0]  mac_w
);

    // -------------------------------------------------------------------------
    // Per-lane state: the held Curr weight, plus a valid/slot tag for whatever
    // the lane's bank read register currently holds (its look-ahead slot).
    // -------------------------------------------------------------------------
    logic [NUM_MULTS-1:0][PID_WIDTH-1:0]   curr_pid;
    logic [NUM_MULTS-1:0][DATA_WIDTH-1:0]  curr_val;   // 2's-complement, $signed() at use
    logic [NUM_MULTS-1:0][SLOT_WIDTH-1:0]  curr_slot;  // bank slot of Curr
    logic [NUM_MULTS-1:0]                  have_curr;
    logic [NUM_MULTS-1:0]                  bank_valid; // rd_val/rd_pid hold a real read
    logic [NUM_MULTS-1:0][SLOT_WIDTH-1:0]  bank_slot;  // slot the bank register holds
    logic                                  running;

    // -------------------------------------------------------------------------
    // Per-lane action (combinational, valid while running)
    // -------------------------------------------------------------------------
    typedef enum logic [1:0] {A_IDLE, A_KEEP, A_BANK, A_SKIP} action_t;
    action_t action [NUM_MULTS];

    logic [NUM_MULTS-1:0]                 need_fetch;       // SKIP lanes -> stall this beat
    logic [NUM_MULTS-1:0]                 want_prefetch;    // idle port -> fetch look-ahead
    logic [NUM_MULTS-1:0][RPTR_WIDTH-1:0] n_weights;        // popcount(wsp): live weight count
    logic [NUM_MULTS-1:0][SLOT_WIDTH-1:0] target_slot;      // bank slot holding b_pid's weight
    logic [NUM_PID-1:0]                   below_mask;
    logic                                 consume;

    always_comb begin
        below_mask = (NUM_PID'(1) << b_pid) - NUM_PID'(1);
        for (int k = 0; k < NUM_MULTS; k++) begin
            logic wsp_hit, keep, bank_hit, next_exists, bank_has_next;

            n_weights[k]   = RPTR_WIDTH'($countones(wsp_q[k]));
            target_slot[k] = SLOT_WIDTH'($countones(wsp_q[k] & below_mask));

            wsp_hit  = running && b_valid && wsp_q[k][b_pid];
            keep     = wsp_hit && have_curr[k] && (b_pid == curr_pid[k]);
            bank_hit = wsp_hit && !keep && bank_valid[k] && (rd_pid[k] == b_pid);

            if (!wsp_hit)
                action[k] = A_IDLE;
            else if (keep)
                action[k] = A_KEEP;
            else if (bank_hit)
                action[k] = A_BANK;
            else
                action[k] = A_SKIP;

            // Prefetch the look-ahead weight (curr_slot+1) whenever the port is
            // idle and the bank does not already hold it -- keeps the next PID
            // transition stall-free. Never issued on a BANK/SKIP cycle (BANK must
            // hold its data; SKIP owns the port for its jump).
            next_exists      = (RPTR_WIDTH'(curr_slot[k]) + RPTR_WIDTH'(1)) < n_weights[k];
            bank_has_next    = bank_valid[k]
                               && (bank_slot[k] == SLOT_WIDTH'(curr_slot[k] + SLOT_WIDTH'(1)));
            need_fetch[k]     = (action[k] == A_SKIP);
            want_prefetch[k]  = running && have_curr[k] && (action[k] != A_BANK)
                                && (action[k] != A_SKIP) && next_exists && !bank_has_next;

            mac_en[k] = (action[k] == A_KEEP) || (action[k] == A_BANK);
            mac_w[k]  = (action[k] == A_KEEP) ? curr_val[k] : rd_val[k];
        end
    end

    // -------------------------------------------------------------------------
    // Per-lane bank read-port driver: a SKIP jumps to target_slot, else an idle
    // port prefetches the look-ahead weight.
    // -------------------------------------------------------------------------
    always_comb begin
        rd_en   = '0;
        rd_slot = '0;
        if (running) begin
            for (int k = 0; k < NUM_MULTS; k++) begin
                if (action[k] == A_SKIP) begin
                    rd_en[k]   = 1'b1;
                    rd_slot[k] = target_slot[k];
                end else if (want_prefetch[k]) begin
                    rd_en[k]   = 1'b1;
                    rd_slot[k] = SLOT_WIDTH'(curr_slot[k] + SLOT_WIDTH'(1));
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // Sequencer
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            running <= 1'b0;
            for (int k = 0; k < NUM_MULTS; k++) begin
                curr_pid[k]         <= '0;
                curr_val[k]         <= '0;
                curr_slot[k]        <= '0;
                have_curr[k]        <= 1'b0;
                bank_valid[k]       <= 1'b0;
                bank_slot[k]        <= '0;
            end
        end else if (wload_done) begin
            // Arm / re-arm: run from the (freshly loaded) banks with an empty
            // window -- the first hit on each lane SKIPs and fetches.
            running <= 1'b1;
            for (int k = 0; k < NUM_MULTS; k++) begin
                have_curr[k]  <= 1'b0;
                bank_valid[k] <= 1'b0;
            end
        end else if (running) begin
            for (int k = 0; k < NUM_MULTS; k++) begin
                // Track what the bank read register will hold next cycle.
                if (rd_en[k]) begin
                    bank_valid[k] <= 1'b1;
                    bank_slot[k]  <= rd_slot[k];
                end

                // On an accepted BANK beat, promote the bank weight to Curr.
                if (consume && action[k] == A_BANK) begin
                    curr_pid[k]  <= rd_pid[k];
                    curr_val[k]  <= rd_val[k];
                    curr_slot[k] <= bank_slot[k];
                    have_curr[k] <= 1'b1;
                end
            end
        end
    end

    assign consume = b_ready;
    assign b_ready = running && b_valid && !(|need_fetch);

endmodule

`default_nettype wire
