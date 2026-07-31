// ---------------------------------------------------------------------
// routing.sv -- APU Routing Module (V2 "act" dataflow)
//
// Drains N_PID FIFO-A lanes (lane j = PID j) in ascending order and HOLDS
// each lane until it is empty before moving to the next -- so a PID run
// arrives at the PEs intact. Each cycle it pops up to NUM_MULTS entries from
// the current lane (they all share PID = cur, with distinct CIDs) and emits
// one M-wide BEAT: {pid, lane_valid[M], act[M], cid[M]}. The beat is pushed
// into every FIFO-B #k whose WSP selects PID cur.
//
// WSP is LSB-first by PID, matching the PE weight bank (pe_mem.wsp_q):
// wsp[k][p] = 1 means PE k has a non-zero weight at PID p. So WSP = 4'b0101
// -> PIDs 0,2. The router reads this straight from each PE's exported WSP.
//
// All-or-nothing multicast: a beat is committed (lane popped + pushed to all
// selected FIFO-Bs) only on a cycle where every selected FIFO-B can accept,
// so no PE drops data.
//
// Framing: pulse start to run one pass; done pulses when the last lane empties.
// Tie start=1 to free-run.
// ---------------------------------------------------------------------

`default_nettype none

module routing #(
    parameter int N_PID     = 9,                        // # FIFO-A lanes (one per PID); = F*F upstream
    parameter int N_PE      = 4,                         // # FIFO-B / PE output ports
    parameter int NUM_MULTS = 4,                         // entries popped / beat width
    parameter int ACT_WIDTH = 8,
    parameter int CID_WIDTH = 6,
    parameter int CNT_WIDTH = 7,                         // FIFO-A occupancy width
    parameter int PID_WIDTH = (N_PID <= 1) ? 1 : $clog2(N_PID),
    parameter int CUR_WIDTH = (N_PID <= 1) ? 1 : $clog2(N_PID),
    parameter int NAV_WIDTH = $clog2(NUM_MULTS + 1)     // 0..NUM_MULTS
)(
    input  logic                                         clk,
    input  logic                                         rst_n,

    input  logic                                         start,
    output logic                                         busy,
    output logic                                         done,

    input  logic [N_PE-1:0][N_PID-1:0]                   wsp,

    // FIFO-A side (N_PID lanes, each an M-wide read port)
    input  logic [N_PID-1:0][NUM_MULTS-1:0][ACT_WIDTH-1:0] a_act,
    input  logic [N_PID-1:0][NUM_MULTS-1:0][CID_WIDTH-1:0] a_cid,
    input  logic [N_PID-1:0][NUM_MULTS-1:0]              a_rvalid,   // thermometer per lane
    input  logic [N_PID-1:0][CNT_WIDTH-1:0]             a_count,    // occupancy per lane
    output logic [N_PID-1:0][NUM_MULTS-1:0]             a_pop,      // rd_ready prefix per lane

    // FIFO-B side (N_PE beat ports)
    input  logic [N_PE-1:0]                              b_ready,
    output logic [N_PE-1:0]                              b_push,
    output logic [N_PE-1:0][PID_WIDTH-1:0]               b_pid,
    output logic [N_PE-1:0][NUM_MULTS-1:0]               b_lane_valid,
    output logic [N_PE-1:0][NUM_MULTS-1:0][ACT_WIDTH-1:0] b_act,
    output logic [N_PE-1:0][NUM_MULTS-1:0][CID_WIDTH-1:0] b_cid
);

    typedef enum logic [1:0] {S_IDLE, S_RUN, S_DONE} state_t;
    state_t               state;
    logic [CUR_WIDTH-1:0] cur;         // lane currently being drained

    logic [NAV_WIDTH-1:0] navail;      // valid entries in cur's read window (0..M)
    logic                 head_valid;
    logic                 all_ready;
    logic                 go;
    logic                 lane_empty;   // cur has nothing resident
    logic                 lane_drained; // this beat pops cur's final entries
    logic                 advance;
    logic [N_PE-1:0]      sel;          // sel[k] = PE#k consumes PID cur

    // Valid entries visible in cur's window (rd_valid is thermometer-coded).
    always_comb begin
        navail = '0;
        for (int i = 0; i < NUM_MULTS; i++)
            if (a_rvalid[cur][i])
                navail = navail + NAV_WIDTH'(1);
    end

    always_comb begin
        head_valid = (state == S_RUN) && (navail != '0);

        // LSB-first by PID (pe_mem.wsp_q): wsp[k][p] = 1 -> PE k wants PID p.
        for (int k = 0; k < N_PE; k++)
            sel[k] = wsp[k][cur];

        all_ready = 1'b1;
        for (int k = 0; k < N_PE; k++)
            if (sel[k] && !b_ready[k])
                all_ready = 1'b0;

        go           = head_valid && all_ready;
        lane_empty   = (state == S_RUN) && (a_count[cur] == '0);
        lane_drained = go && (a_count[cur] == CNT_WIDTH'(navail));
        advance      = lane_empty || lane_drained;

        // Pop the whole visible window from cur when the beat commits.
        a_pop = '0;
        for (int i = 0; i < NUM_MULTS; i++)
            a_pop[cur][i] = go && a_rvalid[cur][i];

        // Broadcast the same beat to every selected PE.
        for (int k = 0; k < N_PE; k++) begin
            b_pid[k]        = PID_WIDTH'(cur);
            b_lane_valid[k] = a_rvalid[cur];
            for (int i = 0; i < NUM_MULTS; i++) begin
                b_act[k][i] = a_act[cur][i];
                b_cid[k][i] = a_cid[cur][i];
            end
            b_push[k] = go && sel[k];
        end

        busy = (state == S_RUN);
        done = (state == S_DONE);
    end

    // Sequencer: walk lanes 0..N_PID-1; hold on a lane until it drains.
    // Synchronous reset to match the rest of the design's reset domain.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE;
            cur   <= '0;
        end else begin
            case (state)
                S_IDLE: begin
                    if (start) begin
                        cur   <= '0;
                        state <= S_RUN;
                    end
                end
                S_RUN: begin
                    if (advance) begin
                        if (cur == CUR_WIDTH'(N_PID-1))
                            state <= S_DONE;
                        else
                            cur <= CUR_WIDTH'(cur + 1);
                    end
                end
                S_DONE: state <= S_IDLE;
                default: state <= S_IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    always_comb begin
        for (int k = 0; k < N_PE; k++) begin
            assert #0 (!b_push[k] || (go && sel[k]))
                else $error("routing: b_push[%0d] without go&&sel", k);
            assert #0 (!go || !sel[k] || b_ready[k])
                else $error("routing: committed but PE#%0d selected & not ready", k);
        end
    end
`endif

endmodule

`default_nettype wire
