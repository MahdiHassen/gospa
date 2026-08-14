// ---------------------------------------------------------------------
// routing.sv -- APU Routing Module (V2 "act" dataflow, batched)
//
// Drains N_PID FIFO-A lanes (lane j = PID j) in ascending order and HOLDS
// each lane until it is empty before moving to the next -- so a PID run
// arrives at the PEs intact. Each cycle it pops up to S2_BEATS*NUM_MULTS
// entries from the current lane (they all share PID = cur, with distinct
// CIDs) and emits up to S2_BEATS M-wide BEATS: {pid, lane_valid[M], act[M],
// cid[M]}. The beat group is pushed into every FIFO-B #k whose WSP selects
// PID cur. With S2_BEATS > 1 the router outruns the PEs (which consume one
// beat/cycle), so FIFO-B builds a backlog that keeps the PEs fed while the
// front end refills FIFO-A with the next channel.
//
// WSP is LSB-first by PID, matching the PE weight bank (pe_mem.wsp_q):
// wsp[k][p] = 1 means PE k has a non-zero weight at PID p. So WSP = 4'b0101
// -> PIDs 0,2. The router reads this straight from each PE's exported WSP.
//
// All-or-nothing multicast: a beat group is committed (lane popped + pushed
// to all selected FIFO-Bs) only on a cycle where every selected FIFO-B can
// accept a full S2_BEATS push, so no PE drops data.
//
// Framing: pulse start to run one pass; done pulses when the last lane
// empties (the PEs may still be consuming FIFO-B backlog at that point).
// Tie start=1 to free-run.
// ---------------------------------------------------------------------

`default_nettype none

module routing #(
    parameter int N_PID     = 9,                        // # FIFO-A lanes (one per PID); = F*F upstream
    parameter int N_PE      = 4,                         // # FIFO-B / PE output ports
    parameter int NUM_MULTS = 4,                         // beat width (activations / beat)
    parameter int S2_BEATS  = 1,                         // beats emitted per cycle
    parameter int ACT_WIDTH = 8,
    parameter int CID_WIDTH = 6,
    parameter int CNT_WIDTH = 7,                         // FIFO-A occupancy width
    parameter int DW_COLW   = 0,                         // >0: depthwise-mosaic
                                                         // demux; cid = {row,
                                                         // col[DW_COLW-1:0]}
    parameter int PID_WIDTH = (N_PID <= 1) ? 1 : $clog2(N_PID),
    parameter int CUR_WIDTH = (N_PID <= 1) ? 1 : $clog2(N_PID),
    localparam int RD_W     = NUM_MULTS * S2_BEATS,      // entries popped / cycle
    localparam int NAV_WIDTH = $clog2(RD_W + 1)          // 0..RD_W
)(
    input  logic                                         clk,
    input  logic                                         rst_n,

    input  logic                                         start,
    output logic                                         busy,
    output logic                                         done,

    input  logic [N_PE-1:0][N_PID-1:0]                   wsp,

    // Depthwise-mosaic demux (used when dw_en; needs DW_COLW > 0): each PE
    // owns a 2-D CID band [r0,r1) x [c0,c1) of the composite output map.
    // A beat group goes to a PE only if some lane's CID is in its band, and
    // b_band_mask marks which lanes are (ANDed into lane_valid downstream).
    input  logic                                         dw_en,
    input  logic [N_PE-1:0][CID_WIDTH-1:0]               band_r0,
    input  logic [N_PE-1:0][CID_WIDTH-1:0]               band_r1,
    input  logic [N_PE-1:0][CID_WIDTH-1:0]               band_c0,
    input  logic [N_PE-1:0][CID_WIDTH-1:0]               band_c1,
    output logic [N_PE-1:0][S2_BEATS-1:0][NUM_MULTS-1:0] b_band_mask,

    // FIFO-A side (N_PID lanes, each an RD_W-wide read port)
    input  logic [N_PID-1:0][RD_W-1:0][ACT_WIDTH-1:0]    a_act,
    input  logic [N_PID-1:0][RD_W-1:0][CID_WIDTH-1:0]    a_cid,
    input  logic [N_PID-1:0][RD_W-1:0]                   a_rvalid,   // thermometer per lane
    input  logic [N_PID-1:0][CNT_WIDTH-1:0]              a_count,    // occupancy per lane
    output logic [N_PID-1:0][RD_W-1:0]                   a_pop,      // rd_ready prefix per lane

    // FIFO-B side: one shared beat group per cycle, gated per PE by b_push.
    // Beat j carries entries [j*M, j*M+M) of the current window; all beats
    // share PID = cur. b_beat_valid is a thermometer (ceil(navail/M) beats).
    input  logic [N_PE-1:0]                              b_ready,    // full S2_BEATS room
    output logic [N_PE-1:0]                              b_push,
    output logic [S2_BEATS-1:0]                          b_beat_valid,
    output logic [PID_WIDTH-1:0]                         b_pid,
    output logic [S2_BEATS-1:0][NUM_MULTS-1:0]           b_lane_valid,
    output logic [S2_BEATS-1:0][NUM_MULTS-1:0][ACT_WIDTH-1:0] b_act,
    output logic [S2_BEATS-1:0][NUM_MULTS-1:0][CID_WIDTH-1:0] b_cid
);

    typedef enum logic [1:0] {S_IDLE, S_RUN, S_DONE} state_t;
    state_t               state;
    logic [CUR_WIDTH-1:0] cur;         // lane currently being drained

    // 2-D CID band membership for the depthwise-mosaic demux.
    function automatic logic band_hit(input int k,
                                      input logic [CID_WIDTH-1:0] cc);
        logic [CID_WIDTH-1:0] rr, col;
        rr  = CID_WIDTH'(cc >> DW_COLW);
        col = cc & CID_WIDTH'((32'(1) << DW_COLW) - 1);
        return (rr >= band_r0[k]) && (rr < band_r1[k]) &&
               (col >= band_c0[k]) && (col < band_c1[k]);
    endfunction

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
        for (int i = 0; i < RD_W; i++)
            if (a_rvalid[cur][i])
                navail = navail + NAV_WIDTH'(1);
    end

    always_comb begin
        head_valid = (state == S_RUN) && (navail != '0);

        // Per-PE per-lane band membership (all-ones outside demux mode).
        for (int k = 0; k < N_PE; k++)
            for (int j = 0; j < S2_BEATS; j++)
                for (int i = 0; i < NUM_MULTS; i++)
                    b_band_mask[k][j][i] = (DW_COLW > 0 && dw_en)
                        ? band_hit(k, a_cid[cur][j * NUM_MULTS + i])
                        : 1'b1;

        // LSB-first by PID (pe_mem.wsp_q): wsp[k][p] = 1 -> PE k wants PID p.
        // In demux mode a PE is selected only if a valid lane is in its band.
        for (int k = 0; k < N_PE; k++) begin
            sel[k] = wsp[k][cur];
            if (DW_COLW > 0 && dw_en)
                sel[k] = wsp[k][cur]
                         && (|(a_rvalid[cur] & RD_W'(b_band_mask[k])));
        end

        all_ready = 1'b1;
        for (int k = 0; k < N_PE; k++)
            if (sel[k] && !b_ready[k])
                all_ready = 1'b0;

        go           = head_valid && all_ready;
        lane_empty   = (state == S_RUN) && (a_count[cur] == '0);
        lane_drained = go && (a_count[cur] == CNT_WIDTH'(navail));
        advance      = lane_empty || lane_drained;

        // Pop the whole visible window from cur when the beat group commits.
        a_pop = '0;
        for (int i = 0; i < RD_W; i++)
            a_pop[cur][i] = go && a_rvalid[cur][i];

        // One shared beat group; beat j takes window entries [j*M, j*M+M).
        b_pid = PID_WIDTH'(cur);
        for (int j = 0; j < S2_BEATS; j++) begin
            b_beat_valid[j] = go && a_rvalid[cur][j * NUM_MULTS];
            for (int i = 0; i < NUM_MULTS; i++) begin
                b_lane_valid[j][i] = a_rvalid[cur][j * NUM_MULTS + i];
                b_act[j][i]        = a_act[cur][j * NUM_MULTS + i];
                b_cid[j][i]        = a_cid[cur][j * NUM_MULTS + i];
            end
        end
        for (int k = 0; k < N_PE; k++)
            b_push[k] = go && sel[k];

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
