// ---------------------------------------------------------------------
// routing.sv -- APU Routing Module 
//
// Drains N_PID FIFO-A lanes (lane j = PID j) in order, one head per cycle,
// and multicasts each {ACT,CID} into every FIFO-B #k whose WSP selects PID j.
//
// WSP is MSB-first by PID: wsp[k][N_PID-1] = PID 0 .. wsp[k][0] = PID N_PID-1, 
// matching the weight-SRAM metadata. So WSP_1 = 4'b1010 -> PIDs 0,2.
//
// All-or-nothing multicast: a head pops (and pushes to all selected FIFO-Bs)
// only on a cycle where every selected FIFO-B is ready, so no PE drops data.
//
// Framing: pulse start to run one pass; done pulses when the last lane empties.
// Tie start=1 to free-run.
// ---------------------------------------------------------------------

module routing #(
    parameter int N_PID = 9,                               // # FIFO-A lanes (one per PID); = F*F upstream
    parameter int N_PE  = 4,                               // # FIFO-B / PE output ports
    parameter int ACT_WIDTH = 8,
    parameter int CID_WIDTH = 6,
    parameter int PID_WIDTH = (N_PID <= 1) ? 1 : $clog2(N_PID),
    parameter int CUR_WIDTH = (N_PID <= 1) ? 1 : $clog2(N_PID)
)(
    input  logic                             clk,
    input  logic                             rst_n,
 
    input  logic                             start,
    output logic                             busy,
    output logic                             done,

    input  logic [N_PE-1:0 ][N_PID-1:0]      wsp,

    // FIFO-A side (N_PID lanes)
    input  logic [N_PID-1:0][ACT_WIDTH-1:0]  a_act,
    input  logic [N_PID-1:0][CID_WIDTH-1:0]  a_cid,
    input  logic [N_PID-1:0]                 a_empty,
    input  logic [N_PID-1:0]                 a_almost_empty,   
    output logic [N_PID-1:0]                 a_pop,

    // FIFO-B side (N_PE ports)
    input  logic [N_PE-1:0 ]                 b_ready,
    output logic [N_PE-1:0 ]                 b_push,
    output logic [N_PE-1:0 ][ACT_WIDTH-1:0]  b_act,
    output logic [N_PE-1:0 ][CID_WIDTH-1:0]  b_cid,
    output logic [N_PE-1:0 ][PID_WIDTH-1:0]  b_pid
);

    typedef enum logic [1:0] {S_IDLE, S_RUN, S_DONE} state_t;
    state_t           state;
    logic [CUR_WIDTH-1:0] cur;     // lane currently being drained

    logic head_valid;
    logic all_ready;
    logic go;                 
    logic last_head;           // this commit drains the lane's final head
    logic [N_PE-1:0] sel;      // sel[k] = PE#k consumes the current lane

    always_comb begin
        head_valid = (state == S_RUN) && !a_empty[cur];

        // MSB-first by PID: wsp[k][N_PID-1] = PID 0 .. wsp[k][0] = PID N_PID-1
        for (int k = 0; k < N_PE; k++)
            sel[k] = wsp[k][N_PID-1 - int'(cur)];

        all_ready = 1'b1;
        for (int k = 0; k < N_PE; k++)
            if (sel[k] && !b_ready[k])
                all_ready = 1'b0;

        go = head_valid && all_ready;
        last_head = go && a_almost_empty[cur];

        a_pop = '0;
        a_pop[cur] = go;

        for (int k = 0; k < N_PE; k++) begin
            b_act[k]  = a_act[cur];
            b_cid[k]  = a_cid[cur];
            b_pid[k]  = PID_WIDTH'(cur);
            b_push[k] = go && sel[k];
        end

        busy = (state == S_RUN);
        done = (state == S_DONE);
    end

    // Sequencer: walk lanes 0..N_PID-1; hold while backpressured.
    always_ff @(posedge clk or negedge rst_n) begin
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
                    if (a_empty[cur] || last_head) begin
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
        assert #0 ($onehot0(a_pop) && (a_pop == '0 || a_pop[cur]))
            else $error("routing: a_pop=%b illegal (cur=%0d)", a_pop, cur);
        for (int k = 0; k < N_PE; k++) begin
            assert #0 (!b_push[k] || (go && sel[k]))
                else $error("routing: b_push[%0d] without go&&sel", k);
            assert #0 (!go || !sel[k] || b_ready[k])
                else $error("routing: committed but PE#%0d selected & not ready", k);
        end
    end
`endif

endmodule
