// =============================================================================
// pe.sv -- GoSPA Processing Element
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// One PE consumes its FIFO-B stream (act_value, PID, CID), multiplies each
// activation by the matching non-zero weight, and accumulates the product into
// a CID-indexed accumulator bank (pe_acc.sv). Output is one channel's E x E map.
//
// Weight reuse (paper Fig. 12): the PE holds the non-zero weights of its filter
// in PID order in a small store, plus a Curr/Next two-register window. Because
// Stage 1 groups activations by PID and Stage 2 drains FIFO-A in PID order, the
// activations arrive in NON-DECREASING PID order, so the window only ever needs
// to move forward.
//
//   - match_curr (b_pid == Curr_PID): reuse Curr weight, accumulate.        KEEP
//   - match_next (b_pid == Next_PID): use Next weight, accumulate, and slide
//                                     the window (Curr <- Next, reload Next). UPDATE
//   - otherwise  (b_pid >  Next_PID): the weight at the current PID has no
//                                     activation this pass -- slide the window
//                                     WITHOUT consuming, and re-check next cycle. SKIP
//
// The SKIP case is what the team's functional `pe_process` gets wrong: it slides
// blindly on any mismatch and multiplies by the wrong weight when a non-zero
// weight position receives no activation (happens with sparse activations). This
// PE handles it correctly -- it is verified against dense convolution, not
// against pe_process. A SKIP costs extra cycles but never extra hardware
// (still a single weight read per cycle, two registers).
//
// Throughput: 1 activation / cycle when no skips; +1 cycle per skipped weight.
// =============================================================================

`default_nettype none

module pe #(
    parameter int N_PID  = 9,    // # kernel positions = F*F (max # weights)
    parameter int N_CID  = 36,   // # output positions = E*E (accumulator banks)
    parameter int DATA_W = 16,   // activation / weight width
    parameter int ACC_W  = 32,   // accumulator width

    // -- Derived --------------------------------------------------------------
    localparam int PID_W  = (N_PID < 2) ? 1 : $clog2(N_PID),
    localparam int CID_W  = (N_CID < 2) ? 1 : $clog2(N_CID),
    localparam int WPTR_W = $clog2(N_PID + 1),
    localparam int PROD_W = 2 * DATA_W
)(
    input  wire  logic                     clk,
    input  wire  logic                     rst_n,

    // -- Weight preload: pulse (pid,val) in PID order, then pulse wload_done ---
    input  wire  logic                     wload_en,
    input  wire  logic [PID_W-1:0]         wload_pid,
    input  wire  logic signed [DATA_W-1:0] wload_val,
    input  wire  logic                     wload_done,   // latch Curr/Next, arm PE

    // -- FIFO-B input stream (act, pid, cid), PID non-decreasing --------------
    input  wire  logic                     b_valid,
    input  wire  logic signed [DATA_W-1:0] b_act,
    input  wire  logic [PID_W-1:0]         b_pid,
    input  wire  logic [CID_W-1:0]         b_cid,
    output logic                           b_ready,

    // -- Drain / output (one beat per CID, 0..N_CID-1) ------------------------
    input  wire  logic                     drain_start,
    output logic                           drain_busy,
    output logic                           drain_done,
    output logic                           out_valid,
    input  wire  logic                     out_ready,
    output logic [CID_W-1:0]               out_cid,
    output logic signed [ACC_W-1:0]        out_acc
);

    // -------------------------------------------------------------------------
    // Sparse weight store (non-zero weights in PID order) + Curr/Next window
    // -------------------------------------------------------------------------
    logic [PID_W-1:0]          wpid [0:N_PID-1];
    logic signed [DATA_W-1:0]  wval [0:N_PID-1];
    logic [WPTR_W-1:0]         wcnt;       // # weights loaded
    logic [WPTR_W-1:0]         wptr;       // read cursor for the Next reload

    logic                      armed;      // weights loaded, PE running
    logic [PID_W-1:0]          curr_pid, next_pid;
    logic signed [DATA_W-1:0]  curr_val, next_val;
    logic                      have_curr, have_next;

    // -------------------------------------------------------------------------
    // Combinational match / dispatch
    // -------------------------------------------------------------------------
    logic match_curr, match_next, consume, advance;
    logic signed [DATA_W-1:0] use_wgt;
    logic signed [PROD_W-1:0] prod;

    always_comb begin
        match_curr = armed && have_curr && b_valid && (b_pid == curr_pid);
        match_next = armed && have_next && b_valid && (b_pid == next_pid);

        // Consume the activation only when a weight matches it.
        consume = match_curr || match_next;

        // Slide the window whenever the head is past Curr (UPDATE or SKIP).
        advance = armed && b_valid && have_curr && (b_pid != curr_pid);

        use_wgt = match_curr ? curr_val : next_val;
        prod    = b_act * use_wgt;

        // If this PE has no weights at all, swallow the stream so it can drain.
        b_ready = consume || (armed && b_valid && !have_curr);
    end

    // -------------------------------------------------------------------------
    // Weight load + window sequencer
    // (async reset to match pe_acc.sv's reset convention)
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wcnt      <= '0;
            wptr      <= '0;
            armed     <= 1'b0;
            have_curr <= 1'b0;
            have_next <= 1'b0;
            curr_pid  <= '0; curr_val <= '0;
            next_pid  <= '0; next_val <= '0;
        end else if (!armed) begin
            // ---- Load phase: append (pid,val); arm on wload_done -------------
            if (wload_en) begin
                wpid[wcnt] <= wload_pid;
                wval[wcnt] <= wload_val;
                wcnt       <= wcnt + 1'b1;
            end
            if (wload_done) begin
                curr_pid  <= wpid[0]; curr_val <= wval[0];
                next_pid  <= wpid[1]; next_val <= wval[1];
                have_curr <= (wcnt > 0);
                have_next <= (wcnt > 1);
                wptr      <= WPTR_W'(2);
                armed     <= 1'b1;
            end
        end else if (advance && !drain_busy) begin
            // ---- Run phase: slide window forward by one weight --------------
            curr_pid  <= next_pid;
            curr_val  <= next_val;
            have_curr <= have_next;
            if (wptr < wcnt) begin
                next_pid  <= wpid[wptr];
                next_val  <= wval[wptr];
                wptr      <= wptr + 1'b1;
                have_next <= 1'b1;
            end else begin
                have_next <= 1'b0;
            end
        end
    end

    // -------------------------------------------------------------------------
    // Accumulator bank (Adil's pe_acc.sv): one MAC add per consumed activation
    // -------------------------------------------------------------------------
    pe_acc #(
        .N_CID(N_CID), .ACC_WIDTH(ACC_W), .PROD_WIDTH(PROD_W)
    ) u_acc (
        .clk        (clk),
        .rst_n      (rst_n),
        .clear      (1'b0),
        .add_en     (consume && !drain_busy),
        .add_cid    (b_cid),
        .add_data   (prod),
        .drain_start(drain_start),
        .drain_busy (drain_busy),
        .drain_done (drain_done),
        .out_valid  (out_valid),
        .out_ready  (out_ready),
        .out_cid    (out_cid),
        .out_acc    (out_acc)
    );

endmodule

`default_nettype wire
