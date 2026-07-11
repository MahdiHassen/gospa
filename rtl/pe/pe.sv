// =============================================================================
// pe.sv -- GoSPA Processing Element (V2: multiple kernels per PE)
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// V2 PE structure (this file is the top-level glue; the two big subsystems
// live in their own modules):
//   - pe_wmem   : shared on-chip weight SRAM (sram.sv) per PE + fill/WSP
//                 bookkeeping. Depth NUM_MULTS*NUM_PID, width DATA_WIDTH+PID_WIDTH; each
//                 word holds {value, pid}. Lane k's kernel lives at SRAM
//                 addresses [k*NUM_PID, (k+1)*NUM_PID). Exports wfill_cnt / wsp_q.
//   - pe_window : per-lane Curr/Next window, the warm-up preload, the per-lane
//                 action-eval (IDLE/RETIRED/KEEP/UPDATE/SLIDE -> mac_en/mac_w),
//                 and the round-robin refill arbiter -- the shared SRAM has one
//                 read port, so at most one lane fetches its new Next per cycle.
//                 While any lane needs a refill (slide) it de-asserts b_ready;
//                 the activation isn't consumed until every non-IDLE /
//                 non-RETIRED lane is in KEEP or UPDATE with a valid window.
//                 Conservative; KEEP-heavy workloads still hit one act/cycle.
//   - pe_lane   : one per output channel -- pipelined multiply + CID-indexed
//                 accumulator + drain.
//   - here      : glue + the array-level drain gating (hold the drain off until
//                 every lane's MAC pipeline has flushed).
//
// Per-lane WSP (NUM_PID bits, from pe_wmem) tells a lane which incoming PIDs
// belong to its kernel; non-WSP PIDs are IDLE (no MAC, no slide).
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
//   1) For each lane k: stream {wfill_we, wfill_lane=k, wfill_pid, wfill_val}
//      once per weight, in PID order. pe_wmem appends to the lane's SRAM slots,
//      counts them, and sets the lane's WSP bit for each wfill_pid -- so the
//      per-lane slot, weight count, and WSP are all DERIVED from this stream;
//      the host drives no separate slot / count / WSP ports.
//   2) Pulse wload_done. pe_window issues 2*NUM_MULTS SRAM reads to pre-load
//      every lane's Curr and Next, then enters S_RUN. A bare wload_done with no
//      new fills re-arms the SAME weights; the first wfill after an arm begins
//      a fresh session (prior counts/WSP cleared).
// =============================================================================

`default_nettype none

module pe #(
    parameter int NUM_MULTS  = 4,    // multiplier lanes per PE (output channels held here)
    parameter int NUM_PID    = 9,    // # kernel positions = F*F (max weights/lane)
    parameter int NUM_CID    = 36,   // # output positions = E*E (accumulator banks)
    parameter int DATA_WIDTH = 16,
    parameter int ACC_WIDTH  = 32,

    // -- Derived --------------------------------------------------------------
    localparam int PID_WIDTH   = (NUM_PID   < 2) ? 1 : $clog2(NUM_PID),
    localparam int CID_WIDTH   = (NUM_CID   < 2) ? 1 : $clog2(NUM_CID),
    localparam int LANE_WIDTH  = (NUM_MULTS < 2) ? 1 : $clog2(NUM_MULTS),
    localparam int RPTR_WIDTH  = $clog2(NUM_PID + 1),
    localparam int WSRAM_DEPTH = NUM_MULTS * NUM_PID,
    localparam int WSRAM_AW    = (WSRAM_DEPTH < 2) ? 1 : $clog2(WSRAM_DEPTH)
)(
    input  logic                                 clk,
    input  logic                                 rst_n,

    // -- Weight SRAM fill -- one weight per cycle; slot, per-lane count, and
    //    WSP are all derived from this stream (see Load sequence above) ------
    input  logic                                 wfill_we,
    input  logic [LANE_WIDTH-1:0]                wfill_lane,
    input  logic [PID_WIDTH-1:0]                 wfill_pid,
    input  logic signed [DATA_WIDTH-1:0]         wfill_val,

    // -- Arm: Curr/Next pre-loaded from the filled weights -------------------
    input  logic                                 wload_done,

    // -- FIFO-B input stream (act, pid, cid), PID monotone --------------------
    input  logic                                 b_valid,
    input  logic signed [DATA_WIDTH-1:0]         b_act,
    input  logic [PID_WIDTH-1:0]                 b_pid,
    input  logic [CID_WIDTH-1:0]                 b_cid,
    output logic                                 b_ready,

    // -- Drain / per-lane output (one NUM_CID-beat stream per lane) ------------
    input  logic                                 drain_start,
    output logic                                 drain_busy,
    output logic                                 drain_done,
    output logic [NUM_MULTS-1:0]                 out_valid,
    output logic [NUM_MULTS-1:0][CID_WIDTH-1:0]  out_cid,
    output logic [NUM_MULTS-1:0][ACC_WIDTH-1:0]  out_acc,
    input  logic [NUM_MULTS-1:0]                 out_ready
);

    // -------------------------------------------------------------------------
    // Signal declarations
    // -------------------------------------------------------------------------
    // Weight-memory / window interface: wmem hands back SRAM read data plus the
    // derived per-lane weight count (wfill_cnt) and WSP (wsp_q); the window
    // drives the SRAM's single read port.
    logic                                   rd_en;
    logic [WSRAM_AW-1:0]                    rd_addr;
    logic [DATA_WIDTH-1:0]                  rd_val;
    logic [PID_WIDTH-1:0]                   rd_pid;
    logic [NUM_MULTS-1:0][RPTR_WIDTH-1:0]   wfill_cnt;
    logic [NUM_MULTS-1:0][NUM_PID-1:0]      wsp_q;

    // Per-lane MAC controls from the window's action-eval (KEEP/UPDATE -> MAC).
    logic [NUM_MULTS-1:0]                   mac_en;
    logic [NUM_MULTS-1:0][DATA_WIDTH-1:0]   mac_w;

    logic consume;   // alias for b_ready

    // Per-lane pe_lane outputs, piped through internal wires so both the module
    // output ports AND drain_busy can read them cleanly (one driver -> readers).
    // mac_en_q gathers each lane's registered accumulate-enable (mac_busy) so
    // the drain can wait for the MAC pipeline to flush.
    logic [NUM_MULTS-1:0]                   pe_out_valid_w;
    logic [NUM_MULTS-1:0][CID_WIDTH-1:0]    pe_out_cid_w;
    logic [NUM_MULTS-1:0][ACC_WIDTH-1:0]    pe_out_acc_w;
    logic [NUM_MULTS-1:0]                   pe_drain_busy_w;
    logic [NUM_MULTS-1:0]                   mac_en_q;

    // Drain gating: deferred until every lane's in-flight product has been
    // accumulated (mac_en_q all 0) and no new product is being formed this
    // cycle (!consume) -- otherwise the trailing pipelined product would be
    // dropped when a lane switches into drain-readout mode. drain_start is
    // latched (drain_req_q) so a single-cycle pulse is not lost while we wait.
    logic drain_req_q;
    logic acc_drain_pulse;

    logic busy_q;   // drain_busy, registered one cycle (for the drain_done edge)

    // -------------------------------------------------------------------------
    // Sequential logic
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n)            
            drain_req_q <= 1'b0;
        else if (acc_drain_pulse)   
            drain_req_q <= 1'b0;
        else if (drain_start)       
            drain_req_q <= 1'b1;
    end

    always_ff @(posedge clk) begin
        if (!rst_n) 
            busy_q <= 1'b0;
        else        
            busy_q <= drain_busy;
    end

    // -------------------------------------------------------------------------
    // Submodules
    // -------------------------------------------------------------------------
    pe_wmem #(
        .NUM_MULTS(NUM_MULTS), .NUM_PID(NUM_PID), .DATA_WIDTH(DATA_WIDTH)
    ) u_wmem (
        .clk        (clk),
        .rst_n      (rst_n),
        .wfill_we   (wfill_we),
        .wfill_lane (wfill_lane),
        .wfill_pid  (wfill_pid),
        .wfill_val  (wfill_val),
        .wload_done (wload_done),
        .rd_en      (rd_en),
        .rd_addr    (rd_addr),
        .rd_val     (rd_val),
        .rd_pid     (rd_pid),
        .wfill_cnt  (wfill_cnt),
        .wsp_q      (wsp_q)
    );

    // The window owns the Curr/Next weight state AND the per-lane action-eval,
    // so it is self-contained about why it stalls; it exports only the MAC
    // controls (mac_en/mac_w) the datapath below needs.
    pe_window #(
        .NUM_MULTS(NUM_MULTS), 
        .NUM_PID(NUM_PID), 
        .DATA_WIDTH(DATA_WIDTH)
    ) u_window (
        .clk            (clk),
        .rst_n          (rst_n),
        .wload_done     (wload_done),
        .b_valid        (b_valid),
        .b_pid          (b_pid),
        .b_ready        (b_ready),
        .wsp_q          (wsp_q),
        .wfill_cnt      (wfill_cnt),
        .rd_val         (rd_val),
        .rd_pid         (rd_pid),
        .rd_en          (rd_en),
        .rd_addr        (rd_addr),
        .mac_en         (mac_en),
        .mac_w          (mac_w)
    );

    genvar k;
    generate
        for (k = 0; k < NUM_MULTS; k++) begin : g_lane
            pe_lane #(
                .NUM_CID(NUM_CID), 
                .DATA_WIDTH(DATA_WIDTH), 
                .ACC_WIDTH(ACC_WIDTH)
            ) u_lane (
                .clk            (clk),
                .rst_n          (rst_n),
                .b_act          (b_act),
                .b_cid          (b_cid),
                .mac_w          (mac_w[k]),
                .mac_en         (mac_en[k]),
                .consume        (consume),
                .drain_busy_any (drain_busy),
                .mac_busy       (mac_en_q[k]),
                .drain_pulse    (acc_drain_pulse),
                .drain_busy     (pe_drain_busy_w[k]),
                .out_ready      (out_ready[k]),
                .out_valid      (pe_out_valid_w[k]),
                .out_cid        (pe_out_cid_w[k]),
                .out_acc        (pe_out_acc_w[k])
            );
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Combinational assigns
    // -------------------------------------------------------------------------
    assign consume         = b_ready;
    assign acc_drain_pulse = (drain_start || drain_req_q) && (mac_en_q == '0) && !consume;

    assign out_valid  = pe_out_valid_w;
    assign out_cid    = pe_out_cid_w;
    assign out_acc    = pe_out_acc_w;
    assign drain_busy = |pe_drain_busy_w;
    assign drain_done = busy_q && !drain_busy;

endmodule

`default_nettype wire
