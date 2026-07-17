// =============================================================================
// pe.sv -- GoSPA Processing Element (V2: multiple kernels per PE)
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// Top-level glue over three submodules, plus the array-level drain gating
// (holds the drain off until every lane's MAC pipeline has flushed):
//   - pe_mem   : per-lane weight SRAM bank + fill/WSP bookkeeping. Lane k's
//                kernel lives in bank k, PID-sorted. Exports wsp_q.
//   - pe_fetch : per lane, one of IDLE/KEEP/BANK/SKIP per beat -> mac_en/mac_w.
//                A SKIP jumps straight to the needed slot (popcount of wsp below
//                b_pid) and stalls one beat; b_ready drops only for SKIP lanes.
//   - pe_lane  : one per output channel -- pipelined multiply + CID-indexed
//                accumulator + drain.
//
// Load sequence (host -> PE): stream {wfill_we, wfill_lane=k, wfill_pid,
// wfill_val} once per weight per lane, in PID order -- slot, count, and WSP are
// all DERIVED from this stream. Pulse wload_done to arm; a bare wload_done
// re-arms the same weights, and the first wfill after an arm starts a fresh
// session (prior count/WSP cleared).
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
    localparam int SLOT_WIDTH  = (NUM_PID < 2) ? 1 : $clog2(NUM_PID)
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
    // Weight-memory / fetch interface: pe_mem hands back per-lane SRAM read data
    // and the WSP (wsp_q); pe_fetch drives each lane's bank read port.
    logic [NUM_MULTS-1:0]                   rd_en;
    logic [NUM_MULTS-1:0][SLOT_WIDTH-1:0]   rd_slot;
    logic [NUM_MULTS-1:0][DATA_WIDTH-1:0]   rd_val;
    logic [NUM_MULTS-1:0][PID_WIDTH-1:0]    rd_pid;
    logic [NUM_MULTS-1:0][NUM_PID-1:0]      wsp_q;

    // Per-lane MAC controls from pe_fetch's action-eval (KEEP/BANK -> MAC).
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
    logic [NUM_MULTS-1:0]                   mac_en_q;   // per-lane "MAC in flight" (drain gate)
    logic [NUM_MULTS-1:0]                   mac_fire;   // per-lane 1-cycle accumulate pulse (perf tally)

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
    pe_mem #(
        .NUM_MULTS(NUM_MULTS), .NUM_PID(NUM_PID), .DATA_WIDTH(DATA_WIDTH)
    ) u_mem (
        .clk        (clk),
        .rst_n      (rst_n),
        .wfill_we   (wfill_we),
        .wfill_lane (wfill_lane),
        .wfill_pid  (wfill_pid),
        .wfill_val  (wfill_val),
        .wload_done (wload_done),
        .rd_en      (rd_en),
        .rd_slot    (rd_slot),
        .rd_val     (rd_val),
        .rd_pid     (rd_pid),
        .wsp_q      (wsp_q)
    );

    // pe_fetch owns the per-lane held weight AND the action-eval, so it is
    // self-contained about why it stalls; it exports only the MAC controls
    // (mac_en/mac_w) the datapath below needs.
    pe_fetch #(
        .NUM_MULTS(NUM_MULTS),
        .NUM_PID(NUM_PID),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_fetch (
        .clk            (clk),
        .rst_n          (rst_n),
        .wload_done     (wload_done),
        .b_valid        (b_valid),
        .b_pid          (b_pid),
        .b_ready        (b_ready),
        .wsp_q          (wsp_q),
        .rd_val         (rd_val),
        .rd_pid         (rd_pid),
        .rd_en          (rd_en),
        .rd_slot        (rd_slot),
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
                .mac_fire       (mac_fire[k]),
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
