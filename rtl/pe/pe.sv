// =============================================================================
// pe.sv -- GoSPA Processing Element (V2 "act" dataflow: one kernel per PE)
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// One kernel per PE, weight-stationary. A single Curr weight, selected by the
// beat's PID, feeds all M multipliers; the M lanes consume M FIFO-B entries that
// share that PID but carry distinct CIDs (per-lane valid masks short beats). One
// WSP per PE is exported to the routing module (Fig 9 switch control).
//   - pe_mem   : single PID-sorted weight bank + WSP bookkeeping.
//   - pe_fetch : Curr/Next selector -> one weight (mac_w) + mac_en per beat.
//   - pe_lane  : M of them; each multiplies its activation by mac_w into its own
//                CID-indexed bank. Per-lane banks give M conflict-free writes.
// The drain streams one CID per beat, summing the M banks (out_acc), and is held
// off until every lane's MAC pipeline has flushed.
//
// Load: stream {wfill_we, wfill_pid, wfill_val} once per weight in PID order;
// pulse wload_done to arm (a bare wload_done re-arms the same weights).
// =============================================================================

`default_nettype none

module pe #(
    parameter int NUM_MULTS  = 4,    // multipliers = activations consumed per beat
    parameter int NUM_PID    = 9,    // # kernel positions = F*F (max weights)
    parameter int NUM_CID    = 36,   // # output positions = E*E (accumulator depth)
    parameter int DATA_WIDTH = 16,
    parameter int ACC_WIDTH  = 32,

    // -- Derived --------------------------------------------------------------
    localparam int PID_WIDTH   = (NUM_PID < 2) ? 1 : $clog2(NUM_PID),
    localparam int CID_WIDTH   = (NUM_CID < 2) ? 1 : $clog2(NUM_CID),
    localparam int SLOT_WIDTH  = (NUM_PID < 2) ? 1 : $clog2(NUM_PID)
)(
    input  logic                                 clk,
    input  logic                                 rst_n,

    // -- Weight SRAM fill -- one weight per cycle, PID order ------------------
    input  logic                                 wfill_we,
    input  logic [PID_WIDTH-1:0]                 wfill_pid,
    input  logic signed [DATA_WIDTH-1:0]         wfill_val,
    input  logic                                 wload_done,

    // -- WSP export to the routing module ------------------------------------
    output logic [NUM_PID-1:0]                   wsp,
    output logic                                 wsp_valid,

    // -- FIFO-B input beat: M activations sharing b_pid, distinct CIDs --------
    input  logic                                 b_valid,
    input  logic [PID_WIDTH-1:0]                 b_pid,
    input  logic [NUM_MULTS-1:0]                 b_lane_valid,   // per-lane real activation
    input  logic [NUM_MULTS-1:0][DATA_WIDTH-1:0] b_act,
    input  logic [NUM_MULTS-1:0][CID_WIDTH-1:0]  b_cid,
    output logic                                 b_ready,

    // -- Drain / output (single CID stream for this PE's output channel) ------
    input  logic                                 drain_start,
    output logic                                 drain_busy,
    output logic                                 drain_done,
    output logic                                 out_valid,
    output logic [CID_WIDTH-1:0]                 out_cid,
    output logic [ACC_WIDTH-1:0]                 out_acc,
    input  logic                                 out_ready
);

    // -------------------------------------------------------------------------
    // Signal declarations
    // -------------------------------------------------------------------------
    // Weight-memory / fetch interface: one shared read port and selected weight.
    logic                    rd_en;
    logic [SLOT_WIDTH-1:0]   rd_slot;
    logic [DATA_WIDTH-1:0]   rd_val;
    logic [PID_WIDTH-1:0]    rd_pid;

    logic                    mac_en;    // beat produces MACs (KEEP/BANK)
    logic [DATA_WIDTH-1:0]   mac_w;     // the one weight all lanes use
    logic                    consume;   // alias for b_ready

    // Per-lane MAC status and drain read-back. mac_fire is an internal
    // observability signal (one pulse per accumulate) for the perf monitor --
    // not a port, so it does not touch the array/synthesis interface.
    logic [NUM_MULTS-1:0]                   mac_busy;
    logic [NUM_MULTS-1:0]                   mac_fire;
    logic [NUM_MULTS-1:0][ACC_WIDTH-1:0]    lane_drain_val;

    // Drain FSM (single stream out, summing the M banks per CID).
    logic                    armed;         // WSP loaded / running
    logic                    draining;
    logic [CID_WIDTH-1:0]    drain_idx;
    logic                    drain_req_q;   // latched drain_start
    logic                    start_drain;
    logic                    busy_q;        // drain_busy delayed (done edge)
    logic signed [ACC_WIDTH-1:0] acc_sum;

    // -------------------------------------------------------------------------
    // Submodules
    // -------------------------------------------------------------------------
    pe_mem #(
        .NUM_PID(NUM_PID), .DATA_WIDTH(DATA_WIDTH)
    ) u_mem (
        .clk        (clk),
        .rst_n      (rst_n),
        .wfill_we   (wfill_we),
        .wfill_pid  (wfill_pid),
        .wfill_val  (wfill_val),
        .wload_done (wload_done),
        .rd_en      (rd_en),
        .rd_slot    (rd_slot),
        .rd_val     (rd_val),
        .rd_pid     (rd_pid),
        .wsp_q      (wsp)
    );

    pe_fetch #(
        .NUM_PID(NUM_PID), .DATA_WIDTH(DATA_WIDTH)
    ) u_fetch (
        .clk        (clk),
        .rst_n      (rst_n),
        .wload_done (wload_done),
        .b_valid    (b_valid),
        .b_pid      (b_pid),
        .b_ready    (b_ready),
        .wsp_q      (wsp),
        .rd_val     (rd_val),
        .rd_pid     (rd_pid),
        .rd_en      (rd_en),
        .rd_slot    (rd_slot),
        .mac_en     (mac_en),
        .mac_w      (mac_w)
    );

    genvar k;
    generate
        for (k = 0; k < NUM_MULTS; k++) begin : g_lane
            logic lane_go;
            // Lane fires when the beat is accepted, produces MACs, this lane
            // carries a real activation, and no drain is in progress.
            assign lane_go = consume && mac_en && b_lane_valid[k] && !draining;

            pe_lane #(
                .NUM_CID(NUM_CID), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)
            ) u_lane (
                .clk       (clk),
                .rst_n     (rst_n),
                .b_act     (b_act[k]),
                .b_cid     (b_cid[k]),
                .mac_w     (mac_w),
                .mac_go    (lane_go),
                .mac_busy  (mac_busy[k]),
                .mac_fire  (mac_fire[k]),
                .drain_idx (drain_idx),
                .drain_val (lane_drain_val[k])
            );
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Drain gating: start only once every lane's in-flight product has been
    // accumulated (mac_busy all 0) and no new product is being formed this cycle
    // (!consume). drain_start is latched so a 1-cycle pulse is not lost.
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n)
            drain_req_q <= 1'b0;
        else if (start_drain)
            drain_req_q <= 1'b0;
        else if (drain_start)
            drain_req_q <= 1'b1;
    end

    // -------------------------------------------------------------------------
    // Drain FSM: stream one CID per accepted beat, summing the M banks.
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            draining  <= 1'b0;
            drain_idx <= '0;
        end else if (!draining) begin
            if (start_drain) begin
                draining  <= 1'b1;
                drain_idx <= '0;
            end
        end else if (out_ready) begin
            if (drain_idx == CID_WIDTH'(NUM_CID-1))
                draining  <= 1'b0;
            else
                drain_idx <= drain_idx + CID_WIDTH'(1);
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n)
            armed <= 1'b0;
        else if (wload_done)
            armed <= 1'b1;
    end

    always_ff @(posedge clk) begin
        if (!rst_n)
            busy_q <= 1'b0;
        else
            busy_q <= drain_busy;
    end

    // -------------------------------------------------------------------------
    // Combinational: sum the M per-lane banks at the drained CID.
    // -------------------------------------------------------------------------
    always_comb begin
        acc_sum = '0;
        for (int i = 0; i < NUM_MULTS; i++)
            acc_sum = acc_sum + $signed(lane_drain_val[i]);
    end

    // -------------------------------------------------------------------------
    // Combinational assigns
    // -------------------------------------------------------------------------
    assign consume     = b_ready;
    assign start_drain = (drain_start || drain_req_q) && !draining
                         && (mac_busy == '0) && !consume;

    assign wsp_valid   = armed;
    assign drain_busy  = draining;
    assign drain_done  = busy_q && !drain_busy;
    assign out_valid   = draining;
    assign out_cid     = drain_idx;
    assign out_acc     = acc_sum;

endmodule

`default_nettype wire
