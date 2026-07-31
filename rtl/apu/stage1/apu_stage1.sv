// =============================================================================
// apu_stage1.sv -- APU Stage 1 (ID Generation), STAGE1_BATCH-wide
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// Accepts up to STAGE1_BATCH (val, x, y) tuples per cycle from the scanner and
// produces the FIFO-A bank. Each lane runs its own combinational front end:
//
//   lane i:  position_encode -> idgen (G x G units)
//                                  |
//                    up to G^2 (valid, a_xy, cid, pid)
//
// The activation SRAM is CSR: only non-zeros are ever stored, so the scanner
// never emits a zero and no zero filter is needed here.
//
// One activation's G^2 idgen lanes always carry DISTINCT PIDs, so a single
// activation contributes AT MOST one pair to any FIFO-A slot. Across the W lanes
// of a batch, up to W pairs can therefore target the same PID in one cycle, so
// each FIFO-A slot takes a contiguous multi-push of length 0..W (lane order =
// scan order, oldest at lane 0). FIFO-A read width stays NUM_MULTS (router pops
// M same-PID entries/cycle); the FIFO PORT_WIDTH is max(NUM_MULTS, W).
//
// All-or-nothing join: the whole batch is consumed only when EVERY targeted
// FIFO-A can accept. If any is full, in_ready drops and no slot is written, so
// no slot is written twice while waiting on another. idgen is PIPE=0
// (combinational) so the fan-out is valid the same cycle as the input.
// =============================================================================

`default_nettype none

module apu_stage1 #(
    parameter int H            = 32,   // activation map size (H x H, square)
    parameter int F            = 3,    // weight kernel size (F x F)
    parameter int S            = 1,    // convolution stride
    parameter int NUM_MULTS    = 4,    // FIFO-A read width (router pops/cycle)
    parameter int DATA_W       = 16,   // activation value width in bits
    parameter int FIFO_D       = 64,   // FIFO-A depth (power of 2)
    parameter int STAGE1_BATCH = 1,    // activations consumed per cycle (>=1)

    // -- Derived widths (mirror idgen.sv so all ports line up) ----------------
    localparam int E        = (H - F)/S + 1,                  // output map size
    localparam int G        = (F + S - 1)/S,                  // ceil(F/S)
    localparam int NUM_UNIT = G*G,                            // G^2 IDGen lanes
    localparam int N_PID    = F*F,                            // # FIFO-A slots
    localparam int IDX_W    = (H   < 2) ? 1 : $clog2(H),      // coord width
    localparam int CID_W    = (E*E < 2) ? 1 : $clog2(E*E),    // CID width
    localparam int PID_W    = (F*F < 2) ? 1 : $clog2(F*F),    // PID width
    localparam int FIFOA_W  = DATA_W + CID_W,                 // FIFO-A payload
    localparam int CNT_W    = $clog2(FIFO_D) + 1,             // FIFO occupancy width
    localparam int W        = STAGE1_BATCH,
    localparam int PW       = (NUM_MULTS > W) ? NUM_MULTS : W  // FIFO port width
)(
    input  wire  logic                              clk,
    input  wire  logic                              rst_n,

    // -- Batched (val, x, y) input from the scanner --------------------------
    input  wire  logic                              in_valid,       // batch present
    input  wire  logic [W-1:0]                      in_lane_valid,  // per-lane real entry
    input  wire  logic [W-1:0][DATA_W-1:0]          in_value,
    input  wire  logic [W-1:0][IDX_W-1:0]           in_x,
    input  wire  logic [W-1:0][IDX_W-1:0]           in_y,
    output logic                                    in_ready,       // whole batch taken

    // -- FIFO-A read ports (one per PID slot, NUM_MULTS-wide; to Stage 2) ------
    output logic [N_PID-1:0][NUM_MULTS-1:0]              fifoa_rd_valid,
    output logic [N_PID-1:0][NUM_MULTS-1:0][FIFOA_W-1:0] fifoa_rd_data,   // {a_xy, cid}
    input  wire  logic [N_PID-1:0][NUM_MULTS-1:0]        fifoa_rd_ready,
    output logic [N_PID-1:0][CNT_W-1:0]                  fifoa_count
);

    // -------------------------------------------------------------------------
    // Signal declarations
    // -------------------------------------------------------------------------
    // Per-lane idgen fan-out (W lanes x G^2 units).
    logic [W-1:0][NUM_UNIT-1:0]               ig_valid;
    logic [W-1:0][NUM_UNIT-1:0][DATA_W-1:0]   ig_axy;
    logic [W-1:0][NUM_UNIT-1:0][CID_W-1:0]    ig_cid;
    logic [W-1:0][NUM_UNIT-1:0][PID_W-1:0]    ig_pid;

    // Per-PID write prefix (contiguous from lane 0), FIFO room, join.
    logic [N_PID-1:0][PW-1:0]              wr_valid_pre;
    logic [N_PID-1:0][PW-1:0][FIFOA_W-1:0] wr_data_pre;
    logic [N_PID-1:0]                     fa_wr_ready;   // FIFO has room
    logic [N_PID-1:0]                     need;          // >=1 pair targets slot
    logic                                 all_ready;     // every needed slot ready
    logic                                 commit;        // batch written this cycle

    // -------------------------------------------------------------------------
    // W parallel front-end chains: position_encode -> idgen
    // -------------------------------------------------------------------------
    genvar i;
    /* verilator lint_off PINCONNECTEMPTY */
    generate
        for (i = 0; i < W; i++) begin : g_lane
            logic              pe_valid;
            logic [DATA_W-1:0] pe_value;
            logic [IDX_W-1:0]  pe_Px, pe_Py, pe_Cx, pe_Cy;

            position_encode #(.H(H), .S(S), .DATA_W(DATA_W)) u_position_encode (
                .in_valid (in_valid && in_lane_valid[i]),
                .in_value (in_value[i]),
                .x        (in_x[i]),
                .y        (in_y[i]),
                .in_ready (),
                .out_valid(pe_valid),
                .out_value(pe_value),
                .Px       (pe_Px),
                .Py       (pe_Py),
                .Cx       (pe_Cx),
                .Cy       (pe_Cy),
                .out_ready(1'b1)
            );

            idgen #(
                .H(H), .F(F), .S(S), .ACT_WIDTH(DATA_W), .PIPE(1'b0)
            ) u_idgen (
                .clk     (clk),
                .rst_n   (rst_n),
                .in_valid(pe_valid),
                .a_xy_in (pe_value),
                .Px      (pe_Px),
                .Py      (pe_Py),
                .Cx      (pe_Cx),
                .Cy      (pe_Cy),
                .valid   (ig_valid[i]),
                .a_xy_out(ig_axy[i]),
                .cid     (ig_cid[i]),
                .pid     (ig_pid[i])
            );
        end
    endgenerate
    /* verilator lint_on PINCONNECTEMPTY */

    // -------------------------------------------------------------------------
    // Route the W x G^2 lanes to the F^2 FIFO-A slots by PID. For each slot,
    // gather targeting pairs in scan order (activation lane 0 first) into a
    // contiguous prefix. <=1 pair per activation lane, so <=W pairs per slot.
    // -------------------------------------------------------------------------
    always_comb begin
        for (int p = 0; p < N_PID; p++) begin
            int unsigned cnt;
            cnt              = 0;
            wr_valid_pre[p]  = '0;
            wr_data_pre[p]   = '0;
            need[p]          = 1'b0;
            for (int j = 0; j < W; j++) begin
                for (int k = 0; k < NUM_UNIT; k++) begin
                    if (ig_valid[j][k] && (ig_pid[j][k] == PID_W'(p))) begin
                        wr_valid_pre[p][cnt] = 1'b1;
                        wr_data_pre[p][cnt]  = {ig_axy[j][k], ig_cid[j][k]};
                        cnt                  = cnt + 1;
                    end
                end
            end
            need[p] = (cnt != 0);
        end
    end

    // All-or-nothing join.
    always_comb begin
        all_ready = 1'b1;
        for (int p = 0; p < N_PID; p++)
            if (need[p] && !fa_wr_ready[p]) all_ready = 1'b0;
    end
    assign commit   = in_valid && all_ready;
    assign in_ready = all_ready;

    // -------------------------------------------------------------------------
    // FIFO-A bank : one FIFO per PID slot. Write width up to W (this batch),
    // read width NUM_MULTS (router). PORT_WIDTH = max(NUM_MULTS, W).
    // -------------------------------------------------------------------------
    genvar p;
    /* verilator lint_off PINCONNECTEMPTY */
    generate
        for (p = 0; p < N_PID; p++) begin : g_fifoa
            logic [PW-1:0]              wr_valid_vec;
            logic [PW-1:0][FIFOA_W-1:0] wr_data_vec;
            logic [PW-1:0]              rd_valid_vec;
            logic [PW-1:0][FIFOA_W-1:0] rd_data_vec;
            logic [PW-1:0]              rd_ready_vec;

            always_comb begin
                wr_valid_vec = commit ? wr_valid_pre[p] : '0;
                wr_data_vec  = wr_data_pre[p];
                // Router reads the low NUM_MULTS lanes; any surplus is idle.
                rd_ready_vec = '0;
                for (int m = 0; m < NUM_MULTS; m++) begin
                    fifoa_rd_valid[p][m] = rd_valid_vec[m];
                    fifoa_rd_data[p][m]  = rd_data_vec[m];
                    rd_ready_vec[m]      = fifoa_rd_ready[p][m];
                end
            end

            fifo #(
                .DATA_WIDTH(FIFOA_W), .DEPTH(FIFO_D), .PORT_WIDTH(PW)
            ) u_fifoa (
                .clk     (clk),
                .rst_n   (rst_n),
                .wr_valid(wr_valid_vec),
                .wr_data (wr_data_vec),
                .wr_ready(fa_wr_ready[p]),
                .rd_valid(rd_valid_vec),
                .rd_data (rd_data_vec),
                .rd_ready(rd_ready_vec),
                .full    (),
                .empty   (),
                .count   (fifoa_count[p])
            );
        end
    endgenerate
    /* verilator lint_on PINCONNECTEMPTY */

endmodule

`default_nettype wire
