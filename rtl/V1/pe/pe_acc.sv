module pe_acc #(
    parameter  int N_CID      = 4,     // # accumulator entries 
    parameter  int ACC_WIDTH  = 32,    // accumulator width 
    parameter  int PROD_WIDTH = 16,    // width of add_data

    localparam int CID_WIDTH  = (N_CID < 2) ? 1 : $clog2(N_CID)
)(
    input  logic                          clk,
    input  logic                          rst_n,

    // accumulate 
    input  logic                          clear,      // zero all accumulators
    input  logic                          add_en,
    input  logic [CID_WIDTH-1:0]          add_cid,
    input  logic signed [PROD_WIDTH-1:0]  add_data,

    // drain 
    input  logic                          drain_start,
    output logic                          drain_busy,
    output logic                          drain_done,
    output logic                          out_valid,
    input  logic                          out_ready,
    output logic [CID_WIDTH-1:0]          out_cid,
    output logic [ACC_WIDTH-1:0]          out_acc
);

    logic signed [ACC_WIDTH-1:0] acc [0:N_CID-1];

    logic                 draining;
    logic [CID_WIDTH-1:0] drain_idx;
    logic                 last_beat;

    assign last_beat  = draining && out_ready && (drain_idx == CID_WIDTH'(N_CID-1));
    assign drain_busy = draining;
    assign drain_done = last_beat; // pulses on the final drained beat
    assign out_valid  = draining;
    assign out_cid    = drain_idx;
    assign out_acc    = acc[drain_idx];

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            draining  <= 1'b0;
            drain_idx <= '0;
            for (int i = 0; i < N_CID; i++) acc[i] <= '0;
        end else if (!draining) begin
            // Accumulate phase (combinational demux+add into the banks).
            if (drain_start) begin
                draining  <= 1'b1;
                drain_idx <= '0;
            end else if (clear) begin
                for (int i = 0; i < N_CID; i++) acc[i] <= '0;
            end else if (add_en) begin
                acc[add_cid] <= acc[add_cid] + ACC_WIDTH'(add_data);
            end
        end else if (out_ready) begin
            // Drain phase: one entry per accepted beat.
            if (drain_idx == CID_WIDTH'(N_CID-1)) 
                draining  <= 1'b0;
            else
                drain_idx <= drain_idx + CID_WIDTH'(1);
        end
    end

endmodule
