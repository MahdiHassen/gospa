`default_nettype none

module fifo #(
    parameter int DATA_WIDTH = 22,   // payload width in bits
    parameter int DEPTH      = 64,   // number of entries; MUST be a power of 2
    parameter int PORT_WIDTH = 1     // entries pushed/popped per cycle; power of 2, must divide DEPTH
) (
    input  wire  logic                              clk,
    input  wire  logic                              rst_n,    // active-low synchronous reset

    // ---- Write / Producer port -----------------------------------------------
    input  wire  logic [PORT_WIDTH-1:0]             wr_valid,  // lane i valid; contiguous prefix from 0
    input  wire  logic [PORT_WIDTH-1:0][DATA_WIDTH-1:0] wr_data,
    output logic                                    wr_ready,  // FIFO can accept a full PORT_WIDTH-wide push

    // ---- Read / Consumer port ------------------------------------------------
    output logic [PORT_WIDTH-1:0]                   rd_valid,  // thermometer-coded from lane 0
    output logic [PORT_WIDTH-1:0][DATA_WIDTH-1:0]   rd_data,
    input  wire  logic [PORT_WIDTH-1:0]             rd_ready,  // contiguous pop request from lane 0

    // ---- Status --------------------------------------------------------------
    output logic                                    full,
    output logic                                    empty,
    output logic [$clog2(DEPTH):0]                  count      // occupancy: 0 .. DEPTH
);

    localparam int PTR_WIDTH         = $clog2(DEPTH);          // global entry-pointer width
    localparam int COUNT_WIDTH       = PTR_WIDTH + 1;           // occupancy width (represent 0..DEPTH)
    localparam int BANK_DEPTH        = DEPTH / PORT_WIDTH;      // entries per physical bank
    localparam int BANK_ADDR_WIDTH   = $clog2(BANK_DEPTH);      // per-bank RAM address width
    localparam int PORT_COUNT_WIDTH  = $clog2(PORT_WIDTH + 1);  // width to represent 0..PORT_WIDTH
    localparam int LANE_INDEX_WIDTH  = (PORT_WIDTH > 1) ? $clog2(PORT_WIDTH) : 1; // width to index a PORT_WIDTH-wide vector

    // Output show-ahead buffer
    localparam int OBUF_DEPTH        = 4 * PORT_WIDTH;
    localparam int OBUF_PTR_WIDTH    = $clog2(OBUF_DEPTH);
    localparam int OBUF_COUNT_WIDTH  = OBUF_PTR_WIDTH + 1;


    logic [DATA_WIDTH-1:0] mem   [0:PORT_WIDTH-1][0:BANK_DEPTH-1];
    logic [DATA_WIDTH-1:0] mem_q [0:PORT_WIDTH-1];             // per-bank RAM output register

    logic [PTR_WIDTH-1:0]      wr_ptr;         // RAM write-issue pointer (global entry index)
    logic [PTR_WIDTH-1:0]      rd_ptr;         // RAM read-issue pointer (global entry index)
    logic [COUNT_WIDTH-1:0]    cnt;            // TOTAL occupancy (RAM + in-flight + obuf)
    logic [COUNT_WIDTH-1:0]    ram_cnt;        // words resident in RAM, not yet read-issued

    logic [PTR_WIDTH-1:0]       write_base;    // wr_ptr's phase within a PORT_WIDTH bank group
    logic [PTR_WIDTH-1:0]       read_base;     // rd_ptr's phase within a PORT_WIDTH bank group
    logic [PTR_WIDTH-1:0]       read_base_q;   // read_base at the time the in-flight read was issued
    logic [PORT_COUNT_WIDTH-1:0] mem_rd_cnt;   // banks read-issued this cycle
    logic [PORT_COUNT_WIDTH-1:0] rd_cnt_q;     // banks read-issued last cycle (landing in mem_q now)

    // Show-ahead output buffer: small register FIFO fed by the banked reads.
    logic [DATA_WIDTH-1:0] obuf [0:OBUF_DEPTH-1];
    logic [OBUF_PTR_WIDTH-1:0] ob_head, ob_tail;
    logic [OBUF_COUNT_WIDTH-1:0] ob_cnt;

    // Handshake qualifiers
    logic                          do_push;      // this cycle's push (if any) is accepted in full
    logic [PORT_COUNT_WIDTH-1:0]   wr_push_cnt;  // lanes requested by the producer this cycle
    logic [PORT_COUNT_WIDTH-1:0]   rd_pop_cnt;   // lanes requested by the consumer this cycle
    logic [COUNT_WIDTH-1:0]        room;         // free slots (DEPTH - cnt)

    // -------------------------------------------------------------------------
    // Lane popcount: counts a contiguous prefix, used for both wr_valid and
    // rd_ready (occupancy math needs "how many", not which individual bits).
    // -------------------------------------------------------------------------
    function automatic logic [PORT_COUNT_WIDTH-1:0] count_ones(input logic [PORT_WIDTH-1:0] bits);
        logic [PORT_COUNT_WIDTH-1:0] total;
        total = '0;
        for (int i = 0; i < PORT_WIDTH; i++) begin
            if (bits[LANE_INDEX_WIDTH'(i)]) begin
                total = total + PORT_COUNT_WIDTH'(1);
            end
        end
        return total;
    endfunction

    // -------------------------------------------------------------------------
    // Write side: one bank instance per lane slot. lane_idx rotates with
    // write_base so that PORT_WIDTH consecutive global indices map bijectively
    // onto the PORT_WIDTH physical banks.
    // -------------------------------------------------------------------------
    genvar wbank;
    generate
        for (wbank = 0; wbank < PORT_WIDTH; wbank++) begin : g_write_bank
            logic [PTR_WIDTH-1:0]       lane_idx;
            logic                       write_enable;
            logic [BANK_ADDR_WIDTH-1:0] write_addr;

            always_comb begin
                lane_idx     = (PTR_WIDTH'(wbank) + PTR_WIDTH'(PORT_WIDTH) - write_base) % PTR_WIDTH'(PORT_WIDTH);
                write_enable = do_push && wr_valid[LANE_INDEX_WIDTH'(lane_idx)];
                write_addr   = BANK_ADDR_WIDTH'((wr_ptr + lane_idx) / PTR_WIDTH'(PORT_WIDTH));
            end

            always_ff @(posedge clk) begin
                if (write_enable) begin
                    mem[wbank][write_addr] <= wr_data[LANE_INDEX_WIDTH'(lane_idx)];
                end
            end
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Read side: one bank instance per lane slot, gated by mem_rd_cnt (how
    // many banks may issue a read this cycle).
    // -------------------------------------------------------------------------
    genvar rbank;
    generate
        for (rbank = 0; rbank < PORT_WIDTH; rbank++) begin : g_read_bank
            logic [PTR_WIDTH-1:0]       lane_idx;
            logic                       read_enable;
            logic [BANK_ADDR_WIDTH-1:0] read_addr;

            always_comb begin
                lane_idx    = (PTR_WIDTH'(rbank) + PTR_WIDTH'(PORT_WIDTH) - read_base) % PTR_WIDTH'(PORT_WIDTH);
                read_enable = (int'(lane_idx) < int'(mem_rd_cnt));
                read_addr   = BANK_ADDR_WIDTH'((rd_ptr + lane_idx) / PTR_WIDTH'(PORT_WIDTH));
            end

            always_ff @(posedge clk) begin
                if (read_enable) begin
                    mem_q[rbank] <= mem[rbank][read_addr];
                end
            end
        end
    endgenerate

    // -------------------------------------------------------------------------
    // How many bank reads to issue this cycle: bounded by data available in
    // RAM, the port width, and room left in the read pipeline + output buffer.
    // -------------------------------------------------------------------------
    always_comb begin : compute_mem_rd_cnt
        int unsigned reserved;
        int unsigned slack;
        int unsigned candidate;

        reserved  = int'(ob_cnt) + int'(rd_cnt_q);
        slack     = (OBUF_DEPTH > reserved) ? (OBUF_DEPTH - reserved) : 0;
        candidate = int'(ram_cnt);
        if (candidate > PORT_WIDTH) begin
            candidate = PORT_WIDTH;
        end
        if (candidate > slack) begin
            candidate = slack;
        end
        mem_rd_cnt = PORT_COUNT_WIDTH'(candidate);
    end

    // -------------------------------------------------------------------------
    // Pointers, counters, and the show-ahead output buffer
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            wr_ptr      <= '0;
            rd_ptr      <= '0;
            cnt         <= '0;
            ram_cnt     <= '0;
            rd_cnt_q    <= '0;
            read_base_q <= '0;
            ob_head     <= '0;
            ob_tail     <= '0;
            ob_cnt      <= '0;
        end else begin
            wr_ptr <= wr_ptr + (do_push ? PTR_WIDTH'(wr_push_cnt) : PTR_WIDTH'(0));
            rd_ptr <= rd_ptr + PTR_WIDTH'(mem_rd_cnt);

            cnt     <= cnt + (do_push ? COUNT_WIDTH'(wr_push_cnt) : COUNT_WIDTH'(0)) - COUNT_WIDTH'(rd_pop_cnt);
            ram_cnt <= ram_cnt + (do_push ? COUNT_WIDTH'(wr_push_cnt) : COUNT_WIDTH'(0)) - COUNT_WIDTH'(mem_rd_cnt);

            // A bank read issued this cycle lands in mem_q next cycle; remember
            // how many and which rotation phase so it can be un-rotated below.
            rd_cnt_q    <= mem_rd_cnt;
            read_base_q <= read_base;

            // Output buffer: append the just-landed bank words (un-rotated back
            // to linear order) at the tail, advance the head by however many the
            // consumer accepted this cycle. Both may occur together.
            for (int j = 0; j < PORT_WIDTH; j++) begin
                if (j < int'(rd_cnt_q)) begin
                    obuf[ob_tail + OBUF_PTR_WIDTH'(j)] <= mem_q[(int'(read_base_q) + j) % PORT_WIDTH];
                end
            end
            ob_tail <= ob_tail + OBUF_PTR_WIDTH'(rd_cnt_q);
            ob_head <= ob_head + OBUF_PTR_WIDTH'(rd_pop_cnt);
            ob_cnt  <= ob_cnt + OBUF_COUNT_WIDTH'(rd_cnt_q) - OBUF_COUNT_WIDTH'(rd_pop_cnt);
        end
    end

    // -------------------------------------------------------------------------
    // Consumer-facing show-ahead window: lane j exposes obuf[head+j], valid
    // while j is within the current output-buffer occupancy.
    // -------------------------------------------------------------------------
    always_comb begin : present_read_window
        for (int j = 0; j < PORT_WIDTH; j++) begin
            rd_valid[LANE_INDEX_WIDTH'(j)] = (j < int'(ob_cnt));
            rd_data[LANE_INDEX_WIDTH'(j)]  = obuf[ob_head + OBUF_PTR_WIDTH'(j)];
        end
    end

    // -------------------------------------------------------------------------
    // Continuous assignments
    // -------------------------------------------------------------------------
    assign wr_push_cnt = count_ones(wr_valid);
    assign rd_pop_cnt  = count_ones(rd_ready & rd_valid);

    assign write_base = wr_ptr % PTR_WIDTH'(PORT_WIDTH);
    assign read_base  = rd_ptr % PTR_WIDTH'(PORT_WIDTH);

    assign room     = COUNT_WIDTH'(DEPTH) - cnt;
    assign wr_ready = (room >= COUNT_WIDTH'(PORT_WIDTH));
    assign do_push  = wr_ready;

    assign full  = (cnt == COUNT_WIDTH'(DEPTH));
    assign empty = (cnt == '0);
    assign count = cnt;

`ifndef SYNTHESIS
    // -------------------------------------------------------------------------
    // Simulation sanity checks (stripped for synthesis)
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst_n) begin
            for (int i = 1; i < PORT_WIDTH; i++) begin
                assert (!(wr_valid[LANE_INDEX_WIDTH'(i)] && !wr_valid[LANE_INDEX_WIDTH'(i-1)]))
                    else $error("fifo: wr_valid must be a contiguous prefix from lane 0");
                assert (!(rd_ready[LANE_INDEX_WIDTH'(i)] && !rd_ready[LANE_INDEX_WIDTH'(i-1)]))
                    else $error("fifo: rd_ready must be a contiguous prefix from lane 0");
            end
            assert (ob_cnt <= OBUF_COUNT_WIDTH'(OBUF_DEPTH))
                else $error("fifo: output buffer overflow ob_cnt=%0d", ob_cnt);
            assert (cnt <= COUNT_WIDTH'(DEPTH))
                else $error("fifo: occupancy overflow cnt=%0d", cnt);
        end
    end
`endif

endmodule

`default_nettype wire
