// =============================================================================
// fifo_tb.sv -- Testbench for fifo.sv
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// Tests use FIFO-A geometry (22-bit payload, depth 64).
// The same fifo.sv handles FIFO-B by changing DATA_WIDTH -- see bottom comment.
//
// Test list:
//   1.  Initial state after reset
//   2.  Single push -- flags and count
//   3.  Single pop  -- data integrity and flags
//   4.  Fill to full
//   5.  Overflow protection  (push when full, count must not change)
//   6.  Drain to empty, verify FIFO order
//   7.  Underflow protection (pop when empty)
//   8.  Simultaneous push + pop (count unchanged)
//   9.  GoSPA FIFO-A scenario (toy example from paper: F=2, H=3, S=1)
//   10. Randomized stress -- 50 push then 50 pop, verify order
//
// Run with Verilator 5.x : make
// Run with Icarus Verilog : make SIM=iverilog
// =============================================================================

`timescale 1ns / 1ps

module fifo_tb;

    // =========================================================================
    // GoSPA-specific data layout constants
    // =========================================================================
    //   FIFO-A payload: { act_value[ACT_W-1:0], cid[CID_W-1:0] }
    //   FIFO-B payload: { act_value[ACT_W-1:0], pid[PID_W-1:0], cid[CID_W-1:0] }
    localparam int ACT_W   = 16;                    // activation value width (paper: 16-bit)
    localparam int CID_W   = 6;                     // CID width (paper: 6 bits)
    localparam int PID_W   = 4;                     // PID width ($clog2(F^2) for F<=4)
    localparam int FIFO_D  = 64;                    // FIFO depth (paper: 64 entries)

    localparam int FIFO_A_W = ACT_W + CID_W;        // 22 bits -- used in this TB
    localparam int FIFO_B_W = ACT_W + PID_W + CID_W;// 26 bits -- shown for reference

    // =========================================================================
    // DUT signals (FIFO-A geometry)
    // =========================================================================
    logic                        clk;
    logic                        rst_n;
    logic                        wr_valid;
    logic [FIFO_A_W-1:0]         wr_data;
    logic                        wr_ready;
    logic                        rd_valid;
    logic [FIFO_A_W-1:0]         rd_data;
    logic                        rd_ready;
    logic                        full;
    logic                        empty;
    logic [$clog2(FIFO_D):0]     count;

    // =========================================================================
    // DUT instantiation
    // =========================================================================
    fifo #(
        .DATA_WIDTH (FIFO_A_W),
        .DEPTH      (FIFO_D)
    ) dut (
        .clk      (clk),
        .rst_n    (rst_n),
        .wr_valid (wr_valid),
        .wr_data  (wr_data),
        .wr_ready (wr_ready),
        .rd_valid (rd_valid),
        .rd_data  (rd_data),
        .rd_ready (rd_ready),
        .full     (full),
        .empty    (empty),
        .count    (count)
    );

    // =========================================================================
    // Clock -- 500 MHz (2 ns period), matching GoSPA target
    // =========================================================================
    initial clk = 1'b0;
    always #1 clk = ~clk;

    // =========================================================================
    // Scoreboard
    // =========================================================================
    int pass_count;
    int fail_count;

    // =========================================================================
    // Utility tasks
    // =========================================================================

    // 1-bit signal check
    task automatic chk(
        input string label,
        input logic  got,
        input logic  expected
    );
        if (got === expected) begin
            $display("  [PASS] %-52s got=%b", label, got);
            pass_count++;
        end else begin
            $display("  [FAIL] %-52s expected=%b  got=%b", label, expected, got);
            fail_count++;
        end
    endtask

    // Data word check
    task automatic chk_data(
        input string               label,
        input logic [FIFO_A_W-1:0] got,
        input logic [FIFO_A_W-1:0] expected
    );
        if (got === expected) begin
            $display("  [PASS] %-52s got=0x%06h", label, got);
            pass_count++;
        end else begin
            $display("  [FAIL] %-52s expected=0x%06h  got=0x%06h",
                     label, expected, got);
            fail_count++;
        end
    endtask

    // Occupancy counter check
    // Both args are 32-bit logic so count (narrow) and FIFO_D (int) both fit cleanly.
    task automatic chk_count(
        input string       label,
        input logic [31:0] got,
        input logic [31:0] expected
    );
        if (got === expected) begin
            $display("  [PASS] %-52s count=%0d", label, got);
            pass_count++;
        end else begin
            $display("  [FAIL] %-52s expected=%0d  got=%0d", label, expected, got);
            fail_count++;
        end
    endtask

    // Wait N clock cycles
    task automatic wait_cycles(input int n);
        repeat (n) @(posedge clk);
    endtask

    // Reset sequence
    task automatic do_reset();
        rst_n    = 1'b0;
        wr_valid = 1'b0;
        rd_ready = 1'b0;
        wr_data  = '0;
        repeat (4) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        @(posedge clk);
    endtask

    // Push one entry.
    // Drives on negedge, samples handshake on posedge -- avoids setup races.
    // Blocks until wr_ready is asserted (back-pressure safe).
    task automatic push(input logic [FIFO_A_W-1:0] data);
        @(negedge clk);
        wr_valid = 1'b1;
        wr_data  = data;
        @(posedge clk);
        while (!wr_ready) @(posedge clk);
        @(negedge clk);
        wr_valid = 1'b0;
        wr_data  = '0;
    endtask

    // Pop one entry. Blocks until rd_valid is asserted.
    task automatic pop(output logic [FIFO_A_W-1:0] data);
        @(negedge clk);
        rd_ready = 1'b1;
        @(posedge clk);
        while (!rd_valid) @(posedge clk);
        data = rd_data;
        @(negedge clk);
        rd_ready = 1'b0;
    endtask

    // =========================================================================
    // Waveform dump
    // =========================================================================
    initial begin
        $dumpfile("fifo_tb.vcd");
        $dumpvars(0, fifo_tb);
    end

    // =========================================================================
    // Main test body
    // =========================================================================
    initial begin : TEST_MAIN

        // Declare all variables at top for maximum tool compatibility
        logic [FIFO_A_W-1:0] popped;
        logic [FIFO_A_W-1:0] rand_entry;
        logic [FIFO_A_W-1:0] ref_queue [$];
        logic [FIFO_A_W-1:0] expected_val;
        int                  rand_errors;

        pass_count = 0;
        fail_count = 0;

        $display("");
        $display("=============================================================");
        $display("  GoSPA FIFO Testbench");
        $display("  DATA_WIDTH=%0d  DEPTH=%0d  Clock=500MHz", FIFO_A_W, FIFO_D);
        $display("  Payload format: { act_value[15:0], cid[5:0] }  (FIFO-A)");
        $display("=============================================================");

        do_reset();

        // ------------------------------------------------------------------
        // TEST 1 -- Initial state after reset
        // ------------------------------------------------------------------
        $display("\n--- TEST 1: Initial state after reset ---");
        chk      ("empty asserted",      empty,    1'b1);
        chk      ("full  deasserted",    full,     1'b0);
        chk      ("rd_valid deasserted", rd_valid, 1'b0);
        chk      ("wr_ready asserted",   wr_ready, 1'b1);
        chk_count("count == 0",          count,    '0);

        // ------------------------------------------------------------------
        // TEST 2 -- Single push
        // ------------------------------------------------------------------
        $display("\n--- TEST 2: Single push ---");
        push(FIFO_A_W'({16'hBEEF, 6'd5}));   // act=0xBEEF, CID=5
        wait_cycles(1);
        chk      ("empty cleared after push", empty, 1'b0);
        chk      ("full still clear",         full,  1'b0);
        chk_count("count == 1",               count, 1);

        // ------------------------------------------------------------------
        // TEST 3 -- Single pop, verify data and empty flag
        // ------------------------------------------------------------------
        $display("\n--- TEST 3: Single pop, verify data ---");
        pop(popped);
        wait_cycles(1);
        chk_data ("popped data",         popped, FIFO_A_W'({16'hBEEF, 6'd5}));
        chk      ("empty re-asserted",   empty,  1'b1);
        chk_count("count == 0",          count,  '0);

        // ------------------------------------------------------------------
        // TEST 4 -- Fill to full
        // ------------------------------------------------------------------
        $display("\n--- TEST 4: Fill to full (DEPTH=%0d) ---", FIFO_D);
        for (int i = 0; i < FIFO_D; i++)
            push(FIFO_A_W'({16'(i), 6'(i % 64)}));
        wait_cycles(1);
        chk      ("full asserted at DEPTH",  full,     1'b1);
        chk      ("wr_ready deasserted",     wr_ready, 1'b0);
        chk_count("count == DEPTH",          count,    FIFO_D);

        // ------------------------------------------------------------------
        // TEST 5 -- Overflow protection: push while full
        // ------------------------------------------------------------------
        $display("\n--- TEST 5: Overflow protection ---");
        @(negedge clk);
        wr_valid = 1'b1;
        wr_data  = {FIFO_A_W{1'b1}};   // garbage all-ones entry
        wait_cycles(3);
        chk      ("wr_ready stays 0 when full", wr_ready, 1'b0);
        chk_count("count unchanged",            count,    FIFO_D);
        @(negedge clk);
        wr_valid = 1'b0;

        // ------------------------------------------------------------------
        // TEST 6 -- Drain to empty, verify FIFO order
        // ------------------------------------------------------------------
        $display("\n--- TEST 6: Drain and verify FIFO order ---");
        for (int i = 0; i < FIFO_D; i++) begin
            logic [FIFO_A_W-1:0] exp;
            exp = FIFO_A_W'({16'(i), 6'(i % 64)});
            pop(popped);
            chk_data($sformatf("entry[%0d]", i), popped, exp);
        end
        wait_cycles(1);
        chk("empty after full drain", empty, 1'b1);

        // ------------------------------------------------------------------
        // TEST 7 -- Underflow protection: pop while empty
        // ------------------------------------------------------------------
        $display("\n--- TEST 7: Underflow protection ---");
        @(negedge clk);
        rd_ready = 1'b1;
        wait_cycles(3);
        chk      ("rd_valid stays 0 when empty", rd_valid, 1'b0);
        chk_count("count stays 0",               count,    '0);
        @(negedge clk);
        rd_ready = 1'b0;

        // ------------------------------------------------------------------
        // TEST 8 -- Simultaneous push + pop (count must be unchanged)
        // ------------------------------------------------------------------
        $display("\n--- TEST 8: Simultaneous push + pop ---");
        for (int i = 0; i < FIFO_D/2; i++)
            push(FIFO_A_W'({16'(i+1), 6'(i % 64)}));
        wait_cycles(1);
        chk_count("pre-fill: count == DEPTH/2", count, FIFO_D/2);

        // Assert both wr_valid and rd_ready on the same clock edge
        @(negedge clk);
        wr_valid = 1'b1;
        wr_data  = FIFO_A_W'({16'hCAFE, 6'd63});
        rd_ready = 1'b1;
        @(posedge clk);   // push and pop fire simultaneously this cycle
        @(negedge clk);
        wr_valid = 1'b0;
        rd_ready = 1'b0;
        wait_cycles(1);
        chk_count("count unchanged after simultaneous push+pop", count, FIFO_D/2);

        // Drain the remaining FIFO_D/2 entries
        for (int i = 0; i < FIFO_D/2; i++)
            pop(popped);
        wait_cycles(1);
        chk("empty after drain", empty, 1'b1);

        // ------------------------------------------------------------------
        // TEST 9 -- GoSPA FIFO-A scenario (paper toy example)
        //
        //   F=2, H=3, S=1
        //   Weight: [[0, 3], [-3, 0]]  --> WSP = [0,1,1,0]
        //   Input:  [[-1,3,1],[0,2,0],[0,0,-2]]
        //
        //   APU Stage-1 feeds FIFO-A[PID=1] with (act, CID) tuples:
        //     activation 3  at (0,1) --> CID=0
        //     activation 1  at (0,2) --> CID=1
        //     activation 2  at (1,1) --> CID=2
        //   These are pushed and must be popped in the same order.
        //   (See reference doc Stage-1 walkthrough table for full derivation.)
        // ------------------------------------------------------------------
        $display("\n--- TEST 9: GoSPA FIFO-A scenario (toy example from paper) ---");
        push(FIFO_A_W'({16'sh0003, 6'd0}));   // act= 3, CID=0
        push(FIFO_A_W'({16'sh0001, 6'd1}));   // act= 1, CID=1
        push(FIFO_A_W'({16'sh0002, 6'd2}));   // act= 2, CID=2

        pop(popped);
        chk_data("entry 0: (act=3, CID=0)", popped, FIFO_A_W'({16'sh0003, 6'd0}));
        pop(popped);
        chk_data("entry 1: (act=1, CID=1)", popped, FIFO_A_W'({16'sh0001, 6'd1}));
        pop(popped);
        chk_data("entry 2: (act=2, CID=2)", popped, FIFO_A_W'({16'sh0002, 6'd2}));

        wait_cycles(1);
        chk("FIFO-A empty after toy example drain", empty, 1'b1);

        // ------------------------------------------------------------------
        // TEST 10 -- Randomized stress: 50 pushes then 50 pops
        // ------------------------------------------------------------------
        $display("\n--- TEST 10: Randomized stress (50 push, 50 pop) ---");
        ref_queue   = '{};
        rand_errors = 0;

        for (int i = 0; i < 50; i++) begin
            rand_entry = FIFO_A_W'($urandom());
            ref_queue.push_back(rand_entry);
            push(rand_entry);
        end
        for (int i = 0; i < 50; i++) begin
            expected_val = ref_queue.pop_front();
            pop(popped);
            if (popped !== expected_val) begin
                $display("  [FAIL] entry[%0d]: expected=0x%06h  got=0x%06h",
                         i, expected_val, popped);
                rand_errors++;
            end
        end

        if (rand_errors == 0) begin
            $display("  [PASS] All 50 randomized entries matched in order");
            pass_count++;
        end else begin
            $display("  [FAIL] %0d mismatches in randomized test", rand_errors);
            fail_count++;
        end
        wait_cycles(1);
        chk("empty after randomized drain", empty, 1'b1);

        // ------------------------------------------------------------------
        // Summary
        // ------------------------------------------------------------------
        $display("");
        $display("=============================================================");
        if (fail_count == 0)
            $display("  ALL TESTS PASSED  (%0d checks)", pass_count);
        else
            $display("  %0d FAILED / %0d PASSED", fail_count, pass_count);
        $display("=============================================================");
        $display("");

        $finish;
    end : TEST_MAIN

    // =========================================================================
    // Timeout watchdog -- kills the sim if a task deadlocks
    // =========================================================================
    initial begin : WATCHDOG
      #500000;   // 500 us at 1 ns timescale
        $display("[WATCHDOG] Simulation timeout -- possible task deadlock.");
        $finish;
    end : WATCHDOG

    // =========================================================================
    // NOTE: Instantiating fifo for FIFO-B geometry (for reference):
    //
    //   fifo #(
    //       .DATA_WIDTH (FIFO_B_W),   // 26 bits: {act[15:0], pid[3:0], cid[5:0]}
    //       .DEPTH      (FIFO_D)      // 64
    //   ) fifo_b_inst ( ... );
    // =========================================================================

endmodule
