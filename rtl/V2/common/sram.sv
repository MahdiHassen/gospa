// =============================================================================
// sram.sv
// Synchronous SRAM Module
//
// Features:
//   - Single-port or simple dual-port configurations (USE_DUAL_PORT param)
//   - Synchronous read: 1-cycle latency, or 2 cycles with OUTPUT_REG=1
//   - Whole-word writes
//   - Configurable data / address widths and depth
//   - Infers FPGA block RAM (OUTPUT_REG=1 maps to the BRAM output register)
//
// Port naming convention:
//   Port A  – read/write
//   Port B  – read-only, active only when USE_DUAL_PORT = 1
// =============================================================================

`timescale 1ns/1ps

module sram #(
    parameter int  DATA_WIDTH    = 32,         // Width of one word
    parameter int  ADDR_WIDTH    = 12,         // log2(depth); depth = 4 K words
    parameter bit  USE_DUAL_PORT = 1'b1,       // 0 = single-port, 1 = dual-port
    parameter bit  OUTPUT_REG    = 1'b1        // Extra pipeline register on outputs
) (
    input  logic clk,
    input  logic rst_n,

    // =========================================================================
    // Port A  –  Read / Write
    // =========================================================================
    input  logic                  a_en,        // Port A clock enable
    input  logic                  a_we,        // Write enable
    input  logic [ADDR_WIDTH-1:0] a_addr,      // Word address
    input  logic [DATA_WIDTH-1:0] a_wdata,     // Write data
    output logic [DATA_WIDTH-1:0] a_rdata,     // Read data (1 or 2 cycle latency)
    output logic                  a_rdata_vld, // Qualifier for read data

    // =========================================================================
    // Port B  –  Read-only (ignored when USE_DUAL_PORT == 0)
    // =========================================================================
    input  logic                  b_en,
    input  logic [ADDR_WIDTH-1:0] b_addr,
    output logic [DATA_WIDTH-1:0] b_rdata,
    output logic                  b_rdata_vld
);

    // -------------------------------------------------------------------------
    // Memory array  (synthesis infers block RAM when OUTPUT_REG == 1)
    // -------------------------------------------------------------------------
    localparam int DEPTH = 2**ADDR_WIDTH;

    logic [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    // -------------------------------------------------------------------------
    // Initialisation
    // -------------------------------------------------------------------------
    initial begin
        foreach (mem[i]) mem[i] = '0;
    end

    // -------------------------------------------------------------------------
    // Internal read-data wires (before optional output register)
    // -------------------------------------------------------------------------
    logic [DATA_WIDTH-1:0] a_rdata_raw, b_rdata_raw;
    logic                  a_vld_raw,   b_vld_raw;

    // -------------------------------------------------------------------------
    // Port A  –  synchronous whole-word write, synchronous read
    // (write-first: a read issued on the same cycle as a write to the same
    //  address returns the newly written data)
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (a_en) begin
            if (a_we)
                mem[a_addr] <= a_wdata;
            a_rdata_raw <= a_we ? a_wdata : mem[a_addr];
            a_vld_raw   <= 1'b1;
        end else begin
            a_vld_raw   <= 1'b0;
            a_rdata_raw <= a_rdata_raw; // hold
        end
    end

    // -------------------------------------------------------------------------
    // Port B  –  read-only, separate address (dual-port only)
    // -------------------------------------------------------------------------
    generate
        if (USE_DUAL_PORT) begin : gen_portB
            always_ff @(posedge clk) begin
                if (b_en) begin
                    b_rdata_raw <= mem[b_addr];
                    b_vld_raw   <= 1'b1;
                end else begin
                    b_vld_raw   <= 1'b0;
                    b_rdata_raw <= b_rdata_raw;
                end
            end
        end else begin : gen_portB_tie
            assign b_rdata_raw = '0;
            assign b_vld_raw   = 1'b0;
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Optional output pipeline register
    // Adding this register causes synthesis to map to BRAM output registers,
    // improving timing on Xilinx/Intel FPGAs.
    // -------------------------------------------------------------------------
    generate
        if (OUTPUT_REG) begin : gen_outreg
            always_ff @(posedge clk) begin
                if (!rst_n) begin
                    a_rdata     <= '0;
                    a_rdata_vld <= 1'b0;
                    b_rdata     <= '0;
                    b_rdata_vld <= 1'b0;
                end else begin
                    a_rdata     <= a_rdata_raw;
                    a_rdata_vld <= a_vld_raw;
                    b_rdata     <= b_rdata_raw;
                    b_rdata_vld <= b_vld_raw;
                end
            end
        end else begin : gen_comb_out
            assign a_rdata     = a_rdata_raw;
            assign a_rdata_vld = a_vld_raw;
            assign b_rdata     = b_rdata_raw;
            assign b_rdata_vld = b_vld_raw;
        end
    endgenerate

endmodule