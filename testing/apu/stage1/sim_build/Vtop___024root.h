// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vtop.h for the primary calling header

#ifndef VERILATED_VTOP___024ROOT_H_
#define VERILATED_VTOP___024ROOT_H_  // guard

#include "verilated.h"

class Vtop__Syms;

class Vtop___024root final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    VL_IN8(clk,0,0);
    VL_IN8(rst_n,0,0);
    VL_IN8(row_ptr_valid,0,0);
    VL_OUT8(row_ptr_ready,0,0);
    VL_IN8(entry_valid,0,0);
    VL_IN8(entry_col,4,0);
    VL_OUT8(entry_ready,0,0);
    VL_OUT8(out_valid,0,0);
    VL_OUT8(out_x,4,0);
    VL_OUT8(out_y,4,0);
    VL_IN8(out_ready,0,0);
    CData/*0:0*/ csr_decode__DOT__clk;
    CData/*0:0*/ csr_decode__DOT__rst_n;
    CData/*0:0*/ csr_decode__DOT__row_ptr_valid;
    CData/*0:0*/ csr_decode__DOT__row_ptr_ready;
    CData/*0:0*/ csr_decode__DOT__entry_valid;
    CData/*4:0*/ csr_decode__DOT__entry_col;
    CData/*0:0*/ csr_decode__DOT__entry_ready;
    CData/*0:0*/ csr_decode__DOT__out_valid;
    CData/*4:0*/ csr_decode__DOT__out_x;
    CData/*4:0*/ csr_decode__DOT__out_y;
    CData/*0:0*/ csr_decode__DOT__out_ready;
    CData/*1:0*/ csr_decode__DOT__state;
    CData/*4:0*/ csr_decode__DOT__row_ctr;
    CData/*0:0*/ csr_decode__DOT__do_emit;
    CData/*0:0*/ __Vtrigrprev__TOP__clk;
    CData/*0:0*/ __VactContinue;
    VL_IN16(row_ptr_data,10,0);
    VL_IN16(entry_value,15,0);
    VL_OUT16(out_value,15,0);
    SData/*10:0*/ csr_decode__DOT__row_ptr_data;
    SData/*15:0*/ csr_decode__DOT__entry_value;
    SData/*15:0*/ csr_decode__DOT__out_value;
    SData/*10:0*/ csr_decode__DOT__row_start;
    SData/*10:0*/ csr_decode__DOT__row_end;
    SData/*10:0*/ csr_decode__DOT__entry_ctr;
    SData/*10:0*/ csr_decode__DOT__entries_in_row;
    IData/*31:0*/ __VstlIterCount;
    IData/*31:0*/ __VicoIterCount;
    IData/*31:0*/ __VactIterCount;
    VlTriggerVec<1> __VstlTriggered;
    VlTriggerVec<1> __VicoTriggered;
    VlTriggerVec<1> __VactTriggered;
    VlTriggerVec<1> __VnbaTriggered;

    // INTERNAL VARIABLES
    Vtop__Syms* const vlSymsp;

    // PARAMETERS
    static constexpr IData/*31:0*/ csr_decode__DOT__H = 0x00000020U;
    static constexpr IData/*31:0*/ csr_decode__DOT__DATA_W = 0x00000010U;

    // CONSTRUCTORS
    Vtop___024root(Vtop__Syms* symsp, const char* v__name);
    ~Vtop___024root();
    VL_UNCOPYABLE(Vtop___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
} VL_ATTR_ALIGNED(VL_CACHE_LINE_BYTES);


#endif  // guard
