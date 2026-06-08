// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table implementation internals

#include "Vtop__Syms.h"
#include "Vtop.h"
#include "Vtop___024root.h"

// FUNCTIONS
Vtop__Syms::~Vtop__Syms()
{

    // Tear down scope hierarchy
    __Vhier.remove(0, &__Vscope_csr_decode);

}

Vtop__Syms::Vtop__Syms(VerilatedContext* contextp, const char* namep, Vtop* modelp)
    : VerilatedSyms{contextp}
    // Setup internal state of the Syms class
    , __Vm_modelp{modelp}
    // Setup module instances
    , TOP{this, namep}
{
    // Configure time unit / time precision
    _vm_contextp__->timeunit(-9);
    _vm_contextp__->timeprecision(-12);
    // Setup each module's pointers to their submodules
    // Setup each module's pointer back to symbol table (for public functions)
    TOP.__Vconfigure(true);
    // Setup scopes
    __Vscope_TOP.configure(this, name(), "TOP", "TOP", 0, VerilatedScope::SCOPE_OTHER);
    __Vscope_csr_decode.configure(this, name(), "csr_decode", "csr_decode", -9, VerilatedScope::SCOPE_MODULE);

    // Set up scope hierarchy
    __Vhier.add(0, &__Vscope_csr_decode);

    // Setup export functions
    for (int __Vfinal = 0; __Vfinal < 2; ++__Vfinal) {
        __Vscope_TOP.varInsert(__Vfinal,"clk", &(TOP.clk), false, VLVT_UINT8,VLVD_IN|VLVF_PUB_RW,0);
        __Vscope_TOP.varInsert(__Vfinal,"entry_col", &(TOP.entry_col), false, VLVT_UINT8,VLVD_IN|VLVF_PUB_RW,1 ,4,0);
        __Vscope_TOP.varInsert(__Vfinal,"entry_ready", &(TOP.entry_ready), false, VLVT_UINT8,VLVD_OUT|VLVF_PUB_RW,0);
        __Vscope_TOP.varInsert(__Vfinal,"entry_valid", &(TOP.entry_valid), false, VLVT_UINT8,VLVD_IN|VLVF_PUB_RW,0);
        __Vscope_TOP.varInsert(__Vfinal,"entry_value", &(TOP.entry_value), false, VLVT_UINT16,VLVD_IN|VLVF_PUB_RW,1 ,15,0);
        __Vscope_TOP.varInsert(__Vfinal,"out_ready", &(TOP.out_ready), false, VLVT_UINT8,VLVD_IN|VLVF_PUB_RW,0);
        __Vscope_TOP.varInsert(__Vfinal,"out_valid", &(TOP.out_valid), false, VLVT_UINT8,VLVD_OUT|VLVF_PUB_RW,0);
        __Vscope_TOP.varInsert(__Vfinal,"out_value", &(TOP.out_value), false, VLVT_UINT16,VLVD_OUT|VLVF_PUB_RW,1 ,15,0);
        __Vscope_TOP.varInsert(__Vfinal,"out_x", &(TOP.out_x), false, VLVT_UINT8,VLVD_OUT|VLVF_PUB_RW,1 ,4,0);
        __Vscope_TOP.varInsert(__Vfinal,"out_y", &(TOP.out_y), false, VLVT_UINT8,VLVD_OUT|VLVF_PUB_RW,1 ,4,0);
        __Vscope_TOP.varInsert(__Vfinal,"row_ptr_data", &(TOP.row_ptr_data), false, VLVT_UINT16,VLVD_IN|VLVF_PUB_RW,1 ,10,0);
        __Vscope_TOP.varInsert(__Vfinal,"row_ptr_ready", &(TOP.row_ptr_ready), false, VLVT_UINT8,VLVD_OUT|VLVF_PUB_RW,0);
        __Vscope_TOP.varInsert(__Vfinal,"row_ptr_valid", &(TOP.row_ptr_valid), false, VLVT_UINT8,VLVD_IN|VLVF_PUB_RW,0);
        __Vscope_TOP.varInsert(__Vfinal,"rst_n", &(TOP.rst_n), false, VLVT_UINT8,VLVD_IN|VLVF_PUB_RW,0);
        __Vscope_csr_decode.varInsert(__Vfinal,"DATA_W", const_cast<void*>(static_cast<const void*>(&(TOP.csr_decode__DOT__DATA_W))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW|VLVF_DPI_CLAY,1 ,31,0);
        __Vscope_csr_decode.varInsert(__Vfinal,"H", const_cast<void*>(static_cast<const void*>(&(TOP.csr_decode__DOT__H))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW|VLVF_DPI_CLAY,1 ,31,0);
        __Vscope_csr_decode.varInsert(__Vfinal,"clk", &(TOP.csr_decode__DOT__clk), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_csr_decode.varInsert(__Vfinal,"do_emit", &(TOP.csr_decode__DOT__do_emit), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_csr_decode.varInsert(__Vfinal,"entries_in_row", &(TOP.csr_decode__DOT__entries_in_row), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,10,0);
        __Vscope_csr_decode.varInsert(__Vfinal,"entry_col", &(TOP.csr_decode__DOT__entry_col), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,4,0);
        __Vscope_csr_decode.varInsert(__Vfinal,"entry_ctr", &(TOP.csr_decode__DOT__entry_ctr), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,10,0);
        __Vscope_csr_decode.varInsert(__Vfinal,"entry_ready", &(TOP.csr_decode__DOT__entry_ready), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_csr_decode.varInsert(__Vfinal,"entry_valid", &(TOP.csr_decode__DOT__entry_valid), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_csr_decode.varInsert(__Vfinal,"entry_value", &(TOP.csr_decode__DOT__entry_value), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,15,0);
        __Vscope_csr_decode.varInsert(__Vfinal,"out_ready", &(TOP.csr_decode__DOT__out_ready), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_csr_decode.varInsert(__Vfinal,"out_valid", &(TOP.csr_decode__DOT__out_valid), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_csr_decode.varInsert(__Vfinal,"out_value", &(TOP.csr_decode__DOT__out_value), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,15,0);
        __Vscope_csr_decode.varInsert(__Vfinal,"out_x", &(TOP.csr_decode__DOT__out_x), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,4,0);
        __Vscope_csr_decode.varInsert(__Vfinal,"out_y", &(TOP.csr_decode__DOT__out_y), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,4,0);
        __Vscope_csr_decode.varInsert(__Vfinal,"row_ctr", &(TOP.csr_decode__DOT__row_ctr), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,4,0);
        __Vscope_csr_decode.varInsert(__Vfinal,"row_end", &(TOP.csr_decode__DOT__row_end), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,10,0);
        __Vscope_csr_decode.varInsert(__Vfinal,"row_ptr_data", &(TOP.csr_decode__DOT__row_ptr_data), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,10,0);
        __Vscope_csr_decode.varInsert(__Vfinal,"row_ptr_ready", &(TOP.csr_decode__DOT__row_ptr_ready), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_csr_decode.varInsert(__Vfinal,"row_ptr_valid", &(TOP.csr_decode__DOT__row_ptr_valid), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_csr_decode.varInsert(__Vfinal,"row_start", &(TOP.csr_decode__DOT__row_start), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,10,0);
        __Vscope_csr_decode.varInsert(__Vfinal,"rst_n", &(TOP.csr_decode__DOT__rst_n), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_csr_decode.varInsert(__Vfinal,"state", &(TOP.csr_decode__DOT__state), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,1,0);
    }
}
