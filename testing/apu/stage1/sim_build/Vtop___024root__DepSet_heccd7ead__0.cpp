// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtop.h for the primary calling header

#include "verilated.h"
#include "verilated_dpi.h"

#include "Vtop___024root.h"

VL_INLINE_OPT void Vtop___024root___ico_sequent__TOP__0(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___ico_sequent__TOP__0\n"); );
    // Body
    vlSelf->csr_decode__DOT__clk = vlSelf->clk;
    vlSelf->csr_decode__DOT__rst_n = vlSelf->rst_n;
    vlSelf->csr_decode__DOT__row_ptr_valid = vlSelf->row_ptr_valid;
    vlSelf->csr_decode__DOT__row_ptr_data = vlSelf->row_ptr_data;
    vlSelf->csr_decode__DOT__entry_valid = vlSelf->entry_valid;
    vlSelf->csr_decode__DOT__entry_value = vlSelf->entry_value;
    vlSelf->csr_decode__DOT__entry_col = vlSelf->entry_col;
    vlSelf->csr_decode__DOT__out_ready = vlSelf->out_ready;
    vlSelf->csr_decode__DOT__entries_in_row = (0x7ffU 
                                               & ((IData)(vlSelf->csr_decode__DOT__row_end) 
                                                  - (IData)(vlSelf->csr_decode__DOT__row_start)));
    vlSelf->row_ptr_ready = ((0U == (IData)(vlSelf->csr_decode__DOT__state)) 
                             | (1U == (IData)(vlSelf->csr_decode__DOT__state)));
    vlSelf->out_value = vlSelf->entry_value;
    vlSelf->out_y = vlSelf->entry_col;
    vlSelf->out_x = vlSelf->csr_decode__DOT__row_ctr;
    vlSelf->entry_ready = ((2U == (IData)(vlSelf->csr_decode__DOT__state)) 
                           & (IData)(vlSelf->out_ready));
    vlSelf->out_valid = ((2U == (IData)(vlSelf->csr_decode__DOT__state)) 
                         & (IData)(vlSelf->entry_valid));
    vlSelf->csr_decode__DOT__row_ptr_ready = vlSelf->row_ptr_ready;
    vlSelf->csr_decode__DOT__out_value = vlSelf->out_value;
    vlSelf->csr_decode__DOT__out_y = vlSelf->out_y;
    vlSelf->csr_decode__DOT__out_x = vlSelf->out_x;
    vlSelf->csr_decode__DOT__entry_ready = vlSelf->entry_ready;
    vlSelf->csr_decode__DOT__out_valid = vlSelf->out_valid;
    vlSelf->csr_decode__DOT__do_emit = ((IData)(vlSelf->out_valid) 
                                        & (IData)(vlSelf->out_ready));
}

void Vtop___024root___eval_ico(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_ico\n"); );
    // Body
    if (vlSelf->__VicoTriggered.at(0U)) {
        Vtop___024root___ico_sequent__TOP__0(vlSelf);
    }
}

void Vtop___024root___eval_act(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_act\n"); );
}

VL_INLINE_OPT void Vtop___024root___nba_sequent__TOP__0(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___nba_sequent__TOP__0\n"); );
    // Init
    SData/*10:0*/ __Vdly__csr_decode__DOT__row_start;
    __Vdly__csr_decode__DOT__row_start = 0;
    CData/*1:0*/ __Vdly__csr_decode__DOT__state;
    __Vdly__csr_decode__DOT__state = 0;
    SData/*10:0*/ __Vdly__csr_decode__DOT__row_end;
    __Vdly__csr_decode__DOT__row_end = 0;
    SData/*10:0*/ __Vdly__csr_decode__DOT__entry_ctr;
    __Vdly__csr_decode__DOT__entry_ctr = 0;
    CData/*4:0*/ __Vdly__csr_decode__DOT__row_ctr;
    __Vdly__csr_decode__DOT__row_ctr = 0;
    // Body
    __Vdly__csr_decode__DOT__entry_ctr = vlSelf->csr_decode__DOT__entry_ctr;
    __Vdly__csr_decode__DOT__row_end = vlSelf->csr_decode__DOT__row_end;
    __Vdly__csr_decode__DOT__row_start = vlSelf->csr_decode__DOT__row_start;
    __Vdly__csr_decode__DOT__row_ctr = vlSelf->csr_decode__DOT__row_ctr;
    __Vdly__csr_decode__DOT__state = vlSelf->csr_decode__DOT__state;
    if (vlSelf->rst_n) {
        if ((0U == (IData)(vlSelf->csr_decode__DOT__state))) {
            if (vlSelf->row_ptr_valid) {
                __Vdly__csr_decode__DOT__row_start 
                    = vlSelf->row_ptr_data;
                __Vdly__csr_decode__DOT__state = 1U;
            }
        } else if ((1U == (IData)(vlSelf->csr_decode__DOT__state))) {
            if (vlSelf->row_ptr_valid) {
                __Vdly__csr_decode__DOT__row_end = vlSelf->row_ptr_data;
                __Vdly__csr_decode__DOT__entry_ctr = 0U;
                if (((IData)(vlSelf->row_ptr_data) 
                     == (IData)(vlSelf->csr_decode__DOT__row_start))) {
                    __Vdly__csr_decode__DOT__row_ctr 
                        = (0x1fU & ((IData)(1U) + (IData)(vlSelf->csr_decode__DOT__row_ctr)));
                    __Vdly__csr_decode__DOT__row_start 
                        = vlSelf->row_ptr_data;
                    __Vdly__csr_decode__DOT__state = 1U;
                } else {
                    __Vdly__csr_decode__DOT__state = 2U;
                }
            }
        } else if ((2U == (IData)(vlSelf->csr_decode__DOT__state))) {
            if (vlSelf->csr_decode__DOT__do_emit) {
                __Vdly__csr_decode__DOT__entry_ctr 
                    = (0x7ffU & ((IData)(1U) + (IData)(vlSelf->csr_decode__DOT__entry_ctr)));
                if (((IData)(vlSelf->csr_decode__DOT__entry_ctr) 
                     == ((IData)(vlSelf->csr_decode__DOT__entries_in_row) 
                         - (IData)(1U)))) {
                    __Vdly__csr_decode__DOT__row_ctr 
                        = (0x1fU & ((IData)(1U) + (IData)(vlSelf->csr_decode__DOT__row_ctr)));
                    __Vdly__csr_decode__DOT__row_start 
                        = vlSelf->csr_decode__DOT__row_end;
                    __Vdly__csr_decode__DOT__state = 1U;
                }
            }
        } else {
            __Vdly__csr_decode__DOT__state = 0U;
        }
    } else {
        __Vdly__csr_decode__DOT__row_ctr = 0U;
        __Vdly__csr_decode__DOT__state = 0U;
        __Vdly__csr_decode__DOT__row_start = 0U;
        __Vdly__csr_decode__DOT__row_end = 0U;
        __Vdly__csr_decode__DOT__entry_ctr = 0U;
    }
    vlSelf->csr_decode__DOT__entry_ctr = __Vdly__csr_decode__DOT__entry_ctr;
    vlSelf->csr_decode__DOT__row_start = __Vdly__csr_decode__DOT__row_start;
    vlSelf->csr_decode__DOT__row_end = __Vdly__csr_decode__DOT__row_end;
    vlSelf->csr_decode__DOT__row_ctr = __Vdly__csr_decode__DOT__row_ctr;
    vlSelf->csr_decode__DOT__state = __Vdly__csr_decode__DOT__state;
    vlSelf->csr_decode__DOT__entries_in_row = (0x7ffU 
                                               & ((IData)(vlSelf->csr_decode__DOT__row_end) 
                                                  - (IData)(vlSelf->csr_decode__DOT__row_start)));
    vlSelf->out_x = vlSelf->csr_decode__DOT__row_ctr;
    vlSelf->row_ptr_ready = ((0U == (IData)(vlSelf->csr_decode__DOT__state)) 
                             | (1U == (IData)(vlSelf->csr_decode__DOT__state)));
    vlSelf->entry_ready = ((2U == (IData)(vlSelf->csr_decode__DOT__state)) 
                           & (IData)(vlSelf->out_ready));
    vlSelf->out_valid = ((2U == (IData)(vlSelf->csr_decode__DOT__state)) 
                         & (IData)(vlSelf->entry_valid));
    vlSelf->csr_decode__DOT__out_x = vlSelf->out_x;
    vlSelf->csr_decode__DOT__row_ptr_ready = vlSelf->row_ptr_ready;
    vlSelf->csr_decode__DOT__entry_ready = vlSelf->entry_ready;
    vlSelf->csr_decode__DOT__out_valid = vlSelf->out_valid;
    vlSelf->csr_decode__DOT__do_emit = ((IData)(vlSelf->out_valid) 
                                        & (IData)(vlSelf->out_ready));
}

void Vtop___024root___eval_nba(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_nba\n"); );
    // Body
    if (vlSelf->__VnbaTriggered.at(0U)) {
        Vtop___024root___nba_sequent__TOP__0(vlSelf);
    }
}

void Vtop___024root___eval_triggers__ico(Vtop___024root* vlSelf);
#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__ico(Vtop___024root* vlSelf);
#endif  // VL_DEBUG
void Vtop___024root___eval_triggers__act(Vtop___024root* vlSelf);
#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__act(Vtop___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__nba(Vtop___024root* vlSelf);
#endif  // VL_DEBUG

void Vtop___024root___eval(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval\n"); );
    // Init
    CData/*0:0*/ __VicoContinue;
    VlTriggerVec<1> __VpreTriggered;
    IData/*31:0*/ __VnbaIterCount;
    CData/*0:0*/ __VnbaContinue;
    // Body
    vlSelf->__VicoIterCount = 0U;
    __VicoContinue = 1U;
    while (__VicoContinue) {
        __VicoContinue = 0U;
        Vtop___024root___eval_triggers__ico(vlSelf);
        if (vlSelf->__VicoTriggered.any()) {
            __VicoContinue = 1U;
            if (VL_UNLIKELY((0x64U < vlSelf->__VicoIterCount))) {
#ifdef VL_DEBUG
                Vtop___024root___dump_triggers__ico(vlSelf);
#endif
                VL_FATAL_MT("/media/esarkar/DATADisk/ECE720_labs_HW_ML/project/team-19/rtl/apu/stage1/csr_decode.sv", 32, "", "Input combinational region did not converge.");
            }
            vlSelf->__VicoIterCount = ((IData)(1U) 
                                       + vlSelf->__VicoIterCount);
            Vtop___024root___eval_ico(vlSelf);
        }
    }
    __VnbaIterCount = 0U;
    __VnbaContinue = 1U;
    while (__VnbaContinue) {
        __VnbaContinue = 0U;
        vlSelf->__VnbaTriggered.clear();
        vlSelf->__VactIterCount = 0U;
        vlSelf->__VactContinue = 1U;
        while (vlSelf->__VactContinue) {
            vlSelf->__VactContinue = 0U;
            Vtop___024root___eval_triggers__act(vlSelf);
            if (vlSelf->__VactTriggered.any()) {
                vlSelf->__VactContinue = 1U;
                if (VL_UNLIKELY((0x64U < vlSelf->__VactIterCount))) {
#ifdef VL_DEBUG
                    Vtop___024root___dump_triggers__act(vlSelf);
#endif
                    VL_FATAL_MT("/media/esarkar/DATADisk/ECE720_labs_HW_ML/project/team-19/rtl/apu/stage1/csr_decode.sv", 32, "", "Active region did not converge.");
                }
                vlSelf->__VactIterCount = ((IData)(1U) 
                                           + vlSelf->__VactIterCount);
                __VpreTriggered.andNot(vlSelf->__VactTriggered, vlSelf->__VnbaTriggered);
                vlSelf->__VnbaTriggered.set(vlSelf->__VactTriggered);
                Vtop___024root___eval_act(vlSelf);
            }
        }
        if (vlSelf->__VnbaTriggered.any()) {
            __VnbaContinue = 1U;
            if (VL_UNLIKELY((0x64U < __VnbaIterCount))) {
#ifdef VL_DEBUG
                Vtop___024root___dump_triggers__nba(vlSelf);
#endif
                VL_FATAL_MT("/media/esarkar/DATADisk/ECE720_labs_HW_ML/project/team-19/rtl/apu/stage1/csr_decode.sv", 32, "", "NBA region did not converge.");
            }
            __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
            Vtop___024root___eval_nba(vlSelf);
        }
    }
}

#ifdef VL_DEBUG
void Vtop___024root___eval_debug_assertions(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_debug_assertions\n"); );
    // Body
    if (VL_UNLIKELY((vlSelf->clk & 0xfeU))) {
        Verilated::overWidthError("clk");}
    if (VL_UNLIKELY((vlSelf->rst_n & 0xfeU))) {
        Verilated::overWidthError("rst_n");}
    if (VL_UNLIKELY((vlSelf->row_ptr_valid & 0xfeU))) {
        Verilated::overWidthError("row_ptr_valid");}
    if (VL_UNLIKELY((vlSelf->row_ptr_data & 0xf800U))) {
        Verilated::overWidthError("row_ptr_data");}
    if (VL_UNLIKELY((vlSelf->entry_valid & 0xfeU))) {
        Verilated::overWidthError("entry_valid");}
    if (VL_UNLIKELY((vlSelf->entry_col & 0xe0U))) {
        Verilated::overWidthError("entry_col");}
    if (VL_UNLIKELY((vlSelf->out_ready & 0xfeU))) {
        Verilated::overWidthError("out_ready");}
}
#endif  // VL_DEBUG
