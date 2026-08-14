# ==============================================================================
# Makefile -- pipelined arithmetic unit cocotb tests
# GoSPA Project -- Team 19, ECE 720 (Spring 2026)
#
#   make MODULE=test_mult_pipe                 # 16x16 signed multiplier
#   make MODULE=test_mac_pipe                  # 16x16 + 32 fused multiply-add
#   make MODULE=test_mult_pipe A_W=8 B_W=8     # different widths
#   make MODULE=test_mult_pipe SIM=verilator
#   make clean
# ==============================================================================

SIM      ?= icarus
TOPLEVEL_LANG ?= verilog

REPO_ROOT := $(realpath ../..)
ARITH     := $(REPO_ROOT)/rtl/V2/common/arith

A_W ?= 16
B_W ?= 16
C_W ?= 32

MODULE ?= test_mult_pipe

ifeq ($(MODULE),test_mac_pipe)
    VERILOG_SOURCES := $(ARITH)/rca_add.sv $(ARITH)/mult_pipe.sv $(ARITH)/mac_pipe.sv
    TOPLEVEL        := mac_pipe
    PARAMS          := A_W=$(A_W) B_W=$(B_W) C_W=$(C_W)
else
    VERILOG_SOURCES := $(ARITH)/rca_add.sv $(ARITH)/mult_pipe.sv
    TOPLEVEL        := mult_pipe
    PARAMS          := A_W=$(A_W) B_W=$(B_W)
endif

ifeq ($(SIM),icarus)
    COMPILE_ARGS += -g2012 $(foreach p,$(PARAMS),-P$(TOPLEVEL).$(p))
else
    COMPILE_ARGS += $(foreach p,$(PARAMS),-G$(p))
    EXTRA_ARGS   += --timing -Wall --Wno-UNOPTFLAT --Wno-WIDTH \
                    --Wno-TIMESCALEMOD --Wno-DECLFILENAME --Wno-UNUSEDSIGNAL \
                    --Wno-PINCONNECTEMPTY
endif

export A_W B_W C_W

include $(shell cocotb-config --makefiles)/Makefile.sim

.PHONY: clean_all
clean_all: clean
	rm -rf __pycache__ *.xml results.xml
