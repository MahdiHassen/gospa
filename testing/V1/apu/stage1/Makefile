# ==============================================================================
# Makefile -- GoSPA APU Stage 1 cocotb Tests
# ==============================================================================
# GoSPA Project -- Team 19, ECE 720 (Spring 2026)
#
# Usage:
#   make MODULE=test_csr_decode     # run csr_decode tests
#   make MODULE=test_zero_act       # run zero_act tests
#   make MODULE=test_csr_decode SIM=icarus   # use Icarus instead
#   make clean                      # remove build artifacts
#
# Dependencies:
#   pip install -r requirements.txt
# ==============================================================================

# -- Simulator (verilator or icarus) ------------------------------------------
SIM      ?= verilator
TOPLEVEL_LANG ?= verilog

# -- Paths --------------------------------------------------------------------
REPO_ROOT   := $(realpath ../../../..)
RTL_DIR     := $(REPO_ROOT)/rtl/V1/apu/stage1

# -- Per-module config (set by MODULE=... on command line) --------------------
ifeq ($(MODULE),test_csr_decode)
    VERILOG_SOURCES := $(RTL_DIR)/csr_decode.sv
    TOPLEVEL        := csr_decode
endif

ifeq ($(MODULE),test_zero_act)
    VERILOG_SOURCES := $(RTL_DIR)/zero_act.sv
    TOPLEVEL        := zero_act
endif

# -- Full Stage-1 chain: csr_decode -> zero_act -> position_encode ------------
#    -> idgen -> FIFO-A bank.  NOTE: idgen.sv crashes Verilator < 5.020, so run
#    this target with SIM=icarus unless your Verilator is new enough.
ifeq ($(MODULE),test_apu_stage1)
    # apu_stage1 now takes (val, x, y) directly (csr_decode lives upstream of
    # the APU as part of the activation SRAM scanner in act_sram_scanner.sv).
    VERILOG_SOURCES := $(REPO_ROOT)/rtl/V1/common/fifo.sv \
                       $(RTL_DIR)/zero_act.sv \
                       $(RTL_DIR)/position_encode.sv \
                       $(RTL_DIR)/idgen.sv \
                       $(RTL_DIR)/apu_stage1.sv
    TOPLEVEL := apu_stage1

    # Layer config (override on the command line): H x H map, F x F kernel, stride S
    H ?= 8
    F ?= 3
    S ?= 1

    ifeq ($(SIM),icarus)
        COMPILE_ARGS += -g2012 -Papu_stage1.H=$(H) -Papu_stage1.F=$(F) -Papu_stage1.S=$(S)
    endif
    ifeq ($(SIM),verilator)
        COMPILE_ARGS += -GH=$(H) -GF=$(F) -GS=$(S)
    endif

    export H
    export F
    export S
endif

# -- Verilator extra flags ----------------------------------------------------
ifeq ($(SIM),verilator)
    EXTRA_ARGS += --timing -Wall --Wno-UNOPTFLAT --Wno-WIDTH --Wno-TIMESCALEMOD --Wno-DECLFILENAME
endif

# -- cocotb boilerplate -------------------------------------------------------
include $(shell cocotb-config --makefiles)/Makefile.sim

# -- Convenience targets ------------------------------------------------------
.PHONY: clean_all help sweep_s1

clean_all: clean
	rm -rf __pycache__ *.xml results.xml

# Run the full Stage-1 chain across several layer configs (Icarus).
# Includes the paper toy example (H=3 F=2 S=1).
sweep_s1:
	@for cfg in "3 2 1" "8 3 1" "8 3 2" "9 3 2" "8 2 2" "10 5 1" "11 5 2"; do \
		set -- $$cfg; \
		echo "=================================================="; \
		echo "  apu_stage1  H=$$1 F=$$2 S=$$3"; \
		echo "=================================================="; \
		$(MAKE) -s clean; \
		$(MAKE) -s MODULE=test_apu_stage1 SIM=icarus H=$$1 F=$$2 S=$$3 || exit 1; \
	done
	@echo "All Stage-1 sweep configs passed."

help:
	@echo ""
	@echo "GoSPA APU Stage 1 -- cocotb test runner"
	@echo "----------------------------------------"
	@echo "  make MODULE=test_csr_decode             Test csr_decode.sv"
	@echo "  make MODULE=test_zero_act               Test zero_act.sv"
	@echo "  make MODULE=test_apu_stage1 SIM=icarus  Test full Stage-1 vs SW model"
	@echo "  make MODULE=test_apu_stage1 SIM=icarus H=8 F=3 S=2   (custom layer)"
	@echo "  make sweep_s1                           Stage-1 across many configs"
	@echo "  make clean                              Remove build artifacts"
	@echo ""
	@echo "  idgen.sv needs Verilator >= 5.020; otherwise use SIM=icarus."
	@echo "Install deps: pip install -r requirements.txt"
	@echo ""
