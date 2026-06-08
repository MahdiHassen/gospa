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
REPO_ROOT   := $(realpath ../../..)
RTL_DIR     := $(REPO_ROOT)/rtl/apu/stage1

# -- Per-module config (set by MODULE=... on command line) --------------------
ifeq ($(MODULE),test_csr_decode)
    VERILOG_SOURCES := $(RTL_DIR)/csr_decode.sv
    TOPLEVEL        := csr_decode
endif

ifeq ($(MODULE),test_zero_act)
    VERILOG_SOURCES := $(RTL_DIR)/zero_act.sv
    TOPLEVEL        := zero_act
endif

# -- Verilator extra flags ----------------------------------------------------
ifeq ($(SIM),verilator)
    EXTRA_ARGS += --timing -Wall --Wno-UNOPTFLAT --Wno-WIDTH --Wno-TIMESCALEMOD
endif

# -- cocotb boilerplate -------------------------------------------------------
include $(shell cocotb-config --makefiles)/Makefile.sim

# -- Convenience targets ------------------------------------------------------
.PHONY: clean_all help

clean_all: clean
	rm -rf __pycache__ *.xml results.xml

help:
	@echo ""
	@echo "GoSPA APU Stage 1 -- cocotb test runner"
	@echo "----------------------------------------"
	@echo "  make MODULE=test_csr_decode    Test csr_decode.sv"
	@echo "  make MODULE=test_zero_act      Test zero_act.sv"
	@echo "  make MODULE=... SIM=icarus     Use Icarus Verilog"
	@echo "  make clean                     Remove build artifacts"
	@echo ""
	@echo "Install deps: pip install -r requirements.txt"
	@echo ""
