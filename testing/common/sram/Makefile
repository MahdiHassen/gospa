# =====================================================================
# Makefile for cocotb test of sram
# ---------------------------------------------------------------------
# Usage:
#   make clean && make                # default config, verilator
#       NOTE: run `make clean` first when switching parameters. 
#   make ADDR_WIDTH=10                # 1 K-deep memory
#   make OUTPUT_REG=0                 # 1-cycle read latency
#   make USE_DUAL_PORT=0              # single-port (port B tests skip)
#   make sweep                        # run several representative configs
#   make SIM=icarus                   # use icarus instead of verilator
#
# =====================================================================

SIM ?= verilator
TOPLEVEL_LANG ?= verilog

DATA_WIDTH    ?= 32
ADDR_WIDTH    ?= 8
USE_DUAL_PORT ?= 1
OUTPUT_REG    ?= 1

TOPLEVEL = sram
MODULE   = tb_sram

# Path to the RTL, relative to this Makefile's location.
RTL_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))../../../rtl/V2/common
VERILOG_SOURCES = $(RTL_DIR)/sram.sv

ifeq ($(SIM),icarus)
    COMPILE_ARGS += -P$(TOPLEVEL).DATA_WIDTH=$(DATA_WIDTH)
    COMPILE_ARGS += -P$(TOPLEVEL).ADDR_WIDTH=$(ADDR_WIDTH)
    COMPILE_ARGS += -P$(TOPLEVEL).USE_DUAL_PORT=$(USE_DUAL_PORT)
    COMPILE_ARGS += -P$(TOPLEVEL).OUTPUT_REG=$(OUTPUT_REG)
    COMPILE_ARGS += -g2012
endif
ifeq ($(SIM),verilator)
    COMPILE_ARGS += -GDATA_WIDTH=$(DATA_WIDTH)
    COMPILE_ARGS += -GADDR_WIDTH=$(ADDR_WIDTH)
    COMPILE_ARGS += -GUSE_DUAL_PORT=$(USE_DUAL_PORT)
    COMPILE_ARGS += -GOUTPUT_REG=$(OUTPUT_REG)
endif

export DATA_WIDTH
export ADDR_WIDTH
export USE_DUAL_PORT
export OUTPUT_REG

include $(shell cocotb-config --makefiles)/Makefile.sim

.PHONY: sweep
sweep:
	@for cfg in "32 8 1 1" "32 8 1 0" "32 8 0 1" "8 4 1 1" "64 10 1 1" "16 6 0 0"; do \
		set -- $$cfg; \
		echo "=================================================="; \
		echo "  DATA_WIDTH=$$1 ADDR_WIDTH=$$2 USE_DUAL_PORT=$$3 OUTPUT_REG=$$4"; \
		echo "=================================================="; \
		$(MAKE) -s clean; \
		$(MAKE) -s DATA_WIDTH=$$1 ADDR_WIDTH=$$2 USE_DUAL_PORT=$$3 OUTPUT_REG=$$4 || exit 1; \
	done
	@echo "All sweep configs passed."
