# =====================================================================
# Makefile for cocotb test of routing (APU Stage 2)
# ---------------------------------------------------------------------
# Usage:
#   make clean && make                       # default config, verilator
#       NOTE: run `make clean` first when switching parameters.
#   make N_PID=9 N_PE=4                      # explicit shape
#   make ACT_WIDTH=16 CID_WIDTH=6            # wider payload
#   make sweep                               # run several representative configs
#   make SIM=icarus                          # use icarus instead of verilator
# =====================================================================

SIM ?= verilator
TOPLEVEL_LANG ?= verilog

# --- router shape (override on command line) ---
N_PID ?= 9
N_PE  ?= 4
ACT_WIDTH ?= 8
CID_WIDTH ?= 6

TOPLEVEL = routing
MODULE   = tb_routing

# Path to the RTL, relative to this Makefile's location.
RTL_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))../../../../rtl/V1/apu/stage2
VERILOG_SOURCES = $(RTL_DIR)/routing.sv

# Put the functional model (sw/functional.py) on PYTHONPATH so the testbench can import it
SW_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))../../sw_v1
export PYTHONPATH := $(SW_DIR):$(PYTHONPATH)

# Pass shape into the RTL as parameter overrides.
# Icarus uses -P<top>.<param>=<val>; verilator uses -G<param>=<val>.
ifeq ($(SIM),icarus)
    COMPILE_ARGS += -P$(TOPLEVEL).N_PID=$(N_PID) -P$(TOPLEVEL).N_PE=$(N_PE)
    COMPILE_ARGS += -P$(TOPLEVEL).ACT_WIDTH=$(ACT_WIDTH) -P$(TOPLEVEL).CID_WIDTH=$(CID_WIDTH)
    COMPILE_ARGS += -g2012
endif
ifeq ($(SIM),verilator)
    COMPILE_ARGS += -GN_PID=$(N_PID) -GN_PE=$(N_PE) -GACT_WIDTH=$(ACT_WIDTH) -GCID_WIDTH=$(CID_WIDTH)
    COMPILE_ARGS += --assert         
endif

#Python Variable Visibility
export N_PID
export N_PE
export ACT_WIDTH
export CID_WIDTH

include $(shell cocotb-config --makefiles)/Makefile.sim

.PHONY: sweep
sweep:
	@for cfg in "9 4 8 6" "4 2 8 6" "8 4 16 6" "16 8 8 8" "1 1 8 6" "9 1 8 6"; do \
		set -- $$cfg; \
		echo "=================================================="; \
		echo "  N_PID=$$1 N_PE=$$2 ACT_WIDTH=$$3 CID_WIDTH=$$4"; \
		echo "=================================================="; \
		$(MAKE) -s clean; \
		$(MAKE) -s N_PID=$$1 N_PE=$$2 ACT_WIDTH=$$3 CID_WIDTH=$$4 || exit 1; \
	done
	@echo "All sweep configs passed."
