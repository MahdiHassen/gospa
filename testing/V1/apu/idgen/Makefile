# =====================================================================
# Makefile for cocotb test of idgen
# ---------------------------------------------------------------------
# Usage:
#   make                      # default config H=8 F=3 S=1 PIPE=0
#   make H=8 F=3 S=2          # different layer
#   make PIPE=1               # registered (pipelined) outputs
#   make sweep                # run several configs
#   make SIM=icarus           # use icarus instead of verilator
# =====================================================================

SIM ?= verilator
TOPLEVEL_LANG ?= verilog

# --- layer config (override on command line) ---
H ?= 8
F ?= 3
S ?= 1
PIPE ?= 0

TOPLEVEL = idgen
MODULE   = tb_idgen

# Path to the RTL, relative to this Makefile's location.
RTL_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))../../../../rtl/V1/apu/stage1
VERILOG_SOURCES = $(RTL_DIR)/idgen.sv

# Pass H/F/S into the RTL as parameter overrides.
# Icarus uses -P<top>.<param>=<val>; verilator uses -G<param>=<val>.
ifeq ($(SIM),icarus)
    COMPILE_ARGS += -P$(TOPLEVEL).H=$(H) -P$(TOPLEVEL).F=$(F) -P$(TOPLEVEL).S=$(S) -P$(TOPLEVEL).PIPE=$(PIPE)
    COMPILE_ARGS += -g2012
endif
ifeq ($(SIM),verilator)
    COMPILE_ARGS += -GH=$(H) -GF=$(F) -GS=$(S) -GPIPE=$(PIPE)
endif

# make H/F/S/PIPE visible to the Python testbench
export H
export F
export S
export PIPE

include $(shell cocotb-config --makefiles)/Makefile.sim

.PHONY: sweep
sweep:
	@for cfg in "8 3 1" "8 3 2" "9 3 2" "7 3 1" "8 2 2" "10 5 1" "11 5 2" "13 11 4"; do \
		set -- $$cfg; \
		echo "=================================================="; \
		echo "  Config H=$$1 F=$$2 S=$$3"; \
		echo "=================================================="; \
		$(MAKE) -s clean; \
		$(MAKE) -s H=$$1 F=$$2 S=$$3 || exit 1; \
	done
	@echo "All sweep configs passed."