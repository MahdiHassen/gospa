# ==============================================================================
# Makefile -- GoSPA PE cocotb tests
# ==============================================================================
# GoSPA Project -- Team 19, ECE 720 (Spring 2026)
#
# Usage:
#   make MODULE=test_pe                 # default H=8 F=3 S=1 (Icarus)
#   make MODULE=test_pe H=8 F=3 S=2     # different layer
#   make sweep_pe                       # several layer configs
#   make MODULE=test_pe SIM=verilator   # (build dir must be exec-friendly)
#   make clean
# ==============================================================================

SIM      ?= icarus
TOPLEVEL_LANG ?= verilog

REPO_ROOT := $(realpath ../..)
PE_DIR    := $(REPO_ROOT)/rtl/pe

# -- Layer config (override on command line) ----------------------------------
H ?= 8
F ?= 3
S ?= 1

# Derived sizes the PE is parameterized by (kernel positions, output positions).
N_PID := $(shell echo $$(( $(F) * $(F) )))
EE    := $(shell echo $$(( ($(H) - $(F)) / $(S) + 1 )))
N_CID := $(shell echo $$(( $(EE) * $(EE) )))

VERILOG_SOURCES := $(PE_DIR)/pe_acc.sv $(PE_DIR)/pe.sv
TOPLEVEL        := pe
MODULE          ?= test_pe

ifeq ($(SIM),icarus)
    COMPILE_ARGS += -g2012 -Ppe.N_PID=$(N_PID) -Ppe.N_CID=$(N_CID)
endif
ifeq ($(SIM),verilator)
    COMPILE_ARGS += -GN_PID=$(N_PID) -GN_CID=$(N_CID)
    EXTRA_ARGS   += --timing -Wall --Wno-UNOPTFLAT --Wno-WIDTH --Wno-TIMESCALEMOD --Wno-DECLFILENAME
endif

# make config visible to the testbench
export H
export F
export S

include $(shell cocotb-config --makefiles)/Makefile.sim

.PHONY: sweep_pe clean_all
clean_all: clean
	rm -rf __pycache__ *.xml results.xml

sweep_pe:
	@for cfg in "8 3 1" "8 3 2" "6 3 1" "7 3 2" "8 2 2" "10 5 1" "9 3 2"; do \
		set -- $$cfg; \
		echo "=================================================="; \
		echo "  pe  H=$$1 F=$$2 S=$$3"; \
		echo "=================================================="; \
		$(MAKE) -s clean; \
		$(MAKE) -s MODULE=test_pe H=$$1 F=$$2 S=$$3 || exit 1; \
	done
	@echo "All PE sweep configs passed."
