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

SIM      ?= verilator
TOPLEVEL_LANG ?= verilog

REPO_ROOT := $(realpath ../..)
PE_DIR    := $(REPO_ROOT)/rtl/pe

# -- Layer config (override on command line) ----------------------------------
H       ?= 8
F       ?= 3
S       ?= 1
N_PE    ?= 4
N_MULTS ?= 1

# Derived sizes the PE is parameterized by (kernel positions, output positions).
N_PID := $(shell echo $$(( $(F) * $(F) )))
EE    := $(shell echo $$(( ($(H) - $(F)) / $(S) + 1 )))
N_CID := $(shell echo $$(( $(EE) * $(EE) )))

MODULE ?= test_pe

# Single PE vs PE array: pick top module + sources from MODULE.
ifeq ($(MODULE),test_pe_array)
    VERILOG_SOURCES := $(REPO_ROOT)/rtl/common/sram.sv \
                       $(PE_DIR)/pe_acc.sv $(PE_DIR)/pe.sv $(PE_DIR)/pe_array.sv
    TOPLEVEL        := pe_array
    PARAMS_ICARUS   := -Ppe_array.N_PID=$(N_PID) -Ppe_array.N_CID=$(N_CID) \
                       -Ppe_array.N_PE=$(N_PE) -Ppe_array.N_MULTS=$(N_MULTS)
    PARAMS_VERILATOR:= -GN_PID=$(N_PID) -GN_CID=$(N_CID) \
                       -GN_PE=$(N_PE) -GN_MULTS=$(N_MULTS)
else
    VERILOG_SOURCES := $(REPO_ROOT)/rtl/common/sram.sv \
                       $(PE_DIR)/pe_acc.sv $(PE_DIR)/pe.sv
    TOPLEVEL        := pe
    PARAMS_ICARUS   := -Ppe.N_PID=$(N_PID) -Ppe.N_CID=$(N_CID) -Ppe.N_MULTS=$(N_MULTS)
    PARAMS_VERILATOR:= -GN_PID=$(N_PID) -GN_CID=$(N_CID) -GN_MULTS=$(N_MULTS)
endif

ifeq ($(SIM),icarus)
    COMPILE_ARGS += -g2012 $(PARAMS_ICARUS)
endif
ifeq ($(SIM),verilator)
    COMPILE_ARGS += $(PARAMS_VERILATOR)
    EXTRA_ARGS   += --timing -Wall --Wno-UNOPTFLAT --Wno-WIDTH --Wno-TIMESCALEMOD \
                    --Wno-DECLFILENAME --Wno-UNUSEDSIGNAL --Wno-EOFNEWLINE
endif

# make config visible to the testbench
export H
export F
export S
export N_PE
export N_MULTS

include $(shell cocotb-config --makefiles)/Makefile.sim

.PHONY: sweep_pe sweep_array clean_all
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

# H F S N_PE per config
sweep_array:
	@for cfg in "8 3 1 4" "8 3 1 8" "8 3 2 4" "6 3 1 2" "8 2 2 4" "10 5 1 3"; do \
		set -- $$cfg; \
		echo "=================================================="; \
		echo "  pe_array  H=$$1 F=$$2 S=$$3 N_PE=$$4"; \
		echo "=================================================="; \
		$(MAKE) -s clean; \
		$(MAKE) -s MODULE=test_pe_array H=$$1 F=$$2 S=$$3 N_PE=$$4 || exit 1; \
	done
	@echo "All PE-array sweep configs passed."
