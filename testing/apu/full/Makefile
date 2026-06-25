# ==============================================================================
# Makefile -- GoSPA Full APU (Stage 1 + Stage 2) cocotb Tests
# ==============================================================================
# GoSPA Project -- Team 19, ECE 720 (Spring 2026)
#
# Usage:
#   make MODULE=test_apu                              # default H=8 F=3 S=1 N_PE=4
#   make MODULE=test_apu SIM=icarus                   # use Icarus
#   make MODULE=test_apu H=8 F=3 S=2 N_PE=8           # custom layer
#   make sweep_apu                                    # run several configs
#
# Dependencies: pip install cocotb
# ==============================================================================

SIM           ?= verilator
TOPLEVEL_LANG ?= verilog
MODULE        ?= test_apu
TOPLEVEL      := apu

# -- Paths --------------------------------------------------------------------
REPO_ROOT := $(realpath ../../..)
RTL_DIR   := $(REPO_ROOT)/rtl

VERILOG_SOURCES := $(RTL_DIR)/common/fifo.sv \
                   $(RTL_DIR)/apu/stage1/csr_decode.sv \
                   $(RTL_DIR)/apu/stage1/zero_act.sv \
                   $(RTL_DIR)/apu/stage1/position_encode.sv \
                   $(RTL_DIR)/apu/stage1/idgen.sv \
                   $(RTL_DIR)/apu/stage1/apu_stage1.sv \
                   $(RTL_DIR)/apu/stage2/routing.sv \
                   $(RTL_DIR)/apu/stage2/apu_stage2.sv \
                   $(RTL_DIR)/apu/apu.sv

# -- Functional-model import: the TB does `import functional` from sw/ -------
export PYTHONPATH := $(REPO_ROOT)/sw:$(PYTHONPATH)

# -- Layer config (override on the command line) -----------------------------
H     ?= 8
F     ?= 3
S     ?= 1
N_PE  ?= 4

ifeq ($(SIM),icarus)
    COMPILE_ARGS += -g2012 -gsupported-assertions \
                    -Papu.H=$(H) -Papu.F=$(F) -Papu.S=$(S) -Papu.N_PE=$(N_PE)
endif
ifeq ($(SIM),verilator)
    COMPILE_ARGS += -GH=$(H) -GF=$(F) -GS=$(S) -GN_PE=$(N_PE)
    EXTRA_ARGS   += --timing -Wall \
                    --Wno-UNOPTFLAT --Wno-WIDTH --Wno-TIMESCALEMOD \
                    --Wno-DECLFILENAME --Wno-UNUSEDSIGNAL --assert
endif

export H
export F
export S
export N_PE

include $(shell cocotb-config --makefiles)/Makefile.sim

# -- Sweep across representative layer configs --------------------------------
.PHONY: sweep_apu clean_all help
sweep_apu:
	@for cfg in "3 2 1 2" "8 3 1 4" "8 3 2 4" "9 3 2 8" "8 2 2 4" "10 5 1 4"; do \
		set -- $$cfg; \
		echo "=================================================="; \
		echo "  apu  H=$$1 F=$$2 S=$$3 N_PE=$$4"; \
		echo "=================================================="; \
		$(MAKE) -s clean; \
		$(MAKE) -s MODULE=test_apu SIM=$(SIM) H=$$1 F=$$2 S=$$3 N_PE=$$4 || exit 1; \
	done
	@echo "All APU sweep configs passed."

clean_all: clean
	rm -rf __pycache__ *.xml results.xml

help:
	@echo ""
	@echo "GoSPA Full APU -- cocotb test runner"
	@echo "------------------------------------"
	@echo "  make MODULE=test_apu                       Default (H=8 F=3 S=1 N_PE=4)"
	@echo "  make MODULE=test_apu SIM=icarus            Use Icarus"
	@echo "  make MODULE=test_apu H=8 F=3 S=2 N_PE=8    Custom layer"
	@echo "  make sweep_apu                             Several configs"
	@echo ""
