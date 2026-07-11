# ==============================================================================
# Makefile -- goSPA full-accelerator cocotb tests
# ==============================================================================
# GoSPA Project -- Team 19, ECE 720 (Spring 2026)
#
# Usage:
#   make MODULE=test_gospa                    # default H=10 F=3 S=2 N_PE=4 N_MULTS=4
#   make mobilenet                            # mobilenet-shape config (canonical)
#   WAVES=1 make mobilenet                    # also dumps a VCD trace
# ==============================================================================

SIM           ?= verilator
TOPLEVEL_LANG ?= verilog
MODULE        ?= test_gospa
TOPLEVEL      := gospa

REPO_ROOT := $(realpath ../..)
RTL_DIR   := $(REPO_ROOT)/rtl

VERILOG_SOURCES := $(RTL_DIR)/common/fifo.sv \
                   $(RTL_DIR)/common/sram.sv \
                   $(RTL_DIR)/apu/stage1/zero_act.sv \
                   $(RTL_DIR)/apu/stage1/position_encode.sv \
                   $(RTL_DIR)/apu/stage1/idgen.sv \
                   $(RTL_DIR)/apu/stage1/apu_stage1.sv \
                   $(RTL_DIR)/apu/stage2/routing.sv \
                   $(RTL_DIR)/apu/stage2/apu_stage2.sv \
                   $(RTL_DIR)/apu/act_sram_scanner.sv \
                   $(RTL_DIR)/apu/wsp_file.sv \
                   $(RTL_DIR)/apu/apu.sv \
                   $(RTL_DIR)/pe/pe_lane.sv \
                   $(RTL_DIR)/pe/pe_window.sv \
                   $(RTL_DIR)/pe/pe_wmem.sv \
                   $(RTL_DIR)/pe/pe.sv \
                   $(RTL_DIR)/pe/pe_array.sv \
                   $(RTL_DIR)/gospa.sv

# Make the functional model and the MobileNet loader importable from the TB.
export PYTHONPATH := $(REPO_ROOT)/sw:$(REPO_ROOT)/testing/ref:$(PYTHONPATH)

# -- Layer / array config (override on the command line) ----------------------
H        ?= 10
F        ?= 3
S        ?= 2
N_PE     ?= 4
N_MULTS  ?= 4
N_ROWS   ?= $(H)
N_NZ_MAX ?= 1024
FIFO_D   ?= 64

ifeq ($(SIM),verilator)
    COMPILE_ARGS += -GH=$(H) -GF=$(F) -GS=$(S) \
                    -GN_PE=$(N_PE) -GN_MULTS=$(N_MULTS) \
                    -GN_ROWS=$(N_ROWS) -GN_NZ_MAX=$(N_NZ_MAX) \
                    -GFIFO_D=$(FIFO_D)
    EXTRA_ARGS   += --timing -Wall \
                    --Wno-UNOPTFLAT --Wno-WIDTH --Wno-TIMESCALEMOD \
                    --Wno-DECLFILENAME --Wno-UNUSEDSIGNAL --Wno-EOFNEWLINE \
                    --Wno-SYNCASYNCNET \
                    --assert
endif
ifeq ($(SIM),icarus)
    COMPILE_ARGS += -g2012 -gsupported-assertions \
                    -Pgospa.H=$(H) -Pgospa.F=$(F) -Pgospa.S=$(S) \
                    -Pgospa.N_PE=$(N_PE) -Pgospa.N_MULTS=$(N_MULTS) \
                    -Pgospa.N_ROWS=$(N_ROWS) -Pgospa.N_NZ_MAX=$(N_NZ_MAX) \
                    -Pgospa.FIFO_D=$(FIFO_D)
endif

ifeq ($(WAVES),1)
    COMPILE_ARGS += +define+DUMP_VCD
    ifeq ($(SIM),verilator)
        EXTRA_ARGS += --trace --trace-structs
    endif
endif

export H
export F
export S
export N_PE
export N_MULTS
export N_ROWS
export N_NZ_MAX
export FIFO_D

include $(shell cocotb-config --makefiles)/Makefile.sim

# -- Canonical MobileNet first-conv config (32 channels = 8 PEs x 4 lanes) ----
.PHONY: mobilenet clean_all help
mobilenet:
	$(MAKE) -s clean
	$(MAKE) -s MODULE=test_gospa SIM=$(SIM) \
		H=10 F=3 S=2 N_PE=8 N_MULTS=4 \
		N_ROWS=10 N_NZ_MAX=1024 FIFO_D=2048

clean_all: clean
	rm -rf __pycache__ *.xml results.xml dump.vcd

help:
	@echo ""
	@echo "goSPA full-accelerator cocotb tests"
	@echo "-----------------------------------"
	@echo "  make MODULE=test_gospa                Default config (small)"
	@echo "  make mobilenet                        MobileNet first-conv (8 PEs x 4 lanes)"
	@echo "  WAVES=1 make mobilenet                Same, dumps dump.vcd"
	@echo "  make MODULE=test_gospa H=... F=... ...   Custom layer"
	@echo ""
