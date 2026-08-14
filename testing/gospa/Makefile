# ==============================================================================
# Usage:              
#   make mobilenet                            # mobilenet-shape config (canonical)
#   make test_gospa                           # End-to-end functionality test.
#   make sweep_gospa                          # default simplified config of Alexnet    conv5
#   make sweep_gospa H=10 F=3 S=2 N_PE=8      # custom instance
#   make conv5                                # Runs complete Alexnet conv5 layer. Takes ~5 mins.
#   FCLK_MHZ=500 make sweep_gospa             # quote GMAC/s at 500 M
#   WAVES=1 make mobilenet                    # also dumps a VCD trace
# 
# ==============================================================================

SIM           ?= verilator
TOPLEVEL_LANG ?= verilog
MODULE        ?= test_gospa
TOPLEVEL      := gospa

REPO_ROOT := $(realpath ../..)
RTL_DIR   := $(REPO_ROOT)/rtl/V2

VERILOG_SOURCES := $(RTL_DIR)/common/fifo.sv \
                   $(RTL_DIR)/common/sram.sv \
                   $(RTL_DIR)/common/arith/rca_add.sv \
                   $(RTL_DIR)/common/arith/mult_pipe.sv \
                   $(RTL_DIR)/apu/stage1/position_encode.sv \
                   $(RTL_DIR)/apu/stage1/idgen.sv \
                   $(RTL_DIR)/apu/stage1/apu_stage1.sv \
                   $(RTL_DIR)/apu/stage2/routing.sv \
                   $(RTL_DIR)/apu/stage2/apu_stage2.sv \
                   $(RTL_DIR)/apu/stage1/act_sram_scanner.sv \
                   $(RTL_DIR)/apu/apu.sv \
                   $(RTL_DIR)/pe/pe_lane.sv \
                   $(RTL_DIR)/pe/pe_fetch.sv \
                   $(RTL_DIR)/pe/pe_mem.sv \
                   $(RTL_DIR)/pe/pe.sv \
                   $(RTL_DIR)/pe/pe_array.sv \
                   $(RTL_DIR)/gospa.sv


export PYTHONPATH := $(REPO_ROOT)/sw:$(REPO_ROOT)/testing/ref:$(PYTHONPATH)

# -- Layer / array config (override on the command line) ----------------------
# Default shape is AlexNet conv5 (13x13 in, F=3, S=1, pad=1 -> H=15 padded).
H            ?= 15
F            ?= 3
S            ?= 1
N_PE         ?= 8
N_MULTS      ?= 4
N_ROWS       ?= $(H)
N_NZ_MAX     ?= 225
FIFO_D       ?= 64
FIFOB_D      ?= $(FIFO_D)
STAGE1_BATCH ?= 4
FILL_W       ?= 16
S2_BEATS     ?= 1
CH_PID       ?= 0
DW_COLW      ?= 0
DRAIN_W      ?= 1

# -- Input tiling (sweep only): tile_h x tile_w <= FIFO_D so any H fits fixed
# on-chip memory. With FIFO_D=128 an 11x11 tile (121 entries) is the largest.
TILE_H       ?= 11
TILE_W       ?= 11

ifeq ($(SIM),verilator)
    COMPILE_ARGS += -GH=$(H) -GF=$(F) -GS=$(S) \
                    -GN_PE=$(N_PE) -GN_MULTS=$(N_MULTS) \
                    -GN_ROWS=$(N_ROWS) -GN_NZ_MAX=$(N_NZ_MAX) \
                    -GFIFO_D=$(FIFO_D) -GFIFOB_D=$(FIFOB_D) \
                    -GSTAGE1_BATCH=$(STAGE1_BATCH) \
                    -GFILL_W=$(FILL_W) -GS2_BEATS=$(S2_BEATS) \
                    -GCH_PID=$(CH_PID) -GDW_COLW=$(DW_COLW) \
                    -GDRAIN_W=$(DRAIN_W)
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
                    -Pgospa.FIFO_D=$(FIFO_D) -Pgospa.STAGE1_BATCH=$(STAGE1_BATCH) \
                    -Pgospa.FILL_W=$(FILL_W) -Pgospa.S2_BEATS=$(S2_BEATS) \
                    -Pgospa.CH_PID=$(CH_PID) -Pgospa.DW_COLW=$(DW_COLW)
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
export STAGE1_BATCH
export FILL_W
export S2_BEATS
export CH_PID
export DW_COLW
export DRAIN_W
export TILE_H
export TILE_W

include $(shell cocotb-config --makefiles)/Makefile.sim

# -- Canonical MobileNet first-conv config (8 output channels = 8 PEs) --------
.PHONY: mobilenet clean_all help
mobilenet:
	$(MAKE) -s clean
	$(MAKE) -s MODULE=test_gospa SIM=$(SIM) \
		H=10 F=3 S=2 N_PE=8 N_MULTS=4 \
		N_ROWS=10 N_NZ_MAX=1024 FIFO_D=2048

# -- Real-workload apple run: MobileNetV2 first conv, 80x80 apple input -------
# N_PE=8 (default) = 32 mults, 4 group passes; N_PE=32 = one pass, 128 mults.
# S2_BEATS=4: router outruns the PEs 4x, FIFO-B backlog hides the front end
# (FIFO_D=4096 so the backlog fits).
.PHONY: apple
apple:
	$(MAKE) -s clean
	$(MAKE) -s MODULE=test_gospa_apple SIM=$(SIM) \
		H=80 F=3 S=2 N_PE=$(N_PE) N_MULTS=4 \
		N_ROWS=80 N_NZ_MAX=8192 FIFO_D=4096 \
		STAGE1_BATCH=$(STAGE1_BATCH) FILL_W=$(FILL_W) S2_BEATS=4

# -- Real MobileNetV2 layers beyond conv1 (real apple-80 tensors) -------------
.PHONY: mobilenet_dw1 mobilenet_pw2 mobilenet_pw3
mobilenet_dw1:
	$(MAKE) -s clean
	LAYER_IDX=1 $(MAKE) -s MODULE=test_gospa_mobilenet SIM=$(SIM) \
		H=40 F=3 S=1 N_PE=8 N_MULTS=4 N_ROWS=40 N_NZ_MAX=2048 FIFO_D=4096 \
		STAGE1_BATCH=$(STAGE1_BATCH) FILL_W=$(FILL_W) S2_BEATS=4

mobilenet_pw2:
	$(MAKE) -s clean
	LAYER_IDX=2 $(MAKE) -s MODULE=test_gospa_mobilenet SIM=$(SIM) \
		H=40 F=1 S=1 N_PE=8 N_MULTS=4 N_ROWS=40 N_NZ_MAX=2048 FIFO_D=4096 \
		STAGE1_BATCH=$(STAGE1_BATCH) FILL_W=$(FILL_W) S2_BEATS=4

mobilenet_pw3:
	$(MAKE) -s clean
	LAYER_IDX=3 $(MAKE) -s MODULE=test_gospa_mobilenet SIM=$(SIM) \
		H=40 F=1 S=1 N_PE=8 N_MULTS=4 N_ROWS=40 N_NZ_MAX=2048 FIFO_D=4096 \
		STAGE1_BATCH=$(STAGE1_BATCH) FILL_W=$(FILL_W) S2_BEATS=4

# -- Base correctness tests (test_gospa.py), single-bank input-tiled ----------
.PHONY: test_gospa
test_gospa:
	$(MAKE) -s clean
	$(MAKE) -s MODULE=test_gospa SIM=$(SIM) \
		H=$(H) F=$(F) S=$(S) N_PE=$(N_PE) N_MULTS=$(N_MULTS) \
		N_ROWS=$(TILE_H) N_NZ_MAX=128 FIFO_D=128 \
		STAGE1_BATCH=$(STAGE1_BATCH) TILE_H=$(TILE_H) TILE_W=$(TILE_W)

# -- Sparsity sweep -- writes gospa_perf.txt (perf_model --sweep-grid style) ---
.PHONY: sweep_gospa
sweep_gospa:
	$(MAKE) -s clean
	FCLK_MHZ=$(FCLK_MHZ) $(MAKE) -s MODULE=test_gospa_sweep SIM=$(SIM) \
		H=$(H) F=$(F) S=$(S) N_PE=$(N_PE) N_MULTS=$(N_MULTS) \
		N_ROWS=$(TILE_H) N_NZ_MAX=128 FIFO_D=128 \
		STAGE1_BATCH=$(STAGE1_BATCH) TILE_H=$(TILE_H) TILE_W=$(TILE_W)

# -- AlexNet conv5 sweep: same shape as sweep_gospa's default (H=15 F=3 S=1),
# but PASSES=384 to match conv5's real Cin (full accumulation depth).
.PHONY: conv5
conv5:
	$(MAKE) -s sweep_gospa PASSES=384

clean_all: clean
	rm -rf __pycache__ *.xml results.xml dump.vcd

help:
	@echo ""
	@echo "goSPA full-accelerator cocotb tests"
	@echo "-----------------------------------"
	@echo "  make test_gospa                       Base correctness tests (input-tiled)"
	@echo "  make mobilenet                        MobileNet first-conv (8 PEs x 4 lanes)"
	@echo "  make sweep_gospa                      Sparsity sweep, default grid (conv5 shape)"
	@echo "  make conv5                            Same, PASSES=384 (conv5's real Cin)"
	@echo "  WAVES=1 make mobilenet                Same, dumps dump.vcd"
	@echo "  make MODULE=test_gospa H=... F=... ...   Custom layer"
	@echo ""
