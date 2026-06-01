# ==============================================================================
# Makefile — GoSPA FIFO Testbench
# ==============================================================================
# GoSPA Project — Team 19, ECE 720 (Spring 2026)
#
# Usage:
#   make                  # simulate with Verilator (default)
#   make SIM=iverilog     # simulate with Icarus Verilog
#   make wave             # open last VCD in GTKWave
#   make lint             # run Verilator lint only (no simulation)
#   make clean            # remove all build artifacts
#   make help             # show this message
#
# Paths are relative to this Makefile's location:
#   RTL lives in  ../../rtl/common/
#   TB  lives in  ./
# ==============================================================================

# ---- Simulator selection (override on command line) --------------------------
SIM ?= verilator

# ---- Paths -------------------------------------------------------------------
REPO_ROOT := $(realpath ../..)
RTL_DIR   := $(REPO_ROOT)/rtl/common
TB_DIR    := $(CURDIR)

# ---- Source files ------------------------------------------------------------
RTL_SRCS  := $(RTL_DIR)/fifo.sv
TB_TOP    := fifo_tb
TB_SRC    := $(TB_DIR)/$(TB_TOP).sv

# ---- Output artefacts --------------------------------------------------------
VCD_FILE  := $(TB_TOP).vcd
VERILATOR_EXE := obj_dir/$(TB_TOP)

# ==============================================================================
# Verilator
# ==============================================================================
VERILATOR       := verilator
VERILATOR_FLAGS := \
    --binary            \
    --timing            \
    -sv                 \
    --trace             \
    --top-module $(TB_TOP) \
    -Wall               \
    --Wno-UNOPTFLAT     \
    --Wno-WIDTH         \
    --Wno-TIMESCALEMOD  \
    --Wno-BLKSEQ        \
    --Wno-UNUSEDPARAM   \
    -o $(TB_TOP)

# ==============================================================================
# Icarus Verilog
# ==============================================================================
IVERILOG       := iverilog
IVERILOG_FLAGS := -g2012 -Wall
VVP            := vvp
IVERILOG_OUT   := $(TB_TOP).vvp

# ==============================================================================
# GTKWave
# ==============================================================================
GTKWAVE := gtkwave

# ==============================================================================
# Targets
# ==============================================================================

.PHONY: all sim sim_verilator sim_iverilog lint wave clean help

all: sim

# ---- Dispatch to the selected simulator -------------------------------------
sim:
ifeq ($(SIM),verilator)
	@$(MAKE) --no-print-directory sim_verilator
else ifeq ($(SIM),iverilog)
	@$(MAKE) --no-print-directory sim_iverilog
else
	$(error Unknown simulator '$(SIM)'. Use SIM=verilator or SIM=iverilog)
endif

# ---- Verilator flow ---------------------------------------------------------
sim_verilator: $(RTL_SRCS) $(TB_SRC)
	@echo "──────────────────────────────────────────"
	@echo " [Verilator] Compiling..."
	@echo "──────────────────────────────────────────"
	$(VERILATOR) $(VERILATOR_FLAGS) $(RTL_SRCS) $(TB_SRC)
	@echo ""
	@echo "──────────────────────────────────────────"
	@echo " [Verilator] Running simulation..."
	@echo "──────────────────────────────────────────"
	$(VERILATOR_EXE)

# ---- Icarus Verilog flow ----------------------------------------------------
sim_iverilog: $(RTL_SRCS) $(TB_SRC)
	@echo "──────────────────────────────────────────"
	@echo " [iverilog] Compiling..."
	@echo "──────────────────────────────────────────"
	$(IVERILOG) $(IVERILOG_FLAGS) -o $(IVERILOG_OUT) $(RTL_SRCS) $(TB_SRC)
	@echo ""
	@echo "──────────────────────────────────────────"
	@echo " [iverilog] Running simulation..."
	@echo "──────────────────────────────────────────"
	$(VVP) $(IVERILOG_OUT)

# ---- Lint only (RTL sources only -- TB is excluded intentionally)
# Testbenches use timing constructs (@posedge, #delay) that are not
# synthesizable and will always trip --lint-only without --timing.
# Lint the RTL to check it is clean for synthesis; simulate to test the TB.
lint: $(RTL_SRCS)
	@echo " [Verilator] Lint RTL only (TB excluded)..."
	$(VERILATOR) --lint-only -sv \
	    --Wno-TIMESCALEMOD \
	    $(RTL_SRCS)
	@echo " Lint clean."

# ---- Open waveform ----------------------------------------------------------
wave:
	@if [ ! -f $(VCD_FILE) ]; then \
	    echo "No VCD found — run 'make' first."; exit 1; \
	fi
	$(GTKWAVE) $(VCD_FILE) &

# ---- Clean ------------------------------------------------------------------
clean:
	rm -rf obj_dir $(IVERILOG_OUT) $(VCD_FILE) *.fst *.lxt *.log

# ---- Help -------------------------------------------------------------------
help:
	@echo ""
	@echo "GoSPA FIFO Testbench — make targets"
	@echo "────────────────────────────────────"
	@echo "  make                  Compile + simulate (Verilator, default)"
	@echo "  make SIM=iverilog     Compile + simulate (Icarus Verilog)"
	@echo "  make lint             Verilator lint check only"
	@echo "  make wave             Open $(VCD_FILE) in GTKWave"
	@echo "  make clean            Remove all build artefacts"
	@echo ""
	@echo "RTL source : $(RTL_DIR)"
	@echo "TB  source : $(TB_DIR)"
	@echo ""
