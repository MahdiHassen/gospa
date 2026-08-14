# ==============================================================================
# Makefile -- GoSPA PE cocotb tests
# ==============================================================================
# GoSPA Project -- Team 19, ECE 720 (Spring 2026)
#
# Targets (each does a fresh build -- the previous sim_build is auto-removed):
#   make pe            # single PE   (test_pe.py)
#   make array         # PE array    (test_pe_array.py)
#   make sweep_pe      # single PE across several layer configs
#   make sweep_array   # PE array across several (layer, NUM_PE) configs
#   make clean         # remove build artifacts
#
# Knobs (override on the command line), with defaults:
#   SIM=verilator   simulator: verilator | icarus
#   H=8 F=3 S=1     layer geometry (input HxH, kernel FxF, stride S)
#   NUM_PE=8          PEs in the array (array target only)
#   NUM_MULTS=4       multiplier lanes per PE (matches the synthesized config)
#   WAVES=1         dump a waveform (default 0): ./dump.vcd (verilator) or
#                   ./dump.fst (icarus), written in this directory
#
#   e.g.  make array SIM=icarus H=8 F=3 S=2 NUM_PE=8
#         make pe WAVES=1              # then open ./dump.vcd
#
# Post-synthesis gate-level sim lives in testing/synthesis/Makefile.
# ==============================================================================

SIM       ?= verilator

H         ?= 8
F         ?= 3
S         ?= 1
NUM_PE    ?= 8
NUM_MULTS ?= 4
WAVES     ?= 0


REPO_ROOT := $(realpath ../..)
RTL       := $(REPO_ROOT)/rtl/V2

E     := $(shell echo $$(( ($(H) - $(F)) / $(S) + 1 )))
NUM_PID := $(shell echo $$(( $(F) * $(F) )))
NUM_CID := $(shell echo $$(( $(E) * $(E) )))

export H F S NUM_PE NUM_MULTS WAVES

COMMON_RTL := $(RTL)/common/arith/rca_add.sv $(RTL)/common/arith/mult_pipe.sv \
              $(RTL)/common/sram.sv $(RTL)/pe/pe_lane.sv \
              $(RTL)/pe/pe_mem.sv $(RTL)/pe/pe_fetch.sv $(RTL)/pe/pe.sv

ifeq ($(DUT),array)
    MODULE          := test_pe_array
    TOPLEVEL        := pe_array
    VERILOG_SOURCES := $(COMMON_RTL) $(RTL)/pe/pe_array.sv
    PARAMS          := NUM_PID=$(NUM_PID) NUM_CID=$(NUM_CID) NUM_MULTS=$(NUM_MULTS) NUM_PE=$(NUM_PE)
else
    MODULE          := test_pe
    TOPLEVEL        := pe
    VERILOG_SOURCES := $(COMMON_RTL)
    PARAMS          := NUM_PID=$(NUM_PID) NUM_CID=$(NUM_CID) NUM_MULTS=$(NUM_MULTS)
endif

# Compiler Conditional
ifeq ($(SIM),icarus)
    COMPILE_ARGS += -g2012 $(foreach p,$(PARAMS),-P$(TOPLEVEL).$(p))
else ifeq ($(SIM),verilator)
    COMPILE_ARGS += $(foreach p,$(PARAMS),-G$(p))
    EXTRA_ARGS   += --timing -Wall --Wno-UNOPTFLAT --Wno-WIDTH \
                    --Wno-TIMESCALEMOD --Wno-DECLFILENAME \
                    --Wno-UNUSEDSIGNAL --Wno-EOFNEWLINE
    ifeq ($(WAVES),1)
        VERILATOR_TRACE := 1
    endif
else
    $(error Unknown SIM='$(SIM)' -- use SIM=verilator or SIM=icarus)
endif

# Default cocotb include
include $(shell cocotb-config --makefiles)/Makefile.sim

.PHONY: pe array sweep_pe sweep_array

# $(1) = friendly name, $(2) = DUT value passed to the include block above
define run_dut
	@echo "=================================================="
	@echo "  $(1)  |  SIM=$(SIM)  H=$(H) F=$(F) S=$(S) NUM_PE=$(NUM_PE) NUM_MULTS=$(NUM_MULTS)  WAVES=$(WAVES)"
	@echo "  -> NUM_PID=$(NUM_PID) NUM_CID=$(NUM_CID) (E=$(E))"
	@echo "=================================================="
	@$(MAKE) -s clean
	@$(MAKE) DUT=$(2) results.xml
endef

pe:
	$(call run_dut,pe,)

array:
	$(call run_dut,array,array)

sweep_pe:
	@for cfg in "8 3 1" "8 3 2" "6 3 1" "7 3 2" "8 2 2" "10 5 1" "9 3 2"; do \
		set -- $$cfg; \
		$(MAKE) pe H=$$1 F=$$2 S=$$3 || exit 1; \
	done
	@echo "All PE sweep configs passed."

sweep_array:
	@for cfg in "8 3 1 4" "8 3 1 8" "8 3 2 4" "6 3 1 2" "8 2 2 4" "10 5 1 3"; do \
		set -- $$cfg; \
		$(MAKE) array H=$$1 F=$$2 S=$$3 NUM_PE=$$4 || exit 1; \
	done
	@echo "All PE-array sweep configs passed."
