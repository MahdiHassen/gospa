#!/usr/bin/env bash
# ==============================================================================
# run_sim.sh — GoSPA FIFO simulation runner
# ==============================================================================
# GoSPA Project — Team 19, ECE 720 (Spring 2026)
#
# Usage:
#   ./run_sim.sh                  # Verilator (default)
#   ./run_sim.sh verilator        # Verilator explicitly
#   ./run_sim.sh iverilog         # Icarus Verilog
#   ./run_sim.sh wave             # run then open waveform in GTKWave
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SIM="${1:-verilator}"

print_banner() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  GoSPA FIFO Simulation Runner                            ║"
    echo "║  Simulator : $SIM"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""
}

check_tool() {
    local tool="$1"
    if ! command -v "$tool" &>/dev/null; then
        echo "[ERROR] '$tool' not found in PATH."
        echo "  Verilator : https://github.com/verilator/verilator (v5.x required)"
        echo "  Icarus    : sudo apt install iverilog  OR  brew install icarus-verilog"
        exit 1
    fi
}

case "$SIM" in
    verilator)
        print_banner
        check_tool verilator
        make SIM=verilator
        ;;
    iverilog)
        print_banner
        check_tool iverilog
        check_tool vvp
        make SIM=iverilog
        ;;
    wave)
        print_banner
        check_tool verilator
        make SIM=verilator
        echo ""
        echo "[GTKWave] Opening waveform..."
        make wave
        ;;
    clean)
        make clean
        echo "Clean done."
        ;;
    help|--help|-h)
        echo ""
        echo "Usage: ./run_sim.sh [verilator|iverilog|wave|clean|help]"
        echo ""
        echo "  verilator   Run with Verilator 5.x (default)"
        echo "  iverilog    Run with Icarus Verilog"
        echo "  wave        Run with Verilator then open GTKWave"
        echo "  clean       Remove build artefacts"
        echo ""
        ;;
    *)
        echo "[ERROR] Unknown option '$SIM'. Run './run_sim.sh help' for usage."
        exit 1
        ;;
esac
