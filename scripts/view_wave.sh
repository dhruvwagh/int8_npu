#!/usr/bin/env bash
# Helper script to open simulation waveforms in GTKWave
# Usage:
#   ./scripts/view_wave.sh           (opens 16x16 array waveform)
#   ./scripts/view_wave.sh 4x4       (opens 4x4 array waveform)
#   ./scripts/view_wave.sh pe        (opens leaf PE waveform)

TARGET="${1:-16x16}"

case "$TARGET" in
  pe)
    FILE="sim_output/tb_npu_pe.vcd"
    ;;
  4x4|array_4x4)
    FILE="sim_output/tb_npu_array_4x4.vcd"
    ;;
  buf|buffer|fetch)
    FILE="sim_output/tb_npu_fetch_buffer.vcd"
    ;;
  seq|sequencer)
    FILE="sim_output/tb_npu_sequencer.vcd"
    ;;
  epi|epilogue|obuf)
    FILE="sim_output/tb_npu_epilogue.vcd"
    ;;
  16x16|array_16x16|array|*)
    FILE="sim_output/tb_npu_array_16x16.vcd"
    ;;
esac

if [ ! -f "$FILE" ]; then
  echo "Error: Waveform file $FILE does not exist. Run simulations first."
  exit 1
fi

echo "Opening $FILE with GTKWave..."
gtkwave "$FILE" &
