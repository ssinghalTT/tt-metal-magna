# Running tests

## Basic examples

FF1 without gelu: `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest tests/didt/test_ff1_matmul.py::test_ff1_matmul -k "without_gelu and 2chips"`

FF1 with gelu: `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest tests/didt/test_ff1_matmul.py::test_ff1_matmul -k "with_gelu and 2chips"`

LM head: `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest tests/didt/test_lm_head_matmul.py::test_lm_head_matmul -k "2chips"`

Resnet Convolution: `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest tests/didt/test_resnet_conv.py::test_resnet_conv -k "2chips"`


TT_MM_THROTTLE_PCT=50

python -m tracy -r -m "pytest tests/didt/test_lm_head_matmul.py::test_lm_head_matmul -k \"1chips\" --iterations 1"


## Variations

### Supported systems

We support N150, N300, T3000, Galaxy (TG) systems, and single chip Blackhole. To choose the system, pass in the following parametrization ids (as shown in the example commands):
- 1chips
- 2chips
- 8chips
- galaxy

NOTE: If running on Galaxy system, remove the WH_ARCH_YAML env variable from the command.

### Targetting specific device

On all multi-device systems, you can target a specific device using its ID in the parametrization `logical_chip_{id}_`:

`WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest tests/didt/test_ff1_matmul.py::test_specific_chip_ff1_matmul -k "without_gelu and 8chips and logical_chip_3_"`

### Targetting specific board

On T3000 systems, you can target a specific board (local and remote chip together) using the ID of the local device in the parametrization `board_id_{id}`:

`WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest tests/didt/test_ff1_matmul.py::test_specific_board_ff1_matmul -k "without_gelu and 8chips and board_id_2"`

### Iterations

By default, we run 100000 iterations of the loop, but you can override that behavior using --iterations option:

`WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest tests/didt/test_ff1_matmul.py::test_ff1_matmul -k "without_gelu and 2chips" --iterations 5000000`

### Determinism

If you wish to check if the output is deterministic, simply pass in the --determinism-check-iteration option - the option tells on how many iterations we do the determinism check. Example:

`WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest tests/didt/test_ff1_matmul.py::test_ff1_matmul -k "without_gelu and 2chips" --determinism-check-iterations 50`

## Blackhole

Tests are only supported with `1chips` ID because multi-chip/board BH has not been brought up yet.

By default the Blackhole workload compute grid is 13x10 (1 column is reserved for fast-dispatch). Adding `--simulate_bh_harvesting` will simulate 2 column harvesting on Blackhole by reducing the compute grid to 11x10.

`WH_ARCH_YAML` is not supported  but setting env var `TT_METAL_ETH_DISPATCH=1` will enable the unharvested workload to run on 14x10 compute grid. Running with `--simulate_bh_harvesting` is not supported with `TT_METAL_ETH_DISPATCH`


## Legacy commands

For backwards compatibility, we still support the commands used so far and their old behavior:

FF1 without gelu: `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/experimental/falcon_7b/tests/test_reproduce_hang_matmul.py -k "test_reproduce_matmul_2d_hang and ff1-hang and 8chips"`

FF1 with gelu: `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest tests/didt/test_sharded_ff1.py -k "test_reproduce_matmul_2d_hang and 8chips"`

LM head: `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/falcon7b/tests/test_falcon_hang.py -k "test_reproduce_lm_head_nd_32 and 8chips"`


reading core (9, 5):
00: 0x00015240 => 0x00baba01
01: 0x00015244 => 0x00000000
02: 0x00015248 => 0xa4bc311b
03: 0x0001524c => 0xa3903f60

TT_MATMUL_STAGGER_TYPE=1 TT_MATMUL_STAGGER_VALUE=500

pytest tests/didt/test_resnet_conv.py::test_resnet_conv -k "1chips" --iterations 1
python -m tracy -r -m "pytest tests/didt/test_resnet_conv.py::test_resnet_conv -k \"1chips\" --iterations 1"

export TT_METAL_ARC_DEBUG_BUFFER_SIZE=16000000

// max
export TT_METAL_ARC_DEBUG_BUFFER_SIZE=100000000

ttp load_arc_dbg_fw
ttp arc_logger --args start=1,pmon_id=0,ro_id=27,stop_on_flatline=1
bin/arc-dbg.py read
python3 -c 'print("\n".join(f"{byte}" for byte in open("out-0-0.bin", "rb").read()))' > out.csv

// bgd-lab-09 board 1
RO_SEL=27
PMON_SEL=0
a = 0.0410675886821849
b = -13.470956232050119
c = 1801.0176411251837


ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_bmm_tilize_col_major_out_blocks.cpp
'ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp'
'ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp'
'ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_conv_activations_padded_with_halo_3x3_weights_v2.cpp'

['ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp';
'ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_receiver.cpp';
'ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp']


sudo apt update
sudo apt install software-properties-common build-essential libyaml-cpp-dev libboost-all-dev libhwloc-dev libzmq3-dev libgtest-dev libgmock-dev xxd -y
pip uninstall -y debuda
pip install git+https://github.com/tenstorrent/tt-debuda.git

ppopovic/supersynced_first_block




# Adding another suspected repro test

`tests/didt/matmul_test_base.py` defines a base class for all tests that encapsulates common behavior - how we run iterations, deallocate, check determinism, sync, etc.  To add a new test, create a new file under the same directory, and then either:
- instantiate object of the base class in case you don't need to change any behavior, just populate dimensions, configs etc (example in `test_ff1_matmul.py`)
- extend the base class to override any behavior that needs to be changed (for now we allow to change the way we generate activations & weights, and setting the seed), and then instantiate object of the new class
