# Install

These instructions will guide you through the installation of TT-Metalium and TT-NN. Note that this page assumes you have already read and followed the instructions in the [Starting Guide](https://docs.tenstorrent.com/getting-started/README.html) and wish to install TT-Metalium manually.

> **Take Note!**
>
> If you are using a release version of this software, check installation instructions packaged with it.
> You can find them in either the release assets for that version, or in the source files for that [version tag](https://github.com/tenstorrent/tt-metal/tags).

There are three options for installing TT-Metalium:

- [Option 1: From Source](#option-1-from-source)

  Installing from source gets developers closer to the metal and the source code.

- [Option 2: From Docker Release Image](#option-2-from-docker-release-image)

  Installing from Docker Release Image is the quickest way to access our APIs and to start running AI models.

- [Option 3: From Wheel](#option-3-from-wheel)

  Install from wheel as an alternative method to get quick access to our APIs and to running AI models.

---

# Option 1: From Source
Install from source if you are a developer who wants to be close to the metal and the source code. Recommended for running the demo models.

## Step 1. Clone the Repository:

```sh
git clone https://github.com/tenstorrent/tt-metal.git --recurse-submodules
```

## Step 2. Invoke our Build Scripts:

```
./build_metal.sh
```

- (recommended) For an out-of-the-box virtual environment to use, execute:
```
./create_venv.sh
source python_env/bin/activate
```

- (optional) Software dependencies for profiling use:
  - Install dependencies:
  ```sh
  sudo apt install pandoc libtbb-dev libcapstone-dev pkg-config
  ```

  - Download and install [Doxygen](https://www.doxygen.nl/download.html), (v1.9 or higher, but less than v1.10)

- Continue to [You Are All Set!](#you-are-all-set)

---

# Option 2: From Docker Release Image
Installing from Docker Release Image is the quickest way to access our APIs and to start running AI models.

Download the latest Docker release from our [Docker registry](https://github.com/orgs/tenstorrent/packages?q=tt-metalium-ubuntu&tab=packages&q=tt-metalium-ubuntu-22.04-release-amd64) page

```sh
docker pull ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc
docker run -it --rm -v /dev/hugepages-1G:/dev/hugepages-1G --device /dev/tenstorrent ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc bash
```

- For more information on the Docker Release Images, visit our [Docker registry page](https://github.com/orgs/tenstorrent/packages?q=tt-metalium-ubuntu&tab=packages&q=tt-metalium-ubuntu-22.04-release-amd64).

- Continue to [You Are All Set!](#you-are-all-set)

---

# Option 3: From Wheel
Install from wheel for quick access to our APIs and to get an AI model running

## Step 1. Download and Install the Latest Wheel:

- Navigate to our [releases page](https://github.com/tenstorrent/tt-metal/releases/latest) and download the latest wheel file for the Tenstorrent card architecture you have installed.

- Install the wheel using your Python environment manager of choice. For example, to install with `pip`:

  ```sh
  pip install <wheel_file.whl>
  ```

## Step 2. (For models users only) Set Up Environment for Models:

To try our pre-built models in `models/`, you must:

  - Install their required dependencies
  - Set appropriate environment variables
  - Set the CPU performance governor to ensure high performance on the host

- This is done by executing the following:
  ```sh
  export PYTHONPATH=$(pwd)
  pip install -r tt_metal/python_env/requirements-dev.txt
  sudo apt-get install cpufrequtils
  sudo cpupower frequency-set -g performance
  ```

---

# You are All Set!

## To verify your installation, try executing a programming example:

- First, set the following environment variables:

  - Run the appropriate command for the Tenstorrent card you have installed:

  | Card             | Command                              |
  |------------------|--------------------------------------|
  | Grayskull        | ```export ARCH_NAME=grayskull```     |
  | Wormhole         | ```export ARCH_NAME=wormhole_b0```   |
  | Blackhole        | ```export ARCH_NAME=blackhole```     |

  - Run:
  ```
  export TT_METAL_HOME=$(pwd)
  export PYTHONPATH=$(pwd)
  ```

 - Finally, try running some examples. Visit Tenstorrent's [TT-NN Basic Examples Page](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/usage.html#basic-examples) or get started with [Simple Kernels on TT-Metalium](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/index.html)

---

# Interested in Contributing?
- For more information on development and contributing, visit Tenstorrent's [CONTRIBUTING.md page](https://github.com/tenstorrent/tt-metal/blob/main/CONTRIBUTING.md).
