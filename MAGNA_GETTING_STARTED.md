# Steps for Ubuntu 22.04:

# If you already have metal dependencies installed, you may skip the dependencies installation and start from [install metal](###install-metal)

## Install dependencies
```
wget https://github.com/tenstorrent/tt-metal/blob/ssinghal/gstreamer-flow/install_dependencies.sh
chmod a+x install_dependencies.sh
sudo apt-get install jq libcapstone-dev
sudo ./install_dependencies.sh
sudo apt install dkms
sudo apt install graphviz
git clone https://github.com/tenstorrent/tt-kmd.git
cd tt-kmd
sudo dkms add .
sudo dkms install "tenstorrent/$(./tools/current-version)"
sudo modprobe tenstorrent
cd ..
```

## tt-flash
```
sudo apt install python3.10-venv

curl https://sh.rustup.rs -sSf | sh  ; . "$HOME/.cargo/env"     ## Install cargo if needed

git clone https://github.com/tenstorrent/tt-flash.git
cd tt-flash
make build
sudo reboot
.env/bin/tt-flash --version   # Check if tt-flash is correctly installed
file_name=$(curl -s "https://raw.githubusercontent.com/tenstorrent/tt-firmware/main/latest.fwbundle")
curl -L -o "$file_name" "https://github.com/tenstorrent/tt-firmware/raw/main/$file_name"
.env/bin/tt-flash flash --fw-tar $file_name
cd ..
```
## tt-smi
```
source tt-flash/.env/bin/activate
pip install git+https://github.com/tenstorrent/tt-smi
tt-smi # To check if tt-smi is installed correctly
```


## install metal
### install metal from the branch:  ssinghal/gstreamer-demo

```
git clone https://github.com/tenstorrent/tt-metal.git --recurse-submodules -b ssinghal/gstreamer-flow
cd tt-metal-magna
./create_venv.sh
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
source python_env/bin/activate
./build_metal.sh -p
```


### UFLD model:

1. Download the ```ufld.tar``` from the shared drive and place it in metal directory under tt-metal.

2. Unzip model implementations. This will place the files on the expected paths as needed.
```
tar -xvf ufld.tar
```

### LRASPP model:

1. Download the ```lraspp.tar``` deom rge shared drive and place it in metal directory under tt-metal.

2. Unzip model implementations. This will place the files on the expected paths as needed.
```
tar -xvf lraspp.tar
```

### DEIT model:

1. Download the ```deit.tar``` deom rge shared drive and place it in metal directory under tt-metal.

2. Unzip model implementations. This will place the files on the expected paths as needed.
```
tar -xvf deit.tar
```


### Please follow the recommended tests in the following MAGNA_READMEs for reproducing our reported results.
```
./models/demos/wormhole/deit_tiny/MAGNA_README.md
```
```
./models/experimental/ufld_v2_rn18like/MAGNA_README.md
```
```
./models/experimental/lraspp/MAGNA_README.md
```

## GStreamer PLUGIN:
### Please note, there is some conflict between python_env to run Gstreamer plugin. You need to install new virtual environment to run gstreamer plugin.

## Build python enviroment for GStreamer with --site-packages option.
```
deactivate
./create_venv_gstreamer.sh
source python_env_gstreamer/bin/activate


sudo apt install python3-gi python3-gi-cairo
sudo apt install python3-gst-1.0 gstreamer1.0-python3-plugin-loader
sudo apt install gstreamer1.0-tool
pip install graphviz numpy_ringbuffer
sudo apt install ubuntu-restricted-extras
sudo apt-get install -y gstreamer1.0-plugins-ugly gstreamer1.0-plugins-bad gstreamer1.0-libav

export GST_PLUGIN_PATH=$PWD/plugin:$PWD/plugins

rm ~/.cache/gstreamer-1.0/registry.x86_64.bin
gst-inspect-1.0 python
```

### OUTPUT:
```
Plugin Details:
  Name                     python
  Description              loader for plugins written in python
  Filename                 /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstpython.so
  Version                  1.20.1
  License                  LGPL
  Source module            gst-python
  Binary package           GStreamer Python
  Origin URL               http://gstreamer.freedesktop.org

  ExampleTransform: ExampleTransform Python
  audioplot: AudioPlotFilter
  deit: Deit Python
  identity_py: Identity Python
  lraspp: LRASPP Python
  mysink: CustomSink
  mytextprepender: MyTextPrepender Python
  py_audiotestsrc: CustomSrc
  py_videomixer: Videomixer
  ufld: UFLD Python

  10 features:
  +-- 10 elements
```
#### Plese note, in case you run into dependecy installation errors and you resolve the errors, make sure to delete the cache by running:
```
rm ~/.cache/gstreamer-1.0/registry.x86_64.bin
```

## Install gstreamer dependencies for the old flow of GStreamer (GStreamer within a python script/not the plug-in)
### you will not need a separate python enviroment for the old flow.
### If still intested in using GStremear as a script and not as a plug-in, you may install GStreamer dependcies within the oringial python_env via:

```
source python_env/bin/activate
sudo apt install libcairo2-dev libxt-dev libgirepository1.0-dev
pip install pycairo PyGObject    # Might show some error but eventaully finds the correct pygobject version # (3.50.0 for me)
sudo apt-get install gstreamer-1.0
sudo apt install python3-gst-1.0
sudo apt install ubuntu-restricted-extras
sudo apt-get install -y gstreamer1.0-plugins-ugly gstreamer1.0-plugins-bad gstreamer1.0-libav
```

## then run:
```
python ufld.py
```
```
python lraspp.py <batch-size>
```
```
python deit.py <batch-size>
```
