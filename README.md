
# Binaural Audio Renderer

A high-performance spatial audio processing tool designed for real-time binaural synthesis.

## Prerequisites

Install the following dependencies using [Homebrew](https://brew.sh):

```bash
brew install fftw cmake hdf5 libsndfile rtaudio yaml-cpp
```

### Required Library: libmysofa
This project requires libmysofa for handling SOFA files. Build and install it from source:

```bash
git clone https://github.com
cd libmysofa
mkdir build && cd build
cmake .. -DBUILD_TESTS=OFF
make
sudo make install
```

## Installation
Once the dependencies are installed, clone this repository and compile the project:

```bash
mkdir build && cd build
cmake ..
make
```

## Usage
Execute the renderer by modifying the file config.yaml and running:

```bash
./binaural <path to config.yaml>
```

### Config.yaml Arguments:
**azimuth:** The horizontal angle in degrees for the initial position of the source.

**elevation**: The vertical angle in degrees for the initial position of the source.

**distance**: Initial distance to the source in meters.

**velocity**: Velocity of the source in [x, y, z] coordinates

**block_sizes**: Configurable processing buffer sizes. They do not need to be uniform.

**sofa_path**: Path to the .sofa file used for extracting BRIR data.

**audio_file**: Path to audio file to use for processing.

**save_output**: Boolean to save the binaural output. Format: <config_save_path>/binaural_az_el_d_vx_vy_vz.wav. 

The velocity values in the wav file will either be in the cartesian "mps_mps_mps" form or the polar "dps_dps_mps" form.

### Coordinate System Notes
Zero degrees faces forward. Positive azimuth runs counter-clockwise. Positive elevation runs upward.

Positive x runs to the left. Positive y runs forward. Positive z runs upward.


