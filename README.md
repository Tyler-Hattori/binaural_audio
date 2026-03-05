# binaural_audio

install:

brew install fftw
brew install cmake
brew install hdf5
brew install libsndfile
brew install rtaudio

git clone https://github.com/hoene/libmysofa.git
cd libmysofa
cd build
cmake .. -DBUILD_TESTS=OFF
make
sudo make install

compile:

cmake ..
make

run:

./binaural



