#include "load_brir.h"
#include "load_audio.h"
#include "play_audio.h"
#include <iostream>
#include <vector>
#include <string>
#include <complex>
#include <algorithm>
#include <fftw3.h>

using std::vector;
using std::string;
using std::complex;

void initialize_BRIR(AudioData& data, const BRIR& brir, const vector<int>& block_sizes) {
    // Clear stereo output buffers and prepare for stereo playback
    data.left = vector<float>(data.mono.size() + 2*block_sizes.back(), 0.0f);
    data.right = vector<float>(data.mono.size() + 2*block_sizes.back(), 0.0f);
    data.use_mono = false;
    data.playhead = 0;

    // Update audio data struct with BRIR and block size info for processing
    data.apply_brir = true;
    data.block_sizes = block_sizes;
    data.brir = brir;

    // Each stage has its own history buffers, overlap buffers, and FFT plans based on its block size
    int stages = data.brir.left.stages.size();
    for (int s = 0; s < stages; ++s) {
        const auto& stage = data.brir.left.stages[s];
        StageData sd;
        
        sd.block_size = stage.block_size;
        sd.fft_size = stage.fft_size;
        sd.bins = stage.bins;
        sd.num_partitions = stage.num_partitions;

        sd.x = fftwf_alloc_real(sd.fft_size);
        sd.X = fftwf_alloc_complex(sd.bins);

        sd.Y_left = fftwf_alloc_complex(sd.bins);
        sd.Y_right = fftwf_alloc_complex(sd.bins);

        sd.overlap_left = fftwf_alloc_real(sd.fft_size);
        sd.overlap_right = fftwf_alloc_real(sd.fft_size);

        sd.X_history.resize(sd.num_partitions);
        for (auto& h : sd.X_history) h.resize(sd.bins);

        sd.fft_plan = fftwf_plan_dft_r2c_1d(sd.fft_size, sd.x, sd.X, FFTW_MEASURE);
        sd.ifft_plan_left = fftwf_plan_dft_c2r_1d(sd.fft_size, sd.Y_left, sd.overlap_left, FFTW_MEASURE);
        sd.ifft_plan_right = fftwf_plan_dft_c2r_1d(sd.fft_size, sd.Y_right, sd.overlap_right, FFTW_MEASURE);

        data.stages.emplace_back(sd);
    }
}

int main(int argc, char* argv[]) {
    // Set default sofa path
    string sofa_path = "/Users/tyler/Documents/asp projects/binaural_audio/data/mit_kemar_normal_pinna.sofa";
    // Set default audio file path
    string audio_file = "/Users/tyler/Documents/asp projects/binaural_audio/data/fairest_of_the_seasons_short.wav";

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <azimuth> <elevation> <block_size1> <block_size2> ..." << std::endl;
        return 1; // Exit with error
    }
    float azimuth = std::stof(argv[1]);
    float elevation = std::stof(argv[2]);
    vector<int> block_sizes;
    for (int i = 3; i < argc; i++) {
        block_sizes.push_back(std::stoi(argv[i]));
    }

    // Load the BRIR for the specified angles and block sizes
    BRIR brir = get_BRIR(sofa_path, azimuth, elevation, block_sizes);
    std::cout << "Loaded BRIR for azimuth " << azimuth << "°, elevation " << elevation << "°\n\n";

    // Load the audio file as a vector of floats (mono)
    AudioData audio_data = load_audio(audio_file);
    audio_data.buffer_size = block_sizes[0]; // Use smallest block size for playback buffer
    vector<float> input_audio = audio_data.mono;
    std::cout << "Loaded audio file: " << audio_file << " with " << input_audio.size() << " samples at " << audio_data.sample_rate << " Hz.\n\n";

    // Playback the original mono audio
    std::cout << "Playing unmodified audio...\n";
    play_audio(audio_data);
    std::cout << "Finished playing unmodified audio.\n\n";

    // Modify audio data for BRIR processing and playback
    initialize_BRIR(audio_data, brir, block_sizes);

    // Apply the BRIR to the input audio file
    std::cout << "Applying BRIR to audio in real-time..." << std::endl;
    play_audio(audio_data);
    std::cout << "Finished playing binaural audio.\n\n";

    // Save the ouput stereo audio

}
    