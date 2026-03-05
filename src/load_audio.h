#pragma once
#include "load_brir.h"
#include <fftw3.h>
#include <vector>
#include <string>

std::vector<float> convert_to_mono(const std::vector<float>& left, const std::vector<float>& right);

// Runtime data for each stage
struct StageData {
    int write_index = 0; // Index for writing incoming audio blocks
    int period;         // How many blocks of the smallest stage fit into this stage's block size
    int counter = 0;    // Counter to track when to process this stage
    int block_size;     // Block size for this stage
    int fft_size;       // FFT size for this stage
    int bins;           // Number of frequency bins (fft_size/2 + 1)
    int num_partitions; // Number of partitions in the BRIR for this stage

    std::vector<std::vector<std::complex<float>>> X_history;    // [partition][bin]
    std::vector<std::complex<float>> Y_left;                    // [bin]
    std::vector<std::complex<float>> Y_right;                   // [bin]

    std::vector<float> fft_input;     // For FFT input (size = fft_size)
    std::vector<float> ifft_output;   // For IFFT output (size = fft_size)

    std::vector<float> overlap_left;    // For overlap-add output (size = block_size)
    std::vector<float> overlap_right;

    fftwf_plan fft_plan;
    fftwf_plan ifft_plan;
};

struct AudioData {
    // Basic data
    std::vector<float> left;
    std::vector<float> right;
    std::vector<float> mono;

    // For playback
    int sample_rate;
    size_t playhead = 0;
    int buffer_size = 128;
    bool use_mono = false;

    // For processing
    bool apply_brir = false;
    std::vector<int> block_sizes; // Length = number of stages
    BRIR brir;

    std::vector<StageData> stages; // One runtime struct per stage
};

AudioData load_audio(const std::string& path);