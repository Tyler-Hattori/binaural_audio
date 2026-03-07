#pragma once
#include "load_brir.h"
#include <fftw3.h>
#include <vector>
#include <string>
#include <complex>

std::vector<float> convert_to_mono(const std::vector<float>& left, const std::vector<float>& right);

// Runtime data for each stage
struct StageData {
    int sample_counter = 0;  // Counter to track when to process this stage
    int block_size;          // Block size for this stage
    int fft_size;            // FFT size for this stage
    int bins;                // Number of frequency bins (fft_size/2 + 1)
    int num_partitions;      // Number of partitions in the BRIR for this stage
    std::string type; // "direct sound", "early reflection", or "late reverb"

    float* x;                 // For FFT input (size = fft_size)
    fftwf_complex* X;         // For FFT output (size = fft_size)
    fftwf_complex* Y_left;    // [bin]
    fftwf_complex* Y_right;   // [bin]
    float* overlap_left;      // For overlap-add IFFT output (size = FFT size)
    float* overlap_right;

    std::vector<std::vector<std::complex<float>>> X_history;    // [partition][bin]
    std::vector<std::vector<std::complex<float>>> H_left;       // [partition][bin]
    std::vector<std::vector<std::complex<float>>> H_right;      // [partition][bin]

    // In case the stage is "direct sound"
    std::vector<std::vector<std::vector<std::complex<float>>>> H_left_history;       // [time][partition][bin]
    std::vector<std::vector<std::vector<std::complex<float>>>> H_right_history;      // [time][partition][bin]

    fftwf_plan fft_plan;
    fftwf_plan ifft_plan_left;
    fftwf_plan ifft_plan_right;
};

struct AudioData {
    // Basic data
    std::vector<float> left;
    std::vector<float> right;
    std::vector<float> mono;
    SofaBRIRData brir_data;

    // For playback
    int sample_rate;
    size_t playhead = 0;
    int buffer_size;
    bool use_mono = true;

    // For BRIR processing
    bool apply_brir = false;
    std::vector<float> ir_left;
    std::vector<float> ir_right;
    std::vector<int> block_sizes;
    std::vector<StageData> stages; // One runtime struct per stage
    int num_stages;

    // To handle moving sources
    std::vector<float> azimuths;
    std::vector<float> elevations;
    std::vector<float> rhos;
    std::vector<std::vector<std::complex<float>>> current_H_left;  // [partition][bin]
    std::vector<std::vector<std::complex<float>>> current_H_right; // [partition][bin]
    float current_az;
    float current_el;
    float reference_dist = 1.4f; // Reference distance for distance-based gain compensation, 1.4m aligns with MIT KEMAR measurements
};

AudioData load_audio(const std::string& path);