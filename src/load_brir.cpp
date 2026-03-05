#include "load_brir.h"
#include <mysofa.h>
#include <iostream>
#include <string>
#include <vector>
#include <complex>
#include <fftw3.h>
#include <numeric>
#include <fstream>

using std::vector;
using std::string;
using std::complex;

struct SofaLoader {
    MYSOFA_HRTF* sofa;
    int sample_rate;
    int filter_length;
    int measurement_count;

    bool load(const string& path) {
        // Check if file exists before calling mysofa_open
        std::ifstream f(path.c_str());
        if (!f.good()) {
            std::cout << "Standard C++ cannot find file at: " << path << "\n";
            return false;
        }

        int err;
        sofa = mysofa_load(path.c_str(), &err);
        if (!sofa || err != MYSOFA_OK) {
            std::cout << "SOFA load failed! Error: " << err << "\n";
            return false;
        }

        filter_length = sofa->N;
        measurement_count = sofa->M;
        if (sofa->DataSamplingRate.values) sample_rate = (int)sofa->DataSamplingRate.values[0];
        else sample_rate = 44100; // Fallback
        
        std::cout << "Manual Load Success. Filter length: " << filter_length << ". Sample rate: " << sample_rate << " Hz\n";
        return true;
    }

    int find_nearest_index(float target_az, float target_el) {
        int best_idx = 0;
        float min_dist = 1e10;
        int C = sofa->C;

        for (int i = 0; i < measurement_count; ++i) {
            float az = sofa->SourcePosition.values[i * C];
            float el = sofa->SourcePosition.values[i * C + 1];

            // Simple Euclidean distance for angles
            float d_az = az - target_az;
            float d_el = el - target_el;
            float dist = d_az * d_az + d_el * d_el;

            if (dist < min_dist) {
                min_dist = dist;
                best_idx = i;
            }
        }
        return best_idx;
    }

    void close() {
        if (sofa) mysofa_free(sofa);
    }
};

void get_IR(SofaLoader& loader, float azimuth, float elevation, vector<float>& left, vector<float>& right) {
    int nearest = loader.find_nearest_index(azimuth, elevation);
    int n = loader.filter_length;
    int receiver_count = loader.sofa->R; 
    float* ir_data = loader.sofa->DataIR.values;

    if (!ir_data) return;

    int base_idx = nearest * receiver_count * n;

    // Direct Copy for Left Channel
    left.assign(ir_data + base_idx, ir_data + base_idx + n);

    // Direct Copy for Right Channel
    if (receiver_count > 1) right.assign(ir_data + base_idx + n, ir_data + base_idx + 2 * n);
    else right = left;
}

PartitionedIR create_stage(const vector<float>& ir, vector<int> block_sizes) {
    // Check if ir.size() < sum(block_sizes) and update block_sizes if necessary
    if (ir.size() < std::accumulate(block_sizes.begin(), block_sizes.end(), 0)) {
        // Update block_sizes to fit ir.size()
        int total_size = 0;
        for (int i = 0; i < block_sizes.size(); ++i) {
            if (total_size + block_sizes[i] >= ir.size()) {
                block_sizes[i] = ir.size() - total_size;
                block_sizes.resize(i + 1);
                break;
            }
            total_size += block_sizes[i];
        }
        // Sort block sizes so the smallest block is first (important for runtime processing)
        std::sort(block_sizes.begin(), block_sizes.end());

        // Print block sizes update
        std::cout << "Updated block sizes to fit IR length: ";
        for (int i = 0; i < block_sizes.size(); ++i) std::cout << block_sizes[i] << " ";
        std::cout << "\n\n";
    } else if (ir.size() > std::accumulate(block_sizes.begin(), block_sizes.end(), 0)) {
        // Add an extra block to cover remaining IR samples
        int remaining = ir.size() - std::accumulate(block_sizes.begin(), block_sizes.end(), 0);
        // Round remaining to nearest power of 2 for efficient FFT
        int next_pow2 = 1;
        while (next_pow2 < remaining) next_pow2 *= 2;
        block_sizes.push_back(next_pow2);
        // Sort block sizes so the smallest block is first (important for runtime processing)
        std::sort(block_sizes.begin(), block_sizes.end());

        std::cout << "Added extra block to cover remaining IR samples. New block sizes: ";
        for (int i = 0; i < block_sizes.size(); ++i) std::cout << block_sizes[i] << " ";
        std::cout << "\n";
        std::cout << "Zero-padded the impulse response to " << ir.size() + (next_pow2 - remaining) << " samples to account for the last block.\n\n";
        // Zero-pad IR to match total block size
        vector<float> padded_ir = ir;
        padded_ir.resize(padded_ir.size() + (next_pow2 - remaining), 0.0f);
        return create_stage(padded_ir, block_sizes); // Recurse with padded IR and updated block sizes
    }

    // The number of stages is determined by the number of unique block sizes
    // The number of partitions for a given stage is how many times that block size is repeated
    std::vector<int> stage_block_sizes;
    std::vector<int> num_partitions;
    for (int i = 0; i < block_sizes.size(); ++i) {
        if (i == 0 || block_sizes[i] != block_sizes[i - 1]) {
            stage_block_sizes.push_back(block_sizes[i]);
            num_partitions.push_back(1);
        } else {
            num_partitions.back()++;
        }
    }
    // Print stage and partition info
    std::cout << "Stage block sizes and partition counts:\n";
    for (size_t i = 0; i < stage_block_sizes.size(); ++i) {
        std::cout << "Stage " << i << ": Block size = " << stage_block_sizes[i] << ", Partitions = " << num_partitions[i] << "\n";
    }
    std::cout << "\n";

    // Partition the IR into
    int start = 0; // Index to track where we are in the IR
    PartitionedIR pir;
    pir.stages.resize(stage_block_sizes.size());
    for (size_t s = 0; s < stage_block_sizes.size(); ++s) {
        auto& stage = pir.stages[s];
        stage.block_size = stage_block_sizes[s];
        stage.fft_size = stage.block_size * 2; // Zero-pad to double the block size for FFT
        stage.bins = stage.fft_size / 2 + 1;
        stage.num_partitions = num_partitions[s];

        vector<float> temp(stage.fft_size, 0.0f); // Zero-padded input for FFT
        vector<complex<float>> spectrum(stage.bins); // Output spectrum for current partition
        fftwf_plan plan = fftwf_plan_dft_r2c_1d( 
            stage.fft_size,
            temp.data(),
            reinterpret_cast<fftwf_complex*>(spectrum.data()),
            FFTW_ESTIMATE);
        
        stage.H.resize(stage.num_partitions);
        for (int p = 0; p < stage.num_partitions; ++p) {
            if (p != 0) std::fill(temp.begin(), temp.end(), 0.0f); // Clear temp buffer for next partition
            std::copy(ir.begin() + start, ir.begin() + start + stage.block_size, temp.begin()); // Copy current block to temp buffer with zero-padding
            start += stage.block_size;

            fftwf_execute(plan); // Take FFT of current partition
            stage.H[p] = spectrum; // Store spectrum in current stage's partition list
        }
        fftwf_destroy_plan(plan);
    }

    return pir;
}

BRIR get_BRIR(string sofa_path, float azimuth, float elevation, vector<int> block_sizes) {
    SofaLoader loader;
    if (!loader.load(sofa_path)) {
        std::cout << "Failed to load SOFA file. Returning empty BRIR.\n";
        return BRIR();
    }

    vector<float> left_ir, right_ir;
    get_IR(loader, azimuth, elevation, left_ir, right_ir);

    std::cout << "\nRequested block sizes: ";
    for (int size : block_sizes) {
        std::cout << size << " ";
    }
    std::cout << "\n\n";

    std::cout << "Left IR length: " << left_ir.size() << "\n";
    PartitionedIR left_stage = create_stage(left_ir, block_sizes);
    std::cout << "Right IR length: " << right_ir.size() << "\n";
    PartitionedIR right_stage = create_stage(right_ir, block_sizes);

    BRIR brir;
    brir.left = left_stage;
    brir.right = right_stage;

    loader.close();
    return brir;
}