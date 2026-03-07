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
        
        std::cout << "Loaded " << measurement_count << " BRIRs. Filter lengths: " << filter_length << ". Sample rate: " << sample_rate << " Hz.\n" << std::endl;
        return true;
    }

    std::tuple<int, int, float> find_closest_indices(float target_az, float target_el) {
        int nearest_idx = 0;
        int second_nearest_idx = 0;
        float smallest_dist = 1e10;
        float second_smallest_dist = 1e10;
        int C = sofa->C;

        for (int i = 0; i < measurement_count; ++i) {
            float az = sofa->SourcePosition.values[i * C + 0];
            float el = sofa->SourcePosition.values[i * C + 1];
            float dist = std::sqrt(std::pow(az - target_az, 2) + std::pow(el - target_el, 2));
            if (dist < smallest_dist) {
                second_smallest_dist = smallest_dist;
                second_nearest_idx = nearest_idx;
                smallest_dist = dist;
                nearest_idx = i;
            } else if (dist < second_smallest_dist) {
                second_smallest_dist = dist;
                second_nearest_idx = i;
            }
        }

        float interpolation_factor = smallest_dist / (smallest_dist + second_smallest_dist);

        // For simplicity, we'll just return the nearest measurement index for both low and high (no interpolation)
        return std::make_tuple(nearest_idx, second_nearest_idx, interpolation_factor);
    }

    void close() {
        if (sofa) mysofa_free(sofa);
    }
};

void get_IR(SofaLoader& loader, float azimuth, float elevation, vector<float>& left, vector<float>& right) {
    auto [nearest_idx, second_nearest_idx, interp_factor] = loader.find_closest_indices(azimuth, elevation);
    int n = loader.filter_length;
    int receiver_count = loader.sofa->R;

    // Get IR data for nearest and second nearest measurements
    float* ir_data = loader.sofa->DataIR.values;
    if (!ir_data) return;
    int base_idx_nearest = nearest_idx * receiver_count * n;
    int base_idx_second = second_nearest_idx * receiver_count * n;

    left.assign(n, 0.0f);
    right.assign(n, 0.0f);

    for (int i = 0; i < n; ++i) {
        float nearest_left = ir_data[base_idx_nearest + i];
        float nearest_right = (receiver_count > 1) ? ir_data[base_idx_nearest + n + i] : nearest_left;
        float second_left = ir_data[base_idx_second + i];
        float second_right = (receiver_count > 1) ? ir_data[base_idx_second + n + i] : second_left;

        // Linear interpolation between the two nearest measurements
        left[i] = nearest_left * (1 - interp_factor) + second_left * interp_factor;
        right[i] = nearest_right * (1 - interp_factor) + second_right * interp_factor;
    }
}

PartitionedIR partition_IR(const vector<float>& ir, vector<int> block_sizes, bool verbose) {
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
        if (verbose) {
            std::cout << "Updated block sizes to fit IR length: ";
            for (int i = 0; i < block_sizes.size(); ++i) std::cout << block_sizes[i] << " ";
            std::cout << "\n\n";
        }
    } else if (ir.size() > std::accumulate(block_sizes.begin(), block_sizes.end(), 0)) {
        // Add an extra block to cover remaining IR samples
        int remaining = ir.size() - std::accumulate(block_sizes.begin(), block_sizes.end(), 0);
        // Round remaining to nearest power of 2 for efficient FFT
        int next_pow2 = 1;
        while (next_pow2 < remaining) next_pow2 *= 2;
        block_sizes.push_back(next_pow2);
        // Sort block sizes so the smallest block is first (important for runtime processing)
        std::sort(block_sizes.begin(), block_sizes.end());

        if (verbose) {
            std::cout << "Added extra block to cover remaining IR samples. New block sizes: ";
            for (int i = 0; i < block_sizes.size(); ++i) std::cout << block_sizes[i] << " ";
            std::cout << "\n";
            std::cout << "Zero-padded the impulse response to " << ir.size() + (next_pow2 - remaining) << " samples to account for the last block.\n\n";
        }
        // Zero-pad IR to match total block size
        vector<float> padded_ir = ir;
        padded_ir.resize(padded_ir.size() + (next_pow2 - remaining), 0.0f);
        return partition_IR(padded_ir, block_sizes, verbose); // Recurse with padded IR and updated block sizes
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
    if (verbose) {
        std::cout << "Stage block sizes and partition counts:\n";
        for (size_t i = 0; i < stage_block_sizes.size(); ++i) {
            std::cout << "Stage " << i << ": Block size = " << stage_block_sizes[i] << ", Partitions = " << num_partitions[i] << "\n";
        }
        std::cout << "\n";
    }

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

BRIR get_BRIR(string sofa_path, float azimuth, float elevation, vector<int> block_sizes, bool verbose) {
    SofaLoader loader;
    if (!loader.load(sofa_path)) {
        std::cout << "Failed to load SOFA file. Returning empty BRIR.\n";
        return BRIR();
    }

    vector<float> left_ir, right_ir;
    get_IR(loader, azimuth, elevation, left_ir, right_ir);

    if (verbose) {
        std::cout << "\nRequested block sizes: ";
        for (int size : block_sizes) std::cout << size << " ";
        if (verbose) std::cout << "\n\n";
    }

    if (verbose) std::cout << "Left IR length: " << left_ir.size() << "\n";
    PartitionedIR left_stage = partition_IR(left_ir, block_sizes, verbose);
    if (verbose) std::cout << "Right IR length: " << right_ir.size() << "\n";
    PartitionedIR right_stage = partition_IR(right_ir, block_sizes, verbose);

    BRIR brir;
    brir.left = left_stage;
    brir.right = right_stage;

    loader.close();
    return brir;
}

SofaBRIRData get_all_BRIR_data(string sofa_path) {
    SofaLoader loader;
    if (!loader.load(sofa_path)) {
        std::cout << "Failed to load SOFA file. Returning empty BRIR.\n";
        return SofaBRIRData();
    }

    SofaBRIRData data;

    int M = loader.sofa->M;
    int N = loader.filter_length;
    data.filter_length = N;
    data.num_measurements = M;
    
    // Extract the left and right IRs from the interleaved sofa data
    std::vector<float> left;
    std::vector<float> right;
    left.assign(N * M, 0.0f);
    right.assign(N * M, 0.0f);
    float* ir_data = loader.sofa->DataIR.values;
    for (int m = 0; m < M; ++m) {
        for (int i = 0; i < N; ++i) {
            left[m * N + i] = ir_data[2 * m * N + i];
            right[m * N + i] = ir_data[2 * m * N + N + i];
        }
    }
    data.left_irs = left;
    data.right_irs = right;

    // Store the az and el angles and rhos corresponding to the IR measurements
    vector<float> azimuths(M);
    vector<float> elevations(M);
    vector<float> rhos(M);
    int C = loader.sofa->C;
    for (int i = 0; i < M; ++i) {
        azimuths[i] = loader.sofa->SourcePosition.values[i * C];
        elevations[i] = loader.sofa->SourcePosition.values[i * C + 1];
        rhos[i] = loader.sofa->SourcePosition.values[i * C + 2];
    }
    data.azimuths = azimuths;
    data.elevations = elevations;
    data.rhos = rhos;

    return data;
}