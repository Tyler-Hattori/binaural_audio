#pragma once
#include <mysofa.h>
#include <vector>
#include <string>
#include <complex>

struct StageInfo {
    int block_size;
    int fft_size;
    int bins;
    int num_partitions;

    std::vector<std::vector<std::complex<float>>> H;  // [partition][bin]
};

struct PartitionedIR {
    std::vector<StageInfo> stages;
};

struct BRIR {
    PartitionedIR left;
    PartitionedIR right;
};

struct SofaBRIRData {
    // IRs for each measurement flattened
    std::vector<float> left_irs;
    std::vector<float> right_irs;

    std::vector<std::vector<std::vector<std::complex<float>>>> H_left;   // [measurement][partition][bin]
    std::vector<std::vector<std::vector<std::complex<float>>>> H_right;  // [measurement][partition][bin]

    int filter_length;
    int num_measurements;

    // Coordinate positions for each measurement
    std::vector<float> azimuths;
    std::vector<float> elevations;
    std::vector<float> rhos;
};

// main.cpp will call
// BRIR get_BRIR(std::string sofa_path, float azimuth, float elevation, std::vector<int> block_sizes, bool verbose);
SofaBRIRData get_all_BRIR_data(std::string sofa_path);