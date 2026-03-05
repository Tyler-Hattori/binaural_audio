#pragma once
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
    std::vector<Stage> stages;
};

struct BRIR {
    PartitionedIR left;
    PartitionedIR right;
};

// The function main.cpp will call
BRIR get_BRIR(std::string sofa_path, float azimuth, float elevation, std::vector<int> block_sizes);