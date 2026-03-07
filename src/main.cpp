#include "load_brir.h"
#include "load_audio.h"
#include "play_audio.h"
#include <iostream>
#include <vector>
#include <string>
#include <complex>
#include <numeric>
#include <algorithm>
#include <fftw3.h>
#include <yaml-cpp/yaml.h>
#include <format>
#include <cmath>
#include "save_audio.h"

using std::vector;
using std::string;
using std::complex;

void initialize_BRIR_settings(AudioData& data, vector<int>& block_sizes) {
    // Modify block sizes
    int ir_length = data.brir_data.filter_length;
    if (ir_length < std::accumulate(block_sizes.begin(), block_sizes.end(), 0)) {
        // Update block_sizes to fit ir_length
        int total_size = 0;
        for (int i = 0; i < block_sizes.size(); ++i) {
            if (total_size + block_sizes[i] >= ir_length) {
                block_sizes[i] = ir_length - total_size;
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
        std::cout << std::endl << std::endl;
    } else if (ir_length > std::accumulate(block_sizes.begin(), block_sizes.end(), 0)) {
        // Add an extra block to cover remaining IR samples
        int remaining = ir_length - std::accumulate(block_sizes.begin(), block_sizes.end(), 0);
        // Partition remaining into powers of 2 for efficient FFT
        while (remaining != 0) {
            int next_pow2 = 1;
            while (next_pow2 <= remaining/2) next_pow2 *= 2;
            remaining -= next_pow2;
            block_sizes.push_back(next_pow2);
        }
        // Sort block sizes so the smallest block is first (important for runtime processing)
        std::sort(block_sizes.begin(), block_sizes.end());

        std::cout << "Added extra blocks to cover remaining IR samples. New block sizes: ";
        for (int i = 0; i < block_sizes.size(); ++i) std::cout << block_sizes[i] << " ";
        std::cout << std::endl;
    }

    // The number of stages is determined by the number of unique block sizes
    // The number of partitions for a given stage is how many times that block size is repeated
    vector<int> stage_block_sizes;
    vector<int> num_partitions;
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
    std::cout << std::endl;

    // Clear stereo output buffers and prepare for stereo playback
    data.left = vector<float>(data.mono.size() + 2*block_sizes.back(), 0.0f);
    data.right = vector<float>(data.mono.size() + 2*block_sizes.back(), 0.0f);
    data.use_mono = false;
    data.playhead = 0;

    // Update audio data struct with BRIR and block size info for processing
    data.apply_brir = true;
    data.block_sizes = block_sizes;
    data.num_stages = stage_block_sizes.size();

    // Each stage has its own history buffers, overlap buffers, and FFT plans based on its block size
    int sample_count = 0;
    for (int s = 0; s < data.num_stages; ++s) {
        StageData sd;
        
        sd.block_size = stage_block_sizes[s];
        sd.fft_size = sd.block_size * 2;
        sd.bins = sd.fft_size / 2 + 1;
        sd.num_partitions = num_partitions[s];

        if (s == 0) sd.type = "direct sound";
        else if (s == 1 || sample_count < 4096) sd.type = "early reflection";
        else sd.type = "late reverb";
        sample_count += sd.num_partitions * sd.block_size;

        sd.x = fftwf_alloc_real(sd.fft_size);
        sd.X = fftwf_alloc_complex(sd.bins);

        sd.Y_left = fftwf_alloc_complex(sd.bins);
        sd.Y_right = fftwf_alloc_complex(sd.bins);

        sd.overlap_left = fftwf_alloc_real(sd.fft_size);
        sd.overlap_right = fftwf_alloc_real(sd.fft_size);

        sd.X_history.resize(sd.num_partitions);
        sd.H_left.resize(sd.num_partitions);
        sd.H_right.resize(sd.num_partitions);
        for (auto& h : sd.X_history) h.resize(sd.bins);
        for (auto& h : sd.H_left) h.resize(sd.bins);
        for (auto& h : sd.H_right) h.resize(sd.bins);

        if (sd.type == "direct sound") {
            sd.H_left_history.resize(sd.num_partitions - 1);
            sd.H_right_history.resize(sd.num_partitions - 1);
            for (int p = 0; p < sd.num_partitions - 1; ++p) {
                sd.H_left_history[p].resize(sd.num_partitions - p - 1);
                sd.H_right_history[p].resize(sd.num_partitions - p - 1);
                for (auto& h : sd.H_left_history[p]) h.resize(sd.bins);
                for (auto& h : sd.H_right_history[p]) h.resize(sd.bins);
            }
        }

        sd.fft_plan = fftwf_plan_dft_r2c_1d(sd.fft_size, sd.x, sd.X, FFTW_MEASURE);
        sd.ifft_plan_left = fftwf_plan_dft_c2r_1d(sd.fft_size, sd.Y_left, sd.overlap_left, FFTW_MEASURE);
        sd.ifft_plan_right = fftwf_plan_dft_c2r_1d(sd.fft_size, sd.Y_right, sd.overlap_right, FFTW_MEASURE);

        data.stages.emplace_back(sd);
    }

    // Allocate space for H_left and H_right buffers
    data.current_H_left.resize(block_sizes.size());
    data.current_H_right.resize(block_sizes.size());
    for (int i = 0; i < block_sizes.size(); ++i) {
        data.current_H_left[i].resize(block_sizes[i] + 1);
        data.current_H_right[i].resize(block_sizes[i] + 1);
    }

    data.brir_data.H_left.resize(data.brir_data.num_measurements);
    for (auto& h : data.brir_data.H_left) {
        h.resize(block_sizes.size());
        for (int i = 0; i < h.size(); ++i) h[i].resize(block_sizes[i] + 1);
    }

    data.brir_data.H_right.resize(data.brir_data.num_measurements);
    for (auto& h : data.brir_data.H_right) {
        h.resize(block_sizes.size());
        for (int i = 0; i < h.size(); ++i) h[i].resize(block_sizes[i] + 1);
    }
}

void partition_BRIR_data(AudioData& data) {
    int partition_count = 0;
    int sample_offset = 0;
    int M = data.brir_data.num_measurements;
    int N = data.brir_data.filter_length;

    for (int s = 0; s < data.num_stages; s++) {
        auto& sd = data.stages[s];
        int bs = sd.block_size;

        // Allocate temporary buffers for this stage's FFT size
        float* left_ir = fftwf_alloc_real(sd.fft_size);
        float* right_ir = fftwf_alloc_real(sd.fft_size);
        fftwf_complex* H_left_tmp = fftwf_alloc_complex(sd.bins);
        fftwf_complex* H_right_tmp = fftwf_alloc_complex(sd.bins);

        // Create plans once per stage
        fftwf_plan p_left = fftwf_plan_dft_r2c_1d(sd.fft_size, left_ir, H_left_tmp, FFTW_MEASURE);
        fftwf_plan p_right = fftwf_plan_dft_r2c_1d(sd.fft_size, right_ir, H_right_tmp, FFTW_MEASURE);

        for (int m = 0; m < M; ++m) {
            for (int p = 0; p < sd.num_partitions; ++p) {
                int src_idx = m * N + sample_offset + (p * bs);

                // Copy and Zero-pad
                memcpy(left_ir, &data.brir_data.left_irs[src_idx], bs * sizeof(float));
                memset(left_ir + bs, 0, bs * sizeof(float));
                memcpy(right_ir, &data.brir_data.right_irs[src_idx], bs * sizeof(float));
                memset(right_ir + bs, 0, bs * sizeof(float));

                fftwf_execute(p_left);
                fftwf_execute(p_right);

                // Cast and Copy to persistent cache
                auto* H_left = reinterpret_cast<std::complex<float>*>(H_left_tmp); // this is unnormalized
                auto* H_right = reinterpret_cast<std::complex<float>*>(H_right_tmp);

                std::copy(H_left, H_left + sd.bins, data.brir_data.H_left[m][partition_count + p].begin());
                std::copy(H_right, H_right + sd.bins, data.brir_data.H_right[m][partition_count + p].begin());
            }
        }
        
        // Clean up stage resources
        sample_offset += sd.num_partitions * sd.block_size;
        partition_count += sd.num_partitions;
        fftwf_destroy_plan(p_left);
        fftwf_destroy_plan(p_right);
        fftwf_free(left_ir); fftwf_free(right_ir);
        fftwf_free(H_left_tmp); fftwf_free(H_right_tmp);
    }
}

void initialize_stages(AudioData& data) {
    int partition_count = 0;
    for (int s = 0; s < data.stages.size(); ++s) {
        for (int p = 0; p < data.stages[s].num_partitions; ++p) {
            for (int b = 0; b < data.stages[s].bins; ++b) {
                data.stages[s].H_left[p][b] = data.current_H_left[partition_count + p][b];
                data.stages[s].H_right[p][b] = data.current_H_right[partition_count + p][b];
            }
        }
        partition_count += data.stages[s].num_partitions;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path to config.yaml> ..." << std::endl;
        return 1; // Exit with error
    }
    // Parse yaml file
    YAML::Node config;
    try {config = YAML::LoadFile(argv[1]);}
    catch (const std::exception& e) {
        std::cerr << "Error loading config file: " << e.what() << std::endl;
        return 1; // Exit with error
    }
    float azimuth = config["azimuth"].as<float>(30.0f);
    float elevation = config["elevation"].as<float>(0.0f);
    float distance = config["distance"].as<float>(1.0f);
    vector<int> block_sizes;
    for (const auto& bs : config["block_sizes"]) {
        block_sizes.push_back(bs.as<int>());
    }
    string velocity_format = config["velocity"]["type"].as<string>("");
    vector<float> velocity;
    if (velocity_format != "cartesian" && velocity_format != "polar") {
        std::cerr << "Invalid velocity type in config. Use 'cartesian' or 'polar'." << std::endl;
        return 1;
    } else if (velocity_format == "cartesian") {
        velocity = {config["velocity"]["x"].as<float>(0.0f),
                    config["velocity"]["y"].as<float>(0.0f),
                    config["velocity"]["z"].as<float>(0.0f)};
    } else {
        velocity = {config["velocity"]["az"].as<float>(0.0f),
                    config["velocity"]["el"].as<float>(0.0f),
                    config["velocity"]["rho"].as<float>(0.0f)};
    }
    string default_sofa_path = "/Users/tyler/Documents/asp projects/binaural_audio/data/mit_kemar_large_pinna.sofa";
    string default_audio_file = "/Users/tyler/Documents/asp projects/binaural_audio/data/fairest_of_the_seasons_short.wav";
    string sofa_path = config["sofa_path"].as<string>(default_sofa_path);
    string audio_file = config["audio_file"].as<string>(default_audio_file);
    bool save_output = config["save_output"].as<bool>(false);
    string save_folder = config["save_folder"].as<string>("./results");

    // Load the audio file as a vector of floats (mono)
    AudioData audio_data = load_audio(audio_file);
    audio_data.buffer_size = block_sizes[0]; // Use smallest block size for playback buffer
    std::cout << "Loaded audio file: " << audio_file << " with " << audio_data.mono.size() << " samples at " << audio_data.sample_rate << " Hz.\n\n";

    // Playback the original mono audio
    std::cout << "Playing unmodified audio...\n";
    play_audio(audio_data);
    std::cout << "Finished playing unmodified audio.\n\n";

    // Calculate angles and distances per block based on the audio source's initial position and velocity
    int num_angles = audio_data.mono.size() / audio_data.buffer_size;
    vector<float> az_angles(num_angles);
    vector<float> el_angles(num_angles);
    vector<float> distances(num_angles);
    if (velocity_format == "polar") {
        for (int i = 0; i < num_angles; ++i) {
            az_angles[i] = std::fmod(azimuth + velocity[0] * i * audio_data.buffer_size / audio_data.sample_rate, 360.0f);
            if (az_angles[i] < 0) az_angles[i] += 360.0f;
            el_angles[i] = std::fmod(elevation + velocity[1] * i * audio_data.buffer_size / audio_data.sample_rate, 360.0f);
            if (el_angles[i] < 0) el_angles[i] += 360.0f;
            distances[i] = distance + velocity[2] * i * audio_data.buffer_size / audio_data.sample_rate;
        }
    } else if (velocity_format == "cartesian") {
        float x_init = distance * cos(elevation * M_PI / 180.0f) * sin(azimuth * M_PI / 180.0f);
        float y_init = distance * cos(elevation * M_PI / 180.0f) * cos(azimuth * M_PI / 180.0f);
        float z_init = distance * sin(elevation * M_PI / 180.0f);
        for (int i = 0; i < num_angles; ++i) {
            float x_coor = x_init + velocity[0] * i * audio_data.buffer_size / audio_data.sample_rate;
            float y_coor = y_init + velocity[1] * i * audio_data.buffer_size / audio_data.sample_rate;
            float z_coor = z_init + velocity[2] * i * audio_data.buffer_size / audio_data.sample_rate;
 
            float r = std::sqrt(x_coor*x_coor + y_coor*y_coor + z_coor*z_coor);
            distances[i] = r;
            float az = std::atan2(x_coor, y_coor) * 180.0f / M_PI;
            if (az < 0) az += 360.0f;
            az_angles[i] = az;
            float el = std::asin(z_coor / (r + 1e-8f)) * 180.0f / M_PI; // Add small value to avoid division by zero
            el_angles[i] = el; // Elevation is in [-90, 90], so no adjustment needed
        }
    }
    audio_data.azimuths = az_angles;
    audio_data.elevations = el_angles;
    audio_data.rhos = distances;

    // Load all BRIRs from the sofa dataset
    audio_data.brir_data = get_all_BRIR_data(sofa_path);

    // Modify audio data for BRIR processing and playback
    initialize_BRIR_settings(audio_data, block_sizes);

    // Partition and FFT all BRIR data before processing
    std::cout << "Partitioning BRIRs and converting to the frequency domain..." << std::endl;
    partition_BRIR_data(audio_data);
    std::cout << "Done.\n" << std::endl;

    // Interpolate initial values for H_left and H_right
    update_current_IR(&audio_data, azimuth, elevation);

    // Initialize IR of each stage
    initialize_stages(audio_data);

    // Print audio source positional info
    std::cout << "Audio source initial position:" << std::endl << "\taz: " << azimuth << " degrees\n\tel: " << elevation << " degrees\n\trho: " << distance << " meters\n" << std::endl;
    std::cout << "Audio source velocity:" << std::endl;
    if (velocity_format == "cartesian") std::cout << "\tvel_x: " << velocity[0] << " m/s\n\tvel_y: " << velocity[1] << " m/s\n\tvel_z: " << velocity[2] << " m/s\n" << std::endl;
    else if (velocity_format == "polar") std::cout << "\tvel_az: " << velocity[0] << " deg/s\n\tvel_el: " << velocity[1] << " deg/s\n\tvel_rho: " << velocity[2] << " m/s\n" << std::endl;

    // Apply the BRIR to the input audio file
    std::cout << "Applying BRIRs to audio in real-time..." << std::endl;
    play_audio(audio_data);
    std::cout << "Finished playing binaural audio.\n\n";

    // Save the ouput stereo audio
    if (save_output) {
        string output_path = "";
        if (velocity_format == "cartesian") output_path = std::format("{}/binaural_{}_{}_{}_{}mps_{}mps_{}mps.wav", save_folder, azimuth, elevation, distance, velocity[0], velocity[1], velocity[2]);
        else output_path = std::format("{}/binaural_{}_{}_{}_{}dps_{}dps_{}mps.wav", save_folder, azimuth, elevation, distance, velocity[0], velocity[1], velocity[2]);
        std::cout << "Saving output audio to: " << output_path << std::endl;
        // Trim the output to the original input length (since we added padding for convolution)
        vector<float> left_trimmed(audio_data.left.begin(), audio_data.left.begin() + audio_data.mono.size());
        vector<float> right_trimmed(audio_data.right.begin(), audio_data.right.begin() + audio_data.mono.size());
        save_stereo_wav(output_path, left_trimmed, right_trimmed, audio_data.sample_rate, 2);
        std::cout << "Finished saving output audio." << std::endl;
    }
}
    