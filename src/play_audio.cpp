#include "load_brir.h"
#include "load_audio.h"
#include <RtAudio.h>
#include <vector>
#include <algorithm>
#include <iostream>

using std::vector;
using std::complex;

void update_current_IR(AudioData *data, float target_az, float target_el) {
    int nearest_idx = 0;
    int second_nearest_idx = 0;
    float smallest_dist = 1e10;
    float second_smallest_dist = 1e10;
    for (int m = 0; m < data->brir_data.num_measurements; ++m) {
        float az = data->brir_data.azimuths[m];
        float el = data->brir_data.elevations[m];
        float dist = std::sqrt(std::pow(az - target_az, 2) + std::pow(el - target_el, 2));
        if (dist < smallest_dist) {
            second_smallest_dist = smallest_dist;
            second_nearest_idx = nearest_idx;
            smallest_dist = dist;
            nearest_idx = m;
        } else if (dist < second_smallest_dist) {
            second_smallest_dist = dist;
            second_nearest_idx = m;
        }
    }
    float interpolation_factor = smallest_dist / (smallest_dist + second_smallest_dist);
    data->current_az = data->brir_data.azimuths[nearest_idx] * (1 - interpolation_factor) + data->brir_data.azimuths[second_nearest_idx] * (interpolation_factor);
    data->current_el = data->brir_data.elevations[nearest_idx] * (1 - interpolation_factor) + data->brir_data.elevations[second_nearest_idx] * (interpolation_factor);

    // Calculate H_left and H_right
    for (int p = 0; p < data->block_sizes.size(); ++p) {
        for (int b = 0; b < data->block_sizes[p] + 1; ++b) {
            data->current_H_left[p][b] = data->brir_data.H_left[nearest_idx][p][b] * (1 - interpolation_factor) +
                                        data->brir_data.H_left[second_nearest_idx][p][b] * interpolation_factor;
            data->current_H_right[p][b] = data->brir_data.H_right[nearest_idx][p][b] * (1 - interpolation_factor) +
                                        data->brir_data.H_right[second_nearest_idx][p][b] * interpolation_factor;
        }
    }
}

int audio_callback(void *output_buffer, void *input_buffer, unsigned int buffer_size,
                    double stream_time, RtAudioStreamStatus status, void *user_data) {
    float *out = static_cast<float*>(output_buffer);
    AudioData *data = static_cast<AudioData*>(user_data);

    // If we've reached the end of the audio data, fill the output with zeros and stop
    if (data->playhead >= data->mono.size()) {
        std::fill(out, out + (buffer_size * 2), 0.0f);
        return 0;
    }

    // Calculate how many samples to process in this callback
    size_t remaining = data->mono.size() - data->playhead;
    size_t to_process = (buffer_size < remaining) ? buffer_size : remaining;

    // Process BRIR convolution if enabled
    if (data->apply_brir) {
        // Get current IR for this instant in time
        int block_idx = std::min((int)(data->playhead / data->buffer_size), (int)(data->azimuths.size() - 1));
        update_current_IR(data, data->azimuths[block_idx], data->elevations[block_idx]);

        // Apply IR in frequency domain
        int partition_count = 0;
        float reference_dist = data->reference_dist;
        float distance_gain = reference_dist / std::max(data->rhos[block_idx], 0.7f); // Simple distance-based gain compensation
        for (size_t s = 0; s < data->stages.size(); ++s) {
            auto& stage_data = data->stages[s];

            // Copy the current block of audio into the stage's input buffer
            int sc = stage_data.sample_counter;
            memcpy(stage_data.x + sc, &data->mono[data->playhead], to_process * sizeof(float));

            // Only process this stage if block_size samples have been accumulated
            stage_data.sample_counter = (stage_data.sample_counter + buffer_size) % stage_data.block_size;
            if (stage_data.sample_counter != 0) continue;

            // Based on the stage type, update H_left and H_right
            int bins = stage_data.bins;
            if (stage_data.type == "direct sound") {
                // Update the IR partitions one at a time
                stage_data.H_left[0] = data->current_H_left[partition_count + 0]; // partition_count should be 0 since currently "direct sound" is only for the first stage
                stage_data.H_right[0] = data->current_H_right[partition_count + 0];
                for (int p = 1; p < stage_data.num_partitions; ++p) {
                    for (int b = 0; b < bins; ++b) {
                        stage_data.H_left[p][b] = stage_data.H_left_history[p - 1][0][b];
                        stage_data.H_right[p][b] = stage_data.H_right_history[p - 1][0][b];
                    }
                }
                // Update IR memory
                for (int p = stage_data.num_partitions - 1; p > 0; --p) {
                    for (int i = 0; i < stage_data.H_left_history[p - 1].size(); ++i) {
                        for (int b = 0; b < bins; ++b) {
                            if (p > 1) {
                                stage_data.H_left_history[p - 1][i][b] = stage_data.H_left_history[p - 2][i + 1][b];
                                stage_data.H_right_history[p - 1][i][b] = stage_data.H_right_history[p - 2][i + 1][b];
                            }
                            else {
                                stage_data.H_left_history[p - 1][i][b] = data->current_H_left[partition_count + 1 + i][b];
                                stage_data.H_right_history[p - 1][i][b] = data->current_H_right[partition_count + 1 + i][b];
                            }
                        }
                    }
                }
            } else if (stage_data.type == "early reflection") {
                // Update every partition of the IR
                for (int p = 0; p < stage_data.num_partitions; ++p) {
                    for (int b = 0; b < bins; ++b) {
                        stage_data.H_left[p][b] = data->current_H_left[partition_count + p][b];
                        stage_data.H_right[p][b] = data->current_H_right[partition_count + p][b];
                    }
                }
            } // Otherwise, the stage type is "late reverb" and no update is made
            partition_count += stage_data.num_partitions;

            // Zero-pad and FFT the input
            memset(stage_data.x + stage_data.block_size, 0, stage_data.block_size * sizeof(float));
            fftwf_execute(stage_data.fft_plan);

            // For the complex multiplication, cast to std::complex for easy math:
            auto* X = reinterpret_cast<std::complex<float>*>(stage_data.X);
            auto* Y_left = reinterpret_cast<std::complex<float>*>(stage_data.Y_left);
            auto* Y_right = reinterpret_cast<std::complex<float>*>(stage_data.Y_right);

            // Store into circular FDL
            std::rotate(stage_data.X_history.rbegin(), stage_data.X_history.rbegin() + 1, stage_data.X_history.rend());
            std::copy(X, X + bins, stage_data.X_history[0].begin());

            // Convolution sum in frequency domain
            float N = static_cast<float>(stage_data.fft_size);
            memset(stage_data.Y_left, 0, bins * sizeof(std::complex<float>));
            memset(stage_data.Y_right, 0, bins * sizeof(std::complex<float>));
            for (int p = 0; p < stage_data.num_partitions; ++p) {
                for (int b = 0; b < bins; ++b) {
                    Y_left[b]  += stage_data.H_left[p][b] * stage_data.X_history[p][b];
                    Y_right[b] += stage_data.H_right[p][b] * stage_data.X_history[p][b];
                }
            }

            // Inverse FFT
            fftwf_execute(stage_data.ifft_plan_left);
            fftwf_execute(stage_data.ifft_plan_right);

            // Overlap-add output to final stereo buffers
            float norm = 1.0f / N;
            for (int n = 0; n < N; ++n) {
                if (data->playhead + buffer_size - stage_data.block_size + n < data->left.size()) {
                    data->left[data->playhead + buffer_size - stage_data.block_size + n] += distance_gain * norm * stage_data.overlap_left[n];
                    data->right[data->playhead + buffer_size - stage_data.block_size + n] += distance_gain * norm * stage_data.overlap_right[n];
                }
            }
        }
    }
    
    // Playback: either mono or stereo output depending on settings
    if (data->use_mono) {
        for (size_t i = 0; i < to_process; ++i) {
            float s = data->mono[data->playhead++];
            *out++ = s; 
            *out++ = s;
        }
    } else {
        for (size_t i = 0; i < to_process; ++i) {
            *out++ = data->left[data->playhead];
            *out++ = data->right[data->playhead++];
        }
    }

    // If we processed less than the buffer size, fill the rest with zeros
    if (to_process < buffer_size) {
        std::fill(out, out + ((buffer_size - to_process) * 2), 0.0f);
    }

    return 0;
}

int play_audio(AudioData& audio_data) {
    RtAudio dac;
    RtAudio::StreamParameters parameters;
    parameters.deviceId = dac.getDefaultOutputDevice();
    parameters.nChannels = 2; // Stereo output

    unsigned int buffer_size_unsigned = static_cast<unsigned int>(audio_data.buffer_size);

    try {
        dac.openStream(&parameters, NULL, RTAUDIO_FLOAT32, audio_data.sample_rate, &buffer_size_unsigned, &audio_callback, static_cast<void*>(&audio_data));
        dac.startStream();
    } catch (const std::exception& e) {
        std::cerr << "RtAudio Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Playing... Press Enter to quit." << std::endl;
    std::cin.get(); // Keep program alive while audio plays

    if (dac.isStreamOpen()) {
        if (dac.isStreamRunning()) dac.stopStream();
        dac.closeStream(); // This blocks until the audio thread is safely shut down
    }

    return 0;
}