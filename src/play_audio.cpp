#include "load_brir.h"
#include "load_audio.h"
#include <RtAudio.h>
#include <vector>
#include <iostream>

using std::vector;
using std::complex;

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
        for (size_t s = 0; s < data->stages.size(); ++s) {
            auto& stage_info = data->brir.left.stages[s];
            auto& stage_data = data->stages[s];

            // Copy the current block of audio into the stage's input buffer
            int sc = stage_data.sample_counter;
            memcpy(stage_data.x + sc, &data->mono[data->playhead], to_process * sizeof(float));

            // Only process this stage if block_size samples have been accumulated
            stage_data.sample_counter = (stage_data.sample_counter + buffer_size) % stage_data.block_size;
            if (stage_data.sample_counter != 0) continue;

            // Zero-pad and FFT
            memset(stage_data.x + stage_data.block_size, 0, stage_data.block_size * sizeof(float));
            fftwf_execute(stage_data.fft_plan);

            // For the complex multiplication, cast to std::complex for easy math:
            auto* X = reinterpret_cast<std::complex<float>*>(stage_data.X);

            // Store into circular FDL
            int bins = stage_data.bins;
            std::rotate(stage_data.X_history.rbegin(), stage_data.X_history.rbegin() + 1, stage_data.X_history.rend());
            std::copy(X, X + bins, stage_data.X_history[0].begin());

            // Convolution sum in frequency domain
            auto* Y_left = reinterpret_cast<std::complex<float>*>(stage_data.Y_left);
            auto* Y_right = reinterpret_cast<std::complex<float>*>(stage_data.Y_right);
            memset(stage_data.Y_left, 0, bins * sizeof(std::complex<float>));
            memset(stage_data.Y_right, 0, bins * sizeof(std::complex<float>));
            for (int p = 0; p < stage_data.num_partitions; ++p) {
                for (int k = 0; k < bins; ++k) {
                    Y_left[k]  += data->brir.left.stages[s].H[p][k] * stage_data.X_history[p][k];
                    Y_right[k] += data->brir.right.stages[s].H[p][k] * stage_data.X_history[p][k];
                }
            }

            // Inverse FFT
            fftwf_execute(stage_data.ifft_plan_left);
            fftwf_execute(stage_data.ifft_plan_right);

            // Overlap-add output to final stereo buffers
            int N = stage_data.fft_size;
            for (int n = 0; n < N; ++n) {
                if (data->playhead + n < data->left.size()) {
                    data->left[data->playhead + n] += stage_data.overlap_left[n] / N; // Normalize by fft_size
                    data->right[data->playhead + n] += stage_data.overlap_right[n] / N;
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