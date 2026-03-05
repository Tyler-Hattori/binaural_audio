#include "load_brir.h"
#include "load_audio.h"
#include <RtAudio.h>
#include <vector>
#include <iostream>

using std::vector;

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

    if (data->apply_brir) {
        for (size_t s = 0; s < data->stages.size(); ++s) {
            auto& stage_info = data->brir.left.stages[s];
            auto& stage_data = data->stages[s];

            // Only process this stage if block_size samples have been accumulated
            if (stage_data.counter != 0) {
                stage_data.counter = (stage_data.counter + 1) % stage_data.period;
                continue;
            }
            stage_data.counter = (stage_data.counter + 1) % stage_data.period;

            int B = stage_info.block_size;
            int N = stage_info.fft_size;
            int bins = stage_info.bins;
            int P = stage_info.num_partitions;

            // 1️⃣ Copy input block
            std::fill(stage_data.time_buffer.begin(), stage_data.time_buffer.end(), 0.0f);

            for (size_t i = 0; i < to_process; ++i)
                stage_data.time_buffer[i] = data->mono[data->playhead + i];

            // 2️⃣ FFT
            fftwf_execute(stage_data.forward_plan);

            // 3️⃣ Store into circular FDL
            stage_data.X_history[stage_data.write_index] =
                std::vector<std::complex<float>>(
                    reinterpret_cast<std::complex<float>*>(stage_data.time_buffer.data()),
                    reinterpret_cast<std::complex<float>*>(stage_data.time_buffer.data()) + bins);

            // 4️⃣ Clear accumulators
            std::fill(stage_data.Y_left.begin(), stage_data.Y_left.end(), {0,0});
            std::fill(stage_data.Y_right.begin(), stage_data.Y_right.end(), {0,0});

            // 5️⃣ Convolution sum
            for (int p = 0; p < P; ++p)
            {
                int read_index = (stage_data.write_index - p + P) % P;

                for (int k = 0; k < bins; ++k)
                {
                    stage_data.Y_left[k]  += stage_info.H[p][k] * stage_data.X_history[read_index][k];
                    stage_data.Y_right[k] += data->brir.right.stages[s].H[p][k] *
                                    stage_data.X_history[read_index][k];
                }
            }

            // 6️⃣ Inverse FFT (Left)
            fftwf_execute(stage_data.inverse_plan);

            for (int i = 0; i < B; ++i)
            {
                float sample = stage_data.time_buffer[i] / N;
                sample += stage_data.overlap_left[i];
                out[2*i] += sample;
                stage_data.overlap_left[i] = stage_data.time_buffer[i + B] / N;
            }

            // 7️⃣ Inverse FFT (Right)
            fftwf_execute(stage_data.inverse_plan);

            for (int i = 0; i < B; ++i)
            {
                float sample = stage_data.time_buffer[i] / N;
                sample += stage_data.overlap_right[i];
                out[2*i+1] += sample;
                stage_data.overlap_right[i] = stage_data.time_buffer[i + B] / N;
            }

            // 8️⃣ Advance circular index
            stage_data.write_index = (stage_data.write_index + 1) % P;
        }


    } else if (data->use_mono) { // Mono output with no processing
        for (size_t i = 0; i < to_process; ++i) {
            float s = data->mono[data->playhead++];
            *out++ = s; 
            *out++ = s;
        }
    } else { // Stereo output with no processing
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

    dac.stopStream();
    dac.closeStream();

    return 0;
}