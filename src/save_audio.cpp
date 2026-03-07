#include <vector>
#include <string>

using std::vector;
using std::string;

bool save_audio_file(const string& filepath, const vector<float>& audio, int sample_rate, int channels) {
    FILE* file = fopen(filepath.c_str(), "wb");
    if (!file) return false;

    // 1. Define the structure
    struct WavHeader {
        char riff[4] = {'R', 'I', 'F', 'F'};
        uint32_t file_size;
        char wave[4] = {'W', 'A', 'V', 'E'};
        char fmt[4] = {'f', 'm', 't', ' '};
        uint32_t fmt_size = 16;
        uint16_t audio_format = 1; 
        uint16_t num_channels;
        uint32_t sample_rate_val;
        uint32_t byte_rate;
        uint16_t block_align;
        uint16_t bits_per_sample = 16;
        char data[4] = {'d', 'a', 't', 'a'};
        uint32_t data_size;
    };

    // 2. Initialize the instance with your variables
    WavHeader header;
    header.num_channels = (uint16_t)channels;
    header.sample_rate_val = (uint32_t)sample_rate;
    header.data_size = (uint32_t)(audio.size() * 2);
    header.file_size = 36 + header.data_size;
    header.byte_rate = sample_rate * channels * 2;
    header.block_align = (uint16_t)(channels * 2);

    fwrite(&header, sizeof(header), 1, file);

    for (float sample : audio) {
        // Clamp and convert to 16-bit PCM
        float clamped = std::max(-1.0f, std::min(1.0f, sample));
        int16_t pcm_sample = static_cast<int16_t>(clamped * 32767.0f);
        fwrite(&pcm_sample, sizeof(int16_t), 1, file);
    }

    fclose(file);
    return true;
}

bool save_stereo_wav(const string& filepath, const vector<float>& audio_left, const vector<float>& audio_right, int sample_rate, int channels) {
    // Interleave left and right channels
    vector<float> interleaved;
    interleaved.reserve(audio_left.size() + audio_right.size());
    for (size_t i = 0; i < audio_left.size(); ++i) {
        interleaved.push_back(audio_left[i]);
        if (i < audio_right.size()) {
            interleaved.push_back(audio_right[i]);
        } else {
            interleaved.push_back(0.0f); // Pad with silence if right channel is shorter
        }
    }
    return save_audio_file(filepath, interleaved, sample_rate, channels);
}