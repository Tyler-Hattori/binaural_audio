#include <sndfile.hh>
#include "load_audio.h"
#include <vector>
#include <string>

using std::vector;
using std::string;

vector<float> convert_to_mono(const vector<float>& left, const vector<float>& right) {
    vector<float> mono;
    mono.resize(left.size());
    for (size_t i = 0; i < left.size(); ++i) {
        mono[i] = (left[i] + right[i]) * 0.5f;
    }
    return mono;
}

AudioData load_audio(const string& path) {
    SndfileHandle file(path.c_str());
    if (file.error()) {
        throw std::runtime_error("Could not open audio file");
    }

    int channels = file.channels();
    int frames = file.frames();
    
    // Read all interleaved samples into a temporary buffer
    vector<float> interleaved(frames * channels);
    file.read(interleaved.data(), interleaved.size());

    AudioData data;
    data.sample_rate = file.samplerate();
    data.left.resize(frames);
    data.right.resize(frames);

    // De-interleave: L, R, L, R -> [L, L...] and [R, R...]
    for (int i = 0; i < frames; ++i) {
        data.left[i] = interleaved[i * channels];
        if (channels > 1) data.right[i] = interleaved[i * channels + 1];
        else data.right[i] = interleaved[i * channels]; // Mono fallback
    }

    data.mono = convert_to_mono(data.left, data.right);

    return data;
}