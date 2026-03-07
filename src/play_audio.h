#pragma once
#include "load_audio.h"

void update_current_IR(AudioData *data, float target_az, float target_el);
int play_audio(AudioData& audio_data);