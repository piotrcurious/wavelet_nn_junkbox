import numpy as np
import soundfile as sf
import os
import random

def generate_tone(freq, duration, sr=16000, harmonics=True):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    x = 0.5 * np.sin(2 * np.pi * freq * t)
    if harmonics:
        # Add some harmonics
        x += 0.2 * np.sin(4 * np.pi * freq * t)
        x += 0.1 * np.sin(6 * np.pi * freq * t)
        x += 0.05 * np.sin(8 * np.pi * freq * t)
    # Apply ADR envelope
    env = np.ones_like(x)
    att_len = int(0.1 * sr)
    rel_len = int(0.2 * sr)
    env[:att_len] = np.linspace(0, 1, att_len)
    env[-rel_len:] = np.linspace(1, 0, rel_len)
    return (x * env).astype(np.float32)

def generate_fm_tone(carrier_freq, mod_freq, mod_index, duration, sr=16000):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    modulator = mod_index * np.sin(2 * np.pi * mod_freq * t)
    x = 0.5 * np.sin(2 * np.pi * carrier_freq * t + modulator)
    return x.astype(np.float32)

def generate_noise(duration, sr=16000):
    return (np.random.randn(int(sr * duration)) * 0.1).astype(np.float32)

def setup_mock_data():
    sr = 16000
    # Violin mock (rich harmonics, some FM for vibrato)
    os.makedirs('data/violin_ref', exist_ok=True)
    for i in range(5):
        base_freq = 440 * (1.2 ** (i - 2))
        v = generate_fm_tone(base_freq, 6, 2, 2.0, sr) # 6Hz vibrato
        sf.write(f'data/violin_ref/v{i}.wav', v, sr)

    # Flute mock (pure-ish tones, some breath noise)
    os.makedirs('data/flute_ref', exist_ok=True)
    for i in range(5):
        base_freq = 660 * (1.1 ** (i - 2))
        f = generate_tone(base_freq, 2.0, sr, harmonics=False)
        noise = generate_noise(2.0, sr) * 0.05
        f += noise
        sf.write(f'data/flute_ref/f{i}.wav', f, sr)

    # Background mock (environmental noise)
    os.makedirs('data/background_scenes', exist_ok=True)
    for i in range(5):
        bg = generate_noise(5.0, sr)
        # add some low frequency rumble
        t = np.linspace(0, 5.0, int(sr * 5.0), endpoint=False)
        rumble = 0.1 * np.sin(2 * np.pi * 50 * t)
        bg += rumble
        sf.write(f'data/background_scenes/bg{i}.wav', bg, sr)

    # MIDI mixtures simulation (mix of instruments)
    os.makedirs('data/midi_mixtures', exist_ok=True)
    for i in range(5):
        v = generate_fm_tone(440 * random.uniform(0.8, 1.2), 6, 2, 3.0, sr)
        f = generate_tone(660 * random.uniform(0.8, 1.2), 3.0, sr)
        bg = generate_noise(3.0, sr) * 0.2
        mix = v + f + bg
        mix = mix / (np.max(np.abs(mix)) + 1e-9)
        sf.write(f'data/midi_mixtures/mix{i}.wav', mix, sr)

if __name__ == "__main__":
    setup_mock_data()
