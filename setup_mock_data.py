import numpy as np
import soundfile as sf
import os
import random
import pretty_midi

def generate_tone(freq, duration, sr=16000, harmonics=True):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    x = 0.5 * np.sin(2 * np.pi * freq * t)
    if harmonics:
        # Add some harmonics with random variations
        x += 0.2 * np.sin(4 * np.pi * freq * t + random.uniform(0, np.pi))
        x += 0.1 * np.sin(6 * np.pi * freq * t + random.uniform(0, np.pi))
        x += 0.05 * np.sin(8 * np.pi * freq * t + random.uniform(0, np.pi))

    # Apply ADSR envelope
    env = np.ones_like(x)
    att_len = int(min(0.1 * sr, len(x)*0.1))
    rel_len = int(min(0.2 * sr, len(x)*0.2))
    env[:att_len] = np.linspace(0, 1, att_len)
    env[-rel_len:] = np.linspace(1, 0, rel_len)
    return (x * env).astype(np.float32)

def generate_fm_tone(carrier_freq, mod_freq, mod_index, duration, sr=16000):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    modulator = mod_index * np.sin(2 * np.pi * mod_freq * t)
    x = 0.5 * np.sin(2 * np.pi * carrier_freq * t + modulator)

    # Envelope
    env = np.ones_like(x)
    att_len = int(min(0.1 * sr, len(x)*0.1))
    rel_len = int(min(0.2 * sr, len(x)*0.2))
    env[:att_len] = np.linspace(0, 1, att_len)
    env[-rel_len:] = np.linspace(1, 0, rel_len)
    return (x * env).astype(np.float32)

def midi_to_audio(midi_path, sr=16000):
    """
    Synthesize MIDI file into audio using simple additive synthesis.
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    audio = np.zeros(int(pm.get_end_time() * sr) + sr)

    for instrument in pm.instruments:
        for note in instrument.notes:
            start = int(note.start * sr)
            duration = note.end - note.start
            if duration <= 0: continue

            freq = 440.0 * (2.0 ** ((note.pitch - 69.0) / 12.0))
            # Use FM for some instruments, simple tone for others
            if instrument.is_drum:
                # white noise for drums
                sig = (np.random.randn(int(duration * sr)) * 0.2).astype(np.float32)
            elif instrument.program < 32: # Strings/Violin like
                sig = generate_fm_tone(freq, 6, 2, duration, sr)
            else: # Flute/Wind like
                sig = generate_tone(freq, duration, sr, harmonics=False)

            if start + len(sig) > len(audio):
                audio = np.pad(audio, (0, start + len(sig) - len(audio) + sr))

            audio[start:start+len(sig)] += sig * (note.velocity / 127.0)

    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / (np.max(np.abs(audio)) + 1e-9)
    return audio.astype(np.float32)

def create_mock_midi(path, n_notes=10):
    pm = pretty_midi.PrettyMIDI()
    # Violin-like
    v_inst = pretty_midi.Instrument(program=40) # Violin
    # Flute-like
    f_inst = pretty_midi.Instrument(program=73) # Flute

    curr_t = 0.0
    for _ in range(n_notes):
        dur = random.uniform(0.5, 1.5)
        v_pitch = random.randint(50, 70)
        f_pitch = random.randint(60, 80)
        v_inst.notes.append(pretty_midi.Note(velocity=80, pitch=v_pitch, start=curr_t, end=curr_t+dur))
        f_inst.notes.append(pretty_midi.Note(velocity=60, pitch=f_pitch, start=curr_t, end=curr_t+dur))
        curr_t += dur * 0.8

    pm.instruments.append(v_inst)
    pm.instruments.append(f_inst)
    pm.write(path)

def setup_mock_data():
    sr = 16000
    os.makedirs('data/violin_ref', exist_ok=True)
    os.makedirs('data/flute_ref', exist_ok=True)
    os.makedirs('data/background_scenes', exist_ok=True)
    os.makedirs('data/midi_mixtures', exist_ok=True)

    # 1. References
    for i in range(5):
        v = generate_fm_tone(440 * (1.2 ** (i - 2)), 6, 2, 2.0, sr)
        sf.write(f'data/violin_ref/v{i}.wav', v, sr)
        f = generate_tone(660 * (1.1 ** (i - 2)), 2.0, sr, harmonics=False)
        sf.write(f'data/flute_ref/f{i}.wav', f, sr)

    # 2. Background
    for i in range(5):
        bg = (np.random.randn(int(sr * 5.0)) * 0.05).astype(np.float32)
        sf.write(f'data/background_scenes/bg{i}.wav', bg, sr)

    # 3. Polyphonic MIDI mixtures
    for i in range(3):
        midi_path = f'data/midi_mixtures/test_{i}.mid'
        create_mock_midi(midi_path)
        audio = midi_to_audio(midi_path, sr)
        sf.write(f'data/midi_mixtures/mix{i}.wav', audio, sr)

if __name__ == "__main__":
    setup_mock_data()
