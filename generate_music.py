"""
Генератор нежной фоновой музыки для свадебного приглашения.
Создаёт зацикленную инструментальную петлю (мягкий пэд + колокольное арпеджио),
прогрессия C - G - Am - F. Сохраняет в assets/wedding-music.wav (16-bit mono).
Зависимости: только numpy + стандартная библиотека.
"""
import wave
import struct
import numpy as np

SR = 44100                      # частота дискретизации
CHORD_DUR = 4.0                 # длительность одного аккорда, сек
MASTER = 0.32                   # общая громкость (тихо, ненавязчиво)

# Аккорды (частоты нот, Гц): C-dur, G-dur, A-moll, F-dur
CHORDS = [
    [261.63, 329.63, 392.00],   # C  (C4 E4 G4)
    [196.00, 246.94, 293.66],   # G  (G3 B3 D4)
    [220.00, 261.63, 329.63],   # Am (A3 C4 E4)
    [174.61, 220.00, 261.63],   # F  (F3 A3 C4)
]

def pad(freqs, n):
    """Тёплый пэд: сумма синусоид + слабая 2-я гармоника. Громкость 'дышит'
    по полусинусу — на стыках аккордов амплитуда = 0 (бесшовный цикл, без щелчков)."""
    t = np.arange(n) / SR
    env = np.sin(np.pi * np.arange(n) / n)      # 0 -> 1 -> 0 за аккорд
    sig = np.zeros(n)
    for f in freqs:
        sig += np.sin(2 * np.pi * f * t)                 # основной тон
        sig += 0.25 * np.sin(2 * np.pi * 2 * f * t)      # мягкая 2-я гармоника
    return sig / len(freqs) * env

def bell(freq, n, decay=0.6):
    """Мягкий колокольчик: синус + слабые обертоны, экспоненциальное затухание."""
    t = np.arange(n) / SR
    env = np.exp(-t / decay)
    s = (np.sin(2 * np.pi * freq * t)
         + 0.4 * np.sin(2 * np.pi * 2 * freq * t)
         + 0.15 * np.sin(2 * np.pi * 3 * freq * t))
    return s * env

def build():
    n_chord = int(CHORD_DUR * SR)
    track = np.zeros(len(CHORDS) * n_chord)

    for i, ch in enumerate(CHORDS):
        start = i * n_chord
        # 1) пэд на весь аккорд
        track[start:start + n_chord] += 0.55 * pad(ch, n_chord)

        # 2) арпеджио октавой выше: ноты по очереди каждые 0.5 c
        notes = [f * 2 for f in ch] + [ch[0] * 4]   # 3 ноты + верхняя точка
        step = int(0.5 * SR)
        for k, nf in enumerate(notes):
            pos = start + k * step
            seg = bell(nf, min(int(1.2 * SR), len(track) - pos))
            # к концу аккорда арпеджио тише (чтобы стык был мягким)
            fade = 1.0 - 0.6 * (k / max(1, len(notes) - 1))
            track[pos:pos + len(seg)] += 0.22 * fade * seg

    # нормализация и общая громкость
    track = track / np.max(np.abs(track)) * MASTER
    return track

def save_wav(samples, path):
    data = np.int16(np.clip(samples, -1, 1) * 32767)
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(struct.pack("<%dh" % len(data), *data.tolist()))

if __name__ == "__main__":
    import os
    os.makedirs("assets", exist_ok=True)
    out = os.path.join("assets", "wedding-music.wav")
    save_wav(build(), out)
    print("OK ->", out)
