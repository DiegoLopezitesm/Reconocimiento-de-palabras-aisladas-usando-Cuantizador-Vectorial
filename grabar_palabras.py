#!/usr/bin/env python3
"""
=============================================================
GRABACIÓN DE PALABRAS — Puzzlebot Navigation
=============================================================
Graba 15 repeticiones de cada palabra a 16 kHz, mono, 16-bit.
Guarda en:
    data/<palabra>/<palabra>_01.wav … <palabra>_15.wav

Dependencias:
    pip install sounddevice scipy numpy

Uso:
    python grabar_palabras.py
    python grabar_palabras.py --word start --start 1  # grabar desde idx 1
=============================================================
"""

import argparse
import os
import time
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav

FS         = 16_000
DURACION   = 2.0          # segundos por grabación
N_TOTAL    = 15           # grabaciones por palabra
DATA_DIR   = "data"

WORDS = [
    "start",   # Iniciar movimiento
    "stop",    # Detener
    "lift",    # Levantar pinza
    "drop",    # Soltar objeto
    "left",    # Girar izquierda
    "right",   # Girar derecha
    "forward", # Avanzar
    "back",    # Retroceder
    "pick",    # Recoger
    "place",   # Colocar
]


def grabar(duracion: float = DURACION, fs: int = FS) -> np.ndarray:
    """Graba `duracion` segundos del micrófono. Retorna ndarray float64."""
    n_samples = int(duracion * fs)
    audio = sd.rec(n_samples, samplerate=fs, channels=1, dtype="float64")
    sd.wait()
    return audio.flatten()

def normalizar_y_guardar(audio: np.ndarray, path: str, fs: int = FS):
    """Normaliza a [-1, 1] y guarda como PCM 16-bit."""
    mx = np.max(np.abs(audio))
    if mx > 0:
        audio = audio / mx
    pcm = (audio * 32767).astype(np.int16)
    wav.write(path, fs, pcm)

def reproducir_feedback(ok: bool):
    """Beep corto de confirmación."""
    t    = np.linspace(0, 0.15, int(0.15 * FS))
    freq = 880 if ok else 440
    tone = 0.3 * np.sin(2 * np.pi * freq * t)
    sd.play(tone, samplerate=FS)
    sd.wait()


def grabar_palabra(word: str, start_idx: int = 1, n_total: int = N_TOTAL):
    word_dir = os.path.join(DATA_DIR, word)
    os.makedirs(word_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  Palabra: '{word.upper()}'")
    print(f"  Grabaciones: {start_idx} – {n_total}")
    print(f"{'='*50}")
    print("  Consejos:")
    print("   • Habitación silenciosa")
    print("   • Misma distancia al micrófono siempre")
    print("   • Pronunciación clara y natural")
    print("   • Duración aprox. 0.5–1.5 s por palabra")
    input("\n  Presiona ENTER cuando estés listo…\n")

    for i in range(start_idx, n_total + 1):
        path = os.path.join(word_dir, f"{word}_{i:02d}.wav")

        while True:
            print(f"  [{i:2d}/{n_total}]  Di: '{word}'  (grabando {DURACION:.1f}s)…", end="", flush=True)
            time.sleep(0.5)          # pausa breve antes de grabar

            audio = grabar()
            normalizar_y_guardar(audio, path)

            # Revisar energía mínima (detectar silencio accidental)
            energia = np.sum(audio ** 2) / len(audio)
            if energia < 1e-5:
                print(" ⚠  Silencio detectado. ¿Grabamos de nuevo? [s/n]: ", end="")
                resp = input().strip().lower()
                if resp != "n":
                    continue

            print(f"  ✓ Guardado: {path}")
            reproducir_feedback(True)
            break

        if i < n_total:
            time.sleep(0.8)   # pausa entre grabaciones

    print(f"\n  ✓  '{word}' completado ({n_total} grabaciones)\n")


def main():
    parser = argparse.ArgumentParser(description="Grabador de palabras para VQ-LPC")
    parser.add_argument("--word",  default=None, help="Grabar solo esta palabra")
    parser.add_argument("--start", type=int, default=1, help="Índice inicial de grabación")
    parser.add_argument("--list",  action="store_true", help="Listar palabras y salir")
    args = parser.parse_args()

    if args.list:
        print("Palabras configuradas:")
        for w in WORDS:
            print(f"  {w}")
        return

    words_to_record = [args.word] if args.word else WORDS

    print("\n╔══════════════════════════════════════════╗")
    print("║   GRABADOR VQ-LPC — Puzzlebot Navigation  ║")
    print("╚══════════════════════════════════════════╝")
    print(f"  Frecuencia de muestreo: {FS} Hz")
    print(f"  Duración por grabación: {DURACION} s")
    print(f"  Grabaciones por palabra: {N_TOTAL}")
    print(f"  Palabras: {', '.join(words_to_record)}")

    for w in words_to_record:
        grabar_palabra(w, start_idx=args.start)

    print("\n  ¡Grabación completa!")
    print(f"  Archivos en: {os.path.abspath(DATA_DIR)}/")


if __name__ == "__main__":
    main()
