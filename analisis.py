#!/usr/bin/env python3

import argparse
import os
import pickle
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pipeline_voz import (
    WORDS, FS, FRAME_SIZE, HOP_SIZE, LPC_ORDER,
    CODEBOOK_SIZES, DATA_DIR, MODEL_DIR, RESULT_DIR,
    load_audio, preemphasis, detect_vad, frame_signal,
    compute_lpc, lpc_to_lsf, extract_features
)


def analizar_palabra(word: str, idx: int = 1):
    """Figura completa del pipeline para una grabación."""
    path = os.path.join(DATA_DIR, word, f"{word}_{idx:02d}.wav")
    if not os.path.exists(path):
        print(f"No se encontró: {path}")
        return

    raw   = load_audio(path)
    pre   = preemphasis(raw)
    seg, _, _, _ = detect_vad(pre)  # nuevo pipeline retorna (seg, energy, threshold, frames)
    frms  = frame_signal(seg)

    t_raw = np.arange(len(raw)) / FS
    t_seg = np.arange(len(seg)) / FS
    energias = np.array([np.sum(f**2) for f in frms])

    # LPC y LSF de la trama con mayor energía
    best_fr = frms[np.argmax(energias)]
    lpc, g  = compute_lpc(best_fr)
    lsf     = lpc_to_lsf(lpc)

    # Espectro LPC vs FFT
    nfft   = 512
    freqs  = np.linspace(0, FS / 2, nfft // 2 + 1)
    fft_sp = 20 * np.log10(np.abs(np.fft.rfft(best_fr, nfft)) + 1e-6)
    a_full = np.concatenate(([1.0], lpc))
    lpc_sp = 20 * np.log10(np.sqrt(g) /
                            (np.abs(np.fft.rfft(a_full, nfft)) + 1e-12))

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(t_raw, raw, lw=0.6, color="steelblue")
    ax1.set_title("Señal cruda"); ax1.set_xlabel("Tiempo (s)")

    ax2 = fig.add_subplot(gs[1, :2])
    ax2.plot(t_seg, seg, lw=0.6, color="darkorange")
    ax2.set_title("Preénfasis + VAD"); ax2.set_xlabel("Tiempo (s)")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(energias, marker="o", ms=3, color="green")
    ax3.set_title("Energía por trama"); ax3.set_xlabel("Trama")
    ax3.axhline(0.02 * energias.max(), color="red", ls="--", label="umbral")
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.specgram(seg, NFFT=256, Fs=FS, noverlap=128, cmap="inferno")
    ax4.set_title("Espectrograma"); ax4.set_xlabel("Tiempo (s)")
    ax4.set_ylabel("Freq (Hz)")

    ax5 = fig.add_subplot(gs[2, :2])
    ax5.plot(freqs, fft_sp,   lw=0.8, alpha=0.6, label="FFT trama", color="gray")
    ax5.plot(freqs, lpc_sp,   lw=1.8, label=f"Envolvente LPC-{LPC_ORDER}", color="crimson")
    ax5.set_xlim(0, FS / 2); ax5.set_ylim(-80, 30)
    ax5.set_title("Espectro vs Envolvente LPC")
    ax5.set_xlabel("Frecuencia (Hz)"); ax5.set_ylabel("dB")
    ax5.legend(fontsize=8)

    ax6 = fig.add_subplot(gs[2, 2])
    lsf_hz = lsf * FS / (2 * np.pi)
    ax6.stem(range(1, LPC_ORDER + 1), lsf_hz, basefmt=" ")
    ax6.set_title("LSF (mejor trama)")
    ax6.set_xlabel("Índice LSF"); ax6.set_ylabel("Frecuencia (Hz)")

    fig.suptitle(f"Análisis: {word}_{idx:02d}.wav", fontsize=14, fontweight="bold")

    os.makedirs(RESULT_DIR, exist_ok=True)
    out = os.path.join(RESULT_DIR, f"analisis_{word}_{idx:02d}.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  Guardado → {out}")


def graficar_codebook(word: str, cb_size: int):
    """Scatter de los codevectores LSF[0] vs LSF[1] para una palabra."""
    model_path = os.path.join(MODEL_DIR, f"codebooks_{cb_size}.pkl")
    if not os.path.exists(model_path):
        print(f"Modelo no encontrado: {model_path}  (entrena primero con pipeline_voz.py)")
        return
    with open(model_path, "rb") as f:
        codebooks = pickle.load(f)

    if word not in codebooks:
        print(f"'{word}' no está en el modelo.")
        return

    cb  = codebooks[word]   # (K, LPC_ORDER)
    # Recopilar también los LSF de entrenamiento
    all_lsf = []
    for i in range(1, 11):
        p = os.path.join(DATA_DIR, word, f"{word}_{i:02d}.wav")
        if os.path.exists(p):
            _, lsf, _ = extract_features(p)
            all_lsf.append(lsf)
    if not all_lsf:
        return
    all_lsf = np.vstack(all_lsf)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(all_lsf[:, 0], all_lsf[:, 1],
               alpha=0.15, s=10, color="cornflowerblue", label="tramas entrenamiento")
    ax.scatter(cb[:, 0], cb[:, 1],
               s=80, color="red", zorder=5, marker="*", label="codevectores")
    ax.set_xlabel("LSF 1 (rad)"); ax.set_ylabel("LSF 2 (rad)")
    ax.set_title(f"Codebook '{word}'  (K={cb_size})")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(RESULT_DIR, f"codebook_{word}_{cb_size}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  Guardado → {out}")


def comparar_codebook_sizes():
    """Barra de exactitud por tamaño de codebook (lee los .pkl del pipeline mejorado)."""
    accs = {}
    for mode in ["mfcc", "lsf", "combined"]:
        summary_path = os.path.join(RESULT_DIR, f"summary_{mode}.pkl")
        if os.path.exists(summary_path):
            with open(summary_path, "rb") as f:
                summary = pickle.load(f)
            for cb_size, acc in summary.items():
                if cb_size not in accs or acc > accs[cb_size]:
                    accs[cb_size] = acc

    if not accs:
        print("No se encontraron logs de exactitud.")
        print("Corre pipeline_mejorado.py primero.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    sizes = list(accs.keys())
    vals  = [accs[s] * 100 for s in sizes]
    bars  = ax.bar([str(s) for s in sizes], vals, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_ylim(0, 110)
    ax.set_xlabel("Tamaño de codebook"); ax.set_ylabel("Exactitud (%)")
    ax.set_title("Comparación de tamaños de codebook")
    plt.tight_layout()
    out = os.path.join(RESULT_DIR, "comparacion_codebooks.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  Guardado → {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--word",  default=None, help="Palabra a analizar")
    parser.add_argument("--idx",   type=int, default=1, help="Índice de grabación (1-15)")
    parser.add_argument("--cb",    type=int, default=None, help="Tamaño de codebook a graficar")
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)

    words = [args.word] if args.word else WORDS[:3]  # por defecto primeras 3

    for w in words:
        print(f"\nAnalizando '{w}'…")
        analizar_palabra(w, args.idx)

    if args.cb:
        for w in words:
            graficar_codebook(w, args.cb)

    comparar_codebook_sizes()


if __name__ == "__main__":
    main()