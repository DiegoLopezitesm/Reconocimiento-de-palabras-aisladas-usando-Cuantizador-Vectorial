#!/usr/bin/env python3

import os, pickle, warnings
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import lfilter
from scipy.fftpack import dct
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────
WORDS          = ["start", "stop", "lift", "drop",
                  "left",  "right", "forward", "back",
                  "pick",  "place"]
FS             = 16_000
FRAME_SIZE     = 320
HOP_SIZE       = 128
LPC_ORDER      = 12
N_MFCC         = 13
CODEBOOK_SIZES = [16, 32, 64]
N_TRAIN        = 10
N_TEST         = 5
DATA_DIR       = "data"
MODEL_DIR      = "models"
RESULT_DIR     = "results"

# ── Modo de features: "lsf" | "mfcc" | "combined"
FEATURE_MODE   = "mfcc"     # ← CAMBIAR AQUÍ para experimentar


# ══════════════════════════════════════════
#  CARGA Y PREÉNFASIS
# ══════════════════════════════════════════
def load_audio(path):
    fs, data = wav.read(path)
    assert fs == FS, f"Se esperaba {FS} Hz, encontrado {fs} en {path}"
    data = data.astype(np.float64)
    if data.ndim > 1:
        data = data.mean(axis=1)
    mx = np.max(np.abs(data))
    return data / mx if mx > 0 else data

def preemphasis(signal, coeff=0.95):
    return lfilter([1.0, -coeff], [1.0], signal)


# ══════════════════════════════════════════
#  ENMARCADO CON HAMMING
# ══════════════════════════════════════════
def frame_signal(signal, frame_size=FRAME_SIZE, hop_size=HOP_SIZE):
    window   = np.hamming(frame_size)
    n_frames = max(1, 1 + (len(signal) - frame_size) // hop_size)
    frames   = np.zeros((n_frames, frame_size))
    for i in range(n_frames):
        s = i * hop_size
        chunk = signal[s: s + frame_size]
        if len(chunk) < frame_size:
            chunk = np.pad(chunk, (0, frame_size - len(chunk)))
        frames[i] = chunk * window
    return frames


# ══════════════════════════════════════════
#  VAD ADAPTATIVO  ← FIX PRINCIPAL
# ══════════════════════════════════════════
def detect_vad(signal, frame_size=FRAME_SIZE, hop_size=HOP_SIZE,
               k_sigma=1.2, margin_frames=4):
    frames   = frame_signal(signal, frame_size, hop_size)
    energy   = np.array([np.sum(f**2) for f in frames])

    # Umbral adaptativo
    e_med    = np.median(energy)
    e_std    = np.std(energy)
    threshold = e_med + k_sigma * e_std

    active   = energy > threshold

    idxs = np.where(active)[0]
    if len(idxs) < 2:
        # Si no se detecta voz, devolver la región de mayor energía
        peak = np.argmax(energy)
        i0   = max(0, peak - 10)
        i1   = min(len(frames) - 1, peak + 10)
    else:
        i0 = max(0, idxs[0]  - margin_frames)
        i1 = min(len(frames) - 1, idxs[-1] + margin_frames)

    start_s = i0 * hop_size
    end_s   = min(i1 * hop_size + frame_size, len(signal))
    seg     = signal[start_s: end_s]

    return seg, energy, threshold, frames

def vad_simple(signal):
    """Wrapper que solo devuelve la señal recortada."""
    seg, _, _, _ = detect_vad(signal)
    return seg


# ══════════════════════════════════════════
#  MFCC  (13 coeficientes)
# ══════════════════════════════════════════
MEL_FILTERS = None   # cache

def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)

def build_mel_filterbank(n_filters=26, n_fft=512, fs=FS):
    global MEL_FILTERS
    if MEL_FILTERS is not None:
        return MEL_FILTERS

    mel_low  = hz_to_mel(0)
    mel_high = hz_to_mel(fs / 2)
    mel_pts  = np.linspace(mel_low, mel_high, n_filters + 2)
    hz_pts   = mel_to_hz(mel_pts)
    bin_pts  = np.floor((n_fft + 1) * hz_pts / fs).astype(int)

    fbank    = np.zeros((n_filters, n_fft // 2 + 1))
    for m in range(1, n_filters + 1):
        for k in range(bin_pts[m - 1], bin_pts[m]):
            fbank[m-1, k] = (k - bin_pts[m-1]) / (bin_pts[m] - bin_pts[m-1] + 1e-9)
        for k in range(bin_pts[m], bin_pts[m + 1] + 1):
            fbank[m-1, k] = (bin_pts[m+1] - k) / (bin_pts[m+1] - bin_pts[m] + 1e-9)

    MEL_FILTERS = fbank
    return fbank

def compute_mfcc(frame, n_mfcc=N_MFCC, n_fft=512):
    """MFCC-13 para una trama."""
    spec     = np.abs(np.fft.rfft(frame, n_fft)) ** 2
    fbank    = build_mel_filterbank(n_fft=n_fft)
    mel_e    = np.dot(fbank, spec)
    mel_e    = np.where(mel_e > 0, mel_e, 1e-10)
    log_mel  = np.log(mel_e)
    mfcc     = dct(log_mel, type=2, norm="ortho")[:n_mfcc]
    return mfcc


# ══════════════════════════════════════════
#  LPC y LSF  (sin cambios)
# ══════════════════════════════════════════
def autocorrelation(frame, order):
    r = np.correlate(frame, frame, mode='full')
    mid = len(r) // 2
    return r[mid: mid + order + 1]

def levinson_durbin(r, order):
    a = np.zeros(order)
    E = r[0]
    if E < 1e-12:
        return a, 1.0
    for i in range(order):
        lam = r[i + 1] - np.dot(a[:i], r[i:0:-1]) if i > 0 else r[1]
        k = lam / (E + 1e-12)
        a_new    = a.copy()
        a_new[i] = k
        if i > 0:
            a_new[:i] = a[:i] + k * a[i - 1::-1][:i]
        a = a_new
        E = max(E * (1.0 - k**2), 1e-12)
    return a, float(E)

def compute_lpc(frame, order=LPC_ORDER):
    r = autocorrelation(frame, order)
    return levinson_durbin(r, order)

def lpc_to_lsf(lpc):
    p   = len(lpc)
    a   = np.concatenate(([1.0], lpc))
    ar  = a[::-1]
    P   = a + ar
    Q   = a - ar
    try:
        P = np.polydiv(P, [1.0,  1.0])[0]
        Q = np.polydiv(Q, [1.0, -1.0])[0]
        rP = np.roots(P)
        rQ = np.roots(Q)
        def unit_upper(roots):
            return np.angle(roots[
                (np.abs(np.abs(roots) - 1.0) < 0.05) & (np.imag(roots) >= 0)
            ])
        lsf = np.sort(np.concatenate([unit_upper(rP), unit_upper(rQ)]))
        if len(lsf) >= p:
            return lsf[:p]
    except Exception:
        pass
    # fallback: distribución uniforme
    return np.linspace(0.1, np.pi - 0.1, p)


# ══════════════════════════════════════════
#  EXTRACCIÓN DE FEATURES
# ══════════════════════════════════════════
def extract_features(path, mode=FEATURE_MODE):
    """
    Retorna matriz de features (F × D) solo de tramas con VOZ.
    mode: "lsf" | "mfcc" | "combined"
    """
    audio  = load_audio(path)
    audio  = preemphasis(audio)
    seg, energy_full, thr, frames_full = detect_vad(audio)
    frames = frame_signal(seg)

    # Energía de cada trama del segmento
    frame_e = np.array([np.sum(f**2) for f in frames])

    feats = []
    for i, fr in enumerate(frames):
        if mode == "mfcc":
            feat = compute_mfcc(fr)
        elif mode == "lsf":
            lpc, _ = compute_lpc(fr)
            feat   = lpc_to_lsf(lpc)
        else:  # combined
            mfcc_f = compute_mfcc(fr)
            lpc, _ = compute_lpc(fr)
            lsf_f  = lpc_to_lsf(lpc)
            # Normalizar cada grupo al rango [0,1] antes de concatenar
            mfcc_n = (mfcc_f - mfcc_f.mean()) / (mfcc_f.std() + 1e-8)
            lsf_n  = lsf_f / np.pi
            feat   = np.concatenate([mfcc_n, lsf_n])
        feats.append(feat)

    feats = np.array(feats) if feats else np.zeros((1, N_MFCC))

    # Normalización CMVN (media y varianza) por utterance → quita canal
    feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-8)

    return feats, frame_e


# ══════════════════════════════════════════
#  LBG  (sin cambios esenciales)
# ══════════════════════════════════════════
def lbg_vq(data, target_size, eps=1e-3, max_iter=150):
    dim      = data.shape[1]
    codebook = data.mean(axis=0, keepdims=True)

    while len(codebook) < target_size:
        perturb  = eps * (np.ones(dim) + 1e-4 * np.random.randn(dim))
        codebook = np.vstack([codebook + perturb, codebook - perturb])
        codebook = codebook[:target_size]

        prev_D = np.inf
        for _ in range(max_iter):
            diff  = data[:, None, :] - codebook[None, :, :]
            dists = np.sum(diff**2, axis=2)
            idx   = np.argmin(dists, axis=1)
            new_cb = np.zeros_like(codebook)
            for k in range(len(codebook)):
                members = data[idx == k]
                new_cb[k] = members.mean(axis=0) if len(members) > 0 else codebook[k]
            D = np.mean(np.min(dists, axis=1))
            codebook = new_cb
            if abs(prev_D - D) / (prev_D + 1e-12) < eps:
                break
            prev_D = D
    return codebook


# ══════════════════════════════════════════
#  ENTRENAMIENTO
# ══════════════════════════════════════════
def train_codebooks(words, codebook_size, n_train=N_TRAIN, mode=FEATURE_MODE):
    codebooks = {}
    for word in words:
        all_feats = []
        for i in range(1, n_train + 1):
            p = os.path.join(DATA_DIR, word, f"{word}_{i:02d}.wav")
            if not os.path.exists(p):
                print(f"  [WARN] No encontrado: {p}")
                continue
            feats, _ = extract_features(p, mode)
            all_feats.append(feats)

        if not all_feats:
            continue
        all_feats = np.vstack(all_feats)
        n_frames  = len(all_feats)
        print(f"  '{word}'  {n_frames} tramas  →  codebook {codebook_size}")
        codebooks[word] = lbg_vq(all_feats, codebook_size)
    return codebooks


# ══════════════════════════════════════════
#  RECONOCIMIENTO  ← SCORE MEJORADO
# ══════════════════════════════════════════
def recognize(path, codebooks, mode=FEATURE_MODE):
    """
    Score = distorsión media ponderada por energía.
    Las tramas más energéticas (núcleo de la palabra) pesan más.
    """
    feats, frame_e = extract_features(path, mode)

    # Peso proporcional a la energía de cada trama
    weights = frame_e + 1e-12
    weights = weights[:len(feats)]          # alinear tamaños
    weights = weights / weights.sum()

    scores = {}
    for word, cb in codebooks.items():
        diff  = feats[:, None, :] - cb[None, :, :]
        dists = np.sum(diff**2, axis=2)        # (F, K)
        min_d = np.min(dists, axis=1)          # (F,)
        scores[word] = float(np.dot(weights, min_d))

    best = min(scores, key=scores.get)
    return best, scores


# ══════════════════════════════════════════
#  EVALUACIÓN
# ══════════════════════════════════════════
def evaluate(words, codebooks, n_train=N_TRAIN, n_total=N_TRAIN + N_TEST,
             mode=FEATURE_MODE):
    label2idx = {w: i for i, w in enumerate(words)}
    y_true, y_pred = [], []

    for word in words:
        for i in range(n_train + 1, n_total + 1):
            p = os.path.join(DATA_DIR, word, f"{word}_{i:02d}.wav")
            if not os.path.exists(p):
                continue
            pred, _ = recognize(p, codebooks, mode)
            y_true.append(label2idx[word])
            y_pred.append(label2idx[pred])
            ok = "✓" if pred == word else "✗"
            print(f"  {word}_{i:02d}  →  {pred:10s}  {ok}")

    n  = len(words)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    acc = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0.0
    return cm, acc


# ══════════════════════════════════════════
#  GRÁFICAS
# ══════════════════════════════════════════
def plot_vad_comparison(word="start", idx=1):
    """Muestra diferencia entre VAD anterior (fijo) y nuevo (adaptativo)."""
    p = os.path.join(DATA_DIR, word, f"{word}_{idx:02d}.wav")
    if not os.path.exists(p):
        return
    audio = preemphasis(load_audio(p))
    frames_all = frame_signal(audio)
    e_all      = np.array([np.sum(f**2) for f in frames_all])

    # VAD anterior: umbral fijo 2% del máximo
    th_old = 0.02 * e_all.max()
    # VAD nuevo: adaptativo
    _, _, th_new, _ = detect_vad(audio)
    seg_new, _, _, _ = detect_vad(audio)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(e_all, color="steelblue")
    axes[0].axhline(th_old, color="red",    ls="--", label=f"Umbral anterior: {th_old:.4f}")
    axes[0].axhline(th_new, color="green",  ls="--", label=f"Umbral adaptativo: {th_new:.4f}")
    axes[0].set_title("Energía por trama")
    axes[0].legend(fontsize=8)
    axes[0].set_xlabel("Trama")

    n_old = np.sum(e_all > th_old)
    n_new = np.sum(e_all > th_new)
    t_all = np.arange(len(audio)) / FS
    t_new = np.arange(len(seg_new)) / FS

    axes[1].plot(t_all, audio, lw=0.5, color="gray", label=f"Completo ({n_old} tramas activas)")
    axes[1].set_title("Umbral anterior (fijo 2%)")
    axes[1].set_xlabel("Tiempo (s)")

    axes[2].plot(t_new, seg_new, lw=0.5, color="darkorange",
                 label=f"Recortado ({n_new} tramas activas)")
    axes[2].set_title("VAD adaptativo (nuevo)")
    axes[2].set_xlabel("Tiempo (s)")

    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Comparación VAD — {word}_{idx:02d}.wav", fontsize=12)
    plt.tight_layout()
    os.makedirs(RESULT_DIR, exist_ok=True)
    out = os.path.join(RESULT_DIR, f"vad_comparacion_{word}.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  VAD comparación → {out}")

def plot_confusion_matrix(cm, words, title, accuracy, path):
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(words))); ax.set_yticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(words, fontsize=11)
    thresh = cm.max() / 2
    for i in range(len(words)):
        for j in range(len(words)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
    ax.set_title(f"{title}\nExactitud: {accuracy*100:.1f}%", fontsize=13)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Guardada → {path}")


# ══════════════════════════════════════════
#  DIAGNÓSTICO DE TRAMAS POST-VAD
# ══════════════════════════════════════════
def diagnostico_frames():
    """
    Imprime cuántas tramas quedan por archivo tras el VAD.
    Ayuda a verificar que el silencio fue eliminado.
    """
    print("\n[ Diagnóstico VAD — tramas por archivo ]")
    print(f"  {'Archivo':<22}  {'Tramas':>7}  {'Duración(ms)':>12}")
    print("  " + "─" * 45)
    for word in WORDS[:3]:   # muestra solo las primeras 3 palabras
        for i in [1, 5, 10]:
            p = os.path.join(DATA_DIR, word, f"{word}_{i:02d}.wav")
            if not os.path.exists(p):
                continue
            audio = preemphasis(load_audio(p))
            seg, _, _, _ = detect_vad(audio)
            n_fr = len(frame_signal(seg))
            dur  = len(seg) / FS * 1000
            flag = "⚠ MUY LARGO" if n_fr > 100 else ("⚠ MUY CORTO" if n_fr < 10 else "✓")
            print(f"  {word}_{i:02d}.wav{'':<12}  {n_fr:>7}  {dur:>10.0f}ms  {flag}")


# ══════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════
def main():
    os.makedirs(MODEL_DIR,  exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  PIPELINE MEJORADO — Modo features: {FEATURE_MODE.upper()}")
    print(f"{'='*60}")

    # Diagnóstico VAD
    diagnostico_frames()

    # Comparación visual VAD
    for w in WORDS:
        if os.path.exists(os.path.join(DATA_DIR, w, f"{w}_01.wav")):
            plot_vad_comparison(w)
            break

    summary = {}
    for cb_size in CODEBOOK_SIZES:
        print(f"\n{'='*60}")
        print(f"  CODEBOOK {cb_size}  |  Features: {FEATURE_MODE}")
        print(f"{'='*60}")

        print("\n[ Entrenamiento ]")
        cbs = train_codebooks(WORDS, cb_size)
        with open(os.path.join(MODEL_DIR, f"cbs_{FEATURE_MODE}_{cb_size}.pkl"), "wb") as f:
            pickle.dump(cbs, f)

        print("\n[ Evaluación ]")
        cm, acc = evaluate(WORDS, cbs)
        summary[cb_size] = acc
        print(f"\n  Exactitud (codebook={cb_size}): {acc*100:.1f}%")
        plot_confusion_matrix(
            cm, WORDS,
            f"Matriz de Confusión [{FEATURE_MODE.upper()}]  K={cb_size}",
            acc,
            os.path.join(RESULT_DIR, f"cm_{FEATURE_MODE}_{cb_size}.png")
        )

    print(f"\n{'='*60}")
    print(f"  RESUMEN — {FEATURE_MODE.upper()}")
    print(f"{'='*60}")
    for size, acc in summary.items():
        bar = "█" * int(acc * 40)
        print(f"  K={size:<3}  {bar:<40}  {acc*100:.1f}%")
    best = max(summary, key=summary.get)
    print(f"\n  ✓ Mejor: K={best}  ({summary[best]*100:.1f}%)")

    # ── Comparar los 3 modos si ya existen resultados previos
    print("\n[ Guarda resultados para comparar modos ]")
    with open(os.path.join(RESULT_DIR, f"summary_{FEATURE_MODE}.pkl"), "wb") as f:
        pickle.dump(summary, f)


if __name__ == "__main__":
    main()
