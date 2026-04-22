# Reconocimiento de Voz con VQ-MFCC — Puzzlebot Navigation

Sistema de reconocimiento de palabras aisladas usando Cuantización Vectorial (VQ) con MFCC-13, aplicado a comandos de navegación para el robot Puzzlebot.

**Exactitud final: 94% (K=32, 10 palabras, mismo locutor)**

---

## Palabras reconocidas

`start` `stop` `lift` `drop` `left` `right` `forward` `back` `pick` `place`

---

## Instalación

```bash
pip install numpy scipy sounddevice matplotlib
```

---

## Uso rápido

```bash
# 1. Grabar las 150 muestras (15 por palabra)
python grabar_palabras.py

# 2. Entrenar codebooks y evaluar
python pipeline_mejorado.py

# 3. Gráficas de diagnóstico
python analisis.py --word start --idx 1
```

---

## Estructura del proyecto

```
voz_puzzlebot/
├── grabar_palabras.py    # Captura de audio (16 kHz, mono)
├── pipeline_mejorado.py  # Pipeline completo: VAD → MFCC → LBG → clasificación
├── analisis.py           # Visualización: señal, espectrograma, codebooks
├── data/
│   └── <palabra>/        # start_01.wav … start_15.wav  (x10 palabras)
├── models/               # Codebooks entrenados (.pkl)
└── results/              # Matrices de confusión y gráficas
```

---

## Pipeline

```
WAV (16 kHz) → Preénfasis → VAD adaptativo → Hamming (320 pts, hop 128)
             → MFCC-13 + CMVN → LBG (K=16/32/64) → Clasificación ponderada
```

| Paso | Parámetro | Valor |
|------|-----------|-------|
| Preénfasis | coeficiente | 0.95 |
| VAD | umbral | mediana + 1.2·σ(energía) |
| Ventana | tamaño / salto | 320 / 128 muestras |
| LPC | orden | 12 |
| MFCC | coeficientes | 13 |
| Codebook | tamaño óptimo | **K = 32** |

---

## Resultados

| Codebook | Exactitud | Errores / 50 |
|----------|-----------|--------------|
| K = 16   | 92%       | 4            |
| K = 32   | **94%**   | **3**        |
| K = 64   | 94%       | 3            |

> K=32 y K=64 dan el mismo resultado — con 10 archivos de entrenamiento, K=64 no aporta más resolución.

Las únicas confusiones sistemáticas son **pick↔lift** (vocal /ɪ/ compartida). Para producción se recomienda reemplazar `pick` por `grab`.

---

## Protocolo de grabación

- Misma persona en las **150 grabaciones**
- Habitación silenciosa, micrófono a ~15 cm
- Archivos `01–10` → entrenamiento · `11–15` → prueba
