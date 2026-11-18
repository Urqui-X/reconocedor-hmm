# Reconocedor de caracteres 2D/3D con HMM — Procesos Estocásticos (603504)

Python 3.11+ · **la lógica de HMM usa solo NumPy** (sin hmmlearn/sklearn/scipy) · GUI con Tkinter.
(`matplotlib` solo se usa en `analyze_similarity.py` para las figuras; no es parte de la app.)

## Ejecución
```bash
pip install -r requirements.txt
python main.py                 # los modelos ya vienen entrenados en models/
python train_models.py         # (opcional) reentrena offline con data/glifo-dataset.json
python analyze_similarity.py   # (opcional) análisis de similitud, figuras en analisis/
python test_engine.py          # Forward/Viterbi contra fuerza bruta y Baum-Welch no decrece
```
Si Tkinter falta en Linux: `sudo apt install python3-tk`.

## Arquitectura
| Archivo | Rol |
|---|---|
| `HMMEngine.py` | HMM λ=(A,B,π), Forward/Backward escalados, Viterbi, Baum-Welch, proyección futura, `HMMClassifier` (27 modelos) |
| `FeatureExtractor.py` | Remuestreo, Chain Code K=16 (+ símbolo «levantar el lápiz»), coordenadas esféricas (Δθ, Δφ·senθ) |
| `Geometry.py` | Rasgos geométricos por letra y refinador por pares (x/y, d/a, p/b…) |
| `Rejection.py` | Umbral 20 % + ajuste absoluto → «no se reconoció ningún gesto» |
| `Session.py` | Lógica en vivo sin GUI: puntos → símbolos → posterior → rechazo → proyección escalada |
| `Training.py`, `train_models.py` | Entrenamiento offline con el dataset real (Baum-Welch) y evaluación |
| `real_dataset.py` | Lee `glifo-dataset.json` (trazos con múltiples segmentos) |
| `Gestures.py` | Garabatos para probar el rechazo |
| `GUI.py`, `main.py` | Menú (A), Reconocedor 2D (B), Reconocedor 3D (C) |
| `analyze_similarity.py` | Análisis de similitud entre trazos (figuras en `analisis/`) |
| `LetterTemplates.py` | Plantillas de letras (iconos de respaldo del panel) |

## Matemática (para el tablero)
- Un HMM por letra, izquierda-derecha (Bakis), N estados ≈ longitud media/3. Observaciones: 16 direcciones + 1 símbolo de levantar el lápiz.
- Forward: α₁(i)=πᵢbᵢ(O₁); α_{t+1}(j)=[Σᵢ αₜ(i)aᵢⱼ]bⱼ(O_{t+1}); log P(O|λ)=Σ log cₜ.
- Posterior: P(c|O₁:t)=softmax(log P(O₁:t|λc)/T), prior uniforme (T=2.5 solo calibra la confianza mostrada).
- Al finalizar: P(O, q_T=N | λ) + log-verosimilitud geométrica; el refinador decide los pares confundibles.
- Proyección: ĉ=argmax; estado actual por Viterbi; transición más probable hasta el final, 1/(1−a_jj) pasos por estado (acotado a 8), símbolo = media circular de b_j; paso escalado a tu tamaño.
- Baum-Welch multi-secuencia (γ, ξ).

## Uso de la GUI
- **Siempre visible el modo:** cursor de lápiz (dibujar) / mano (rotar, 3D) / cruz de 4 flechas (arrastrando la esfera), borde y etiqueta de color.
- **2D:** puedes levantar el mouse entre trazos; un clic sin mover dibuja un punto. **Limpiar lienzo** borra lo dibujado y conserva el último resultado; **Reiniciar captura** además reinicia reconocedor y panel; **Finalizar letra** (o Enter) cierra la clasificación.
- **3D:** clic izq. dibuja; rotar con clic der., Shift o tecla R; rueda = zoom. Reiniciar captura también restablece la vista.
- **Sentido del trazo:** se aprende como se escribió (punto verde del icono = inicio): `l` e `i` de arriba hacia abajo.
- Si ninguna letra supera el 20 % (o el trazo no se parece a ninguna), se muestra «No se reconoció ningún gesto».

## Resultados (letras NO vistas, entrenado con datos reales)
2D: 98.8 % · 3D: 97.2 % (letra terminada, sin rechazo). Con rechazo activo: 2D 97.0 % bien / 0.5 % error aceptado; 3D 95.2 % / 0.5 %. Un trazo rechazado por no parecerse a ninguna letra muestra 0 % en todas las letras.
