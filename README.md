# SideDetectorINE by MicroTec

Detector de lado (frente / reverso) para credenciales INE mexicanas a partir de una foto. Devuelve también la orientación (0° o 180°), una confianza entre 0 y 1, y la imagen ya rectificada en proporción de credencial.

Funciona sobre fotos tomadas a mano alzada: tolera rotación, perspectiva, fondos heterogéneos y orientación volteada. No requiere GPU.


## Características

- **Clasificación frente / reverso** con reglas en cascada sobre varias señales: rostro, QR, densidad de QR, líneas tipo MRZ y franja roja vertical.
- **Detección de orientación** (0° vs 180°): clasifica la imagen normal y rotada 180° y se queda con la hipótesis más fuerte.
- **Localización del card por contorno**: detección de cuadrilátero por bordes (Canny + `approxPolyDP`) con fallback a segmentación por brillo y un segundo fallback "agresivo" (Otsu + `minAreaRect`) cuando la confianza queda baja.
- **Warp en proporción canónica de credencial** (1000 × 630).
- **Detección facial con YuNet** (`face_detection_yunet_2023mar.onnx`) y fallback a Haar Cascade si el modelo no está disponible.
- **Sin dependencias pesadas**: sólo `opencv-python` y `numpy`.


## Requisitos

- Python 3.8+
- `opencv-python` (incluye `cv2.FaceDetectorYN` y `cv2.QRCodeDetector`)
- `numpy`
- Modelo YuNet: `face_detection_yunet_2023mar.onnx`, colocado **junto a `sideDetectorINE_Module.py`**.

```bash
pip install opencv-python numpy
```

El modelo YuNet se descarga del repo oficial de OpenCV Zoo:

```
https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
```

> Si el `.onnx` no está, el módulo cae automáticamente a Haar Cascade (incluido en `opencv-python`). Funciona, pero con menor recall facial.


## Estructura del proyecto

```
.
├── sideDetectorINE_Module.py          # Módulo detector (lógica de visión)
├── tester_ine.py                      # CLI para procesar carpetas
└── face_detection_yunet_2023mar.onnx  # Modelo YuNet (descargar aparte)
```


## Uso

### 1. CLI por carpeta - `tester_ine.py`

Procesa una carpeta de fotos y las separa en subcarpetas `_frente` y `_reverso`:

```bash
python tester_ine.py /ruta/a/fotos
```

Por defecto **copia** los archivos. Para mover:

```bash
python tester_ine.py /ruta/a/fotos --move
```

Resultado:

```
fotos/                  # entrada original
fotos_frente/           # creadas por el script
fotos_reverso/
```

Salida en consola, una línea por imagen:

```
[ 1/42] ine_001.jpg -> frente  (conf=0.95)
[ 2/42] ine_002.jpg -> reverso (conf=0.90)
[ 3/42] ine_003.jpg -> SKIP (no se pudo leer la imagen)
...

Frente: 28  Reverso: 13  Skip: 1
```

### 2. CLI del módulo (modo verbose)

El módulo trae su propio `__main__` con scores detallados:

```bash
python sideDetectorINE_Module.py /ruta/a/fotos --verbose
python sideDetectorINE_Module.py una_foto.jpg --min-confidence 0.6
```

Flags:

- `--min-confidence FLOAT`: si la confianza queda por debajo, el lado se reporta como `unknown`.
- `--skip-warp`: usa la imagen tal cual (cuando ya viene recortada en proporción de credencial).
- `--verbose`: imprime todos los scores intermedios (`n_faces_0`, `qr_density_180`, `mrz_lines_0`, etc.).


### 3. Como librería

**Uso rápido** (un solo lado):

```python
from sideDetectorINE_Module import detect_side, INESide

side = detect_side("foto.jpg")
if side is INESide.FRONT:
    print("frente")
elif side is INESide.BACK:
    print("reverso")
else:
    print("no se pudo determinar")
```

**Uso completo** (con scores y warp):

```python
from sideDetectorINE_Module import INEDetector

det = INEDetector(min_confidence=0.5)
r = det.detect("foto.jpg")

if not r.ok:
    print("error:", r.error)
else:
    print(r.side.value)        # "front" / "back" / "unknown"
    print(r.confidence)        # 0.0 - 1.0
    print(r.orientation)       # 0 o 180
    print(r.scores)             # dict con todas las señales
    # r.warped       -> ndarray BGR 1000x630 (credencial rectificada y derecha)
    # r.card_quad    -> 4 puntos del cuadrilátero detectado en la imagen original
```

También acepta `np.ndarray` BGR directamente, no sólo rutas:

```python
import cv2
img = cv2.imread("foto.jpg")
r = det.detect(img)
```


## Parámetros del `INEDetector`

| Parámetro          | Default | Qué hace                                                                                                                                                    |
| ------------------ | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `min_confidence`   | `0.0`   | Si la confianza queda debajo, el lado se reporta como `INESide.UNKNOWN` aunque internamente haya una hipótesis.                                             |
| `skip_warp`        | `False` | No intenta localizar el card. Útil cuando ya tienes recortes limpios.                                                                                       |
| `face_detector`    | `None`  | Callable `(BGR ndarray) -> [(x, y, w, h), ...]`. Permite inyectar otro detector facial (por ejemplo MediaPipe).                                             |
| `yunet_model`      | `None`  | Ruta a un `.onnx` de YuNet alterno. Default: el que esté junto al módulo.                                                                                   |
| `retry_threshold`  | `0.6`   | Si el primer intento queda con confianza menor a esto (o es `BACK` sin rostro y poco seguro), reintenta con un warp más agresivo. Pon `0.0` para apagarlo.  |


## Cómo funciona

El pipeline a alto nivel:

1. **Localizar el card** en la foto y rectificarlo a 1000×630 (`_warp_card`):
   - Canny + contornos + `approxPolyDP` buscando un cuadrilátero.
   - Fallback por brillo (HSV) si no aparece un contorno claro.
   - Fallback "agresivo" (Otsu + `minAreaRect`) si la confianza final queda baja.
2. **Calcular señales** sobre el warp en orientación 0° y 180° (`_score_orientation`):
   - `n_faces`, `face_area_frac`: rostro en la mitad izquierda (zona de la foto del INE).
   - `n_qrs`, `qr_density`: QRs detectados por OpenCV y densidad de píxeles oscuros en las ROIs típicas del reverso.
   - `mrz_lines`: número de "líneas largas y delgadas" en la franja inferior, similares a un MRZ.
   - `red_strip`: promedio del canal rojo dominante en la franja vertical izquierda (característica del frente del INE).
3. **Reglas en cascada** para elegir lado y orientación (`_classify_with_orientation`). De más fuerte a más débil:
   1. QR + MRZ en la **misma** orientación → reverso (conf ≈ 0.95).
   2. `qr_density ≥ 0.30` → reverso (conf ≈ 0.90).
   3. Rostro en la ROI izquierda → frente (conf 0.85 – 0.95, sube si además hay franja roja).
   4. `qr_density ≥ 0.22` → reverso (conf ≈ 0.70).
   5. Algún QR detectado → reverso (conf ≈ 0.70).
   6. 3+ líneas MRZ → reverso (conf ≈ 0.75).
   7. MRZ + densidad QR media en la misma orientación → reverso (conf ≈ 0.65).
   8. Fallback: gana la orientación con más franja roja → frente (conf 0.15 – 0.80 según red strip).

Las señales de reverso **deben corroborarse dentro de la misma orientación** (un QR en rot 180 + MRZ en rot 0 no cuenta como evidencia conjunta, suele ser un OVD fantasma más renglones del frente).


## Notas y limitaciones

- El warp asume **landscape**: si el card aparece muy vertical, `_order_points_landscape` lo fuerza horizontal antes de aplicar la transformada.
- La detección de QR usa `cv2.QRCodeDetector`, que en versiones de OpenCV menores a 4.5 puede ser inestable. 
- La señal de "franja roja" depende de iluminación. Bajo luz muy amarilla o blanco quemado, conviene apoyarse en la confianza y filtrar con `min_confidence`.
- `skip_warp=True` sólo tiene sentido si las imágenes ya vienen recortadas y enderezadas a proporción de credencial.


## Licencia

The Unlicense.
