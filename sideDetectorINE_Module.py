"""
Mini Documentación :D

Uso rapido:
    - side = sideDetectorINE_Module.detect_side("foto.jpg")
Uso con scores:
    - r = sideDetectorINE_Module.INEDetector().detect("foto.jpg")
    - print(r.side, r.confidence, r.orientation)
"""

from __future__ import annotations
 
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
 
import cv2
import numpy as np
 
 
class INESide(str, Enum):
    FRONT = "front"
    BACK = "back"
    UNKNOWN = "unknown"
 
 
@dataclass
class INEDetectionResult:
    side: INESide
    confidence: float
    orientation: int = 0
    scores: Dict[str, float] = field(default_factory=dict)
    warped: Optional[np.ndarray] = None
    card_quad: Optional[np.ndarray] = None
    ok: bool = True
    error: Optional[str] = None
 
 
_WARP_W = 1000
_WARP_H = 630
 
_BACK_QR_ROIS = (
    (0.05, 0.22, 0.34, 0.69),
    (0.34, 0.22, 0.63, 0.69),
    (0.64, 0.22, 0.82, 0.52),
)
_FRONT_FACE_ROI = (0.04, 0.18, 0.40, 0.85)
 
_BACK_MRZ_MIN = 2
_BACK_MRZ_STRONG = 3
_BACK_QR_DENSITY_MIN = 0.20
_BACK_QR_DENSITY_MID = 0.22
_BACK_QR_DENSITY_STRONG = 0.30
_BACK_QR_COUNT_MIN = 1
_FRONT_FACE_MIN_AREA = 0.005
_FRONT_RED_MIN = 5.0
 
_DEFAULT_YUNET_MODEL = Path(__file__).parent / "face_detection_yunet_2023mar.onnx"
 
 
@dataclass
class _WarpResult:
    warped: np.ndarray
    M: np.ndarray
    quad: np.ndarray
 
 
def _order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    rect = np.zeros((4, 2), dtype=np.float32)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect
 
 
def _order_points_landscape(pts):
    """Ordena 4 puntos forzando orientacion landscape (lado largo horizontal).
 
    Si el quad esta rotado >45 grados, _order_points lo dejaria portrait.
    Aqui detecto eso y roto el orden 90 grados para forzar landscape, que es
    lo que necesita el warp del INE.
    """
    rect = _order_points(pts)
    tl, tr, br, bl = rect
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    if height > width:
        rect = np.array([bl, tl, tr, br], dtype=np.float32)
    return rect
 
 
def _find_card_quad_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    v = float(np.median(gray))
    edges = cv2.Canny(gray, int(max(0, 0.66 * v)), int(min(255, 1.33 * v)))
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    edges = cv2.erode(edges, np.ones((5, 5), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    H, W = img.shape[:2]
    for c in cnts[:15]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.20 * H * W:
            return approx.reshape(4, 2)
    c = cnts[0]
    if cv2.contourArea(c) > 0.25 * H * W:
        return cv2.boxPoints(cv2.minAreaRect(c))
    return None
 
 
def _find_card_quad_bright(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2].astype(np.float32)
    S = hsv[:, :, 1].astype(np.float32)
    v_thr = float(np.percentile(V, 60))
    s_thr = float(np.percentile(S, 80))
    mask = ((V > v_thr) & (S < max(190.0, s_thr))).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 9)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return cv2.boxPoints(cv2.minAreaRect(max(cnts, key=cv2.contourArea)))
 
 
def _warp_card(img, out_w=_WARP_W, out_h=_WARP_H):
    H, W = img.shape[:2]
    quad = _find_card_quad_edges(img)
    if quad is None or cv2.contourArea(np.array(quad, dtype=np.float32)) < 0.12 * H * W:
        quad = _find_card_quad_bright(img)
    if quad is None:
        return None
    rect = _order_points_landscape(quad)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return _WarpResult(warped=cv2.warpPerspective(img, M, (out_w, out_h)), M=M, quad=rect)
 
 
def _warp_card_aggressive(img, out_w=_WARP_W, out_h=_WARP_H):
    """Intento mas agresivo de localizar el card: usa minAreaRect sobre el
    contorno mas grande despues de una segmentacion robusta. Pensado como
    fallback cuando el warp normal da baja confianza (card muy rotado, o
    contorno irregular)."""
    H, W = img.shape[:2]
 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if mask.mean() > 127:
        mask = cv2.bitwise_not(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8), iterations=1)
 
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
 
    valid = [c for c in cnts if 0.10 * H * W < cv2.contourArea(c) < 0.95 * H * W]
    if not valid:
        return None
    c = max(valid, key=cv2.contourArea)
 
    rect_min = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect_min)
    rect = _order_points_landscape(box)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return _WarpResult(warped=cv2.warpPerspective(img, M, (out_w, out_h)), M=M, quad=rect)
 
 
def _crop_norm(img, roi):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = roi
    X0 = max(0, min(w - 1, int(x0 * w)))
    X1 = max(1, min(w, int(x1 * w)))
    Y0 = max(0, min(h - 1, int(y0 * h)))
    Y1 = max(1, min(h, int(y1 * h)))
    return img[Y0:Y1, X0:X1].copy()
 
 
def _qr_black_fraction(bgr):
    if bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7)
    return float((thr > 0).mean())
 
 
def _back_qr_density(warped):
    scores = [_qr_black_fraction(_crop_norm(warped, roi)) for roi in _BACK_QR_ROIS]
    return float(np.mean(scores)) if scores else 0.0
 
 
def _detect_qr_codes(warped):
    detector = cv2.QRCodeDetector()
    try:
        retval, points = detector.detectMulti(warped)
        if retval and points is not None:
            return int(len(points))
    except cv2.error:
        pass
    try:
        ret, points = detector.detect(warped)
        if ret and points is not None:
            return 1
    except cv2.error:
        pass
    return 0
 
 
def _mrz_line_score(warped):
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    region = gray[int(h * 0.60): int(h * 0.98), :]
    region = cv2.GaussianBlur(region, (3, 3), 0)
    thr = cv2.adaptiveThreshold(region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    merged = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (45, 3)), iterations=1)
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)), iterations=1)
    cnts, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = 0
    for c in cnts:
        x, y, wc, hc = cv2.boundingRect(c)
        if wc > 0.60 * w and hc < 0.35 * region.shape[0]:
            lines += 1
    return int(lines)
 
 
def _left_red_score(warped):
    h, w = warped.shape[:2]
    strip = warped[:, 0: max(1, int(0.08 * w))]
    strip = strip[int(0.05 * h): int(0.95 * h), :]
    if strip.size == 0:
        return 0.0
    b, g, r = cv2.split(strip.astype(np.float32))
    return float(np.mean(r - 0.5 * (g + b)))
 
 
FaceDetectorFn = Callable[[np.ndarray], List[Tuple[int, int, int, int]]]
 
_YUNET: Optional["cv2.FaceDetectorYN"] = None
_YUNET_MODEL_PATH: Optional[Path] = None
_HAAR: Optional[cv2.CascadeClassifier] = None
 
 
def _make_yunet(model_path: Path):
    if not model_path.exists():
        return None
    try:
        return cv2.FaceDetectorYN.create(
            str(model_path), "", (_WARP_W, _WARP_H),
            score_threshold=0.6, nms_threshold=0.3, top_k=50,
        )
    except cv2.error:
        return None
 
 
def _get_haar() -> Optional[cv2.CascadeClassifier]:
    """Cascade Haar como fallback. Viene incluido con opencv-python."""
    global _HAAR
    if _HAAR is None:
        import os
        path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if not os.path.exists(path):
            return None
        c = cv2.CascadeClassifier(path)
        if c.empty():
            return None
        _HAAR = c
    return _HAAR
 
 
def _haar_detect(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    cascade = _get_haar()
    if cascade is None:
        return []
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    min_face = max(40, int(0.10 * w))

    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(min_face, min_face),
    )
    return [tuple(map(int, f)) for f in faces]
 
 
def _yunet_detector_fn(model_path: Path) -> FaceDetectorFn:
    """FaceDetectorFn que usa YuNet con Haar como fallback. Cachea instancias."""
 
    def detect(warped: np.ndarray) -> List[Tuple[int, int, int, int]]:
        global _YUNET, _YUNET_MODEL_PATH
        if _YUNET is None or _YUNET_MODEL_PATH != model_path:
            _YUNET = _make_yunet(model_path)
            _YUNET_MODEL_PATH = model_path
        out = []
        if _YUNET is not None:
            h, w = warped.shape[:2]
            _YUNET.setInputSize((w, h))
            retval, faces = _YUNET.detect(warped)
            if faces is not None:
                for f in faces:
                    x, y, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                    x, y = max(0, x), max(0, y)
                    fw, fh = max(0, fw), max(0, fh)
                    if fw > 0 and fh > 0:
                        out.append((x, y, fw, fh))

        if not out:
            out = _haar_detect(warped)
        return out
 
    return detect
 
 
def _face_score(warped, detector, restrict_to_left=True):
    faces = detector(warped)
    if not faces:
        return 0, 0.0
    h, w = warped.shape[:2]
    if restrict_to_left:
        x0, y0, x1, y1 = _FRONT_FACE_ROI
        roi_box = (x0 * w, y0 * h, x1 * w, y1 * h)
        valid = []
        for (fx, fy, fw, fh) in faces:
            cx, cy = fx + fw / 2.0, fy + fh / 2.0
            if roi_box[0] <= cx <= roi_box[2] and roi_box[1] <= cy <= roi_box[3]:
                valid.append((fx, fy, fw, fh))
        faces = valid
    if not faces:
        return 0, 0.0
    max_area_frac = max(fw * fh for (_, _, fw, fh) in faces) / float(h * w)
    return len(faces), float(max_area_frac)
 
 
def _score_orientation(warped, face_detector):
    n_faces, face_area = _face_score(warped, face_detector, restrict_to_left=True)
    return {
        "n_faces": float(n_faces),
        "face_area_frac": float(face_area),
        "n_qrs": float(_detect_qr_codes(warped)),
        "qr_density": float(_back_qr_density(warped)),
        "mrz_lines": float(_mrz_line_score(warped)),
        "red_strip": float(_left_red_score(warped)),
    }
 
 
def _classify_with_orientation(warped, face_detector):
    """Decide lado y orientacion con reglas en cascada.
 
    Las senales de "back" deben corroborarse DENTRO de la misma orientacion:
    un QR detectado en rot 180 + MRZ en rot 0 no se considera corroboracion
    real, suelen ser falsos positivos (OVD fantasma + renglones del frente).
 
    Tiers:
      1. QR + MRZ misma orientacion -> back  (corroboracion fuerte)
      2. qr_density >= 0.30 -> back  (densidad asi solo da QR de verdad)
      3. Rostro en ROI -> front  (gana sobre senales debiles aisladas)
      4. qr_density >= 0.22 -> back  (alta para frente, casi seguro back)
      5. QR detectado solo -> back
      6. 3+ lineas MRZ -> back
      7. MRZ + qr_density misma orientacion -> back
      8. Fallback: franja roja -> front
    """
    warped180 = cv2.rotate(warped, cv2.ROTATE_180)
    s0 = _score_orientation(warped, face_detector)
    s180 = _score_orientation(warped180, face_detector)
    cands = [(s0, warped, 0), (s180, warped180, 180)]
 
    max_qr_d = max(s0["qr_density"], s180["qr_density"])
    max_n_qrs = max(s0["n_qrs"], s180["n_qrs"])
    max_mrz = max(s0["mrz_lines"], s180["mrz_lines"])
 
    has_face_0 = s0["n_faces"] >= 1 and s0["face_area_frac"] >= _FRONT_FACE_MIN_AREA
    has_face_180 = s180["n_faces"] >= 1 and s180["face_area_frac"] >= _FRONT_FACE_MIN_AREA
    has_face = has_face_0 or has_face_180
 
    qr_and_mrz_same = (
        (s0["n_qrs"] >= _BACK_QR_COUNT_MIN and s0["mrz_lines"] >= _BACK_MRZ_MIN)
        or (s180["n_qrs"] >= _BACK_QR_COUNT_MIN and s180["mrz_lines"] >= _BACK_MRZ_MIN)
    )
    qrd_and_mrz_same = (
        (s0["qr_density"] >= _BACK_QR_DENSITY_MIN and s0["mrz_lines"] >= _BACK_MRZ_MIN)
        or (s180["qr_density"] >= _BACK_QR_DENSITY_MIN and s180["mrz_lines"] >= _BACK_MRZ_MIN)
    )
 
    def _pick_back():
        return max(range(2), key=lambda i: (cands[i][0]["mrz_lines"], cands[i][0]["qr_density"], cands[i][0]["n_qrs"]))
 
    def _pick_face():
        if has_face_0 and has_face_180:
            return 0 if s0["face_area_frac"] >= s180["face_area_frac"] else 1
        return 0 if has_face_0 else 1
 
    if qr_and_mrz_same:
        side, idx, confidence = INESide.BACK, _pick_back(), 0.95
    elif max_qr_d >= _BACK_QR_DENSITY_STRONG:
        side, idx, confidence = INESide.BACK, _pick_back(), 0.90
    elif has_face:
        side = INESide.FRONT
        idx = _pick_face()
        chosen_red = cands[idx][0]["red_strip"]
        confidence = 0.85 + 0.10 * float(chosen_red >= _FRONT_RED_MIN)
    elif max_qr_d >= _BACK_QR_DENSITY_MID:
        side, idx, confidence = INESide.BACK, _pick_back(), 0.70
    elif max_n_qrs >= _BACK_QR_COUNT_MIN:
        side, idx, confidence = INESide.BACK, _pick_back(), 0.70
    elif max_mrz >= _BACK_MRZ_STRONG:
        side, idx, confidence = INESide.BACK, _pick_back(), 0.75
    elif qrd_and_mrz_same:
        side, idx, confidence = INESide.BACK, _pick_back(), 0.65
    else:
        side = INESide.FRONT
        idx = max(range(2), key=lambda i: cands[i][0]["red_strip"])
        chosen_red = cands[idx][0]["red_strip"]
        chosen_qr_d = cands[idx][0]["qr_density"]

        if max(s0["qr_density"], s180["qr_density"]) < 0.05:
            confidence = 0.15
        elif chosen_red >= _FRONT_RED_MIN and chosen_qr_d < 0.15:
            confidence = 0.55 + min(0.25, chosen_red / 60.0)
        else:
            confidence = 0.30
 
    chosen_scores, chosen_warp, orientation = cands[idx]
    out_scores = {f"{k}_chosen": v for k, v in chosen_scores.items()}
    for k, v in s0.items():
        out_scores[f"{k}_0"] = v
    for k, v in s180.items():
        out_scores[f"{k}_180"] = v
    return side, orientation, chosen_warp, out_scores, float(confidence)
 
 
class INEDetector:
    """Detector de lado para credenciales INE.
 
    Args:
        min_confidence: si la confianza es menor, devuelve INESide.UNKNOWN.
        skip_warp: salta la deteccion de contorno (si la foto ya viene recortada).
        face_detector: callable opcional (BGR -> [(x,y,w,h),...]) para usar otro modelo.
        yunet_model: ruta al .onnx de YuNet. Default: junto al modulo.
        retry_threshold: si la confianza del primer intento queda debajo de
            esto, reintenta con un warp mas agresivo (Otsu + minAreaRect).
            Util cuando el card esta rotado >30 grados o el contorno es
            irregular. Pon 0.0 para desactivar.
    """
 
    def __init__(self, min_confidence=0.0, skip_warp=False, face_detector=None,
                 yunet_model=None, retry_threshold=0.6):
        self.min_confidence = float(min_confidence)
        self.skip_warp = bool(skip_warp)
        self.retry_threshold = float(retry_threshold)
        if face_detector is not None:
            self._face_detector = face_detector
        else:
            model_path = Path(yunet_model) if yunet_model else _DEFAULT_YUNET_MODEL
            self._face_detector = _yunet_detector_fn(model_path)
 
    def detect(self, image: Union[str, Path, np.ndarray]) -> INEDetectionResult:
        img = self._load(image)
        if img is None:
            return INEDetectionResult(
                side=INESide.UNKNOWN, confidence=0.0,
                ok=False, error="no se pudo leer la imagen",
            )
        if self.skip_warp:
            warped0 = self._resize_to_canon(img)
            quad = None
        else:
            wr = _warp_card(img)
            if wr is None:
                warped0 = self._resize_to_canon(img)
                quad = None
            else:
                warped0 = wr.warped
                quad = wr.quad
        side, orientation, chosen, scores, confidence = _classify_with_orientation(
            warped0, self._face_detector,
        )
 
        if not self.skip_warp and self.retry_threshold > 0.0:
            no_face_anywhere = (
                scores.get("n_faces_0", 0) < 1 and scores.get("n_faces_180", 0) < 1
            )
            uncertain_back_no_face = (
                side == INESide.BACK and confidence < 0.85 and no_face_anywhere
            )
            if confidence < self.retry_threshold or uncertain_back_no_face:
                wr2 = _warp_card_aggressive(img)
                if wr2 is not None:
                    side2, orientation2, chosen2, scores2, confidence2 = _classify_with_orientation(
                        wr2.warped, self._face_detector,
                    )

                    found_face_now = (
                        scores2.get("n_faces_0", 0) >= 1 or scores2.get("n_faces_180", 0) >= 1
                    )
                    if confidence2 > confidence or (found_face_now and no_face_anywhere):
                        side, orientation, chosen, scores, confidence = (
                            side2, orientation2, chosen2, scores2, confidence2,
                        )
                        quad = wr2.quad
                        scores["used_aggressive_warp"] = 1.0
 
        if confidence < self.min_confidence:
            side = INESide.UNKNOWN
        return INEDetectionResult(
            side=side, confidence=confidence, orientation=orientation,
            scores=scores, warped=chosen, card_quad=quad, ok=True,
        )
 
    def detect_side(self, image) -> INESide:
        return self.detect(image).side
 
    @staticmethod
    def _load(image):
        if isinstance(image, np.ndarray):
            return image
        return cv2.imread(str(image))
 
    @staticmethod
    def _resize_to_canon(img):
        return cv2.resize(img, (_WARP_W, _WARP_H), interpolation=cv2.INTER_AREA)
 
 
_DEFAULT_DETECTOR: Optional[INEDetector] = None
 
 
def _default_detector() -> INEDetector:
    global _DEFAULT_DETECTOR
    if _DEFAULT_DETECTOR is None:
        _DEFAULT_DETECTOR = INEDetector()
    return _DEFAULT_DETECTOR
 
 
def detect_side(image) -> INESide:
    return _default_detector().detect(image).side
 
 
def detect(image) -> INEDetectionResult:
    return _default_detector().detect(image)
 
 
__all__ = [
    "INESide", "INEDetectionResult", "INEDetector",
    "FaceDetectorFn", "detect", "detect_side",
]
 
 
def _main():
    import argparse
    ap = argparse.ArgumentParser(description="Detecta si un INE es frente o reverso.")
    ap.add_argument("input", help="Imagen o carpeta")
    ap.add_argument("--min-confidence", type=float, default=0.0)
    ap.add_argument("--skip-warp", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
 
    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"No encuentro {inp}")
 
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    paths = [inp] if inp.is_file() else sorted(
        p for p in inp.rglob("*") if p.is_file() and p.suffix.lower() in exts
    )
 
    det = INEDetector(min_confidence=args.min_confidence, skip_warp=args.skip_warp)
    for p in paths:
        r = det.detect(p)
        if not r.ok:
            print(f"{p}: ERROR {r.error}")
            continue
        print(f"{p}: {r.side.value} (conf={r.confidence:.2f}, rot={r.orientation})")
        if args.verbose:
            for k, v in r.scores.items():
                print(f"    {k}: {v:.3f}")
 
 
if __name__ == "__main__":
    _main()
