import argparse
import shutil
import sys
from pathlib import Path

from sideDetectorINE_Module import INEDetector, INESide

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def main():
    ap = argparse.ArgumentParser(description="Separa INEs en frente/reverso")
    ap.add_argument("input", help="Carpeta con fotos")
    ap.add_argument("--move", action="store_true", help="Mover en vez de copiar")
    args = ap.parse_args()

    src = Path(args.input).resolve()
    if not src.is_dir():
        sys.exit(f"ERROR: {src} no es una carpeta")

    images = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not images:
        sys.exit(f"No hay imagenes en {src}")

    front_dir = src.parent / f"{src.name}_frente"
    back_dir = src.parent / f"{src.name}_reverso"
    front_dir.mkdir(exist_ok=True)
    back_dir.mkdir(exist_ok=True)

    print(f"Encontre {len(images)} imagenes en {src}")
    print(f"Salida: {front_dir.name}/  y  {back_dir.name}/\n")

    det = INEDetector()
    op = shutil.move if args.move else shutil.copy2
    counts = {"frente": 0, "reverso": 0, "skip": 0}
    width = len(str(len(images)))

    for i, p in enumerate(images, 1):
        r = det.detect(p)
        prefix = f"[{i:>{width}}/{len(images)}] {p.name}"
        if not r.ok:
            print(f"{prefix} -> SKIP ({r.error})")
            counts["skip"] += 1
            continue
        if r.side == INESide.FRONT:
            op(str(p), str(front_dir / p.name))
            counts["frente"] += 1
            print(f"{prefix} -> frente  (conf={r.confidence:.2f})")
        elif r.side == INESide.BACK:
            op(str(p), str(back_dir / p.name))
            counts["reverso"] += 1
            print(f"{prefix} -> reverso (conf={r.confidence:.2f})")
        else:
            counts["skip"] += 1
            print(f"{prefix} -> SKIP (UNKNOWN)")

    print(f"\nFrente: {counts['frente']}  Reverso: {counts['reverso']}  Skip: {counts['skip']}")


if __name__ == "__main__":
    main()
