from __future__ import annotations
import pathlib
from pathlib import Path
import shutil
import sys
from datetime import datetime
from typing import Iterable, Set
import logging
from tqdm import tqdm
from ultralytics import YOLO

# Пути и параметры
Path("photos").mkdir(parents=True, exist_ok=True)
Path("photos_vehicle").mkdir(parents=True, exist_ok=True)
Path("weights").mkdir(parents=True, exist_ok=True)
SOURCE_DIR = pathlib.Path("./photos")
DEST_DIR = pathlib.Path("./photos_vehicle")
ARCHIVE_DIR = pathlib.Path("./archives")
WEIGHTS = pathlib.Path("./weights/yolov3u.pt")
IMG_SIZE = 640
DEVICE = 0
CONF_THRES = 0.85
VEHICLE_IDS: Set[int] = {2, 5, 7}  # car, bus, truck (COCO ids)
EXTS = {".png"}

logger = logging.getLogger()

# Итератор по изображениям
def iter_images(root: pathlib.Path) -> Iterable[pathlib.Path]:
    for path in root.rglob("*"):
        if path.suffix.lower() in EXTS:
            yield path


# Синхронизация деревьев каталогов
def sync_trees(src: pathlib.Path, dst: pathlib.Path) -> None:
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif not target.exists():
            shutil.copy2(path, target)


# Удаление пустых директорий
def remove_empty_dirs(root: pathlib.Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


# Проверка, содержит ли изображение транспорт
def is_vehicle(model: YOLO, img_path: pathlib.Path) -> bool:
    pred = model.predict(
        source=str(img_path),
        imgsz=IMG_SIZE,
        device=DEVICE,
        conf=0.01,
        verbose=False,
    )[0]

    if pred.boxes.xyxy.shape[0] == 0:
        return False

    xyxy = pred.boxes.xyxy
    if hasattr(xyxy, "cpu"):
        xyxy = xyxy.cpu().numpy()
    widths = xyxy[:, 2] - xyxy[:, 0]
    heights = xyxy[:, 3] - xyxy[:, 1]
    areas = widths * heights

    max_idx = int(areas.argmax())
    max_conf = float(pred.boxes.conf[max_idx])
    max_cls = int(pred.boxes.cls[max_idx])

    return (max_conf >= CONF_THRES) and (max_cls in VEHICLE_IDS)


# Архивирование папки photos и её очистка
def archive_and_clean_photos(dir_path: pathlib.Path) -> None:
    if not dir_path.exists():
        return

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"{dir_path.name}_backup_{timestamp}.zip"
    archive_path = ARCHIVE_DIR / archive_name
    shutil.make_archive(str(archive_path).replace(".zip", ""), "zip", dir_path)

    for item in dir_path.rglob("*"):
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        except Exception as e:
            logger.error(f"⚠ Ошибка при удалении {item}: {e}")


def main() -> None:
    if not SOURCE_DIR.exists():
        sys.exit(f"SOURCE_DIR not found: {SOURCE_DIR}")

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"⮞ Syncing {SOURCE_DIR} → {DEST_DIR}")
    sync_trees(SOURCE_DIR, DEST_DIR)

    logger.info("⮞ Loading model:", WEIGHTS)
    model = YOLO(str(WEIGHTS))

    images = list(iter_images(DEST_DIR))
    if not images:
        sys.exit("No images found in DEST_DIR.")

    removed = 0
    for img in tqdm(images, unit="img", desc="Filtering"):
        try:
            if not is_vehicle(model, img):
                img.unlink()
                removed += 1
        except Exception as exc:
            logger.error(f"\n⚠ {img} — {exc}")

    remove_empty_dirs(DEST_DIR)
    kept = len(images) - removed
    logger.info(f"\n✓ Done! Kept {kept}, removed {removed}.")

    logger.info("⮞ Archiving and cleaning SOURCE_DIR …")
    archive_and_clean_photos(SOURCE_DIR)
    logger.info("✓ SOURCE_DIR archived to './archives' and cleaned.")


if __name__ == "__main__":
    main()