#!/usr/bin/env python3
"""
Gabungkan dataset uang Roboflow (format YOLO) menjadi satu dataset 7 kelas, 640x640.

Yang dilakukan:
  1. Baca semua subfolder di dataset/ yang punya data.yaml (train/valid/test dari Roboflow).
  2. Remap kelas BERDASARKAN NAMA ke 7 kelas: 1000..100000 (indeks 0-6).
     - Gambar yang memuat kelas '75000' dibuang seluruhnya.
     - Baris label kelas lain yang tidak dikenal (mis. 'nominal') dibuang.
     - Gambar yang labelnya kosong setelah pemrosesan dibuang (--keep-empty untuk mempertahankan).
  3. Kelompokkan salinan augmentasi dari satu foto sumber (nama sebelum '.rf.'), lalu
     (opsional) gabungkan foto sumber yang sama antar dataset memakai perceptual hash.
  4. Bagi ulang train/val/test PER KELOMPOK (tanpa kebocoran data), stratified per kelas.
     Val/test selalu 1 gambar per foto sumber.
  5. Mode train:
       aug3x : semua salinan augmentasi yang ada (maks --max-copies per foto sumber)
       same  : 1 gambar per foto sumber (jumlah train = jumlah foto sumber di train)
       both  : hasilkan keduanya (val/test identik, hanya train yang berbeda)
  6. Resize ke 640x640 (gambar 640 disalin apa adanya). Label tidak berubah karena
     koordinat YOLO ternormalisasi dan semua gambar sudah persegi.

Contoh:
  python aggregate_dataset.py --mode both
  python aggregate_dataset.py --mode same --ratios 0.7 0.2 0.1 --overwrite

"""
import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # tqdm opsional
    def tqdm(it, **kw):
        return it

KEEP = ["1000", "2000", "5000", "10000", "20000", "50000", "100000"]
KEEP_IDX = {n: i for i, n in enumerate(KEEP)}
DROP_IMAGE = {"75000", "nominal"}                      # gambar yang memuat kelas ini dibuang
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_DIRS = {"train": False, "valid": True, "val": True, "test": True}  # folder -> asal evaluasi (tanpa augmentasi)
SPLITS = ["train", "val", "test"]


@dataclass
class Rec:
    dataset: str
    img: Path
    eval_origin: bool
    group: str
    boxes: list
    stem: str


@dataclass
class Group:
    gid: str
    recs: list = field(default_factory=list)
    rep: Rec = None
    vec: np.ndarray = None


# ----------------------------------------------------------------------------- load
def slug(s):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_")


def norm_name(n):
    n = str(n).strip()
    digits = re.sub(r"\D", "", n)
    return digits if digits else n.lower()


def load_names(path):
    names = yaml.safe_load(open(path, encoding="utf-8"))["names"]
    items = names.items() if isinstance(names, dict) else enumerate(names)
    return {int(k): norm_name(v) for k, v in items}


def parse_label(path, names, stats, ds):
    boxes, drop = [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        name = names.get(int(float(p[0])))
        if name in DROP_IMAGE:
            drop = True
            continue
        if name not in KEEP_IDX:
            stats[f"{ds}: baris label kelas lain dibuang"] += 1
            continue
        v = [float(t) for t in p[1:]]
        if len(v) == 4:
            x, y, w, h = v
        else:  # poligon -> bounding box
            xs, ys = v[0::2], v[1::2]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            x, y, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
            stats[f"{ds}: poligon diubah ke bbox"] += 1
        x0, x1 = max(0.0, x - w / 2), min(1.0, x + w / 2)
        y0, y1 = max(0.0, y - h / 2), min(1.0, y + h / 2)
        w, h = x1 - x0, y1 - y0
        if w <= 1e-4 or h <= 1e-4:
            stats[f"{ds}: bbox degenerate dibuang"] += 1
            continue
        boxes.append((KEEP_IDX[name], (x0 + x1) / 2, (y0 + y1) / 2, w, h))
    return boxes, drop


def collect(root, keep_empty, stats):
    recs = []
    dirs = sorted(d for d in root.iterdir() if d.is_dir() and (d / "data.yaml").exists())
    if not dirs:
        sys.exit(f"Tidak ada dataset (folder dengan data.yaml) di {root}")
    for i, d in enumerate(dirs):
        names, tag = load_names(d / "data.yaml"), f"ds{i}"
        print(f"[{tag}] {d.name}  kelas: {sorted(set(names.values()))}")
        for sd, is_eval in SPLIT_DIRS.items():
            idir = d / sd / "images"
            if not idir.is_dir():
                continue
            for img in sorted(p for p in idir.iterdir() if p.suffix.lower() in IMG_EXT):
                lbl = d / sd / "labels" / (img.stem + ".txt")
                stats[f"{d.name}: gambar ditemukan"] += 1
                if not lbl.exists():
                    stats[f"{d.name}: label hilang (dilewati)"] += 1
                    continue
                boxes, drop = parse_label(lbl, names, stats, d.name)
                if drop:
                    stats[f"{d.name}: gambar kelas dibuang (75000)"] += 1
                    continue
                if not boxes and not keep_empty:
                    stats[f"{d.name}: label kosong (dibuang)"] += 1
                    continue
                base = img.stem.split(".rf.")[0]
                uid = hashlib.md5(str(img.relative_to(root)).encode()).hexdigest()[:8]
                recs.append(Rec(d.name, img, is_eval, f"{tag}:{base}", boxes,
                                f"{tag}_{slug(base)[:50]}_{uid}"))
    return recs


# ----------------------------------------------------------------------------- dedup
def dhash(path):
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return np.uint64(int(np.packbits(bits).view(">u8")[0]))


def popcount(x):
    u = np.uint64
    with np.errstate(over="ignore"):
        x = x - ((x >> u(1)) & u(0x5555555555555555))
        x = (x & u(0x3333333333333333)) + ((x >> u(2)) & u(0x3333333333333333))
        x = (x + (x >> u(4))) & u(0x0F0F0F0F0F0F0F0F)
        return (x * u(0x0101010101010101)) >> u(56)


def merge_duplicates(groups, thr):
    """Gabungkan grup dari DATASET BERBEDA yang foto sumbernya hampir identik."""
    keys = sorted(groups)
    ds = np.array([groups[k].recs[0].dataset for k in keys])
    hashes = np.array([dhash(groups[k].recs[0].img) for k in tqdm(keys, desc="hash")], dtype=np.uint64)
    parent = {k: k for k in keys}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for i in range(len(keys) - 1):
        d = popcount(hashes[i + 1:] ^ hashes[i])
        for j in np.nonzero((d <= thr) & (ds[i + 1:] != ds[i]))[0]:
            parent[find(keys[i + 1 + j])] = find(keys[i])
    merged = {}
    for k in keys:
        r = find(k)
        merged.setdefault(r, Group(r)).recs.extend(groups[k].recs)
    print(f"Dedup: {len(groups)} grup -> {len(merged)} grup ({len(groups) - len(merged)} digabung)")
    return merged


# ----------------------------------------------------------------------------- split
def split_groups(groups, ratios, seed=0):
    """Greedy stratified split per grup; kelas langka diproses lebih dulu."""
    rng = random.Random(seed)
    act = [i for i, r in enumerate(ratios) if r > 0]
    gl = list(groups.values())
    glob = sum((g.vec for g in gl), np.zeros(len(KEEP)))
    tgt = np.array([r * glob for r in ratios])
    cur = np.zeros_like(tgt)
    n_t, n_c = np.array(ratios) * len(gl), np.zeros(3)

    def rarity(g):
        return glob[g.vec > 0].min() if (g.vec > 0).any() else 1e18

    assign = {}
    for g in sorted(gl, key=lambda g: (rarity(g), rng.random())):
        tot = g.vec.sum()
        w = g.vec / tot if tot else g.vec
        best, bs = None, -1e18
        for s in act:
            dc = float(np.sum(w * (tgt[s] - cur[s]) / np.maximum(tgt[s], 1.0)))
            dn = (n_t[s] - n_c[s]) / max(n_t[s], 1.0)
            if dc + 0.5 * dn > bs:
                best, bs = s, dc + 0.5 * dn
        assign[g.gid] = SPLITS[best]
        cur[best] += g.vec
        n_c[best] += 1
    return assign


# ----------------------------------------------------------------------------- export
def write_one(job):
    rec, split, out, imgsz = job
    dst = out / "images" / split / f"{rec.stem}.jpg"
    with Image.open(rec.img) as im:
        if im.size == (imgsz, imgsz) and im.format == "JPEG":
            shutil.copyfile(rec.img, dst)
        else:
            im.convert("RGB").resize((imgsz, imgsz), Image.Resampling.LANCZOS).save(dst, quality=95)
    lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in rec.boxes]
    (out / "labels" / split / f"{rec.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def export(mode, groups, assign, a):
    out = Path(f"{a.out}_{mode}")
    if out.exists():
        if not a.overwrite:
            sys.exit(f"{out} sudah ada. Gunakan --overwrite atau ganti --out.")
        shutil.rmtree(out)
    active = [s for s, r in zip(SPLITS, a.ratios) if r > 0]
    for s in active:
        (out / "images" / s).mkdir(parents=True)
        (out / "labels" / s).mkdir(parents=True)

    rng = random.Random(a.seed + 1)
    jobs, job_groups = [], []
    for g in groups.values():
        s = assign[g.gid]
        if s == "train" and mode == "aug3x":
            others = [r for r in g.recs if r is not g.rep]
            chosen = [g.rep] + rng.sample(others, min(a.max_copies - 1, len(others)))
        else:
            chosen = [g.rep]
        for r in chosen:
            jobs.append((r, s, out, a.imgsz))
            job_groups.append((s, g.gid))

    with ThreadPoolExecutor(a.workers) as ex:
        list(tqdm(ex.map(write_one, jobs), total=len(jobs), desc=f"tulis {mode}"))

    # data.yaml
    y = [f"path: {out.resolve().as_posix()}", "train: images/train", "val: images/val"]
    if a.ratios[2] > 0:
        y.append("test: images/test")
    y += [f"nc: {len(KEEP)}", "names:"] + [f"  {i}: '{n}'" for i, n in enumerate(KEEP)]
    (out / "data.yaml").write_text("\n".join(y) + "\n", encoding="utf-8")

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "file", "dataset", "source", "group"])
        for (r, s, _, _), (_, gid) in zip(jobs, job_groups):
            w.writerow([s, f"{r.stem}.jpg", r.dataset, r.img.name, gid])

    validate_and_report(out, jobs, job_groups, active)
    print(f"Selesai: {out}/data.yaml\n")


def validate_and_report(out, jobs, job_groups, active):
    gsets = defaultdict(set)
    for s, gid in job_groups:
        gsets[s].add(gid)
    for i, s1 in enumerate(active):
        for s2 in active[i + 1:]:
            assert not (gsets[s1] & gsets[s2]), f"Kebocoran grup antara {s1} dan {s2}"
    for s in active:
        for img in (out / "images" / s).glob("*.jpg"):
            lbl = out / "labels" / s / (img.stem + ".txt")
            assert lbl.exists(), f"label hilang: {lbl}"
            for line in lbl.read_text().splitlines():
                p = line.split()
                assert len(p) == 5 and 0 <= int(p[0]) < len(KEEP) and all(0 <= float(t) <= 1 for t in p[1:]), lbl

    inst, imgs = defaultdict(Counter), defaultdict(Counter)
    for r, s, _, _ in jobs:
        for c in {b[0] for b in r.boxes}:
            imgs[s][c] += 1
        for b in r.boxes:
            inst[s][b[0]] += 1
    n_img = Counter(s for _, s, _, _ in jobs)
    print(f"\n{'split':6} {'gambar':>7} {'foto_sumber':>11} | " + " ".join(f"{n:>7}" for n in KEEP) + "   (jumlah instance)")
    report = {}
    for s in active:
        print(f"{s:6} {n_img[s]:7d} {len(gsets[s]):11d} | " + " ".join(f"{inst[s][i]:7d}" for i in range(len(KEEP))))
        report[s] = {"images": n_img[s], "source_photos": len(gsets[s]),
                     "instances": {KEEP[i]: inst[s][i] for i in range(len(KEEP))},
                     "images_per_class": {KEEP[i]: imgs[s][i] for i in range(len(KEEP))}}
        miss = [KEEP[i] for i in range(len(KEEP)) if inst[s][i] == 0]
        if miss:
            print(f"  PERINGATAN: split '{s}' tidak punya kelas {miss}")
    (out / "report.json").write_text(json.dumps(report, indent=2))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="dataset", help="folder berisi dataset Roboflow (default: dataset)")
    ap.add_argument("--out", default="merged", help="prefix folder output; hasil: <out>_<mode>")
    ap.add_argument("--mode", choices=["aug3x", "same", "both"], default="aug3x")
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.8, 0.1, 0.1], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--max-copies", type=int, default=3, help="maks salinan augmentasi per foto sumber (mode aug3x)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-dedup", action="store_true", help="nonaktifkan deteksi foto sumber sama antar dataset")
    ap.add_argument("--dedup-threshold", type=int, default=3, help="jarak Hamming maks dHash (default 3)")
    ap.add_argument("--keep-empty", action="store_true", help="pertahankan gambar tanpa label")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if abs(sum(a.ratios) - 1) > 1e-6 or a.ratios[0] <= 0 or a.ratios[1] <= 0:
        sys.exit("--ratios harus berjumlah 1, dan train & val > 0")

    stats = Counter()
    recs = collect(Path(a.src), a.keep_empty, stats)
    print()
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    print(f"Gambar valid setelah filter: {len(recs)}")

    groups = {}
    for r in recs:
        groups.setdefault(r.group, Group(r.group)).recs.append(r)
    print(f"Foto sumber (grup): {len(groups)}")
    if not a.no_dedup:
        groups = merge_duplicates(groups, a.dedup_threshold)

    rng = random.Random(a.seed)
    for g in groups.values():
        pool = [r for r in g.recs if r.eval_origin] or g.recs   # utamakan gambar tanpa augmentasi
        g.rep = rng.choice(sorted(pool, key=lambda r: r.stem))
        g.vec = np.zeros(len(KEEP))
        for b in g.rep.boxes:
            g.vec[b[0]] += 1

    assign = split_groups(groups, a.ratios, a.seed)
    for mode in (["aug3x", "same"] if a.mode == "both" else [a.mode]):
        export(mode, groups, assign, a)


if __name__ == "__main__":
    main()