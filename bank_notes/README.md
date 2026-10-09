# Training Ground for Embedded Models - Bank Notes

## Dataset
- Detail dataset terdapat pada `bank_notes/dataset/notes.txt`, termasuk URL ke setiap dataset pada Roboflow.
- Unzip dan pindahkan ketiga folder dataset ke dalam folder `dataset/`.
- Dataset asli masih belum selaras, jalankan script `aggregate_dataset.py` untuk menghasilkan satu dataset training baru dengan `py aggregate_dataset.py --mode aug3x --overwrite --ratios 0.7 0.15 0.15`.
```bash
# Windows
cd bank_notes
py aggregate_dataset.py --mode aug3x --overwrite --ratios 0.7 0.15 0.15
```

## Training
- Sesuaikan *path* ke dataset baru dan jalankan *notebook* `train.ipynb`.
- Model dilatih menggunakan base model YOLO26 dan di-*export* ke format TFLite INT8.