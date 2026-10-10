# Training Ground for Embedded Models

## Setup
- Aktivasi venv dan *install* semua library dengan `pip install -r requirements.txt` di *root project*.
```bash
# Windows
py -m venv .venv
.\.venv\Scripts\activate 
pip install -r requirements.txt
```
- Jalankan nbstripout untuk menghindari commit notebook beserta dengan outputnya.
```bash
nbstripout --install
```

## Notes
Untuk NVIDIA GPU, panduan install PyTorch dapat dilihat pada [link berikut](https://pytorch.org/get-started/locally/), kemudian install ulang PyTorch.
```bash
pip uninstall -y torch torchvision
# Copy-paste dari web
pip install torch torchvision --index-url https://your.pytorch.download.url
```