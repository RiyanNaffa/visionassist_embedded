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