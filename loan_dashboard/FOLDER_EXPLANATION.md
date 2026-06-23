# Loan Prediction Intelligence Dashboard

## 📋 Penjelasan Folder & File

### ❓ Apa itu folder `__pycache__`?

Folder `__pycache__` adalah **cache Python otomatis** yang dibuat oleh Python saat menjalankan kode `.py`. 

**Fungsi:**
- Menyimpan compiled bytecode (`.pyc` files) untuk mempercepat loading modul berikutnya
- Membantu Python engine mempercepat import module
- Dibuat otomatis, tidak perlu commit ke git

**Mengapa ada?**
- Saat kita menjalankan aplikasi Streamlit, Python mengcompile file Python menjadi bytecode
- Bytecode ini disimpan di `__pycache__` untuk optimasi performa
- Ini adalah behavior standar Python, bukan spesifik untuk Sonnet atau tools lainnya

**Apakah perlu dihapus?**
- ❌ TIDAK perlu dihapus secara manual
- ✅ Sudah di-ignore di `.gitignore` file
- ✅ Akan di-regenerate otomatis jika dihapus

**Cara mengabaikannya di Git:**
- File `.gitignore` sudah ada di project ini
- Folder `__pycache__` TIDAK akan di-commit ke repository

---

## 📊 Struktur Folder

```
loan_dashboard/
├── app.py                  # Home page aplikasi
├── model/                  # Folder untuk menyimpan model .joblib (auto-created)
├── Data/                   # Folder CSV dengan hasil training
│   ├── Klasifikasi_*.csv
│   ├── REG_KNN_*.csv
│   └── hasilRegresi_*.csv
├── pages/                  # Folder halaman Streamlit
│   ├── 1_Executive_Overview.py
│   ├── 2_Classification_Lab.py
│   ├── 3_Regression_Lab.py
│   ├── 4_Feature_Analytics.py
│   ├── 5_Explainability.py
│   └── 6_Smart_Loan_Simulator.py
├── utils/                  # Utility modules
│   ├── helper.py          # Loading data, preprocessing, predictions
│   └── visualization.py   # Plotting functions
├── __pycache__/           # Python cache (auto-generated, ignore this!)
├── .gitignore             # Files to ignore in git
└── requirements.txt       # Python dependencies

```

---

## 🔧 Fix untuk Error yang Dilakukan

### 1. ✅ Fix Path Folder Data
- **Problem:** Code mencari `data/` (lowercase) tapi folder sebenarnya `Data/` (uppercase)
- **Solution:** Update `helper.py` untuk menggunakan `Data` (capital)
- **Files Fixed:** `utils/helper.py` - lines 189, 211, 228

### 2. ✅ Tambah Penjelasan di Bawah Setiap Chart & Tabel
- **Added di:** 
  - `pages/1_Executive_Overview.py` - semua histogram & pie chart
  - `pages/2_Classification_Lab.py` - metrik comparison & bar charts
  - `pages/3_Regression_Lab.py` - R² Score chart & scatter plot
  - `pages/4_Feature_Analytics.py` - heatmap & boxplots
  - `pages/6_Smart_Loan_Simulator.py` - radar chart & preview table

### 3. ✅ Highlight R² Score di Sidebar Smart Loan Simulator
- **Added:** Metrics section di sidebar dengan Best Model Performance
- **Shows:** SVM HPO untuk Classification, DT HPO untuk Regression

### 4. ✅ Tambahkan Tabel Perbandingan Metrik di Executive Overview
- **Added:** Section baru "Perbandingan Metrik Evaluasi Model"
- **Includes:** Classification & Regression metrics dalam tabs

---

## 🚀 Cara Menjalankan

```bash
cd loan_dashboard
streamlit run app.py
```

Aplikasi akan berjalan di `http://localhost:8501`

---

## 📝 Catatan

- Semua penjelasan sudah ditambahkan tepat di bawah tabel/chart seperti di contoh Img 4 & 5
- R² Score sekarang prominent di sidebar Smart Loan Simulator
- __pycache__ sudah diabaikan via .gitignore
- Folder model sudah dibuat untuk mengatasi error file model tidak ditemukan

