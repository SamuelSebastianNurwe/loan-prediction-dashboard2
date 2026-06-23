# Cara Menjalankan Loan Dashboard

Project ini sudah dibuat agar membaca file langsung dari folder `data/` dan `model/`.

## 1. Struktur folder wajib

```text
loan_dashboard/
├── app.py
├── pages/
├── utils/
├── data/
│   ├── train_u6lujuX_CVtuZ9i.csv
│   ├── Klasifikasi_DT-HPO.csv
│   ├── Klasifikasi_NN_NoHPO.csv
│   ├── Klasifikasi_SVM-NoHPO.csv
│   ├── hasilRegresi_DT_NoHPO.csv
│   ├── hasilRegresi_NN_NoHPO.csv
│   └── hasilRegresi_SVM_NoHpo.csv
└── model/
    ├── clf_dt_hpo.joblib
    ├── clf_nn_nohpo.joblib
    ├── clf_svm_nohpo.joblib
    ├── reg_dt_nohpo.joblib
    ├── reg_nn_nohpo.joblib
    └── reg_svm_nohpo.joblib
```

## 2. Install library

```bash
pip install -r requirements.txt
```

Disarankan memakai Python 3.11 atau 3.12.

## 3. Jalankan Streamlit

```bash
streamlit run app.py
```

## 4. Perbaikan utama yang sudah diterapkan

1. Semua halaman memakai file CSV dari folder `data/`.
2. Semua simulator memakai model `.joblib` dari folder `model/`.
3. Preprocessing simulator dibuat dinamis mengikuti `feature_names_in_` dari model, karena fitur DT/NN/SVM tidak sama.
4. Mode batch CSV/Excel bisa memakai:
   - data mentah seperti dataset asli, atau
   - data yang sudah numeric/scaled sesuai kolom model.
5. Setiap chart/diagram diberi penjelasan fungsi dan interpretasi agar mudah dipresentasikan di tiap page.

## 5. Catatan penting label klasifikasi

Pada model klasifikasi dashboard ini:

- `0` = Disetujui
- `1` = Ditolak

Karena itu hasil prediksi di simulator dikonversi menjadi label `Disetujui` atau `Ditolak`.
