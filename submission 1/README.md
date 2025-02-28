# Laporan Proyek Machine Learning - Andika Bintang Nursalih

## 1. Domain Proyek

Proyek ini bertujuan untuk mengembangkan model machine learning dalam mendeteksi kanker payudara. Deteksi dini kanker payudara sangat penting karena dapat meningkatkan tingkat kelangsungan hidup pasien melalui penanganan dan perawatan yang cepat. Melalui pemanfaatan data morfologi dan tekstur sel, proyek ini mencoba mengklasifikasikan tumor sebagai benign atau malignant dengan akurasi tinggi.

**Alasan Masalah:**  
- Kanker payudara adalah salah satu penyebab kematian tertinggi pada wanita di dunia.  
- Deteksi dini dengan teknologi cerdas dapat membantu dokter membuat keputusan terapi yang lebih tepat.

**Referensi Terkait:**  
- [Breast Cancer Detection Research](https://www.sciencedirect.com/science/article/abs/pii/S0933365722000410) – Penelitian tentang penerapan machine learning dalam diagnosis kanker payudara.  
- [Early Detection of Breast Cancer](https://repository.uhn.ac.id/handle/123456789/4689) – Studi tentang pentingnya deteksi dini dan pengaruhnya terhadap kesembuhan pasien.

---

## 2. Business Understanding

### Problem Statements
- **Masalah 1:** Bagaimana cara mengklasifikasikan tumor payudara sebagai benign atau malignant dengan akurasi tinggi?
- **Masalah 2:** Bagaimana membandingkan performa berbagai algoritma machine learning untuk memilih model terbaik yang mendukung keputusan klinis?

### Goals
- **Goal 1:** Menghasilkan model klasifikasi yang memiliki nilai evaluasi (accuracy, precision, recall, F1-score) optimal.
- **Goal 2:** Menyusun proses data preparation yang sistematis guna menghasilkan data yang berkualitas untuk training.
- **Goal 3:** Menyediakan solusi yang dapat diukur secara kuantitatif untuk mendukung proses diagnosis kanker payudara.

### Solution Statements
- **Solution 1:** Mengimplementasikan dan membandingkan tiga model klasifikasi (Random Forest, KNN, dan Boosting [AdaBoost]) untuk mengidentifikasi tumor.
- **Solution 2:** Melakukan perbaikan (improvement) dengan hyperparameter tuning pada setiap model untuk mengoptimalkan performa.
- **Solution 3:** Menggunakan evaluasi berbasis metrik (accuracy, precision, recall, dan F1-score) untuk mengukur kinerja dan memilih model terbaik.

---

## 3. Data Understanding

Dataset yang digunakan adalah **Breast Cancer Dataset** yang berisi data numerik mengenai karakteristik sel payudara. Dataset ini memiliki informasi seperti:

- **Jumlah Data:** Misalnya, 569 sampel.
- **Kondisi Data:** Terdiri dari fitur numerik seperti `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, dll., serta variabel target `diagnosis` (B untuk benign dan M untuk malignant).

**Sumber Data:**  
Dataset diunduh dari:  
[Breast Cancer Dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset?resource=download)

### Variabel-Fitur Utama
- **diagnosis:** Label untuk mengidentifikasi tumor (B = Benign, M = Malignant)
- **Fitur Numerik:**  
  - `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, `smoothness_mean`, dan seterusnya.

### Kode Eksplorasi Data (EDA)

```python
# Mount Google Drive dan import library
from google.colab import drive
drive.mount('/content/drive/')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Membaca dataset
data_path = '/content/drive/MyDrive/Dataset/breast-cancer.csv'
df = pd.read_csv(data_path)

# Melihat data awal, informasi, dan statistik deskriptif
print(df.head())
print(df.info())
print(df.describe())

# Cek missing values dan ukuran dataset
print("Missing values per kolom:\n", df.isna().sum())
print("Ukuran dataset:", df.shape)
```

---

## 4. Data Preparation

Tahapan data preparation dilakukan secara berurutan untuk memastikan data siap digunakan dalam pemodelan:

1. **Pembersihan Data:**  
   - Memeriksa nilai yang hilang (missing values) dan memastikan konsistensi data.

2. **Penghapusan Kolom Tidak Relevan:**  
   - Menghapus kolom `id` karena tidak memberikan informasi yang berguna untuk klasifikasi.
   
   ```python
   df.drop(['id'], axis=1, inplace=True)
   ```

3. **Konversi Label:**  
   - Mengubah variabel `diagnosis` dari format huruf ('B', 'M') menjadi format numerik (0, 1).
   
   ```python
   diagnosis_mapping = {'B': 0, 'M': 1}
   df['diagnosis'] = df['diagnosis'].map(diagnosis_mapping)
   print(df['diagnosis'].value_counts())
   ```

4. **Pembagian Data:**  
   - Memisahkan dataset menjadi data latih dan data uji dengan rasio 80:20.
   
   ```python
   from sklearn.model_selection import train_test_split

   X = df.drop('diagnosis', axis=1)
   y = df['diagnosis']

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
   print(f"Total sampel: {len(X)}")
   print(f"Sampel data latih: {len(X_train)}")
   print(f"Sampel data uji: {len(X_test)}")
   ```

5. **Standarisasi Fitur:**  
   - Menggunakan `StandardScaler` untuk menormalkan distribusi fitur numerik.
   
   ```python
   from sklearn.preprocessing import StandardScaler

   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   ```

**Alasan Tahapan:**  
Tahapan di atas dilakukan untuk memastikan bahwa model tidak bias karena perbedaan skala antar fitur dan untuk menangani data yang tidak konsisten.

---

## 5. Modeling

Dalam tahap pemodelan, tiga algoritma digunakan untuk mengklasifikasikan data:

### Random Forest
- **Deskripsi:**  
  Menggunakan ensemble decision trees yang mampu menangani variansi data dan mencegah overfitting.
- **Parameter Kunci:**  
  - Jumlah pohon (n_estimators)  
  - Kedalaman pohon (max_depth)
- **Kelebihan:**  
  - Stabil dan robust terhadap overfitting.
- **Kekurangan:**  
  - Memerlukan waktu training yang lebih lama dibandingkan model sederhana.
  
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Inisialisasi dan pelatihan model
RF = RandomForestClassifier(random_state=5)
RF.fit(X_train_scaled, y_train)

# Prediksi dan evaluasi
RF_pred = RF.predict(X_test_scaled)
print("Akurasi Random Forest:", accuracy_score(y_test, RF_pred))
print(classification_report(y_test, RF_pred))
```

[Cuplikan-layar-2025-02-28-184328.png](https://postimg.cc/Wdbf0y88)

### K-Nearest Neighbors (KNN)
- **Deskripsi:**  
  KNN mengklasifikasikan sampel berdasarkan kedekatan jarak dengan tetangga terdekat.
- **Parameter Kunci:**  
  - Jumlah tetangga (n_neighbors)
- **Kelebihan:**  
  - Sederhana dan mudah diimplementasikan.
- **Kekurangan:**  
  - Performa menurun pada dataset dengan jumlah fitur yang besar.

```python
cm_knn = confusion_matrix(y_test, prediksi_knn)
plt.figure(figsize=(6,4))
sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix - KNN")
plt.show()
```

[Cuplikan-layar-2025-02-28-191351.png](https://postimg.cc/YGp3wGP3)

### Boosting (Menggunakan AdaBoost)
- **Deskripsi:**  
  Algoritma boosting yang menggabungkan beberapa model lemah menjadi model yang lebih kuat.
- **Parameter Kunci:**  
  - Jumlah estimator (n_estimators)  
  - Tingkat pembelajaran (learning_rate)
- **Kelebihan:**  
  - Mampu meningkatkan performa model secara signifikan.
- **Kekurangan:**  
  - Sensitif terhadap noise dalam data.

```python
cm_boost = confusion_matrix(y_test, boost_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm_boost, annot=True, fmt="d", cmap="Oranges")
plt.title("Confusion Matrix - Boosting")
plt.show()
```
  
[Cuplikan-layar-2025-02-28-191821.png](https://postimg.cc/wt9JtKkx)

**Improvement Model:**  
Setelah pelatihan awal, dilakukan hyperparameter tuning untuk setiap model guna mendapatkan performa optimal. Perbandingan metrik evaluasi kemudian digunakan untuk memilih model terbaik.

---

## 6. Evaluation

Evaluasi model dilakukan dengan menggunakan beberapa metrik utama:

- **Accuracy:**  
  Mengukur persentase prediksi yang benar.  
  *Formula:*  
  \( \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} \)

- **Precision:**  
  Mengukur seberapa tepat prediksi positif.  
  *Formula:*  
  \( \text{Precision} = \frac{TP}{TP + FP} \)

- **Recall:**  
  Mengukur kemampuan model menangkap semua sampel positif.  
  *Formula:*  
  \( \text{Recall} = \frac{TP}{TP + FN} \)

- **F1-Score:**  
  Rata-rata harmonis dari precision dan recall.  
  *Formula:*  
  \( F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \)

### Hasil Evaluasi

Tabel berikut menunjukkan hasil evaluasi untuk masing-masing model:

| Model           | Accuracy   | Precision (0) | Recall (0) | F1-Score (0) | Precision (1) | Recall (1) | F1-Score (1) |
|-----------------|------------|---------------|------------|--------------|---------------|------------|--------------|
| **RandomForest**| 0.973684   | 0.956522      | 0.937500   | 0.977778     | 0.967742      | 0.937500   | *(Detail)*   |
| **KNN**         | 0.964912   | 0.942857      | 0.916667   | 0.970588     | 0.956522      | 0.916667   | *(Detail)*   |
| **Boosting**    | 0.973684   | 0.956522      | 0.937500   | 0.977778     | 0.967742      | 0.937500   | *(Detail)*   |

*(Sisipkan tabel evaluasi lengkap dan grafik perbandingan performa di sini jika ada.)*

### Interpretasi Hasil
- **RandomForest** dan **Boosting** menunjukkan nilai akurasi tertinggi (0.973684) dan memiliki performa yang sangat baik secara keseluruhan.  
- **KNN** memiliki akurasi sedikit lebih rendah (0.964912), sehingga model RandomForest dan Boosting menjadi kandidat utama untuk implementasi solusi.
