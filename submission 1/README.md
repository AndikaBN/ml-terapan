# Laporan Proyek Machine Learning - Andika Bintang Nursalih

## Domain Proyek

Kanker payudara merupakan salah satu penyakit yang dapat berakibat fatal, terutama jika tidak terdeteksi sejak dini. Penyakit ini terjadi akibat pertumbuhan sel abnormal yang berasal dari kelenjar susu di payudara. Perkembangannya disebabkan oleh pembelahan sel yang lebih cepat dari normal, di mana sel-sel lama pada duktus lactiferi tidak mati dan terus digantikan oleh sel-sel baru yang tumbuh secara tidak terkendali. Akibatnya, sel-sel ini dapat menginvasi jaringan sehat di sekitarnya.  
Menurut International Agency for Research on Cancer (IARC), jumlah kasus kanker di dunia terus meningkat setiap tahunnya. Pada tahun 2008, terdapat 12,7 juta kasus kanker yang dilaporkan, dan angka ini terus bertambah hingga mencapai 18,1 juta kasus pada tahun 2018. Tidak hanya jumlah kasus, angka kematian akibat kanker juga meningkat dari 7,6 juta pada tahun 2008 menjadi 9,6 juta pada tahun 2018.  
Khusus untuk kanker payudara, data dari IARC menunjukkan bahwa penyakit ini banyak menyerang wanita, dengan tingkat kematian mencapai 627.000 kasus di seluruh dunia pada tahun 2018. Di Indonesia, berdasarkan data Riset Kesehatan Dasar (RISKESDAS) tahun 2018, insiden kanker payudara tercatat sebesar 42,1 per 100.000 penduduk dengan angka kematian rata-rata 17 per 100.000 penduduk. Angka ini meningkat dibandingkan tahun 2012, di mana insiden kanker payudara hanya sebesar 12,1 per 100.000 penduduk dengan total kematian 522.000 jiwa.  
Meningkatnya angka kejadian kanker payudara menunjukkan pentingnya kesadaran akan deteksi dini dan pengobatan yang lebih efektif. Oleh karena itu, pengembangan model machine learning menjadi solusi yang dapat membantu dokter dalam mengidentifikasi kanker payudara lebih awal. Model ini bertujuan untuk memprediksi apakah seseorang yang terdiagnosis kanker payudara memiliki jenis kanker yang ganas atau jinak. Dengan adanya model ini, diharapkan proses diagnosis menjadi lebih cepat dan akurat, sehingga memungkinkan intervensi medis lebih dini. 

  Format Referensi: [Hubungan Tingkat Pengetahuan Mengenai Kanker Payudara dengan Upaya Pencegahan dengan Pemeriksaan Payudara Sendiri pada Wanita Usia Subur di Puskesmas Rantau Laban Kota Tebing Tinggi](http://repository.uhn.ac.id/handle/123456789/4689) 
  Format Referensi: [GAMBARAN KUALITAS HIDUP PASIEN KANKER PAYUDARA](http://repo.poltekkesbandung.ac.id/1505/) 

## Business Understanding

### Problem Statements

Di sektor kesehatan, terutama dalam diagnosis kanker payudara, deteksi dini sangat penting untuk mengurangi angka kematian dan meningkatkan efektivitas perawatan. Namun, proses diagnosis manual sering kali memakan waktu dan berpotensi mengalami kesalahan, yang dapat menghambat penanganan tepat waktu. Oleh karena itu, masalah bisnis yang ingin diselesaikan adalah bagaimana memanfaatkan teknologi machine learning untuk membantu dokter mendiagnosis kanker payudara secara cepat dan akurat, sehingga proses perawatan dapat dioptimalkan dan biaya operasional berkurang.

### Goals

- Efisiensi Diagnosis: Mengembangkan model yang mampu memberikan prediksi diagnosis dengan akurasi di atas 90%, sehingga proses identifikasi kanker dapat dilakukan dengan cepat.
- Peningkatan Kualitas Perawatan: Memastikan bahwa model dapat mengurangi kesalahan diagnosis, mendukung keputusan medis, dan pada akhirnya meningkatkan outcome perawatan pasien.
- Optimalisasi Biaya Operasional: Mengurangi ketergantungan pada metode diagnosis manual yang memakan waktu dan sumber daya, sehingga dapat menekan biaya operasional di fasilitas kesehatan.

    ### Solution statements
    Solusi yang diusulkan adalah membangun beberapa model klasifikasi menggunakan algoritma seperti Random Forest, K-Nearest Neighbors (KNN), dan AdaBoost. Model-model tersebut akan dievaluasi dan dibandingkan berdasarkan metrik seperti akurasi, f1-score, precision, dan recall. Hasil evaluasi ini kemudian digunakan untuk memilih model terbaik yang tidak hanya unggul secara teknis tetapi juga memiliki dampak positif terhadap peningkatan proses diagnosis dan efektivitas layanan kesehatan.


## Data Understanding
![kaggle](https://i.postimg.cc/2SyPvwxP/image.png)
Data pada project ini menggunakan data yang bersumber pada sebuah situs kaggle, dimana fokus pada data tersebut menjelaskan faktor-faktor yang akan mempengaruhi sebuah penyakit kanker payudara bersifat ganas dan jinak.
Informasi dataset dapat dilihat pada tabel dibawah ini :
Jenis | Keterangan
--- | ---
Sumber | [Kaggle Dataset : Cancer Breast Dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)
Lisensi | CC0: Public Domain
Kategori | Cancer, Women, Healthcare
Jenis dan Ukuran Berkas | CSV (124.57 kB)

Pada berkas yang diunduh yakni cancer-breast.csv berisi 569 rows × 32 columns. Kolom-kolom tersebut terdiri dari 1 buah kolom bertipe objek dan 31 buah kolom bertipe numerik (tipe data float64). Untuk penjelasan mengenai variabel-variable pada dataset cancer breast ini dapat dilihat sebagai berikut:
- **id** merupakan parameter bernilai unique. Parameter ini tidak penting untuk dimasukkan kedalam model, oleh karena itu parameter ini di drop.
- **diagnosis** merupakan fitur target pada dataset ini, bertipe object yang terdiri dari (M,B). Dimana data tersebut menjelaskan diagnosis kanker bersifat Ganas (M) atau Jinak (B)
- **radius_mean** merupakan fitur yg merepresentasikan nilai rata-rata jarak dari pusat ke titik pada keliling sekitar payudara/benjolan
- **texture_mean** merupakan fitur yg merepresentasikan standar deviasi nilai skala abu-abu atau rata-rata Tekstur Permukaan
- **perimeter_mean** merupakan rata-rata keliling
- **area_mean** merupakan fitur yg merepresentasikan Rata-rata Luas Lobes
- **smoothness_mean** merupakan fitur yg merepresentasikan Rata-rata Tingkat Kehalusan
- **compactness_mean** merupakan fitur yg merepresentasikan Rata-rata Kekompakan atau keliling² / luas — 1.0
- **concavity_mean** merupakan fitur yg merepresentasikan rata-rata kecekungan atau keparahan bagian cekung dari contour
- **concave points_mean** merupakan fitur yg merepresentasikan rata-rata titik cekung atau jumlah bagian cekung dari contour
- **symmetry_mean** merupakan fitur yg merepresentasikan rata-rata Simetri
- **fractal_dimension_mean** merupakan fitur yg merepresentasikan rata-rata dimensi fraktal atau "*coastline approximation* — 1"
- **radius_se** merupakan fitur yg merepresentasikan radius standard error
- **texture_se** merupakan fitur yg merepresentasikan texture standard error
- **perimeter_se** merupakan fitur yg merepresentasikan perimeter standard error 
- **area_se** merupakan fitur yg merepresentasikan luas standar error
- **smoothness_se** merupakan fitur yg merepresentasikan smoothness standard error
- **compactness_se** merupakan fitur yg merepresentasikan compactness standard error
- **concavity_se** merupakan fitur yg merepresentasikan concavity standard error
- **concave points_se** merupakan fitur yg merepresentasikan titik cekung standard error
- **symmetry_se** merupakan fitur yg merepresentasikan symmetry standard error
- **fractal_dimension_se** merupakan fitur yg merepresentasikan fractal dimension standard error
- **radius_worst** merupakan fitur yg merepresentasikan radius terendah
- **texture_worst** merupakan fitur yg merepresentasikan texture terendah
- **perimeter_worst** merupakan fitur yg merepresentasikan perimeter terendah
- **area_worst** merupakan fitur yg merepresentasikan area terendah
- **smoothness_worst** merupakan fitur yg merepresentasikan tingkat kehalusan terendah
- **compactness_worst** merupakan fitur yg merepresentasikan compactness terendah
- **concavity_worst** merupakan fitur yg merepresentasikan kecekungan terendah
- **concave points_worst** merupakan fitur yg merepresentasikan titik cekung terendah
- **symmetry_worst** merupakan fitur yg merepresentasikan symmetry terendah
- **fractal_dimension_worst** merupakan fitur yg merepresentasikan fractional dimensi terendah

Berikut beberapa tahapan sebelum visualisasi data pada data preparation sebagai berikut:
- Meload Dataset ke dalam sebuah Dataframe menggunakan pandas
- ``` df.info()``` digunakan untuk mengecek tipe kolom pada dataset
- ```df.isna().sum()``` digunakan untuk mengecek apakah ada kolom yg kosong, pada dataset ini nilai kosong tidak ditemukan
- ```df.describe()``` digunakan utk mendapatkan info mengenai dataset terhadap nilai rata-rata, median, banyaknya data, nilai Q1 hingga Q3 dan lain-lain.

Berikut beberapa tahapan visualisasi data pada data preparation:
- Pertama membagi dataset kedalam 2 bentuk variable, yaitu variable untuk kolom tipe numerik dan variable kolom untuk tipe object
- Kemudian melakukan visualisasi distribusi categorial, dimana ini digunakan untuk menghitung jumlah sample Kanker Ganas/positif (M) dan kanker Jinak/negatif (B). pada project ini terdapat 357 jumlah data sampel kanker jinak (B) dan 212 data sample kanker ganas (M)
![img](https://i.postimg.cc/3JBdFYGt/Cuplikan-layar-2025-03-02-115126.png)
- lalu melakukan visualisasi distribusi numerik, yg dapat dilihat lebih rinci sebagai berikut:
![img](https://i.postimg.cc/d073FCxf/Cuplikan-layar-2025-03-01-172153.png)
- Selanjutnya visualisasi dilakukan untuk mengetahui korelasi antar fitur yg terdapat pada dataset, untuk selengkapnya sebagai berikut:
![img](https://i.postimg.cc/GpZ2m20m/Cuplikan-layar-2025-03-01-171239.png)

## Data Preparation
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
Tahapan di atas dilakukan untuk memastikan bahwa model tidak bias karena perbedaan skala antar fitur dan untuk menangani data yang tidak konsisten.

## Modeling
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

![img](https://i.postimg.cc/6pdJxJGk/Cuplikan-layar-2025-02-28-184328.png)

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

![img](https://i.postimg.cc/mD96C7N0/Cuplikan-layar-2025-02-28-191351.png)

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
  
![img](https://i.postimg.cc/8C6b0NJv/Cuplikan-layar-2025-02-28-191821.png)

**Improvement Model:**  
Setelah pelatihan awal, dilakukan hyperparameter tuning untuk setiap model guna mendapatkan performa optimal. Perbandingan metrik evaluasi kemudian digunakan untuk memilih model terbaik.

## Evaluation
Setelah melakukan pelatihan dan evaluasi, diperoleh hasil metrik sebagai berikut:
- Random Forest: Akurasi 97,37%, dengan nilai f1-score, precision, dan recall yang tinggi untuk kedua kelas.
- KNN dan Boosting: Performa mendekati model Random Forest, namun terdapat perbedaan kecil pada metrik recall.

Komparasi metrik ini menunjukkan bahwa model Random Forest memberikan performa yang konsisten dan unggul di hampir semua aspek evaluasi, sehingga dipilih sebagai model terbaik.

### Hubungan dengan Business Understanding  
Evaluasi model mengkonfirmasi bahwa solusi yang diusulkan telah menjawab problem statement dan mencapai goals yang diharapkan:
- **Menjawab Problem Statement:** Model Random Forest mampu mempercepat proses diagnosis dengan akurasi yang tinggi, sehingga mendukung deteksi dini kanker payudara dan mengurangi kesalahan diagnosis.  
- **Mencapai Goals:**  
  - **Efisiensi Diagnosis:** Dengan akurasi lebih dari 97%, model ini memungkinkan diagnosis yang cepat dan andal.  
  - **Peningkatan Kualitas Perawatan:** Akurasi dan nilai evaluasi tinggi mendukung keputusan medis yang lebih tepat, berpotensi meningkatkan outcome perawatan pasien.  
  - **Optimalisasi Biaya Operasional:** Penggunaan model ini dapat mengurangi waktu dan sumber daya yang diperlukan untuk proses diagnosis manual, sehingga memberikan dampak positif pada pengurangan biaya operasional.  
- **Dampak Solusi:** Implementasi model Random Forest diharapkan memberikan kontribusi signifikan dalam meningkatkan efektivitas layanan kesehatan dengan mengoptimalkan proses diagnosis, yang secara langsung dapat menurunkan angka kematian dan meningkatkan kualitas hidup pasien.

Secara keseluruhan, evaluasi menunjukkan bahwa model Random Forest tidak hanya unggul dalam performa teknis tetapi juga memiliki dampak positif yang jelas terhadap kebutuhan dan tujuan bisnis dalam diagnosis kanker payudara.
**---Ini adalah bagian akhir laporan---**
