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

Berdasarkan latar belakang di atas, berikut ini rumusan masalah yang dapat diselesaikan pada proyek ini:  

- Bagaimana cara melakukan pra-pemrosesan pada data penyakit kanker payudara yang akan digunakan untuk membuat model yang baik?  
- Bagaimana cara membuat model untuk memprediksi penyakit kanker payudara ganas atau jinak pada manusia dengan menggunakan machine learning?  
- Bagaimana cara memilih/membuat algoritma yang mampu menghasilkan nilai akurasi di atas 90%?  

### Goals

- Melakukan pra-pemrosesan dengan baik agar dapat digunakan dalam pembuatan model.  
- Mengetahui cara membuat model machine learning untuk memprediksi penyakit kanker payudara ganas dan jinak pada wanita.  
- Membuat model machine learning dengan nilai akurasi yang mencapai 90%.  

    ### Solution statements
    Solusi yang dapat dilakukan untuk memenuhi tujuan dari proyek ini di antaranya:  
    * Untuk pra-pemrosesan data dapat dilakukan beberapa teknik, di antaranya:  
    * Melakukan _drop_ kolom pada kolom ID.  
    * Mengatasi masalah data yang kosong dengan melakukan pengecekan terlebih dahulu lalu menggantinya dengan nilai rata-rata atau median (dalam proyek ini, tidak ditemukan data yang kosong).  
    * Melakukan encoding terhadap kolom yang bertipe _object_.  
    * Membagi dataset menjadi dua bagian dengan rasio 80% untuk data latih dan 20% untuk data uji.  
    * Melakukan _Standard Scaler_.  

    * Untuk pembuatan model dipilih penggunaan model dengan algoritma Random Forest dan K-Nearest Neighbor. Algoritma tersebut dipilih karena mudah digunakan dan cocok untuk kasus ini. Berikut cara kerja, kelebihan, dan kekurangan dari kedua algoritma:  
    * **Cara kerja Algoritma Random Forest** [[4]](https://repository.usd.ac.id/35513/):  
        * Memilih k sampel dataset secara acak dengan pengembalian.  
        * Menggunakan dataset untuk membangun _decision tree_ ke-i.  
        * Mengulangi langkah di atas sebanyak k kali.  
    * **Kelebihan dan kekurangan Algoritma Random Forest** [[5]](https://eprints.umm.ac.id/39299/):  
        * **Kelebihan:** Dapat mengatasi _noise_ dan _missing value_ serta mampu menangani data dalam jumlah besar.  
        * **Kekurangan:** Interpretasi sulit dan memerlukan tuning model yang tepat.  
    * **Cara kerja Algoritma K-Nearest Neighbor** [[6]](https://publikasi.dinus.ac.id/index.php/jais/article/view/1189/):  
        * Menentukan jumlah tetangga terdekat K.  
        * Menghitung jarak dokumen _testing_ ke dokumen _training_.  
        * Mengurutkan data berdasarkan jarak Euclidean terkecil.  
        * Menentukan kelompok _testing_ berdasarkan label pada K.  
    * **Kelebihan dan kekurangan Algoritma K-Nearest Neighbor** [[7]](https://simdos.unud.ac.id/uploads/file_penelitian_1_dir/721bdb509a6f0bb9ccca6d7374b86759.pdf):  
        * **Kelebihan:** Tangguh terhadap _training_ data yang _noisy_ dan efektif jika data latihnya besar.  
        * **Kekurangan:**  
            * Perlu menentukan nilai parameter K.  
            * Pemilihan jarak dan atribut yang optimal tidak selalu jelas.  
            * Biaya komputasi tinggi karena perlu menghitung jarak setiap sampel uji ke seluruh sampel latih.


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
![img](https://i.postimg.cc/SRf5411z/image.png)
- lalu melakukan visualisasi distribusi numerik, yg dapat dilihat lebih rinci sebagai berikut:
![img](https://i.postimg.cc/Bvyf7zRS/image.png)
- Selanjutnya visualisasi dilakukan untuk mengetahui korelasi antar fitur yg terdapat pada dataset, untuk selengkapnya sebagai berikut:
![img](https://i.ibb.co/2gg4qvP/image.png)

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
Pada proyek ini, model yang dikembangkan adalah kasus klasifikasi dan menggunakan metriks akurasi, *f1-score*, *recall* dan *precision*. Berikut hasil pengukuran model yang dipilih yaitu model yang menggunakan algoritma Random Forest metriks akurasi, _f1-score_, _recall_ dan _precision_.
![RF](https://i.postimg.cc/d1tp88yV/Cuplikan-layar-2025-03-01-151435.png)
* Akurasi
    Akurasi merupakan metrik untuk menghitung persentase dari total data yang diidentifikasi dan dinilai benar. Rumus akurasi sebagai berikut:
    ![Image of Dataset](https://i.postimg.cc/NFx1VcgJ/akurasi.png)
    * _True Positive_ (TP):
    Kasus dimana model merupakan data positif yang diprediksi benar. Contohnya, pasien menderita kanker (class 1) dan dari model yang dibuat memprediksi pasien tersebut menderita kanker (class 1).
    * _True Negative_ (TN):
    Kasus dimana model merupakan data negatif yang diprediksi benar. Contohnya, pasien tidak menderita kanker (class 2) dan dari model yang dibuat memprediksi pasien tersebut tidak menderita kanker (class 2).
    * _False Positive_ (FP) - **Type I Error** :
    Kasus dimana model merupakan data negatif namun diprediksi sebagai data positif. Contohnya, pasien tidak menderita kanker (class 2) tetapi dari model yang telah memprediksi pasien tersebut menderita kanker (class 1).
    * _False Negative_ (FN) - **Type II Error** :
    Kasus dimana model merupakan data negatif namun diprediksi sebagai data positif. Contohnya, pasien tidak menderita kanker (class 2) tetapi dari model yang telah memprediksi pasien tersebut menderita kanker (class 1).
* _Precision_
    _Precision_ merupakan metrik untuk memprediksi benar positif dari keseluruhan hasil yang diprediksi positf. Rumus _precision_ sebagai berikut:
    ![Image of Dataset](https://i.postimg.cc/mzwZLjdM/precision.png)
* _Recall_
    _Recall_ merupakan metrik untuk memprediksi benar positif dibandingkan dengan keseluruhan data yang benar positif. Rumus _precision_ sebagai berikut:
    ![Image of Dataset](https://i.postimg.cc/K38GRTVW/recall.png)
* _f1-score_
    _f1-score_ merupakan metrik untuk perbandingan rata-rata precision dan recall yang dibobotkan. Rumus _f1-score_ sebagai berikut:
    ![Image of Dataset](https://i.postimg.cc/Fzm9ztjQ/f1-score.png)

**---Ini adalah bagian akhir laporan---**
