# Sistem Rekomendasi Makanan - Andika Bintang Nursalih

## Domain Proyek

### **Latar Belakang**

Industri kuliner kini mengalami pertumbuhan pesat dengan semakin banyaknya restoran, rumah makan, dan kafe yang bermunculan. Ketersediaan ragam pilihan makanan yang sangat banyak seringkali membuat konsumen bingung dalam menentukan pilihan yang sesuai dengan selera dan kebutuhan mereka. Di sisi lain, pelaku usaha di bidang kuliner juga membutuhkan cara untuk meningkatkan profitabilitas dan menarik lebih banyak pelanggan. Oleh karena itu, diperlukan suatu sistem rekomendasi makanan yang dapat menyajikan saran yang *personalized* berdasarkan preferensi pengguna serta karakteristik makanan itu sendiri.

Pada proyek ini, sistem rekomendasi dibangun menggunakan dua pendekatan utama:  
- **Content-Based Filtering**: Menggunakan informasi deskriptif (misalnya, kategori makanan) untuk menemukan kemiripan antar item.  
- **Collaborative Filtering**: Menggunakan data rating pengguna untuk memprediksi selera dan memberikan rekomendasi yang lebih personal.

Referensi: [Food Recommendation System](https://www.kaggle.com/datasets/schemersays/food-recommendation-system)

## Business Understanding

Proyek ini bertujuan untuk membantu konsumen dalam menentukan pilihan makanan yang sesuai dengan preferensinya, serta memberikan insight bagi pelaku usaha kuliner untuk meningkatkan penjualan.

### *Problem Statement*
- Bagaimana memberikan rekomendasi makanan yang *personalized* sehingga sesuai dengan preferensi masing-masing pengguna?
- Bagaimana memanfaatkan data rating dan informasi deskriptif makanan untuk membangun sistem rekomendasi yang akurat dan efisien?

### *Project Goals*
- Mengembangkan sistem rekomendasi makanan yang mampu menyajikan saran secara personal.
- Mengimplementasikan dua pendekatan utama (content-based filtering dan collaborative filtering) untuk mengoptimalkan performa rekomendasi.
  
### *Solution Statement*
- **Content-Based Filtering**: Menerapkan teknik TF-IDF Vectorizer untuk mengekstrak fitur penting dari kategori makanan, dilanjutkan dengan perhitungan *cosine similarity* guna mengidentifikasi makanan yang mirip.
- **Collaborative Filtering**: Membangun model neural network dengan embedding layer (RecommenderNet) untuk mempelajari interaksi antara pengguna dan makanan serta memprediksi rating.
- Evaluasi dilakukan dengan menggunakan metrik *Precision* untuk content-based dan *Root Mean Squared Error* (RMSE) untuk collaborative filtering.

## Data Understanding

Data yang digunakan bersumber dari [Food Recommendation System](https://www.kaggle.com/datasets/schemersays/food-recommendation-system) di Kaggle. Dataset ini terdiri dari dua bagian:

- **Data Makanan**: Berisi informasi seperti `Food_ID`, `Name`, dan `C_Type` (kategori makanan).
- **Data Rating**: Berisi informasi mengenai rating yang diberikan oleh pengguna terhadap makanan, dengan fitur `User_ID`, `Food_ID`, dan `Rating`.

Contoh data makanan:
| Food_ID | Name                  | C_Type       |
|---------|-----------------------|--------------|
| 1       | summer squash salad   | Healthy Food |
| 2       | chicken minced salad  | Healthy Food |
| 3       | sweet chilli almonds  | Snack        |

Contoh data rating:
| User_ID | Food_ID | Rating |
|---------|---------|--------|
| 1       | 88      | 4      |
| 1       | 46      | 3      |
| 1       | 24      | 5      |

Langkah-langkah eksplorasi data meliputi:
1. Memuat dataset dan melakukan pemeriksaan deskriptif (jumlah baris, kolom, dan nilai unik).
2. Melakukan visualisasi distribusi kategori makanan dan distribusi rating untuk mendapatkan insight awal.
3. Menggabungkan data makanan dengan data rating berdasarkan kolom `Food_ID` guna mendapatkan informasi lengkap untuk setiap makanan.

## Data Preparation

Tahapan persiapan data yang dilakukan adalah sebagai berikut:

1. **Pembersihan Data**:
   - Mengoreksi inkonsistensi pada kolom kategori (misalnya mengganti `' Korean'` menjadi `'Korean'`).
   - Menghapus baris data yang mengandung missing values pada kedua dataset.

2. **Penggabungan Data**:
   - Menggabungkan dataset rating dengan dataset makanan menggunakan `Food_ID` sebagai kunci utama sehingga setiap rating memiliki informasi tambahan seperti nama dan kategori makanan.

3. **Encoding**:
   - Mengubah `User_ID` dan `Food_ID` ke dalam format numerik agar dapat digunakan pada model collaborative filtering.

4. **Normalisasi**:
   - Menormalisasi rating ke dalam rentang tertentu (misalnya 0 hingga 1) untuk memudahkan training model dengan fungsi aktivasi sigmoid.

5. **Splitting Data**:
   - Memisahkan data menjadi training set (80%) dan validation set (20%) untuk keperluan pelatihan model.

## Modeling

### ***Content-Based Filtering***
Pada pendekatan content-based filtering, langkah-langkah yang dilakukan antara lain:
- **Fitur Ekstraksi**: Menggunakan TF-IDF Vectorizer untuk mengubah kategori makanan menjadi representasi numerik.
- **Perhitungan Similaritas**: Menghitung *cosine similarity* antara vektor TF-IDF sehingga dapat diketahui derajat kemiripan antar makanan.
- **Fungsi Rekomendasi**: Dibuat fungsi yang menerima nama makanan sebagai input dan mengembalikan daftar rekomendasi berdasarkan kemiripan.

Contoh kode:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(data['category'])
cosine_sim = cosine_similarity(tfidf_matrix)
```

### ***Collaborative Filtering***
Pada pendekatan collaborative filtering, model yang dikembangkan menggunakan neural network dengan embedding layer:
- **Model RecommenderNet**: Dibangun dengan embedding layer untuk pengguna dan makanan, ditambah dengan bias masing-masing.
- **Operasi Dot Product**: Menghitung dot product antara embedding pengguna dan makanan, kemudian menambahkan bias untuk menghasilkan prediksi rating.
- **Fungsi Aktivasi**: Menggunakan fungsi sigmoid untuk mengonversi output ke rentang [0,1].
- **Training**: Model dilatih menggunakan loss function Binary Crossentropy, optimizer Adam dengan learning rate 0.0001, dan dievaluasi menggunakan metrik RMSE.

Contoh kode:
```python
class RecommenderModel(tf.keras.Model):
    def __init__(self, total_users, total_foods, embed_size, **kwargs):
        super(RecommenderModel, self).__init__(**kwargs)
        self.user_embed = layers.Embedding(total_users, embed_size, 
                                           embeddings_initializer='he_normal',
                                           embeddings_regularizer=keras.regularizers.l2(1e-6))
        self.user_bias = layers.Embedding(total_users, 1)
        self.food_embed = layers.Embedding(total_foods, embed_size, 
                                           embeddings_initializer='he_normal',
                                           embeddings_regularizer=keras.regularizers.l2(1e-6))
        self.food_bias = layers.Embedding(total_foods, 1)
    
    def call(self, inputs):
        u_vec = self.user_embed(inputs[:, 0])
        u_bias = self.user_bias(inputs[:, 0])
        f_vec = self.food_embed(inputs[:, 1])
        f_bias = self.food_bias(inputs[:, 1])
        
        dot = tf.reduce_sum(u_vec * f_vec, axis=1, keepdims=True)
        x_out = dot + u_bias + f_bias
        return tf.nn.sigmoid(x_out)
```

## Evaluation

### *Content-Based Filtering*
Evaluasi dilakukan dengan menggunakan metrik **Precision**, yang mengukur akurasi rekomendasi.  
Persamaan Precision:
$$Precision = \frac{TP}{TP + FP}$$  
Di mana *True Positive* (TP) adalah jumlah rekomendasi relevan dan *False Positive* (FP) adalah jumlah rekomendasi yang tidak relevan.

### *Collaborative Filtering*
Evaluasi model collaborative filtering dilakukan menggunakan **Root Mean Squared Error (RMSE)**, yang mengukur rata-rata kesalahan prediksi rating.  
Persamaan RMSE:
$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

Grafik pelatihan model menunjukkan tren penurunan RMSE seiring bertambahnya epoch, yang mengindikasikan bahwa model mampu belajar dari data dengan baik.

## Kesimpulan dan Saran
- Sistem rekomendasi yang dikembangkan menggunakan kedua pendekatan (content-based dan collaborative filtering) mampu memberikan saran makanan yang *personalized* dan relevan.
- Evaluasi dengan metrik Precision menunjukkan bahwa rekomendasi berbasis konten memiliki tingkat akurasi yang tinggi, sedangkan model collaborative filtering menunjukkan RMSE yang rendah, menandakan prediksi rating yang akurat.
- Untuk pengembangan selanjutnya, disarankan untuk mengeksplorasi pendekatan *hybrid* yang menggabungkan kedua metode guna mengoptimalkan performa rekomendasi serta meningkatkan pengalaman pengguna.

## Daftar Pustaka

[1] Jabar Digital Service, "Jumlah Usaha Restoran, Rumah Makan, dan Cafe Berdasarkan Kabupaten/Kota di Jawa Barat," Jabarprov.go.id, 2021. https://opendata.jabarprov.go.id

[2] SEDLÁK, Matúš. *Content-based Recommender System for Food Recipes*. Masaryk University, 2022. [Online]. Available: https://is.muni.cz/th/vdqix/

[3] Vivek, M; Manju, N; Vijay, M. (2023). *Machine Learning Based Food Recipe Recommendation System*.

[4] schemersays, "Food Recommendation System," Kaggle.com, 2022. https://www.kaggle.com/datasets/schemersays/food-recommendation-system

---