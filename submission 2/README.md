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

Proyek ini menggunakan dua *dataset* utama, yaitu **dataset makanan** dan **dataset rating**, yang keduanya bersumber dari [Food Recommendation System](https://www.kaggle.com/datasets/schemersays/food-recommendation-system).

### 1. Dataset Makanan

1. **Jumlah Data**  
   - Berdasarkan pemanggilan `df_makanan.shape`, didapatkan hasil **(400, 5)**.  
     Ini berarti **dataset makanan** memiliki **400 baris** dan **5 kolom**.

2. **Struktur Data**  
   - Hasil `df_makanan.info()` memberikan gambaran berikut:
     ```
     <class 'pandas.core.frame.DataFrame'>
     RangeIndex: 400 entries, 0 to 399
     Data columns (total 5 columns):
      #   Column    Non-Null Count  Dtype 
      ---  ------    --------------  ----- 
      0   Food_ID   400 non-null    int64 
      1   Name      400 non-null    object
      2   C_Type    400 non-null    object
      3   Veg_Non   400 non-null    object
      4   Describe  400 non-null    object
     dtypes: int64(1), object(4)
     memory usage: 15.8+ KB
     ```
     Terlihat bahwa tidak ada *missing value* pada setiap kolom di dataset makanan (masing-masing kolom memiliki 400 *non-null* entries).

3. **Uraian Fitur**  
   Berikut adalah deskripsi singkat mengenai setiap fitur pada dataset makanan:  
   - **Food_ID**: Kode unik untuk setiap makanan.  
   - **Name**: Nama makanan.  
   - **C_Type**: Kategori atau jenis makanan (misalnya *Indian*, *Snack*, *Dessert*, dll.).  
   - **Veg_Non**: Penanda apakah makanan tersebut vegetarian (*veg*) atau non-vegetarian (*non-veg*).  
   - **Describe**: Deskripsi makanan, misalnya bahan dan cara pembuatan.

4. **Distribusi Kategori Makanan**  
   Melalui perintah `df_makanan['C_Type'].value_counts().sort_values().plot(kind='barh')`, dihasilkan visualisasi gambar di bawah ini yang menunjukkan jumlah makanan pada setiap kategori. Dapat diamati bahwa **Indian**, **Healthy Food**, dan **Dessert** menjadi kategori terbanyak pada dataset ini.

![img1](https://github.com/user-attachments/assets/3170ebc2-e099-408a-a90d-694b70c091cb)

### 2. Dataset Rating

1. **Jumlah Data**  
   - Berdasarkan pemanggilan `df_rating.shape`, didapatkan hasil **(512, 3)**.  
     Ini berarti **dataset rating** memiliki **512 baris** dan **3 kolom**.

2. **Struktur Data**  
   - Hasil `df_rating.info()` memberikan gambaran berikut:
     ```
     <class 'pandas.core.frame.DataFrame'>
     RangeIndex: 512 entries, 0 to 511
     Data columns (total 3 columns):
      #   Column   Non-Null Count  Dtype  
      ---  ------   --------------  -----  
      0   User_ID  511 non-null    float64
      1   Food_ID  511 non-null    float64
      2   Rating   511 non-null    float64
     dtypes: float64(3)
     memory usage: 12.1 KB
     ```
     Terlihat bahwa terdapat 1 baris data *missing* untuk masing-masing kolom (511 *non-null* dari total 512 baris).

3. **Uraian Fitur**  
   Berikut adalah deskripsi singkat mengenai setiap fitur pada dataset rating:  
   - **User_ID**: Kode unik untuk setiap pengguna yang memberikan rating.  
   - **Food_ID**: Kode unik makanan yang dirujuk (sesuai dengan *Food_ID* pada dataset makanan).  
   - **Rating**: Nilai penilaian yang diberikan oleh pengguna, umumnya dalam skala 1 hingga 10.

4. **Distribusi Rating**  
   Melalui perintah `sns.displot(df_rating['Rating'], kde=True, bins=10)`, dihasilkan visualisasi gambar di bawah ini yang menunjukkan sebaran rating yang diberikan oleh pengguna. Rating tampak tersebar pada rentang 1–10, dengan beberapa puncak di nilai 3, 5, dan 10.
   
![img2](https://github.com/user-attachments/assets/ae1bfaa4-5162-4228-af7e-12a2cce35164)

## Data Preparation

Pada tahap Data Preparation, dilakukan serangkaian langkah untuk memastikan data yang akan digunakan dalam model bersih dan siap untuk diproses. Tahapan yang dilakukan meliputi:

### 1. Handling Missing Values & Penggabungan Data

Untuk menggabungkan informasi rating dengan data makanan, kedua dataset di-*merge* berdasarkan kolom `Food_ID`. Setelah itu, dilakukan pengecekan missing values dan penghapusan baris yang tidak lengkap.

```python
df_gabungan = pd.merge(df_rating, df_makanan[['Food_ID', 'Name', 'C_Type']], on='Food_ID', how='left')
print(df_gabungan.head())
print(df_gabungan.isnull().sum())
df_gabungan = df_gabungan.dropna()
```

**Output:**

```
   User_ID  Food_ID  Rating                        Name    C_Type
0      1.0     88.0     4.0     peri peri chicken satay     Snack
1      1.0     46.0     3.0     steam bunny chicken bao  Japanese
2      1.0     24.0     5.0  green lentil dessert fudge   Dessert
3      1.0     25.0     4.0          cashew nut cookies   Dessert
4      2.0     49.0     1.0        christmas tree pizza   Italian
```

```
User_ID    1
Food_ID    1
Rating     1
Name       1
C_Type     1
dtype: int64
```

### 2. Handling Duplicate & Sorting Data

Data hasil penggabungan kemudian diurutkan berdasarkan `Food_ID`. Setelah itu, duplikat dihapus agar setiap makanan hanya muncul sekali. Selain itu, terdapat penyesuaian nilai pada kolom `C_Type` (misalnya, mengubah `'Healthy Food'` menjadi `'Healthy_Food'`).

```python
df_urut = df_gabungan.sort_values('Food_ID', ascending=True)
print(df_urut.head())
print(len(df_urut['Food_ID'].unique()))
print(df_urut['C_Type'].unique())
```

**Output:**

```
     User_ID  Food_ID  Rating                  Name        C_Type
376     71.0      1.0    10.0   summer squash salad  Healthy Food
253     49.0      1.0     5.0   summer squash salad  Healthy Food
116     22.0      2.0     5.0  chicken minced salad  Healthy Food
200     39.0      2.0    10.0  chicken minced salad  Healthy Food
50       9.0      2.0     3.0  chicken minced salad  Healthy Food
```

```
309
```

```
['Healthy Food' 'Snack' 'Dessert' 'Japanese' 'Indian' 'French' 'Mexican'
 'Italian' 'Chinese' 'Beverage' 'Thai']
```

### 3. Pembuatan DataFrame Baru untuk Content-Based Filtering

Setelah pembersihan, dibuatlah dataframe baru untuk content-based filtering dengan mengekstrak kolom `Food_ID`, `Name`, dan `C_Type`. Data ini kemudian dikonversi ke format yang lebih sederhana dengan mengganti nama kolom (misalnya, `C_Type` menjadi `category`).

```python
list_id = df_prep['Food_ID'].tolist()
list_nama = df_prep['Name'].tolist()
list_kategori = df_prep['C_Type'].tolist()

print(len(list_id), len(list_nama), len(list_kategori))

df_makanan_baru = pd.DataFrame({
    'id': list_id,
    'food_name': list_nama,
    'category': list_kategori
})

print(df_makanan_baru.head())
print(df_makanan_baru.sample(5))
```

**Output:**

```
309 309 309
```

Contoh tampilan awal dataframe:
```
    id             food_name      category
0  1.0   summer squash salad  Healthy_Food
1  2.0  chicken minced salad  Healthy_Food
2  3.0  sweet chilli almonds         Snack
3  4.0       tricolour salad  Healthy_Food
4  5.0        christmas cake       Dessert
```

Contoh tampilan sample:
```
        id                               food_name      category
282  283.0                       veg hakka noodles        Chinese
60    61.0  crunchy vegetable dal sattu croquettes        Italian
90    91.0                         chicken biryani         Indian
46    47.0                       meat lovers pizza        Italian
207  208.0    fennel scented sweet banana fritters       Dessert
```

### 4. Ekstraksi Fitur dengan TF-IDF

Untuk mengukur pentingnya kata pada kolom `category`, digunakan metode TF-IDF. Tahap ini menghasilkan matriks fitur yang nantinya digunakan untuk menghitung derajat kemiripan antar makanan (cosine similarity).

```python
tfidf_vect = TfidfVectorizer()
tfidf_vect.fit(df_makanan_baru['category'])
print(tfidf_vect.get_feature_names_out())

tfidf_matrix = tfidf_vect.fit_transform(df_makanan_baru['category'])
print(tfidf_matrix.shape)
print(tfidf_matrix.todense())
```

**Output:**

```
['beverage' 'chinese' 'dessert' 'french' 'healthy_food' 'indian' 'italian'
 'japanese' 'mexican' 'snack' 'thai']
```

```
(309, 11)
```

Contoh tampilan matriks TF-IDF (dense):
```
[[0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 1. 0.]
 ...
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 1. 0.]]
```

Untuk memberikan gambaran yang lebih terperinci, ditampilkan pula sample subset dari dataframe TF-IDF:

```python
df_tfidf = pd.DataFrame(tfidf_matrix.todense(), 
                        columns=tfidf_vect.get_feature_names_out(),
                        index=df_makanan_baru['food_name'])
print(df_tfidf.sample(11, axis=1).sample(10, axis=0))
```

**Output (Contoh Tampilan):**

```
                                   healthy_food  japanese  snack  thai  \
food_name                                                                
crispy herb chicken                         0.0       0.0    0.0   0.0   
sticky rum chicken wings                    0.0       0.0    1.0   0.0   
chicken dragon                              0.0       0.0    0.0   0.0   
spanish artichoke and spinach dip           0.0       0.0    0.0   0.0   
summer squash salad                         1.0       0.0    0.0   0.0   
holi special ice tea thandai                0.0       0.0    0.0   0.0   
chicken gilafi kebab                        0.0       0.0    0.0   0.0   
methi chicken masala                        0.0       0.0    0.0   0.0   
andhra pan fried pomfret                    0.0       0.0    0.0   0.0   
whole wheat cake                            1.0       0.0    0.0   0.0   

                                   indian  beverage  french  mexican  dessert  \
food_name                                                                       
crispy herb chicken                   0.0       0.0     0.0      0.0      0.0   
sticky rum chicken wings              0.0       0.0     0.0      0.0      0.0   
chicken dragon                        0.0       0.0     0.0      0.0      0.0   
spanish artichoke and spinach dip     0.0       0.0     0.0      1.0      0.0   
summer squash salad                   0.0       0.0     0.0      0.0      0.0   
holi special ice tea thandai          0.0       1.0     0.0      0.0      0.0   
chicken gilafi kebab                  1.0       0.0     0.0      0.0      0.0   
methi chicken masala                  1.0       0.0     0.0      0.0      0.0   
andhra pan fried pomfret              1.0       0.0     0.0      0.0      0.0   
whole wheat cake                      0.0       0.0     0.0      0.0      0.0   

                                   italian  chinese  
food_name                                            
crispy herb chicken                    1.0      0.0  
sticky rum chicken wings               0.0      0.0  
chicken dragon                         0.0      1.0  
spanish artichoke and spinach dip      0.0      0.0  
summer squash salad                    0.0      0.0  
holi special ice tea thandai             0.0      0.0  
chicken gilafi kebab                   0.0      0.0  
methi chicken masala                   0.0      0.0  
andhra pan fried pomfret               0.0      0.0  
whole wheat cake                       0.0      0.0  
```

### 5. Pembuatan Dictionary untuk Collaborative Filtering

Untuk melakukan encoding pada data rating, dibuatlah dictionary yang memetakan `User_ID` dan `Food_ID` ke indeks numerik. Hal ini akan digunakan dalam model collaborative filtering.

```python
user_list = df_rating['User_ID'].unique().tolist()
print('User_ID list:', user_list)

user2idx = {user: idx for idx, user in enumerate(user_list)}
idx2user = {idx: user for idx, user in enumerate(user_list)}
print('Mapping User_ID:', user2idx)
print('Mapping indeks ke User_ID:', idx2user)

food_list = df_rating['Food_ID'].unique().tolist()
food2idx = {food: idx for idx, food in enumerate(food_list)}
idx2food = {idx: food for idx, food in enumerate(food_list)}
```

**Output:**

```
User_ID list: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, ... , 100.0]
```

```
Mapping User_ID: {1.0: 0, 2.0: 1, 3.0: 2, 4.0: 3, 5.0: 4, 6.0: 5, ... , 100.0: 99}
Mapping indeks ke User_ID: {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 6.0, ... , 99: 100.0}
```

### 6. Persiapan Data untuk Training pada Collaborative Filtering

Setelah encoding, data rating dipersiapkan untuk training dengan melakukan mapping ulang ke variabel baru dan normalisasi rating. Data kemudian diacak dan dibagi menjadi training set dan validation set.

```python
df_rating['user'] = df_rating['User_ID'].map(user2idx)
df_rating['food'] = df_rating['Food_ID'].map(food2idx)

n_users = len(user2idx)
n_foods = len(idx2food)
print(n_users, n_foods)

df_rating['rating'] = df_rating['Rating'].astype(np.float32)
min_rat = df_rating['rating'].min()
max_rat = df_rating['rating'].max()
print('Jumlah User: {}, Jumlah Food: {}, Rating Min: {}, Rating Max: {}'.format(n_users, n_foods, min_rat, max_rat))

df_rating = df_rating.sample(frac=1, random_state=42)

X_data = df_rating[['user', 'food']].values
y_data = df_rating['rating'].apply(lambda x: (x - min_rat) / (max_rat - min_rat)).values

split_index = int(0.8 * df_rating.shape[0])
X_train, X_val = X_data[:split_index], X_data[split_index:]
y_train, y_val = y_data[:split_index], y_data[split_index:]
print(X_data, y_data)
```

**Output yang Diharapkan:**

- Tampilan jumlah pengguna dan jumlah makanan:
  ```
  100 309
  ```
- Informasi rating:
  ```
  Jumlah User: 100, Jumlah Food: 309, Rating Min: 1.0, Rating Max: 10.0
  ```
- Tampilan array `X_data` dan `y_data` sebagai hasil mapping dan normalisasi rating.



## Modeling and Results

### Content-Based Filtering

Pada pendekatan content-based filtering, sistem rekomendasi menghitung derajat kemiripan antar makanan menggunakan *cosine similarity* dari matriks yang dihasilkan di tahap Data Preparation. Fungsi rekomendasi menerima input berupa nama makanan dan mengembalikan daftar makanan dengan kemiripan tertinggi berdasarkan kategori.

Contoh fungsi rekomendasi:

```python
def rekomendasi_makanan(nama_makanan, sim_data=df_cosine, items=df_makanan_baru[['food_name', 'category']], top=5):
    # Mengambil indeks kemiripan
    idx_array = sim_data.loc[:, nama_makanan].to_numpy().argpartition(range(-1, -top, -1))
    rekomendasi = sim_data.columns[idx_array[-1:-(top+2):-1]]
    rekomendasi = rekomendasi.drop(nama_makanan, errors='ignore')
    return pd.DataFrame(rekomendasi).merge(items).head(top)
```

Contoh pemanggilan:
```python
print(df_makanan_baru[df_makanan_baru['food_name'] == 'banana chips'])
print(rekomendasi_makanan('banana chips'))
```

**Output:**

Data makanan dengan nama *banana chips*:
| id    | food_name    | category |
|-------|--------------|----------|
| 306.0 | banana chips | Snack    |

Rekomendasi untuk "banana chips":
| Food Name                                     | Category |
|-----------------------------------------------|----------|
| puffed rice                                   | Snack    |
| californian breakfast benedict                | Snack    |
| banana phirni tartlets with fresh strawberries | Snack    |
| baked raw banana samosa                       | Snack    |
| baked multigrain murukku                        | Snack    |

### Collaborative Filtering

Pada pendekatan collaborative filtering, model neural network dibangun dengan embedding layer untuk pengguna dan makanan. Model ini menghitung prediksi rating dengan mengoperasikan dot product antara embedding pengguna dan makanan, ditambah bias, dan diaktivasi dengan fungsi sigmoid sehingga output berada pada rentang [0,1].

Contoh struktur model:
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

Model dilatih selama 50 epoch dengan loss Binary Crossentropy dan optimizer Adam (learning rate 0.0001). Evaluasi dilakukan menggunakan metrik RMSE.

Setelah pelatihan, model digunakan untuk memprediksi rating dan menghasilkan rekomendasi bagi seorang sample user. Berikut adalah contoh output rekomendasi untuk *User_ID* 6:

**Makanan yang Pernah Dinilai Tinggi oleh User (User: 6)**
| Food Name                                | Category  |
|------------------------------------------|-----------|
| baked namak para                         | Snack     |
| red wine braised mushroom flatbread      | Italian   |
| spiced almond banana jaggery cake        | Dessert   |
| berry parfait hazelnut white chocolate sable | Dessert   |

**Top 10 Rekomendasi Makanan untuk User 6**
| Food Name                     | Category      |
|-------------------------------|---------------|
| chocolate nero cookies        | Dessert       |
| spinach and feta crepes       | French        |
| sugar free modak              | Japanese      |
| crispy herb chicken           | Italian       |
| vegetable bruschetta          | Italian       |
| easy bread poha               | Indian        |
| egg paratha                   | Indian        |
| eggless coffee cupcakes       | Dessert       |
| double chocolate easter cookies | Dessert     |
| black rice                    | Healthy_Food  |

---

## Evaluation

### Content-Based Filtering

Evaluasi sistem rekomendasi berbasis konten dilakukan menggunakan metrik **Precision**, yang mengukur proporsi rekomendasi yang relevan. Berdasarkan evaluasi pada sample, diperoleh:

$$Precision = \frac{TP}{TP + FP} = \frac{5}{5+0} = 1$$

Dengan demikian, nilai Precision adalah **1** (atau 100%), yang menunjukkan bahwa seluruh rekomendasi yang diberikan oleh sistem berbasis konten relevan dengan preferensi pengguna.

### Collaborative Filtering

Model collaborative filtering dievaluasi dengan metrik **Root Mean Squared Error (RMSE)**, yang mengukur rata-rata kesalahan prediksi rating. Rumus RMSE adalah:

$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

Berdasarkan hasil pelatihan selama 50 epoch, diperoleh nilai sebagai berikut:

- **RMSE pada data training (epoch terakhir):** 0.2623  
- **RMSE pada data validasi (epoch terakhir):** 0.3264  

Grafik pelatihan menunjukkan tren penurunan RMSE secara konsisten seiring bertambahnya epoch, yang mengindikasikan bahwa model mampu belajar dengan baik dari data. Nilai RMSE yang rendah pada data training menunjukkan bahwa model cukup akurat dalam memprediksi rating, meskipun adanya gap kecil antara data training dan validasi menunjukkan potensi overfitting ringan.

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
