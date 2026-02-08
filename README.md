# Decision Tree Tutorial

## 📚 Penjelasan Decision Tree (Pohon Keputusan)

### Apa itu Decision Tree?
Decision Tree adalah algoritma pembelajaran mesin yang digunakan untuk klasifikasi dan regresi. Algoritma ini bekerja dengan membuat model keputusan yang mirip dengan pohon (tree structure), di mana setiap node dalam pohon merepresentasikan fitur, setiap cabang merepresentasikan aturan keputusan, dan setiap daun (leaf) merepresentasikan hasil/kelas.

### Cara Kerja Decision Tree

#### 1. **Pemilihan Fitur Terbaik (Feature Selection)**
   - Algoritma memilih fitur yang paling dapat membedakan data menjadi kelas yang berbeda
   - Menggunakan kriteria seperti Information Gain (IG) atau Gini Impurity
   - Fitur dengan nilai IG tertinggi dipilih sebagai root node

#### 2. **Pemisahan Data (Data Splitting)**
   - Data dibagi menjadi subset berdasarkan nilai fitur yang dipilih
   - Setiap pemisahan menciptakan sub-pohon baru
   - Proses ini dilakukan secara rekursif (berulang) untuk setiap subset

#### 3. **Perhitungan Information Gain**
   ```
   Information Gain = Entropy(Parent) - Weighted Average of Entropy(Children)
   
   Entropy = -Σ(p_i * log2(p_i))
   p_i = proporsi sampel kelas i
   ```

#### 4. **Berhenti Splitting**
   - Ketika semua sampel dalam node memiliki kelas yang sama
   - Ketika kedalaman pohon mencapai batas maksimal
   - Ketika jumlah sampel kurang dari minimum samples per node
   - Ketika tidak ada peningkatan Information Gain

#### 5. **Prediksi (Prediction)**
   - Data baru dilewatkan melalui pohon dari root sampai leaf node
   - Kelas pada leaf node menjadi hasil prediksi

---

## 🔨 Step-by-Step Membangun Decision Tree

### Step 1: Persiapan Data
```python
# Load dataset (Iris dari sklearn)
# Split menjadi fitur (X) dan target (y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```

### Step 2: Membuat Model
```python
# Inisialisasi Decision Tree Classifier
model = DecisionTreeClassifier()
```

### Step 3: Training Model
```python
# Fit/train model dengan data training
model.fit(X_train, y_train)
```
Pada tahap ini:
- Algoritma membangun pohon keputusan secara otomatis
- Menemukan fitur terbaik dan threshold optimal untuk setiap split
- Membuat struktur pohon hingga kondisi berhenti terpenuhi

### Step 4: Evaluasi Model
```python
# Prediksi dengan data test
y_pred = model.predict(X_test)

# Hitung akurasi
accuracy = accuracy_score(y_test, y_pred)
```

### Step 5: Interpretasi Hasil
```python
# Lihat kedalaman pohon
print(f"Kedalaman pohon: {model.get_depth()}")

# Lihat jumlah leaf nodes
print(f"Jumlah daun (leaves): {model.get_n_leaves()}")

# Lihat feature importance
print(f"Pentingnya fitur: {model.feature_importances_}")
```

---

## 📊 Contoh: Dataset Iris
Dataset Iris memiliki 150 sampel dengan 4 fitur:
1. Sepal Length (panjang sepal)
2. Sepal Width (lebar sepal)
3. Petal Length (panjang petal)
4. Petal Width (lebar petal)

Target: 3 kelas iris (Setosa, Versicolor, Virginica)

---

## ✅ File Code
Lihat file `decision_tree.py` untuk implementasi lengkap dengan penjelasan di setiap baris kode.

Halo branch
