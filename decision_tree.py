# ============================================================================
# DECISION TREE CLASSIFIER - TUTORIAL LENGKAP
# ============================================================================
# Program ini mendemonstrasikan cara membangun dan menggunakan Decision Tree
# untuk klasifikasi dengan dataset Iris dari scikit-learn
# ============================================================================

# Import library untuk menyembunyikan warning yang tidak penting
import warnings
warnings.filterwarnings('ignore')

# Import library yang diperlukan
import numpy as np              # Untuk operasi numerik array
import pandas as pd             # Untuk manipulasi data (DataFrame)

from sklearn import datasets    # Untuk dataset bawaan sklearn
from sklearn.metrics import accuracy_score  # Untuk menghitung akurasi model
from sklearn.tree import DecisionTreeClassifier  # Algoritma Decision Tree
from sklearn.model_selection import train_test_split  # Split data train-test

# ============================================================================
# STEP 1: PERSIAPAN DATA (Data Preparation)
# ============================================================================

# Load dataset Iris dari scikit-learn
# Dataset ini berisi 150 sampel bunga iris dengan 4 fitur numerik
iris = datasets.load_iris()

# Buat DataFrame pandas untuk mempermudah manipulasi data
# np.c_ menggabungkan fitur (iris['data']) dengan target (iris['target'])
data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

print("Dataset Iris:")
print(f"Jumlah sampel: {data.shape[0]}")
print(f"Jumlah fitur: {data.shape[1] - 1}")
print(f"Nama fitur: {list(iris['feature_names'])}")
print(f"Kelas target: {list(set(data['target']))}")
print()

# Pisahkan fitur (X) dan target (y)
# X: berisi 4 fitur iris (sepal length, sepal width, petal length, petal width)
# y: berisi kelas target (0=Setosa, 1=Versicolor, 2=Virginica)
X = data.drop('target', axis=1)  # Fitur - hapus kolom target
y = data[['target']]              # Target - hanya kolom target

print(f"Fitur (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print()

# ============================================================================
# STEP 2: SPLIT DATA MENJADI TRAINING DAN TESTING (Data Splitting)
# ============================================================================

# Pisahkan data menjadi 70% training dan 30% testing
# random_state=42 untuk memastikan reproducibility (hasil yang sama setiap kali)
X_train, X_test, y_train, y_test = train_test_split(
    X,                  # Data fitur
    y,                  # Data target
    test_size = 0.3,    # 30% untuk testing, 70% untuk training
    random_state = 42   # Seed untuk random number generator
)

print(f"Training set size: {X_train.shape[0]} sampel")
print(f"Testing set size: {X_test.shape[0]} sampel")
print()

# ============================================================================
# STEP 3: MEMBUAT DAN TRAINING MODEL DECISION TREE (Model Creation & Training)
# ============================================================================

# Inisialisasi Decision Tree Classifier dengan parameter default
# Decision Tree akan secara otomatis:
# - Memilih fitur terbaik untuk setiap split
# - Mencari threshold optimal untuk membagi data
# - Membuat struktur pohon hingga semua leaf pure atau stopping criteria terpenuhi
model = DecisionTreeClassifier()

# FIT: Train model dengan data training
# Pada tahap ini, algoritma membangun pohon keputusan lengkap
model.fit(X_train, y_train)

print("Model Decision Tree berhasil dibangun!")
print(f"Kedalaman pohon (tree depth): {model.get_depth()}")
print(f"Jumlah leaf nodes (daun): {model.get_n_leaves()}")
print()

# ============================================================================
# STEP 4: MEMBUAT PREDIKSI (Prediction)
# ============================================================================

# Prediksi kelas untuk data testing menggunakan model yang sudah dilatih
# Model akan mengikuti jalur pohon dari root sampai leaf untuk setiap sampel
y_pred_dt = model.predict(X_test)

print(f"Prediksi pertama 5 sampel: {y_pred_dt[:5]}")
print(f"Nilai sebenarnya pertama 5 sampel: {y_test.values[:5].flatten()}")
print()

# ============================================================================
# STEP 5: EVALUASI MODEL (Model Evaluation)
# ============================================================================

# Hitung akurasi: persentase prediksi yang benar
# Akurasi = (jumlah prediksi benar) / (total prediksi)
accuracy = accuracy_score(y_test, y_pred_dt)

print(f'Akurasi Model: {accuracy:.4f} ({accuracy*100:.2f}%)')
print()

# ============================================================================
# BONUS: ANALISIS FITUR IMPORTANCE (Feature Importance)
# ============================================================================

# Lihat seberapa penting setiap fitur dalam membuat keputusan
# Nilai lebih tinggi = fitur lebih penting untuk prediksi
feature_importance = model.feature_importances_
feature_names = iris['feature_names']

print("Pentingnya Fitur (Feature Importance):")
for name, importance in zip(feature_names, feature_importance):
    print(f"{name:20s}: {importance:.4f}")