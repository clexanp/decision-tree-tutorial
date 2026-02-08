# Import warnings untuk mengelola pesan warning
import warnings
# Abaikan semua warning agar output lebih bersih
warnings.filterwarnings('ignore')

# Import NumPy untuk operasi array dan perhitungan numerik
import numpy as np
# Import Pandas untuk membuat dan memanipulasi DataFrame
import pandas as pd

# Import datasets dari sklearn untuk mengakses dataset bawaan seperti iris
from sklearn import datasets
# Import accuracy_score untuk menghitung akurasi prediksi model
from sklearn.metrics import accuracy_score
# Import DecisionTreeClassifier - algoritma utama untuk klasifikasi
from sklearn.tree import DecisionTreeClassifier

# ===== TAHAP 1: MEMUAT DAN MEMPERSIAPKAN DATA =====
# Memuat dataset Iris dari sklearn (150 sampel bunga dengan 4 fitur)
iris = datasets.load_iris()

# Menggabungkan fitur (iris['data']) dan label target (iris['target']) menjadi satu array
# np.c_ menggabungkan array secara horizontal (column-wise)
# Membuat DataFrame dengan kolom fitur dan target untuk memudahkan manipulasi
data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     # Menambahkan nama untuk setiap kolom (4 fitur + target)
                     columns= iris['feature_names'] + ['target'])

# Memisahkan fitur (X) dari DataFrame dengan menghapus kolom 'target'
# X berisi 4 fitur: sepal length, sepal width, petal length, petal width
X = data.drop('target', axis=1)
# Memisahkan target (y) - berisi kelas iris (0=Setosa, 1=Versicolor, 2=Virginica)
y = data[['target']]

# ===== TAHAP 2: MEMBAGI DATA MENJADI TRAINING DAN TESTING =====
# Import fungsi train_test_split untuk memisahkan data secara otomatis
from sklearn.model_selection import train_test_split

# Membagi data menjadi 2 bagian: 70% untuk training, 30% untuk testing
# random_state=42 memastikan pembagian yang sama setiap kali dijalankan (reproducibility)
X_train, X_test,y_train,y_test = train_test_split(X,
                                                # Data fitur yang akan dibagi
                                                y,
                                                # Data target yang akan dibagi
                                                test_size = 0.3,
                                                # 30% data untuk testing
                                                random_state = 42)
                                                # Seed agar hasil selalu sama

# ===== TAHAP 3: MEMBUAT DAN MELATIH MODEL DECISION TREE =====
# Membuat instance/objek DecisionTreeClassifier dengan parameter default
# Algoritma akan secara otomatis memilih fitur terbaik dan threshold untuk setiap split
model = DecisionTreeClassifier()

# Melatih (fit) model menggunakan data training
# Di sini algoritma membangun struktur pohon keputusan lengkap
model.fit(X_train, y_train)

# ===== TAHAP 4: MELAKUKAN PREDIKSI =====
# Menggunakan model yang sudah dilatih untuk memprediksi kelas sampel test
# Model akan mengikuti jalur pohon dari root sampai leaf untuk setiap sampel
y_pred_dt = model.predict(X_test)

# ===== TAHAP 5: EVALUASI MODEL =====
# Menghitung akurasi dengan membandingkan prediksi dengan nilai sebenarnya
# Akurasi = (jumlah prediksi benar) / (total prediksi) x 100%
print('Akurasi',accuracy_score(y_test, y_pred_dt))