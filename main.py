import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

data_karyawan = {
    "ID": [1, 2, 3, 4, 5, 6, 7, 8],
    "Lama_Bekerja": [2, 5, 1, 8, 3, 6, 1, 4], # Dalam tahun
    "Jam_Lembur": [40, 10, 50, 5, 35, 15, 45, 10], # Jam per bulan
    "Skor_Performa": [75, 88, 60, 95, 70, 85, 65, 90], # Skala 1-100
    "Gaji_Juta": [6, 12, 5, 20, 7, 15, 5, 10], # Dalam juta Rupiah
    "Resign": [1, 0, 1, 0, 1, 0, 1, 0] # Target AI: 1 = Resign, 0 = Bertahan
}

### 
# **Analogi:** Bayangkan kamu punya catatan 8 karyawan di kertas. `pd.DataFrame` mengubah catatan itu menjadi **tabel rapi** yang bisa diproses komputer.
# ID | Lama_Bekerja | Jam_Lembur | Skor_Performa | Gaji_Juta | Resign
# 1  |      2       |     40     |      75       |     6     |   1
# 2  |      5       |     10     |      88       |    12     |   0
# ###

df = pd.DataFrame(data_karyawan)

y = df["Resign"]
X = df.drop(columns=["Resign", "ID"])

# X = soal (pertanyaan/petunjuk): lama kerja, jam lembur, dll
# y = kunci jawaban: resign atau tidak

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# **Analogi:** Kamu punya **8 soal latihan ujian.**
# - **`test_size=0.25`** → 25% data = **2 data untuk ujian** (test), 75% = **6 data untuk belajar** (train)
# - **`random_state=42`** → Pengacakan yang **konsisten/reproducible** — supaya setiap kali dijalankan hasilnya sama
# ```
# Total 8 data
# ├── X_train / y_train → 6 data (bahan belajar model)
# └── X_test  / y_test  → 2 data (soal ujian model)

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_test)

# **Analogi:** Decision Tree seperti **diagram alur/flowchart** yang dibuat otomatis oleh komputer.

# Model belajar membuat pertanyaan seperti:
# ```
# Jam_Lembur > 30?
# ├── YA  → Gaji_Juta < 8?
# │         ├── YA  → RESIGN ✅
# │         └── TIDAK → Bertahan ❌
# └── TIDAK → Bertahan ❌

print("Prediksi Resign:", y_pred)
print("Index y_test:", y_test.index.tolist())
print("Nilai y_test:", y_test.values)
print("Nilai y_pred:", y_pred)

#Output:
#array = ["ID1", "ID2", "ID3", "ID4", "ID5", "ID6", "ID7", "ID8"]
#index:    0      1      2      3      4      5      6      7
#Jadi karyawan yang bertahan adalah dengan ID 2 dan 6

plt.figure(figsize=(10, 6))

plot_tree(
    dtree,
    feature_names=X.columns,
    class_names=["Bertahan", "Resign"],
    filled=True,
    rounded=True,
)

plt.show()