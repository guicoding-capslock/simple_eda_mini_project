import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data_karyawan = {
    "ID": [1, 2, 3, 4, 5, 6, 7, 8],
    "Lama_Bekerja": [2, 5, 1, 8, 3, 6, 1, 4], # Dalam tahun
    "Jam_Lembur": [40, 10, 50, 5, 35, 15, 45, 10], # Jam per bulan
    "Skor_Performa": [75, 88, 60, 95, 70, 85, 65, 90], # Skala 1-100
    "Gaji_Juta": [6, 12, 5, 20, 7, 15, 5, 10], # Dalam juta Rupiah
    "Resign": [1, 0, 1, 0, 1, 0, 1, 0] # Target AI: 1 = Resign, 0 = Bertahan
}

df = pd.DataFrame(data_karyawan)

y = df["Resign"]
X = df.drop(columns=["Resign", "ID"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_test)

print("Prediksi Resign:", y_pred)
print(y_test)