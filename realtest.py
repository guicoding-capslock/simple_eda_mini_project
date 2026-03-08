# Mencegah Pelanggan Kabur (Customer Churn)
# Sebuah perusahaan E-Commerce sedang bermasalah karena banyak pelanggan yang berhenti
# berbelanja dan pindah ke aplikasi pesaing (fenomena ini disebut Churn). Mereka memintamu mencari tahu
# penyebab utama pelanggan kabur menggunakan Data Analysis, lalu membangun AI untuk memprediksinya.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data_ecommerce = {
    "ID_User": [101, 102, 103, 104, 105, 106, 107, 108],
    "Total_Belanja_Juta": [5.2, 0.5, 4.8, 1.2, 6.0, 0.8, 5.5, 1.0], # Total belanja bulan lalu
    "Jumlah_Komplain": [0, 4, 1, 3, 0, 5, 0, 3], # Komplain ke Customer Service
    "Rating_Aplikasi": [5, 2, 4, 3, 5, 1, 4, 2], # Bintang 1-5 di PlayStore
    "Promo_Digunakan": [10, 2, 8, 1, 12, 0, 9, 2], # Berapa kali pakai voucher gratis ongkir/diskon
    "Churn": [0, 1, 0, 1, 0, 1, 0, 1] # Target AI: 1 = Kabur, 0 = Setia
}

df = pd.DataFrame(data_ecommerce)

df_cor = df.corr()

# Disini akan menampilkan korelasi kolom negatif dan positif paling tinggi dengan kolom Churn
# Ketika di print, kolom paling banyak negatif adalah "Total_Belanja_Juta" dan yang paling banyak positif adalah "Jumlah_Komplain"
# Ini berarti penyebab banyak pelanggan kabur adalah karena banyak komplain dan juga sedikit pembeli yang membeli dalam jumlah total belanja juta

y = df["Churn"]
X = df.drop(columns=["Churn", "ID_User"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

dftree = DecisionTreeClassifier()
dftree = dftree.fit(X_train, y_train)

y_pred = dftree.predict(X_test)

print("Prediksi Churn:", y_pred)
print("Index Kabur:", y_test.index.tolist())
