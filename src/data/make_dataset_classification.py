from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import os

os.environ["PYSPARK_PYTHON"] = r"C:\Users\josia\AppData\Local\Programs\Python\Python311\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\josia\AppData\Local\Programs\Python\Python311\python.exe"

# Iniciar Spark
spark = SparkSession.builder \
    .appName("MiApp") \
    .config("spark.network.timeout", "800s") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .config("spark.hadoop.io.native.lib.available", "false") \
    .getOrCreate()

# Leer archivo Excel (solo se puede hacer con pandas)
df_pd = pd.read_excel(r"D:\Documentos\Blue_tab_prueba\proyecto_analisis_datos\data\raw\default_dataset.xls", header=1)

# Eliminar columnas de septiembre (data leakage)
df_pd.drop(columns=['PAY_AMT1', 'BILL_AMT1', 'PAY_0'], inplace=True)

# PCA sobre PAY variables
pca_pay = PCA(n_components=1)
df_pd['PAY_PCA'] = pca_pay.fit_transform(df_pd[['PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']])
df_pd.drop(columns=['PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'], inplace=True)

# PCA sobre BILL_AMT variables
pca_bill = PCA(n_components=1)
df_pd['BILL_AMT_PCA'] = pca_bill.fit_transform(df_pd[['BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']])
df_pd.drop(columns=['BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6'], inplace=True)

# Heatmap (opcional)
corr_matrix = df_pd.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Matriz de Correlación")
plt.show()

# Normalización
scaler = MinMaxScaler()
df_pd[df_pd.columns] = scaler.fit_transform(df_pd[df_pd.columns])

# División entrenamiento/prueba antes del balanceo
train_df, test_df = train_test_split(
    df_pd,
    test_size=0.2,
    stratify=df_pd['default payment next month'],
    random_state=42
)

# Balancear solo el conjunto de entrenamiento
def balance_train_data(train_df, target_col='default payment next month', random_state=42):
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    print("Distribución original en entrenamiento:", Counter(y))

    # Contar clases
    class_counts = Counter(y)
    maj_class = max(class_counts, key=class_counts.get)
    min_class = min(class_counts, key=class_counts.get)

    # Punto medio entre ambas clases
    midpoint = (class_counts[maj_class] + class_counts[min_class]) // 2

    print(f"Aplicando undersampling a {maj_class} hasta {midpoint} y oversampling a {min_class} hasta {midpoint}")

    # Undersampling de la clase mayoritaria
    under = RandomUnderSampler(sampling_strategy={maj_class: midpoint}, random_state=random_state)
    X_under, y_under = under.fit_resample(X, y)

    # Oversampling de la clase minoritaria hasta el mismo punto
    smote = SMOTE(sampling_strategy={min_class: midpoint}, random_state=random_state)
    X_balanced, y_balanced = smote.fit_resample(X_under, y_under)

    print("Distribución balanceada:", Counter(y_balanced))

    df_balanced = pd.DataFrame(X_balanced, columns=X.columns)
    df_balanced[target_col] = y_balanced
    return df_balanced

# Aplicar balanceo
train_df_balanced = balance_train_data(train_df)

# Guardar como Parquet
#train_df_balanced.to_parquet("proyecto_analisis_datos/data/processed/default_classification_train.parquet", index=False)
#test_df.to_parquet("proyecto_analisis_datos/data/processed/default_classification_test.parquet", index=False)
