from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import MinMaxScaler, VectorAssembler, PCA
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import math
from pyspark.sql.types import IntegerType, DoubleType
import pandas as pd
from pyspark.ml.functions import vector_to_array

# Inicializar SparkSession
import os
import sys


os.environ["PYSPARK_PYTHON"] = r"C:\Users\josia\AppData\Local\Programs\Python\Python311\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\josia\AppData\Local\Programs\Python\Python311\python.exe"

label_col = "PAY_AMT4"

spark = SparkSession.builder \
    .appName("MiApp") \
    .config("spark.network.timeout", "800s") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .config("spark.hadoop.io.native.lib.available", "false") \
    .getOrCreate()

# Cargar datos desde Excel a Pandas (PySpark no lee directamente Excel)
import pandas as pd
df_pd = pd.read_excel(r"D:\Documentos\Blue_tab_prueba\proyecto_analisis_datos\data\raw\default_dataset.xls", header=1)

# Convertir a Spark DataFrame
df = spark.createDataFrame(df_pd)

# Eliminar columnas que pueden causar data leakage
cols_to_drop = ['PAY_3', 'PAY_2', 'PAY_0', 'BILL_AMT3', 'BILL_AMT2', 'BILL_AMT1',
                'PAY_AMT3','PAY_AMT2','PAY_AMT1','default payment next month']
df = df.drop(*cols_to_drop)

# Eliminar duplicados
df = df.dropDuplicates()

# PCA para PAY_4, PAY_5, PAY_6
assembler_pay = VectorAssembler(inputCols=['PAY_4', 'PAY_5', 'PAY_6'], outputCol="PAY_features")
pca_pay = PCA(k=1, inputCol="PAY_features", outputCol="PAY_PCA")
# PCA para BILL_AMT4, BILL_AMT5, BILL_AMT6
assembler_bill = VectorAssembler(inputCols=['BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6'], outputCol="BILL_features")
pca_bill = PCA(k=1, inputCol="BILL_features", outputCol="BILL_AMT_PCA")

# Crear Pipeline
pipeline = Pipeline(stages=[assembler_pay, pca_pay, assembler_bill, pca_bill])
model = pipeline.fit(df)
df = model.transform(df)

# Eliminar columnas originales de PCA
df = df.drop('PAY_4', 'PAY_5', 'PAY_6', 'PAY_features')
df = df.drop('BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'BILL_features')

# Filtrar las columnas que tienen tipo numérico

feature_cols = [col for col in df.columns if col != label_col]
print("feature_cols: ", feature_cols)





df.select("PAY_AMT4").summary().show()

#Ensamblar y escalar solo las features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
scaler = MinMaxScaler(inputCol="features_raw", outputCol="features")

pipeline = Pipeline(stages=[assembler, scaler])
pipeline_model = pipeline.fit(df)
df_scaled = pipeline_model.transform(df)

# Añadir de nuevo la columna PAY_AMT4
df_scaled = df_scaled.withColumn("scaled_array", vector_to_array("features"))

# Crear columnas individuales para las features escaladas
scaled_col_names = [f"feature_{i}" for i in range(len(feature_cols))]
for i, name in enumerate(scaled_col_names):
    df_scaled = df_scaled.withColumn(name, col("scaled_array")[i])


# Mantener únicamente las columnas necesarias + el label
df_scaled = df_scaled.select(*scaled_col_names, label_col)
print(df_scaled.columns)
df_scaled.select("PAY_AMT4").summary().show()
# Separar train y test
train_df, test_df = df_scaled.randomSplit([0.8, 0.2], seed=42)
# Guardar como Parquet
#train_df.write.mode("overwrite").parquet("D:\Documentos\Blue_tab_prueba\proyecto_analisis_datos\data\processed\PAY_AMT4_regression_train.parquet")
#test_df.write.mode("overwrite").parquet("proyecto_analisis_datosdata/data/processed/PAY_AMT4_regression_test.parquet")
train_df.toPandas().to_parquet(
    "proyecto_analisis_datos/data/processed/PAY_AMT4_regression_train.parquet"
)
train_df.toPandas().to_parquet(
    "proyecto_analisis_datos/data/processed/PAY_AMT4_regression_test.parquet"
)
#output_path = "D:\Documentos\Blue_tab_prueba\proyecto_analisis_datos\data\processed\PAY_AMT4_regression_train.parquet"
#train_df.write.mode("overwrite").parquet(output_path)
# Finalizar sesión de Spark
spark.stop()
