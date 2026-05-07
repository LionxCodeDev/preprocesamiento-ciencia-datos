import pandas as pd
import numpy as np

def limpiar_y_codificar(df):
    df_copy = df.copy()
    
    # 1. Imputación de valores nulos
    columnas_numeros = df_copy.select_dtypes(include=[np.number]).columns
    columnas_letras = df_copy.select_dtypes(include=['object', 'category']).columns

    df_copy[columnas_numeros] = df_copy[columnas_numeros].fillna(df_copy[columnas_numeros].mean())

    for col in columnas_letras:
        moda_valor = df_copy[col].mode()
        df_copy[col] = df_copy[col].fillna(moda_valor[0])
        
    # 2. Manejo de la columna Age (conversión y limpieza)
    df_copy['Age'] = pd.to_numeric(df_copy['Age'], errors='coerce')
    if df_copy['Age'].isnull().any():
        df_copy['Age'] = df_copy['Age'].fillna(df_copy['Age'].median())
    df_copy['Age'] = df_copy['Age'].astype(int)

    # 3. Codificación de variables categóricas
    df_copy['Attrition'] = np.where(df_copy['Attrition'] == 'No', 0, 1)
    df_copy['Gender'] = df_copy['Gender'].map({'Female': 0, 'Male': 1})
    df_copy['OverTime'] = np.where(df_copy['OverTime'] == 'No', 0, 1)
    df_copy['BusinessTravel'] = df_copy['BusinessTravel'].map({'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2})
    df_copy['Over18'] = 1

    # One Hot Encoding
    columnas_ohe = ['Department', 'EducationField', 'JobRole', 'MaritalStatus']
    df_copy = pd.get_dummies(df_copy, columns=columnas_ohe, prefix=[col[:3] for col in columnas_ohe])

    # Convertir booleanos a enteros (para los dummies)
    for col in df_copy.columns:
        if df_copy[col].dtype == 'bool':
            df_copy[col] = df_copy[col].astype(int)
            
    return df_copy

def normalizar_datos(df):
    df_norm = df.copy()
    for col in df_norm.columns:
        minimo = df_norm[col].min()
        maximo = df_norm[col].max()
        rango = maximo - minimo
        if rango != 0:
            df_norm[col] = (df_norm[col] - minimo) / rango
        else:
            df_norm[col] = 1.0
    return df_norm