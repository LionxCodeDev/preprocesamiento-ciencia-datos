import pandas as pd
import numpy as np

def preprocesamiento_completo(path_csv):
    # Carga de datos
    df = pd.read_csv(path_csv)
    
    # Limpieza: Imputar nulos con la media en columnas numéricas
    df = df.fillna(df.mean(numeric_only=True))
    
    # Codificación de la variable objetivo
    if 'Attrition' in df.columns:
        df['Attrition'] = np.where(df['Attrition'] == 'No', 0, 1)
    
    print("Proceso de preprocesamiento terminado.")
    return df

if __name__ == "__main__":
    # Prueba con tu archivo
    resultado = preprocesamiento_completo("data/employer.csv")
    print(resultado.head())