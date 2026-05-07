import pandas as pd
from src.limpieza_datos import limpiar_y_codificar, normalizar_datos
from src.entrenamiento_modelo import entrenar_clasificacion, entrenar_regresion

def ejecutar_pipeline():
    # 1. Cargar datos
    print("Cargando datos...")
    df = pd.read_csv("data/employer.csv")

    # 2. Preprocesamiento
    print("Limpiando y codificando...")
    df_limpio = limpiar_y_codificar(df)
    
    print("Normalizando...")
    df_final = normalizar_datos(df_limpio)

    # 3. Entrenamiento y Evaluación
    print("Iniciando fase de modelos...\n")
    modelo_clf, columnas = entrenar_clasificacion(df_final)
    modelo_reg = entrenar_regresion(df_final)

    # 4. Importancia de variables (Justificación)
    print("\nTop 5 variables más influyentes (Clasificación):")
    importancia = pd.Series(modelo_clf.feature_importances_, index=columnas).sort_values(ascending=False).head(5)
    print(importancia)

if __name__ == "__main__":
    ejecutar_pipeline()