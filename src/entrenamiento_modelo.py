import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score

def entrenar_clasificacion(df):
    X = df.drop(columns=['Attrition']) 
    y = df['Attrition']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    importancia = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
    plt.figure(figsize=(10,6))
    importancia.plot(kind='barh', color='skyblue')
    plt.title('Top 10 Variables más influyentes (Attrition)')
    plt.xlabel('Importancia')
    plt.tight_layout()
    
    # Guardamos la gráfica en la carpeta results
    plt.savefig('results/importancia_variables.png')
    print("Gráfica guardada en results/importancia_variables.png")
    
    y_pred = model.predict(X_test)
    
    print("--- REPORTE DE CLASIFICACIÓN (ATTRITION) ---")
    print(f"Precisión Global (Accuracy): {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))
    
    return model, X.columns

def entrenar_regresion(df):
    # El objetivo es el salario (MonthlyIncome)
    y = df['MonthlyIncome']
    X = df.drop(columns=['MonthlyIncome'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("\n--- REPORTE DE REGRESIÓN (MONTHLY INCOME) ---")
    print(f"Error Cuadrático Medio (MSE): {mean_squared_error(y_test, y_pred):.4f}")
    print(f"Coeficiente de determinación (R2): {r2_score(y_test, y_pred):.2f}")
    
    return model