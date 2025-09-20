import streamlit as st
import joblib
import pickle
import numpy as np
import psycopg2

#ostgresql://postgres.vbeuhmiiygpljvqwqiyo:[YOUR-PASSWORD]@aws-1-us-east-2.pooler.supabase.com:5432/postgres

USER = "postgres.vbeuhmiiygpljvqwqiyo" #os.getenv("user")
PASSWORD = "alfre943553109"# os.getenv("password")
HOST = "aws-1-us-east-2.pooler.supabase.com" #os.getenv("host")
PORT = "5432" #os.getenv("port")
DBNAME = "postgres" #os.getenv("dbname")


# Configuración de la página
st.set_page_config(page_title="Predictor de Iris", page_icon="🌸")
# Connect to the database
#result = None  # <- inicializamos

try:
    connection = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
        
    )
    print("Connection successful!")
    
    # Create a cursor to execute SQL queries
    cursor = connection.cursor()
    
    # Example query
    cursor.execute("SELECT NOW();")
    result = cursor.fetchone()
    print("Current Time:", result)
    # Close the cursor and connection
    cursor.close()
    connection.close()
    print("Connection closed.")

except Exception as e:
    st.write(str(e))



# Función para cargar los modelos
@st.cache_resource
def load_models():
    try:
        model = joblib.load('components/iris_model.pkl')
        scaler = joblib.load('components/iris_scaler.pkl')
        with open('components/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model, scaler, model_info
    except FileNotFoundError:
        st.error("No se encontraron los archivos del modelo en la carpeta 'models/'")
        return None, None, None

# Título
st.title("🌸 Predictor de Especies de Iris")

# Cargar modelos
model, scaler, model_info = load_models()

if model is not None:
    # Inputs
    st.header("Ingresa las características de la flor:")
    st.write(result)

    longitud_sepalo = st.number_input("Longitud del Sépalo (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    ancho_sepalo = st.number_input("Ancho del Sépalo (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    longitud_petalo = st.number_input("Longitud del Pétalo (cm)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
    ancho_petalo = st.number_input("Ancho del Pétalo (cm)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    
    # Botón de predicción
    if st.button("Predecir Especie"):
        # Preparar datos
        features = np.array([[longitud_sepalo, ancho_sepalo, longitud_petalo, ancho_petalo]])
        
        # Estandarizar
        features_scaled = scaler.transform(features)
        
        # Predecir
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Mostrar resultado
        target_names = model_info['target_names']
        predicted_species = target_names[prediction]
        
        st.success(f"Especie predicha: **{predicted_species}**")
        st.write(f"Confianza: **{max(probabilities):.1%}**")

          # Guardar en la base de datos
        try:
            connection = psycopg2.connect(
                user=USER,
                password=PASSWORD,
                host=HOST,
                port=PORT,
                dbname=DBNAME
            )
            cursor = connection.cursor()
            cursor.execute(
                """
                INSERT INTO table_iris (longitud_petalo, longitud_sepalo, ancho_petalo, ancho_sepalo, prediction)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (longitud_petalo, longitud_sepalo, ancho_petalo, ancho_sepalo, prediction)
            )
            connection.commit()
            cursor.close()
            connection.close()
            st.success(" Datos guardados en Supabase correctamente")
        except Exception as e:
            st.error(f"Error al guardar en la base de datos: {e}")
        
        # Mostrar todas las probabilidades
        st.write("Probabilidades:")
        for species, prob in zip(target_names, probabilities):
            st.write(f"- {species}: {prob:.1%}")




