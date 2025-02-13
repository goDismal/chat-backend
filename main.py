from flask import Flask, request, jsonify
import pandas as pd
import requests
import faiss
import numpy as np
import os
import io
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 📌 Cargar la clave de OpenAI
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 📌 Inicializar el modelo de GPT-4o-mini
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)
embeddings_model = OpenAIEmbeddings()

# 📌 Inicializar Flask
app = Flask(__name__)

# 📌 URL del CSV con embeddings
CSV_URL = "https://raw.githubusercontent.com/goDismal/chat-backend/refs/heads/main/EmbeddingsEntrevistas.csv"

def load_embeddings():
    response = requests.get(CSV_URL)
    if response.status_code != 200:
        raise Exception("No se pudo descargar el archivo CSV.")
    
    # Carga el CSV en un DataFrame de forma optimizada
    df = pd.read_csv(io.StringIO(response.text), dtype={"Embeddings": str})
    
    # Convierte los embeddings en arrays sin cargar toda la columna en memoria
    df["Embeddings"] = df["Embeddings"].apply(lambda x: np.array(eval(x), dtype=np.float32))
    
    return df

df = load_embeddings()

# 📌 Preparar FAISS
embedding_dim = len(df["Embeddings"].iloc[0])
index = faiss.IndexFlatL2(embedding_dim)
embeddings = np.vstack(df["Embeddings"].values).astype(np.float32)
index.add(embeddings)

# 📌 Ruta para recibir preguntas
@app.route("/chat", methods=["POST"])
def chat_with_gpt():
    user_message = request.json.get("message", "")
    
    # 📌 Convertir la pregunta en un embedding y buscar en FAISS
    user_embedding = np.array(embeddings_model.embed_query(user_message), dtype=np.float32).reshape(1, -1)
    _, idx = index.search(user_embedding, k=3)
    retrieved_texts = df.iloc[idx[0]]["Text"].values  # Extraer textos relevantes
    
    # 📌 Generar respuesta con GPT
    context = "\n".join(retrieved_texts)
    prompt = f"""
    Eres un asistente basado en entrevistas con candidatos presidenciales de Ecuador.
    Proporciona respuestas claras y detalladas en el idioma de la pregunta.
    
    Contexto relevante:
    {context}
    
    Pregunta: {user_message}
    
    Respuesta:"""
    
    response = llm.invoke(prompt)
    return jsonify({"response": response})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Fly.io asigna un puerto automáticamente
    app.run(host="0.0.0.0", port=port)
