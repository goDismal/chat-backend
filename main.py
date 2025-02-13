from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import requests
import faiss
import numpy as np
import os
import io
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import uvicorn

# 📌 Cargar la clave de OpenAI
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 📌 Verificar que la clave de API está presente
if not OPENAI_API_KEY:
    raise ValueError("🚨 ERROR: La clave OPENAI_API_KEY no está configurada!")

# 📌 Inicializar FastAPI
app = FastAPI()

# 📌 Modelo de datos para la solicitud de chat
class ChatRequest(BaseModel):
    message: str

# 📌 URL del CSV con embeddings
CSV_URL = "https://raw.githubusercontent.com/goDismal/chat-backend/refs/heads/main/EmbeddingsEntrevistas.csv"

# 📌 Cargar embeddings de manera diferida
df = None
index = None
embeddings_model = None

def load_embeddings():
    global df, index, embeddings_model
    if df is not None:
        return  # Ya está cargado

    print("📥 Descargando embeddings...")
    response = requests.get(CSV_URL)
    if response.status_code != 200:
        raise Exception("No se pudo descargar el archivo CSV.")

    df = pd.read_csv(io.StringIO(response.text), dtype={"Embeddings": str})
    df["Embeddings"] = df["Embeddings"].apply(lambda x: np.array(eval(x), dtype=np.float32))

    # 📌 Preparar FAISS
    embedding_dim = len(df["Embeddings"].iloc[0])
    index = faiss.IndexFlatL2(embedding_dim)
    embeddings = np.vstack(df["Embeddings"].values).astype(np.float32)
    index.add(embeddings)

    # 📌 Inicializar el modelo de embeddings
    embeddings_model = OpenAIEmbeddings()
    print("✅ Embeddings cargados correctamente.")

# 📌 Ruta de prueba para verificar que la API está activa
@app.get("/")
async def health_check():
    return {"status": "ok"}

# 📌 Ruta para recibir preguntas
@app.post("/chat")
async def chat_with_gpt(request: ChatRequest):
    global df, index, embeddings_model
    if df is None:
        load_embeddings()

    user_message = request.message

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

    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)
    response = llm.invoke(prompt)

    return {"response": response}

# 📌 Ejecutar la aplicación con Uvicorn en Fly.io
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Fly.io asigna el puerto automáticamente
    uvicorn.run(app, host="0.0.0.0", port=port)
