from fastapi import FastAPI
import pandas as pd
import requests
import faiss
import numpy as np
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langchain_community.embeddings import OpenAIEmbeddings


# 📌 Cargar la clave de OpenAI desde Railway
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 📌 Inicializar el modelo de GPT-4o-mini
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)

app = FastAPI()

# 📌 URL del archivo CSV con embeddings (¡Reemplaza con la URL correcta!)
CSV_URL = "https://raw.githubusercontent.com/goDismal/chat-backend/refs/heads/main/EmbeddingsEntrevistas.csv"

# 📌 Descargar y cargar el CSV
def load_embeddings():
    response = requests.get(CSV_URL)
    if response.status_code != 200:
        raise Exception("No se pudo descargar el archivo CSV.")
    
    with open("EmbeddingsEntrevistas.csv", "wb") as f:
        f.write(response.content)

    df = pd.read_csv("EmbeddingsEntrevistas.csv")
    return df

df = load_embeddings()

# 📌 Preparar FAISS para búsqueda rápida
embedding_dim = df.shape[2] - 1  # Asumimos que la primera columna es texto
index = faiss.IndexFlatL2(embedding_dim)
index.add(np.array(df.iloc[:, 1:].values, dtype=np.float32))  # Añadir embeddings a FAISS

# 📌 Esquema para recibir preguntas
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_with_gpt(request: ChatRequest):
    user_message = request.message

    # 📌 Convertir la pregunta en un embedding
    embedding_model = OpenAIEmbeddings()
    user_embedding = np.array(embedding_model.embed_query(user_message), dtype=np.float32).reshape(1, -1)

    # 📌 Buscar los textos más similares en FAISS
    _, idx = index.search(user_embedding, k=3)
    retrieved_texts = df.iloc[idx[0], 0].values  # Obtener textos relevantes

    # 📌 Construir el contexto y generar respuesta con GPT
    context = "\n".join(retrieved_texts)
    prompt = f"Contexto relevante:\n{context}\n\nPregunta: {user_message}\n\nRespuesta:"
    response = llm.invoke(prompt)

    return {"response": response}
