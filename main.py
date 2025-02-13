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

# 📌 Cargar la clave de OpenAI
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 📌 Inicializar el modelo de GPT-4o-mini
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)
embeddings_model = OpenAIEmbeddings()

# 📌 Inicializar FastAPI
app = FastAPI()

# 📌 URL del CSV con embeddings
CSV_URL = "https://raw.githubusercontent.com/goDismal/chat-backend/refs/heads/main/EmbeddingsEntrevistas.csv"

def load_embeddings():
    response = requests.get(CSV_URL)
    if response.status_code != 200:
        raise Exception("No se pudo descargar el archivo CSV.")
    
    df = pd.read_csv(pd.compat.StringIO(response.text))
    df["Embeddings"] = df["Embeddings"].apply(lambda x: np.array(eval(x)))
    return df

df = load_embeddings()

# 📌 Preparar FAISS
embedding_dim = len(df["Embeddings"].iloc[0])
index = faiss.IndexFlatL2(embedding_dim)
embeddings = np.vstack(df["Embeddings"].values).astype(np.float32)
index.add(embeddings)

# 📌 Esquema para recibir preguntas
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_with_gpt(request: ChatRequest):
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
    
    response = llm.invoke(prompt)
    return {"response": response}
