from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import os
import requests
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# Documentos estáticos de exemplo
documents = [
    {"id": 1, "text": "Google CEO, Lex Luthor, announced a new headquarters in Metropolis."},
    {"id": 2, "text": "Apple launches invisible iPhone under the leadership of Bruce Wayne."},
    {"id": 3, "text": "Amazon opens an underground distribution center in Gotham City."},
    {"id": 4, "text": "Meta hires Clark Kent as the new head of innovation."},
    {"id": 5, "text": "SGE consists of 9 employees, with Carla as its head."}
]

# Modelo de embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = {
    doc["id"]: model.encode(doc["text"], convert_to_tensor=True)
    for doc in documents
}

class QueryRequest(BaseModel):
    query: str

def gerar_resposta_ollama(prompt: str, modelo: str = "llama3") -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": modelo,
            "prompt": prompt,
            "stream": False
        }
    )
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Ollama erro: {response.text}")
    return response.json().get("response", "Sem resposta do modelo local.")

@app.post("/query")
def query_rag(request: QueryRequest):
    # 1) Calcula embedding da query e encontra o documento mais relevante
    query_embedding = model.encode(request.query, convert_to_tensor=True)
    best_doc = None
    best_score = float("-inf")
    for doc in documents:
        score = util.cos_sim(query_embedding, doc_embeddings[doc["id"]])
        if score > best_score:
            best_score = score
            best_doc = doc

    prompt = (
        f"You are an AI assistant. Answer based ONLY on this document:\n"
        f"{best_doc['text']}\n\nUser: {request.query}\nAssistant:"
    )

    # 2) Chama apenas o Ollama
    resposta = gerar_resposta_ollama(prompt, modelo="llama3")
    return {"response": resposta}

    # --- Código Hugging Face comentado abaixo ---
    # api_key = os.getenv("MY_HF_KEY")
    # if not api_key:
    #     raise HTTPException(status_code=500, detail="API key is missing.")
    #
    # url = "https://api-inference.huggingface.co/models/google/gemma-2-2b-it"
    # headers = {
    #     "Authorization": f"Bearer {api_key}",
    #     "Content-Type": "application/json",
    #     "x-use-cache": "true"
    # }
    # data = {"inputs": prompt}
    # response = requests.post(url, headers=headers, json=data)
    # if response.status_code != 200:
    #     raise HTTPException(status_code=response.status_code, detail="Hugging Face error: " + response.text)
    #
    # res = response.json()
    # if isinstance(res, list):
    #     return {"response": res[0].get("generated_text", "No response found")}
    # else:
    #     return {"response": res.get("generated_text", "No response found")}
