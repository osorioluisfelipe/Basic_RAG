from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import os
from dotenv import load_dotenv
import requests
from huggingface_hub import InferenceClient


load_dotenv()

app = FastAPI()

documents =[
    {
        "id": 1,
        "text": "Google CEO, Lex Luthor, announced a new headquarters in Metropolis."
    },
    {
        "id": 2,
        "text": "Apple launches invisible iPhone under the leadership of Bruce Wayne."
    },
    {
        "id": 3,
        "text": "Amazon opens an underground distribution center in Gotham City."
    },
    {
        "id": 4,
        "text": "Meta hires Clark Kent as the new head of innovation."
    },
    {
        "id": 5,
        "text": "SGE consists of 9 employees, with Carla as its head."
    }
]


model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embedings = {
    doc["id"]: model.encode(doc["text"], convert_to_tensor=True) for doc in documents
    }

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query_rag(request: QueryRequest):
    query_embeding = model.encode(request.query, convert_to_tensor=True)
    best_doc = {}
    best_score = float("-inf")

    for doc in documents:
        score = util.cos_sim(query_embeding, doc_embedings[doc["id"]])
        if score > best_score:
            best_score = score
            best_doc = doc
    prompt = f"You are an AI assistant. Answer based ONLY on this document: {best_doc['text']}\n\nUser: {request.query}\nAssistant:"

    try:
        api_key=os.getenv("MY_HF_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="API key is missing.")
        url = "https://api-inference.huggingface.co/models/google/gemma-2-2b-it"


        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "x-use-cache": "true"
        }

        data = {
            "inputs": prompt  # Enviar apenas o texto de entrada para o modelo
        }

        response = requests.post(url, headers=headers, json=data)

        # Verifique o status e o conteúdo da resposta
        if response.status_code != 200:
            # Imprimir resposta para depuração
            print(f"Error {response.status_code}: {response.text}")
            raise HTTPException(status_code=response.status_code, detail="Failed to get a response.")

        res = response.json()

        # Verifique a estrutura da resposta para diagnosticar corretamente
        # print("Resposta da API:", res)

        if isinstance(res, list):
            # Se a resposta for uma lista, extraímos o texto da primeira entrada
            return res[0].get("generated_text", "No response found")
        else:
            # Caso a resposta seja um dicionário, tentamos acessar diretamente
            return {"response": res.get("generated_text", "No response found")}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
