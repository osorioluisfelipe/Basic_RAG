from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

documents = [
    {
        "id": 1,
        "text": "O CEO do Google, Lex Luthor, anunciou nova sede em Metrópolis."
    },
    {
        "id": 2,
        "text": "Apple lança iPhone invisível sob liderança de Bruce Wayne."
    },
    {
        "id": 3,
        "text": "Amazon inaugura centro de distribuição subterrâneo em Gotham City."
    },
    {
        "id": 4,
        "text": "Meta contrata Clark Kent como novo chefe de inovação."
    },
    {
        "id": 5,
        "text": "O SGE é composto por 9 servidores, sendo Carla sua chefe"
    }
]

model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embedings = {
    doc["id"]: model.encode(doc["text"], convert_to_tensor=True) for doc in documents
    }
