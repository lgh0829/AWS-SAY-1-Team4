from fastapi import FastAPI
from app.api import webhook

app = FastAPI()

app.include_router(webhook.router, prefix="/api/webhook", tags=["webhook"])

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI is working!"}
