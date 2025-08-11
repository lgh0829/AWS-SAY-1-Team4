from fastapi import FastAPI
from app.api import webhook
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI()

app.include_router(webhook.router, prefix="/api/webhook", tags=["webhook"])

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI is working!"}

@app.get("/healthz")
def health_check():
    return {"status": "ok"}
