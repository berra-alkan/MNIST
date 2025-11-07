# MNIST — SSE vs Streamable HTTP

Bu mini proje, model eğitim sürecinde anlık log ve görsel çıktıların kullanıcıya aktarımı için  
iki farklı yaklaşımı içerir: Server-Sent Events (SSE) ve Streamable HTTP.

## Klasörler
sse/ → SSE tabanlı FastAPI demo, 
Streamable_HTTP/ → Streamable HTTP tabanlı demo

## Çalıştırma
```bash
# Ortam
pip install fastapi uvicorn tensorflow matplotlib numpy

# SSE
cd sse && uvicorn app:app --reload --port 8000

# Streamable HTTP
cd Streamable_HTTP && uvicorn api:app --reload --port 8000
