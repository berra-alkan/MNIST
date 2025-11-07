# api.py
import asyncio, json
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from train import train_mnist  # AYRI modülü çağırıyoruz

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def index():
    # basit demo sayfası
    return FileResponse("index.html")

@app.post("/train")
async def train(request: Request):
    """
    Tek endpoint — Streamable HTTP:
    - İstek body’sinde params al
    - Cevap: Content-Type: text/event-stream (SSE formatında satırlar)
      ama *POST yanıtı* üzerinden akıtıyoruz => Streamable HTTP.
    """
    params = await request.json() if request.headers.get("content-type","").startswith("application/json") else {}
    q: asyncio.Queue[str] = asyncio.Queue()

    loop = asyncio.get_running_loop()

    def emit(kind: str, **kw):
        # train içinden gelen event'leri SSE satırına çevir
        if kind == "log":
            msg = f"data: {kw.get('text','')}\n\n"
        elif kind == "image":
            payload = json.dumps({"name": kw.get("name"), "b64": kw.get("b64")})
            msg = f"event: image\ndata: {payload}\n\n"
        else:
            msg = f"data: [WARN] unknown event {kind}\n\n"
        loop.call_soon_threadsafe(q.put_nowait, msg)

    async def run_in_thread():
        # bloklayan eğitimi thread’e at
        with ThreadPoolExecutor(max_workers=1) as pool:
            await loop.run_in_executor(pool, train_mnist, params, emit)
        await q.put("event: done\ndata: {}\n\n")

    async def streamer():
        # client yeniden bağlanırsa yumuşak retry önerisi
        yield "retry: 2000\n\n"
        # arka planda eğitimi başlat
        asyncio.create_task(run_in_thread())
        try:
            while True:
                item = await q.get()
                yield item
        except asyncio.CancelledError:
            pass

    return StreamingResponse(streamer(), media_type="text/event-stream")
