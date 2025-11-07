# app.py
import asyncio, io, json, base64
import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import Callback

app = FastAPI()

# index.html'i kökten servis edelim
@app.get("/")
def root():
    return FileResponse("index.html")

# ---- basit kanal yönetimi (run_id -> asyncio.Queue) ----
CHANNELS: dict[str, asyncio.Queue] = {}

async def push(run_id: str, msg: dict):
    q = CHANNELS.get(run_id)
    if q:
        await q.put(msg)

# ---- SSE endpoint ----
@app.get("/stream")
async def stream(run_id: str):
    q = asyncio.Queue()
    CHANNELS[run_id] = q

    async def gen():
        # reconnect eden client için önerilen retry süresi
        yield "retry: 2000\n\n"
        try:
            while True:
                msg = await q.get()
                t = msg.get("type")
                if t == "log":
                    line = msg["text"]
                    yield f"data: {line}\n\n"
                elif t == "image":
                    payload = json.dumps({"name": msg["name"], "b64": msg["b64"]})
                    yield f"event: image\ndata: {payload}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            # client ayrıldıysa kanalı temizle
            if CHANNELS.get(run_id) is q:
                del CHANNELS[run_id]

    return StreamingResponse(gen(), media_type="text/event-stream")

# ---- Eğitimi başlat ----
@app.post("/start")
async def start(run_id: str):
    # eğitim zaten varsa tekrar başlatma
    if run_id in CHANNELS and not CHANNELS[run_id].empty():
        await push(run_id, {"type": "log", "text": "[WARN] Bu run için kanal zaten açık."})

    loop = asyncio.get_running_loop()

    def emit(kind: str, **kw):
        # eğitim thread'inden güvenle SSE kuyruğuna yaz
        loop.call_soon_threadsafe(asyncio.create_task, push(run_id, {"type": kind, **kw}))

    # bloklayan eğitimi arka planda thread'e at
    asyncio.create_task(asyncio.to_thread(train_and_emit, emit))

    return JSONResponse({"ok": True})

# ---- MNIST AE + log/görsel üretimi ----
def build_dense_autoencoder(input_dim=784, latent_dim=32):
    inp = layers.Input(shape=(input_dim,), name="input_flat")
    x = layers.Dense(128, activation="relu")(inp)
    x = layers.Dense(64, activation="relu")(x)
    z = layers.Dense(latent_dim, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(z)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(input_dim, activation="sigmoid")(x)
    return models.Model(inp, out)

def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def train_and_emit(emit):
    np.random.seed(42); tf.random.set_seed(42)

    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0).reshape((-1, 28*28))
    x_test  = (x_test.astype("float32")  / 255.0).reshape((-1, 28*28))

    model = build_dense_autoencoder(input_dim=784, latent_dim=32)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    class LogCb(Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            def f(x):
                try: return f"{float(x):.4f}"
                except: return "0.0000"
            line = f"epoch={epoch+1} acc={f(logs.get('accuracy'))} val_acc={f(logs.get('val_accuracy'))} loss={f(logs.get('loss'))} val_loss={f(logs.get('val_loss'))}"
            emit("log", text=line)

    emit("log", text="[INFO] Eğitim başlıyor…")
    history = model.fit(
        x_train, x_train,
        validation_data=(x_test, x_test),
        epochs=10, batch_size=256, shuffle=True,
        callbacks=[LogCb()], verbose=0
    )
    emit("log", text="[INFO] Eğitim bitti, görseller hazırlanıyor…")

    # Rekonstrüksiyon
    x_hat = model.predict(x_test, verbose=0)

    # 1) Recon grid figürü
    n = 10
    fig = plt.figure(figsize=(2*n, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        plt.imshow(x_test[i].reshape(28,28), cmap="gray"); plt.axis("off"); ax.set_title("in", fontsize=8)
        ax = plt.subplot(2, n, i+1+n)
        plt.imshow(x_hat[i].reshape(28,28), cmap="gray"); plt.axis("off"); ax.set_title("out", fontsize=8)
    recon_b64 = fig_to_b64(fig)
    emit("image", name="recon_grid", b64=recon_b64)

    # 2) Hata histogramı
    err = np.mean((x_test - x_hat) ** 2, axis=1)
    thr = np.percentile(err, 95)
    fig = plt.figure(figsize=(5,3.2))
    plt.hist(err, bins=60); plt.axvline(thr, linestyle="--")
    plt.title(f"Reconstruction MSE (95p={thr:.5f})"); plt.xlabel("MSE"); plt.ylabel("count")
    hist_b64 = fig_to_b64(fig)
    emit("image", name="error_hist", b64=hist_b64)

    emit("log", text="[DONE] Görseller gönderildi.")
