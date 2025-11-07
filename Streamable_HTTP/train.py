# train.py
import io, base64
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import Callback

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def build_dense_autoencoder(input_dim=784, latent_dim=32):
    inp = layers.Input(shape=(input_dim,), name="input_flat")
    x = layers.Dense(128, activation="relu")(inp)
    x = layers.Dense(64, activation="relu")(x)
    z = layers.Dense(latent_dim, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(z)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(input_dim, activation="sigmoid")(x)
    return models.Model(inp, out)

def train_mnist(params: dict, emit):
    """
    params: {"epochs":int, "batch_size":int, "latent_dim":int}
    emit(kind:str, **kwargs) -> backend’in verdiği callback.
      - emit("log", text="...") 
      - emit("image", name="recon_grid", b64="data:image/png;base64,...")
    """
    np.random.seed(42); tf.random.set_seed(42)

    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0).reshape((-1, 28*28))
    x_test  = (x_test.astype("float32")  / 255.0).reshape((-1, 28*28))

    model = build_dense_autoencoder(input_dim=784, latent_dim=int(params.get("latent_dim", 32)))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    class LogCb(Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            def f(x):
                try: return f"{float(x):.4f}"
                except: return "0.0000"
            emit("log", text=f"epoch={epoch+1} acc={f(logs.get('accuracy'))} "
                             f"val_acc={f(logs.get('val_accuracy'))} "
                             f"loss={f(logs.get('loss'))} val_loss={f(logs.get('val_loss'))}")

    emit("log", text="[INFO] Eğitim başlıyor…")
    model.fit(
        x_train, x_train,
        validation_data=(x_test, x_test),
        epochs=int(params.get("epochs", 5)),
        batch_size=int(params.get("batch_size", 256)),
        shuffle=True, callbacks=[LogCb()], verbose=0
    )
    emit("log", text="[INFO] Eğitim bitti, görseller hazırlanıyor…")

    x_hat = model.predict(x_test, verbose=0)

    # Recon grid
    n = 10
    fig = plt.figure(figsize=(2*n, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        plt.imshow(x_test[i].reshape(28,28), cmap="gray"); plt.axis("off"); ax.set_title("in", fontsize=8)
        ax = plt.subplot(2, n, i+1+n)
        plt.imshow(x_hat[i].reshape(28,28), cmap="gray"); plt.axis("off"); ax.set_title("out", fontsize=8)
    emit("image", name="recon_grid", b64=_fig_to_b64(fig))

    # Error hist
    err = np.mean((x_test - x_hat) ** 2, axis=1)
    thr = float(np.percentile(err, 95))
    fig = plt.figure(figsize=(5,3.2))
    plt.hist(err, bins=60); plt.axvline(thr, linestyle="--")
    plt.title(f"Reconstruction MSE (95p={thr:.5f})"); plt.xlabel("MSE"); plt.ylabel("count")
    emit("image", name="error_hist", b64=_fig_to_b64(fig))

    emit("log", text="[DONE] Görseller gönderildi.")
