import requests
import os

base_url = "https://cdn.jsdelivr.net/npm/openwakeword-wasm-browser@0.1.1/models/"
target_dir = r"C:\Users\GVSCH\AppData\Local\Programs\Python\Python313\Lib\site-packages\openwakeword\resources\models"

files = [
    "hey_jarvis_v0.1.onnx",
    "melspectrogram.onnx",
    "embedding_model.onnx"
]

for filename in files:
    url = base_url + filename
    path = os.path.join(target_dir, filename)
    print(f"Downloading {filename} to {path}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(path, "wb") as f:
            f.write(response.content)
        print(f"Successfully downloaded {filename}")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
