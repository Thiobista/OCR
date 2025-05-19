import threading
import time
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image
import cv2
from ultralytics import YOLO
from pyngrok import ngrok
import uvicorn

# ========== 1. FastAPI App ==========
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== 2. Load Vocabulary ==========
def load_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f.readlines()]
    special_tokens = ['<blank>', '<unk>']
    for token in special_tokens:
        if token not in vocab:
            vocab.append(token)
    if ' ' not in vocab:
        vocab.append(' ')
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for char, idx in char2idx.items()}
    return vocab, char2idx, idx2char

vocab_path = '/users/thiobista/ocr/vocab(1).txt'
vocab, char2idx, idx2char = load_vocab(vocab_path)
blank_index = char2idx['<blank>']

# ========== 3. Define CRNN Model ==========
class CRNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, 1, 1), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64, 128, 3, 1, 1), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(128, 256, 3, 1, 1), torch.nn.ReLU(), torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(256, 256, 3, 1, 1), torch.nn.ReLU(), torch.nn.MaxPool2d((2, 1), (2, 1)),
            torch.nn.Conv2d(256, 512, 3, 1, 1), torch.nn.ReLU(), torch.nn.BatchNorm2d(512),
            torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.ReLU(), torch.nn.MaxPool2d((2, 1), (2, 1)),
            torch.nn.Conv2d(512, 512, 2, 1, 0)
        )
        self.lstm = torch.nn.LSTM(512, 256, num_layers=2, bidirectional=True, dropout=0.3)
        self.fc = torch.nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.conv(x)
        x = x.squeeze(2).permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return torch.nn.functional.log_softmax(x, dim=2)

# ========== 4. Load CRNN Model ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
crnn_model = CRNN(num_classes=len(vocab)).to(device)
crnn_model_path = '/users/thiobista/ocr/crnn_amharic_thiob(1).pth'
crnn_model.load_state_dict(torch.load(crnn_model_path, map_location=device))
crnn_model.eval()

# ========== 5. Load YOLO Model ==========
yolo_model_path = '/users/thiobista/ocr/amharic_yolov8l(1).pt'
yolo_model = YOLO(yolo_model_path)

# ========== 6. Define Image Preprocessing ==========
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
])

def preprocess_image(image):
    return transform(image).unsqueeze(0).to(device)

# ========== 7. Decode Prediction ==========
def decode_prediction(prediction):
    pred_indices = prediction.argmax(2).permute(1, 0)
    decoded_texts = []
    for indices in pred_indices:
        decoded = []
        prev_idx = -1
        for idx in indices:
            idx = idx.item()
            if idx != prev_idx and idx != blank_index:
                decoded.append(idx2char[idx])
            prev_idx = idx
        decoded_texts.append(''.join(decoded))
    return decoded_texts[0]

# ========== 8. YOLO Detection ==========
def detect_words(image_path):
    results = yolo_model(image_path, max_det=1000)
    img = cv2.imread(image_path)
    word_data = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cropped_img = img[y1:y2, x1:x2]
            word_image = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            word_data.append({'box': (x1, y1, x2, y2), 'image': word_image})
    return word_data

# ========== 9. FastAPI Endpoint ==========
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # Save uploaded image
    image_path = "temp_image.jpg"
    with open(image_path, "wb") as f:
        content = await image.read()
        f.write(content)

    # Open the image with PIL to get its size
    pil_image = Image.open(image_path)
    width, height = pil_image.size

    # Detect words using YOLO
    word_data = detect_words(image_path)

    # Recognize text for each detected word
    for word in word_data:
        word_image = word['image']
        preprocessed_image = preprocess_image(word_image)
        with torch.no_grad():
            outputs = crnn_model(preprocessed_image)
        prediction = decode_prediction(outputs)
        word['prediction'] = prediction

    # Prepare response (with paper size)
    response = {
        "paper_size": {"width": width, "height": height},
        "words": [{'box': word['box'], 'text': word['prediction']} for word in word_data]
    }
    return JSONResponse(content=response)

# ========== 10. Run App with public URL ==========
def run_uvicorn():
    uvicorn.run(app, host="0.0.0.0", port=5000)

if __name__ == "__main__":
    # Start Uvicorn in a background thread
    server_thread = threading.Thread(target=run_uvicorn)
    server_thread.daemon = True
    server_thread.start()

    # Wait for server to start
    time.sleep(3)

    # Start ngrok tunnel
    public_url = ngrok.connect(5000)
    print(f" * ngrok tunnel available at: {public_url}")

    # Keep the main thread alive
    print(" * Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")