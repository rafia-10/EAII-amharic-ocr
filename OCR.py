import os
import shutil
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

# ------------------------ CONFIG ------------------------
CRNN_LOC = 'best_crnn (1).pth'
YOLO_MODEL_PATH = 'best.pt'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_CROPPED_FOLDER = 'cropped_words'
BW_IMG_LOC = 'BW_read_img.jpg'
BLANK = 302

# ------------------------ LOAD MODELS ------------------------
yolo_model = YOLO(YOLO_MODEL_PATH)

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(nc,64,3,1,1), nn.ReLU(True), nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,1,1), nn.ReLU(True), nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,1,1), nn.ReLU(True),
            nn.Conv2d(256,256,3,1,1), nn.ReLU(True), nn.MaxPool2d((2,2),(2,1),(0,1)),
            nn.Conv2d(256,512,3,1,1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512,512,3,1,1), nn.BatchNorm2d(512), nn.ReLU(True), nn.MaxPool2d((2,2),(2,1),(0,1)),
            nn.Conv2d(512,512,2,1,0), nn.ReLU(True)
        )
        self.rnn1 = nn.LSTM(512, nh, bidirectional=True, dropout=0.3)
        self.rnn2 = nn.LSTM(nh*2, nh, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(nh*2, nclass)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.squeeze(2).permute(2, 0, 1)
        recurrent, _ = self.rnn1(conv)
        recurrent, _ = self.rnn2(recurrent)
        output = self.fc(recurrent)
        return output.permute(1, 0, 2)

CRNN_MODEL = CRNN(imgH=32, nc=1, nclass=303, nh=256).to(DEVICE)
checkpoint = torch.load(CRNN_LOC, map_location=DEVICE)
CRNN_MODEL.load_state_dict(checkpoint)
CRNN_MODEL.eval()

# ------------------------ AMHARIC MAPPING ------------------------
amharic_mapping = {
    # Basic consonants + vowels
    0: 'ሀ', 1: 'ሁ', 2: 'ሂ', 3: 'ሃ', 4: 'ሄ', 5: 'ህ', 6: 'ሆ',
    7: 'ለ', 8: 'ሉ', 9: 'ሊ', 10: 'ላ', 11: 'ሌ', 12: 'ል', 13: 'ሎ', 14: 'ሏ',
    15: 'ሐ', 16: 'ሑ', 17: 'ሒ', 18: 'ሓ', 19: 'ሔ', 20: 'ሕ', 21: 'ሖ', 22: 'ሗ',
    23: 'መ', 24: 'ሙ', 25: 'ሚ', 26: 'ማ', 27: 'ሜ', 28: 'ም', 29: 'ሞ', 30: 'ሟ',
    31: 'ሠ', 32: 'ሡ', 33: 'ሢ', 34: 'ሣ', 35: 'ሤ', 36: 'ሥ', 37: 'ሦ', 38: 'ሧ',
    39: 'ረ', 40: 'ሩ', 41: 'ሪ', 42: 'ራ', 43: 'ሬ', 44: 'ር', 45: 'ሮ', 46: 'ሯ',
    47: 'ሰ', 48: 'ሱ', 49: 'ሲ', 50: 'ሳ', 51: 'ሴ', 52: 'ስ', 53: 'ሶ', 54: 'ሷ',
    55: 'ሸ', 56: 'ሹ', 57: 'ሺ', 58: 'ሻ', 59: 'ሼ', 60: 'ሽ', 61: 'ሾ', 62: 'ሿ',
    63: 'ቀ', 64: 'ቁ', 65: 'ቂ', 66: 'ቃ', 67: 'ቄ', 68: 'ቅ', 69: 'ቆ', 70: 'ቋ',
    71: 'በ', 72: 'ቡ', 73: 'ቢ', 74: 'ባ', 75: 'ቤ', 76: 'ብ', 77: 'ቦ', 78: 'ቧ',
    79: 'ቨ', 80: 'ቩ', 81: 'ቪ', 82: 'ቫ', 83: 'ቬ', 84: 'ቭ', 85: 'ቮ', 86: 'ቯ',
    87: 'ተ', 88: 'ቱ', 89: 'ቲ', 90: 'ታ', 91: 'ቴ', 92: 'ት', 93: 'ቶ', 94: 'ቷ',
    95: 'ቸ', 96: 'ቹ', 97: 'ቺ', 98: 'ቻ', 99: 'ቼ', 100: 'ች', 101: 'ቾ', 102: 'ቿ',
    103: 'ኀ', 104: 'ኁ', 105: 'ኂ', 106: 'ኃ', 107: 'ኄ', 108: 'ኅ', 109: 'ኆ',
    110: 'ነ', 111: 'ኑ', 112: 'ኒ', 113: 'ና', 114: 'ኔ', 115: 'ን', 116: 'ኖ', 117: 'ኗ',
    118: 'ኘ', 119: 'ኙ', 120: 'ኚ', 121: 'ኛ', 122: 'ኜ', 123: 'ኝ', 124: 'ኞ', 125: 'ኟ',
    126: 'አ', 127: 'ኡ', 128: 'ኢ', 129: 'ኣ', 130: 'ኤ', 131: 'እ', 132: 'ኦ', 133: 'ኧ',
    134: 'ከ', 135: 'ኩ', 136: 'ኪ', 137: 'ካ', 138: 'ኬ', 139: 'ክ', 140: 'ኮ', 141: 'ኯ',
    142: 'ኸ', 143: 'ኹ', 144: 'ኺ', 145: 'ኻ', 146: 'ኼ', 147: 'ኽ', 148: 'ኾ', 149: 'ዀ', 150: 'ዃ',
    151: 'ወ', 152: 'ዉ', 153: 'ዊ', 154: 'ዋ', 155: 'ዌ', 156: 'ው', 157: 'ዎ', 158: 'ዏ',
    159: 'ዐ', 160: 'ዑ', 161: 'ዒ', 162: 'ዓ', 163: 'ዔ', 164: 'ዕ', 165: 'ዖ',
    166: 'ዘ', 167: 'ዙ', 168: 'ዚ', 169: 'ዛ', 170: 'ዜ', 171: 'ዝ', 172: 'ዞ', 173: 'ዟ',
    174: 'ዠ', 175: 'ዡ', 176: 'ዢ', 177: 'ዣ', 178: 'ዤ', 179: 'ዥ', 180: 'ዦ', 181: 'ዧ',
    182: 'የ', 183: 'ዩ', 184: 'ዪ', 185: 'ያ', 186: 'ዬ', 187: 'ይ', 188: 'ዮ',
    189: 'ደ', 190: 'ዱ', 191: 'ዲ', 192: 'ዳ', 193: 'ዴ', 194: 'ድ', 195: 'ዶ', 196: 'ዷ',
    197: 'ጀ', 198: 'ጁ', 199: 'ጂ', 200: 'ጃ', 201: 'ጄ', 202: 'ጅ', 203: 'ጆ', 204: 'ጇ',
    205: 'ገ', 206: 'ጉ', 207: 'ጊ', 208: 'ጋ', 209: 'ጌ', 210: 'ግ', 211: 'ጎ', 212: 'ጏ',
    213: 'ጠ', 214: 'ጡ', 215: 'ጢ', 216: 'ጣ', 217: 'ጤ', 218: 'ጥ', 219: 'ጦ', 220: 'ጧ',
    221: 'ጨ', 222: 'ጩ', 223: 'ጪ', 224: 'ጫ', 225: 'ጬ', 226: 'ጭ', 227: 'ጮ', 228: 'ጯ',
    229: 'ጰ', 230: 'ጱ', 231: 'ጲ', 232: 'ጳ', 233: 'ጴ', 234: 'ጵ', 235: 'ጶ', 236: 'ጷ',
    237: 'ጸ', 238: 'ጹ', 239: 'ጺ', 240: 'ጻ', 241: 'ጼ', 242: 'ጽ', 243: 'ጾ', 244: 'ጿ',
    245: 'ፀ', 246: 'ፁ', 247: 'ፂ', 248: 'ፃ', 249: 'ፄ', 250: 'ፅ', 251: 'ፆ', 252: 'ፇ',
    253: 'ፈ', 254: 'ፉ', 255: 'ፊ', 256: 'ፋ', 257: 'ፌ', 258: 'ፍ', 259: 'ፎ', 260: 'ፏ',
    261: 'ፐ', 262: 'ፑ', 263: 'ፒ', 264: 'ፓ', 265: 'ፔ', 266: 'ፕ', 267: 'ፖ', 268: 'ፗ',

    # Special characters / punctuation / numbers
    269: "!", 270: ":-", 271: "<", 272: "(", 273: "«", 274: "፥", 275: "%", 276: "»", 277: ")",
    278: ">", 279: ".", 280: "+", 281: "፣", 282: "-", 283: "።", 284: "/", 
    285: "0", 286: "1", 287: "2", 288: "3", 289: "4", 290: "5", 291: "6", 292: "7", 293: "8", 294: "9", 
    295: "፡", 296: "፤", 297: "...", 298: "*", 299: "#", 300: "?"
}

# ------------------------ UTILITIES ------------------------
def to_black_and_white(image_path, save_path=None, white_thresh=200):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, white_thresh, 255, cv2.THRESH_BINARY)
    if save_path:
        cv2.imwrite(save_path, bw)
    return bw

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read {img_path}")
    h, w = img.shape
    new_w = max(int(32 * (w / h)), 32)
    img = cv2.resize(img, (new_w, 32))
    img = ((img / 255.0) - 0.5) / 0.5
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    return img

def ctc_greedy_decoder(output, idx_to_char, blank=BLANK):
    preds = output.softmax(2).argmax(2)[0].cpu().numpy().tolist()
    decoded = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            decoded.append(idx_to_char.get(p, "�"))
        prev = p
    return "".join(decoded)

def CNNR_Interface(img_path):
    img = preprocess_image(img_path)
    with torch.no_grad():
        output = CRNN_MODEL(img)
    return ctc_greedy_decoder(output, AMHARIC_MAPPING, blank=BLANK)

def clear_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)
    for f in os.listdir(folder_path):
        path = os.path.join(folder_path, f)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print(f"⚠️ Failed to delete {path}: {e}")

def YOLO_cropper(image_path, output_folder, row_tolerance=0.6):
    os.makedirs(output_folder, exist_ok=True)
    results = yolo_model.predict(source=image_path, conf=0.2, iou=0.2, save=False, show=False)
    boxes = results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 else []
    if len(boxes) == 0:
        return []

    crops_info = [(int(x1), int(y1), int(x2), int(y2), (y1+y2)/2, y2-y1) for x1, y1, x2, y2 in boxes]
    crops_info.sort(key=lambda b: b[4])

    # Group into rows
    rows, current_row = [], [crops_info[0]]
    for box in crops_info[1:]:
        row_yc = np.median([b[4] for b in current_row])
        row_h = np.median([b[5] for b in current_row])
        if abs(box[4] - row_yc) <= row_tolerance * row_h:
            current_row.append(box)
        else:
            rows.append(current_row)
            current_row = [box]
    rows.append(current_row)
    for row in rows:
        row.sort(key=lambda b: b[0])
    ordered_boxes = [b for row in rows for b in row]

    saved_paths = []
    for idx, (x1, y1, x2, y2, _, _) in enumerate(ordered_boxes, start=1):
        crop = cv2.imread(image_path)[y1:y2, x1:x2]
        save_path = os.path.join(output_folder, f"word_{idx}.jpg")
        cv2.imwrite(save_path, crop)
        saved_paths.append(save_path)
    return saved_paths

# ------------------------ PIPELINE ------------------------
def pipeline(img_path, user_id=None, bw=True):
    cropped_folder = os.path.join(BASE_CROPPED_FOLDER, str(user_id or "default"))
    if bw:
        to_black_and_white(img_path, save_path=BW_IMG_LOC)
        img_path_to_use = BW_IMG_LOC
    else:
        img_path_to_use = img_path

    clear_folder(cropped_folder)
    locs = YOLO_cropper(img_path_to_use, cropped_folder)
    if not locs:
        return "❌ No text detected."

    detected_text = ''
    for loc in locs:
        detected_text += ' ' + CNNR_Interface(loc)
    return detected_text.strip()

# ------------------------ TEST ------------------------
if __name__ == "__main__":
    print(pipeline('sample.jpg', user_id="test", bw=True))
