from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple, Optional
import math
import numpy as np
import io
import base64

app = FastAPI(title="PaanaAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://paanaai.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── xT Grid ────────────────────────────────────────────────────────────────
xT_grid = np.array([
    [0.006,0.007,0.008,0.009,0.010,0.011,0.012,0.013,0.014,0.016,0.018,0.021,0.024,0.028,0.033,0.039],
    [0.007,0.008,0.009,0.010,0.012,0.013,0.014,0.016,0.018,0.020,0.023,0.027,0.031,0.037,0.044,0.052],
    [0.008,0.009,0.011,0.012,0.014,0.016,0.018,0.020,0.023,0.026,0.030,0.035,0.041,0.048,0.057,0.067],
    [0.009,0.011,0.012,0.014,0.016,0.018,0.021,0.024,0.027,0.031,0.036,0.042,0.049,0.058,0.069,0.081],
    [0.010,0.012,0.014,0.016,0.018,0.021,0.024,0.027,0.031,0.036,0.042,0.049,0.057,0.067,0.079,0.093],
    [0.011,0.013,0.016,0.018,0.021,0.024,0.027,0.031,0.036,0.042,0.049,0.057,0.067,0.079,0.093,0.110],
    [0.011,0.013,0.016,0.018,0.021,0.024,0.027,0.031,0.036,0.042,0.049,0.057,0.067,0.079,0.093,0.110],
    [0.010,0.012,0.014,0.016,0.018,0.021,0.024,0.027,0.031,0.036,0.042,0.049,0.057,0.067,0.079,0.093],
    [0.009,0.011,0.012,0.014,0.016,0.018,0.021,0.024,0.027,0.031,0.036,0.042,0.049,0.058,0.069,0.081],
    [0.008,0.009,0.011,0.012,0.014,0.016,0.018,0.020,0.023,0.026,0.030,0.035,0.041,0.048,0.057,0.067],
    [0.007,0.008,0.009,0.010,0.012,0.013,0.014,0.016,0.018,0.020,0.023,0.027,0.031,0.037,0.044,0.052],
    [0.006,0.007,0.008,0.009,0.010,0.011,0.012,0.013,0.014,0.016,0.018,0.021,0.024,0.028,0.033,0.039]
])

def getxT(x1, y1):
    xi = min(int(x1) // 4, 15)
    yi = min(int(y1) // 4, 11)
    return float(xT_grid[yi][xi])

def xTDiff(x1, y1, x2, y2):
    return getxT(x2, y2) - getxT(x1, y1)

def getOffsideLine(team_b):
    maxi = float('-inf')
    maxi2 = float('-inf')
    for dx, dy in team_b:
        if dx > maxi:
            maxi2 = maxi
            maxi = dx
        elif dx > maxi2:
            maxi2 = dx
    return maxi2

def getAngle(ox, oy, nx, ny, dx, dy):
    pass_angle = math.degrees(math.atan2(ny-oy, nx-ox))
    def_angle  = math.degrees(math.atan2(dy-oy, dx-ox))
    diff = (def_angle - pass_angle + 180) % 360 - 180
    return diff

def getConeAngle(distance):
    if distance < 10:
        return 4
    elif distance < 20:
        return 5
    else:
        return 6

def getReceiverRisk(nx, ny, team_b, distance):
    risk = 0
    radius = distance * 0.2
    for dx, dy in team_b:
        dist = math.dist((dx,dy),(nx,ny))
        if   dist <= radius * 0.125: risk += 8
        elif dist <= radius * 0.25:  risk += 4
        elif dist <= radius * 0.5:   risk += 2
        elif dist <= radius:         risk += 1
        elif dist <= radius * 2:     risk += 0.5
        elif dist <= radius * 4:     risk += 0.25
        elif dist <= radius * 8:     risk += 0.125
    return risk

def getRisk(ox, oy, nx, ny, team_b):
    distance = math.dist((ox,oy),(nx,ny))
    cone_angle = getConeAngle(distance)
    interceptor_in_lane = False
    for dx, dy in team_b:
        def_dist = math.dist((ox,oy),(dx,dy))
        if def_dist >= distance:
            continue
        angle_diff = getAngle(ox, oy, nx, ny, dx, dy)
        if abs(angle_diff) <= cone_angle:
            interceptor_in_lane = True
            break
    risk = 0
    if interceptor_in_lane:
        risk += getReceiverRisk(nx, ny, team_b, distance + distance/20)
        risk += distance * 0.4
    else:
        risk += getReceiverRisk(nx, ny, team_b, distance)
        risk+= distance * 0.2
    pass_type = "lofted" if interceptor_in_lane else "ground"
    return risk, pass_type

class Player(BaseModel):
    x: float
    y: float

class BestPassRequest(BaseModel):
    team_a: List[Player]
    team_b: List[Player]
    ball_carrier_index: int
    mode: int  # 0-4

class BestPassResponse(BaseModel):
    receiver_index: int  # -2 = shoot/keep, >=0 = pass
    pass_type: str
    message: str
    score: float

@app.post("/api/best-pass", response_model=BestPassResponse)
def best_pass(req: BestPassRequest):
    team_a = [(p.x, p.y) for p in req.team_a]
    team_b = [(p.x, p.y) for p in req.team_b]

    ball = team_a[req.ball_carrier_index]
    x, y = ball

    offside_line = getOffsideLine(team_b)
    maxTD = 0.110 - 0.006
    maxRisk = 35

    # Mode weights — exactly from original notebook
    mode = max(0, min(4, req.mode))
    if   mode == 0: xf, yf = 0.3, 0.7
    elif mode == 1: xf, yf = 0.4, 0.6
    elif mode == 2: xf, yf = 0.5, 0.5
    elif mode == 3: xf, yf = 0.6, 0.4
    elif mode == 4: xf, yf = 0.7, 0.3

    maxi = float('-inf')
    bestpass = -1
    best_idx = -1
    Type = "ground"

    i = 0
    for dx, dy in team_a:
        i += 1
        if i - 1 == req.ball_carrier_index:
            continue
        if dx > max(offside_line, x):
            continue
        if math.dist((x, y), (dx, dy)) > 45:
            continue
        xTD = xTDiff(x, y, dx, dy)
        risk, ttype = getRisk(x, y, dx, dy, team_b)
        risk = risk / maxRisk
        xTD = xTD / maxTD
        res = xTD * xf - risk * yf
        if res > maxi:
            maxi = res
            bestpass = (dx, dy)
            Type = ttype
            best_idx = i - 1

    if bestpass == -1:
        if math.dist((64, 24), ball) < 15:
            return BestPassResponse(receiver_index=-2, pass_type="shoot",
                                    message="Go for a Shot!", score=0)
        return BestPassResponse(receiver_index=-2, pass_type="keep",
                                message="Keep the ball", score=0)

    # bestpass == ball in the notebook means no pass improved things.
    # In the API coords are floats so we check maxi <= 0 instead:
    # a non-positive score means every pass goes backward or is too risky.
    if maxi < 0:
        if math.dist((64, 24), ball) < 15:
            return BestPassResponse(receiver_index=-2, pass_type="shoot",
                                    message="Go for a Shot!", score=0)
        if maxi < -0.15:
            return BestPassResponse(receiver_index=-2, pass_type="keep",
                                    message="Keep the ball", score=0)

    return BestPassResponse(
        receiver_index=best_idx,
        pass_type=Type,
        message=f"Best pass to player {best_idx + 1} ({Type} pass)",
        score=round(maxi, 4)
    )


# ─── Role Detection Endpoint ─────────────────────────────────────────────────
try:
    import torch
    import torchvision.transforms as transforms
    from torchvision import models
    from PIL import Image
    import torch.nn as nn
    import torch.nn.functional as F

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.bn1   = nn.BatchNorm2d(32)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.bn2   = nn.BatchNorm2d(64)
            self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.pool3 = nn.MaxPool2d(2, 2)
            self.bn3   = nn.BatchNorm2d(128)
            self.dropout = nn.Dropout(0.5)
            self.fc1 = nn.Linear(128 * 16 * 8, 512)
            self.fc2 = nn.Linear(512, 128)
            self.fc3 = nn.Linear(128, 9)

        def forward(self, x):
            x = self.pool1(F.relu(self.bn1(self.conv2(F.relu(self.conv1(x))))))
            x = self.pool2(F.relu(self.bn2(self.conv4(F.relu(self.conv3(x))))))
            x = self.pool3(F.relu(self.bn3(self.conv6(F.relu(self.conv5(x))))))
            x = x.view(x.size(0), -1)
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.dropout(F.relu(self.fc2(x)))
            x = self.fc3(x)
            return x

    CLASS_NAMES = [
        'Attacking Midfielder', 'Central Midfielder', 'Centre Back',
        'Defensive Midfielder', 'Left Back', 'Left Winger',
        'Right Back', 'Right Winger', 'Striker'
    ]

    # Leader roles = more advanced/creative; Learner = defensive/supporting
    LEADER_ROLES = {'Attacking Midfielder', 'Striker', 'Left Winger', 'Right Winger', 'Central Midfielder'}

    transform = transforms.Compose([
        transforms.Resize((128, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = torch.device('cpu')
    model = CNN().to(device)

    import os
    MODEL_PATH = os.environ.get("MODEL_PATH", "RoleDetectionModel.pth")
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        MODEL_LOADED = True
        print(f"Model loaded from {MODEL_PATH}")
    else:
        MODEL_LOADED = False
        print(f"Model not found at {MODEL_PATH} — role detection will use demo mode")

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    MODEL_LOADED = False
    print("PyTorch not available — role detection in demo mode")

@app.post("/api/detect-role")
async def detect_role(file: UploadFile = File(...)):
    contents = await file.read()

    if not TORCH_AVAILABLE or not MODEL_LOADED:
        # Demo mode: return a plausible random result
        import random
        CLASS_NAMES_DEMO = [
            'Attacking Midfielder', 'Central Midfielder', 'Centre Back',
            'Defensive Midfielder', 'Left Back', 'Left Winger',
            'Right Back', 'Right Winger', 'Striker'
        ]
        LEADER_ROLES_DEMO = {'Attacking Midfielder', 'Striker', 'Left Winger', 'Right Winger', 'Central Midfielder'}
        role = random.choice(CLASS_NAMES_DEMO)
        confs = np.random.dirichlet(np.ones(9) * 2)
        top_idx = CLASS_NAMES_DEMO.index(role)
        confs[top_idx] = max(confs) + 0.2
        confs = confs / confs.sum()
        all_confs = {CLASS_NAMES_DEMO[i]: round(float(confs[i]) * 100, 1) for i in range(9)}
        return {
            "role": role,
            "type": "Leader" if role in LEADER_ROLES_DEMO else "Learner",
            "confidence": round(float(confs[top_idx]) * 100, 1),
            "all_confidences": all_confs,
            "demo_mode": True
        }

    try:
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            probs  = torch.softmax(output, dim=1)[0]
        idx  = probs.argmax().item()
        role = CLASS_NAMES[idx]
        all_confs = {CLASS_NAMES[i]: round(float(probs[i]) * 100, 1) for i in range(9)}
        return {
            "role": role,
            "type": "Leader" if role in LEADER_ROLES else "Learner",
            "confidence": round(float(probs[idx]) * 100, 1),
            "all_confidences": all_confs,
            "demo_mode": False
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_LOADED, "torch": TORCH_AVAILABLE}
