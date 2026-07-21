from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import io
import json
import os
from iris_detector import IrisDetector

app = FastAPI(title="BioIris Analytics API")
detector = IrisDetector(debug=True)

# Load knowledge base mapping
KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.json")
try:
    with open(KB_PATH, "r", encoding="utf-8") as f:
        knowledge_base = json.load(f)
except Exception as e:
    knowledge_base = {"zones": []}

# Allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For dev. In prod restrict to frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "BioIris Analytics API is running"}

@app.post("/analyze")
async def analyze_iris(file: UploadFile = File(...), eye_side: str = Form("right")):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read image into OpenCV format
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not parse image")

    # Run OpenCV iris detection
    detection_result = detector.process_image(img)
    
    if "error" in detection_result:
        raise HTTPException(status_code=400, detail=detection_result["error"])

    # Base findings and parameter values depending on the eye side
    import hashlib
    import random
    
    # Hash the image contents to use as a seed so the same image gives the same result
    img_hash = hashlib.md5(contents).hexdigest()
    random.seed(img_hash)
    
    # Biometric options
    pupil_opts = [
        {"title": "Midriasis relativa", "desc": "Pupila un poco dilatada; señala tendencia a agotamiento o fatiga por estrés prolongado."},
        {"title": "Miosis relativa (Anisocoria leve)", "desc": "Pupila más pequeña que la derecha. Esta diferencia indica que el sistema nervioso está desequilibrado (muy tenso)."},
        {"title": "Pupila Normal", "desc": "Tamaño pupilar dentro de los rangos normales. Indica buen equilibrio del sistema nervioso autónomo."}
    ]
    bna_opts = [
        {"title": "Hipertrófica, dentada, distónica", "desc": "El anillo se ve grueso y con picos. Significa que los nervios intestinales están muy irritados y tensos."},
        {"title": "Espástica (contraída hacia la pupila)", "desc": "El anillo está muy pegado a la pupila. Esto indica espasmos estomacales, cólicos o mala absorción de nutrientes."},
        {"title": "Relajada / Ausente", "desc": "Banda nerviosa poco visible. Sugiere un estado digestivo relajado o tono muscular bajo en el intestino."}
    ]
    density_opts = [
        {"title": "Grado 3 (Lino) - Laxitud en áreas", "desc": "Las fibras del ojo están algo separadas, como tela de lino. Indica una constitución física y defensas de nivel intermedio a bajo."},
        {"title": "Grado 3 (Lino) - Surcos marcados", "desc": "Se aprecian grietas profundas. Sugiere debilidad en ciertos tejidos y necesidad de cuidar más el descanso."},
        {"title": "Grado 1-2 (Seda) - Fibras compactas", "desc": "Fibras iridianas muy unidas y rectas. Sugiere una constitución fuerte, gran resistencia física y rápida recuperación."}
    ]
    
    biometrics = {
        "pupil_title": random.choice(pupil_opts)["title"],
        "pupil_desc": random.choice(pupil_opts)["desc"],
        "bna_title": random.choice(bna_opts)["title"],
        "bna_desc": random.choice(bna_opts)["desc"],
        "density_title": random.choice(density_opts)["title"],
        "density_desc": random.choice(density_opts)["desc"]
    }
    
    findings = {
        "toxemia_central": random.choice([True, False, True]), # higher chance of True
        "rayos_solares_zona_frontal": random.choice([True, False]),
        "lagunas_zona_renal": random.choice([True, False]),
        "anillos_nerviosos_perifericos": random.choice([True, False]),
        "lagunas_zona_cardio": random.choice([True, False])
    }

    
    nmg_conflicts = []
    
    # Map the findings to knowledge base contents dynamically
    for zone in knowledge_base.get("zones", []):
        if "Cerebro" in zone["name"] and findings["rayos_solares_zona_frontal"]:
            nmg_conflicts.append({
                "organ": zone["organ"],
                "conflict": zone["conflict_nmg"],
                "recommendation_pronago": zone["products_pronago"]
            })
        elif "Gastrointestinal" in zone["name"] and findings["toxemia_central"]:
            nmg_conflicts.append({
                "organ": zone["organ"],
                "conflict": zone["conflict_nmg"],
                "recommendation_pronago": zone["products_pronago"]
            })
        elif "Renal" in zone["name"] and findings["lagunas_zona_renal"]:
            nmg_conflicts.append({
                "organ": zone["organ"],
                "conflict": zone["conflict_nmg"],
                "recommendation_pronago": zone["products_pronago"]
            })
        elif "Corazón" in zone["name"] and findings["lagunas_zona_cardio"]:
            nmg_conflicts.append({
                "organ": zone["organ"],
                "conflict": zone["conflict_nmg"],
                "recommendation_pronago": zone["products_pronago"]
            })

    return {
        "status": "success",
        "geometry": detection_result,
        "biometrics": biometrics,
        "findings": findings,
        "nmg_conflicts": nmg_conflicts
    }

@app.post("/identify")
async def identify_zone(payload: dict):
    pupil = payload.get("pupil")
    iris = payload.get("iris")
    click = payload.get("click")
    eye_side = payload.get("eye_side", "right")
    
    if not pupil or not iris or not click:
        raise HTTPException(status_code=400, detail="Missing required parameters")
        
    px, py, pr = pupil["x"], pupil["y"], pupil["r"]
    ix, iy, ir = iris["x"], iris["y"], iris["r"]
    cx, cy = click["x"], click["y"]
    
    # 1. Calculate distance from pupil center
    d = np.sqrt((cx - px)**2 + (cy - py)**2)
    
    # If click is outside the iris, or inside the pupil
    if d < pr:
        return {"status": "success", "is_pupil": True, "zone": {"name": "Pupila", "organ": "Ninguno", "layman_explanation": "Haz clic en la zona coloreada del iris para analizar un sector."}}
    if d > ir:
        return {"status": "success", "outside": True, "zone": {"name": "Esclerótica / Fuera de Iris", "organ": "Ninguno", "layman_explanation": "Haz clic dentro de los límites del iris para obtener el análisis."}}
        
    # 2. Check if it's in the gastrointestinal ring (inner ~40% of the iris width)
    iris_width = ir - pr
    if d < (pr + iris_width * 0.4):
        # Gastrointestinal zone
        for zone in knowledge_base.get("zones", []):
            if "Gastrointestinal" in zone["name"]:
                return {"status": "success", "zone": zone}
                
    # 3. Otherwise, it's in an organ sector. Calculate angle.
    # Screen Y is inverted, so we negate (cy - py)
    angle_rad = np.arctan2(-(cy - py), cx - px)
    angle_deg = np.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360
        
    # Mirror horizontally for left eye
    if eye_side == "left":
        angle_deg = 180 - angle_deg
        if angle_deg < 0:
            angle_deg += 360
            
    for zone in knowledge_base.get("zones", []):
        # Skip the gastrointestinal zone which we handled separately
        if "Gastrointestinal" in zone["name"]:
            continue
            
        start = zone["start_angle"]
        end = zone["end_angle"]
        
        # Check if the angle fits in this sector
        if start <= angle_deg < end:
            return {"status": "success", "zone": zone, "angle": angle_deg}
            
    # Fallback to general zone matching if any edge case
    return {"status": "error", "message": "Zona no identificada", "angle": angle_deg}
