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
    if eye_side == "right":
        biometrics = {
            "pupil_title": "Midriasis relativa",
            "pupil_desc": "Pupila un poco dilatada; señala tendencia a agotamiento o fatiga por estrés prolongado.",
            "bna_title": "Hipertrófica, dentada, distónica",
            "bna_desc": "El anillo se ve grueso y con picos. Significa que los nervios intestinales están muy irritados y tensos.",
            "density_title": "Grado 3 (Lino) - Laxitud en áreas",
            "density_desc": "Las fibras del ojo están algo separadas, como tela de lino. Indica una constitución física y defensas de nivel intermedio a bajo."
        }
        findings = {
            "toxemia_central": True,
            "rayos_solares_zona_frontal": True,
            "lagunas_zona_renal": True,
            "anillos_nerviosos_perifericos": False,
            "lagunas_zona_cardio": False
        }
    else:  # left eye
        biometrics = {
            "pupil_title": "Miosis relativa (Anisocoria leve)",
            "pupil_desc": "Pupila más pequeña que la derecha. Esta diferencia indica que el sistema nervioso está desequilibrado (muy tenso).",
            "bna_title": "Espástica (contraída hacia la pupila)",
            "bna_desc": "El anillo está muy pegado a la pupila. Esto indica espasmos estomacales, cólicos o mala absorción de nutrientes.",
            "density_title": "Grado 3 (Lino) - Surcos marcados",
            "density_desc": "Se aprecian grietas profundas. Sugiere debilidad en ciertos tejidos y necesidad de cuidar más el descanso."
        }
        findings = {
            "toxemia_central": True,
            "rayos_solares_zona_frontal": False,
            "lagunas_zona_renal": True,
            "anillos_nerviosos_perifericos": True,
            "lagunas_zona_cardio": True
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
