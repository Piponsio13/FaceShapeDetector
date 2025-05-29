import cv2, time, torch, numpy as np
from ultralytics import YOLO

# ── pesos ──────────────────────────────────────────────────────────────────────
DET_WEIGHTS = "/Users/felipe/Documents/Programacion/IA/yolov10n-face.pt"
CLS_WEIGHTS = "/Users/felipe/Documents/Programacion/IA/best.pt"

detector   = YOLO(DET_WEIGHTS)
classifier = YOLO(CLS_WEIGHTS)

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")
classifier.model.to(device).eval()
detector.model.to(device).eval()

names = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
CONF_THR = 0        # confianza mínima para mostrar etiqueta
ALPHA    = 0.3      # suavizado exponencial

# Recomendaciones según forma de cara
recommendations = {
    'Heart': {
        'cortes': '- Bob largo con capas que empiezan a la altura de la barbilla\n'
                 '- Flequillo lateral suave y desfilado\n'
                 '- Cortes en capas largas que añaden volumen en los lados',
        'lentes': '- Aviador clásico con puente fino\n'
                 '- Cat-eye con bordes suavizados\n'
                 '- Lentes redondos con detalles en la parte superior'
    },
    'Oblong': {
        'cortes': '- Bob a la altura de la mandíbula con flequillo recto\n'
                 '- Capas medianas que empiezan en la mejilla\n'
                 '- Corte shaggy con textura y volumen en los lados',
        'lentes': '- Rectangulares anchos con bordes suaves\n'
                 '- Wayfarer grandes que cubran desde las cejas\n'
                 '- Oversized cuadrados con esquinas redondeadas'
    },
    'Oval': {
        'cortes': '- Pixie con flequillo largo y texturizado\n'
                 '- Bob asimétrico con capas largas\n'
                 '- Corte largo en capas con volumen en la corona',
        'lentes': '- Aviador clásico tamaño medio\n'
                 '- Wayfarer proporcionados al rostro\n'
                 '- Geométricos balanceados con tu rostro'
    },
    'Round': {
        'cortes': '- Bob angular con largos asimétricos\n'
                 '- Capas largas que empiezan debajo de la barbilla\n'
                 '- Corte pixie con volumen en la corona y lados ajustados',
        'lentes': '- Rectangulares con esquinas definidas\n'
                 '- Cuadrados con puente alto\n'
                 '- Geométricos angulares que alargan el rostro'
    },
    'Square': {
        'cortes': '- Capas suaves que empiezan en la mejilla\n'
                 '- Bob ondulado con flequillo lateral largo\n'
                 '- Corte en capas con textura y movimiento',
        'lentes': '- Redondos con puente fino\n'
                 '- Ovalados con bordes suaves\n'
                 '- Sin marco con formas curvas'
    }
}

# ── funciones auxiliares ──────────────────────────────────────────────────────
def align_and_crop(img, box, kp, out=224, pad=0.2):
    """Recorta la cara y la alinea con los ojos horizontales."""
    x1,y1,x2,y2 = box
    w,h = x2-x1, y2-y1
    cx,cy = x1+w/2, y1+h/2
    side  = max(w,h)*(1+pad*2)
    x1,y1,x2,y2 = int(cx-side/2),int(cy-side/2),int(cx+side/2),int(cy+side/2)
    H,W = img.shape[:2]
    face = img[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]
    if face.size == 0:
        return None
    le,re = kp[0], kp[1]      # left‑eye, right‑eye
    dx,dy = re[0]-le[0], re[1]-le[1]
    angle = np.degrees(np.arctan2(dy,dx))
    M = cv2.getRotationMatrix2D((face.shape[1]/2, face.shape[0]/2), angle, 1.0)
    face = cv2.warpAffine(face, M, (face.shape[1], face.shape[0]),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)
    return cv2.resize(face, (out,out), interpolation=cv2.INTER_AREA)

def preprocess(face):
    """BGR→RGB, 224×224, float32 0‑1, C×H×W."""
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224,224), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(face).permute(2,0,1).float() / 255.0
    return tensor.unsqueeze(0)           # (1,3,224,224)

def put_text_multiline(img, text, org, font_face, font_scale, color, thickness):
    """Dibuja texto multilínea en la imagen."""
    lines = text.split('\n')
    line_height = cv2.getTextSize('A', font_face, font_scale, thickness)[0][1] + 10
    y = org[1]
    for line in lines:
        cv2.putText(img, line, (org[0], y), font_face, font_scale, color, thickness)
        y += line_height

# ── cámara ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara.")
print("✅  Cámara abierta. Pulsa ESC para salir.")

while True:
    ok, frame = cap.read()
    if not ok:
        time.sleep(0.05)
        continue

    det = detector(frame, imgsz=256, conf=0.2, verbose=False)[0]
    boxes = det.boxes
    kps   = det.keypoints          # puede ser None

    # reiniciamos suavizado cada cuadro (una cara)
    smooth = torch.zeros(len(names))

    if boxes is not None:
        for i, box in enumerate(boxes):
            x1,y1,x2,y2 = map(int, box.xyxy[0])

            # dibuja SIEMPRE la caja del detector
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            # recorte + alineación si hay keypoints
            if kps is not None and kps.xy is not None:
                kp_xy = kps.xy[i]        # (N,2)
                face  = align_and_crop(frame, (x1,y1,x2,y2), kp_xy)
            else:
                face  = frame[y1:y2, x1:x2]

            if face is None:
                continue

            tensor = preprocess(face).to(device)

            with torch.inference_mode():
                logits = classifier.model(tensor)
                logits = logits[0] if isinstance(logits, tuple) else logits
                probs  = torch.softmax(logits, 1).cpu().squeeze()

            smooth = ALPHA*smooth + (1-ALPHA)*probs
            cls  = int(smooth.argmax())
            conf = smooth[cls].item()

            if conf >= CONF_THR:
                face_shape = names[cls]
                label = f"{face_shape} {conf*100:.1f}%"
                cv2.putText(frame, label, (x1, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                
                # Mostrar recomendaciones
                if face_shape in recommendations:
                    recs = recommendations[face_shape]
                    rec_text = f"Recomendaciones para rostro {face_shape}:\n"
                    rec_text += f"CORTES DE CABELLO:\n{recs['cortes']}\n"
                    rec_text += f"\nMONTURAS DE LENTES:\n{recs['lentes']}"
                    put_text_multiline(frame, rec_text, (frame.shape[1]-600, 30),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Face‑Shape (ESC)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()