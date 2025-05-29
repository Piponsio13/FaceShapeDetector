# Face Shape Analyzer

Este proyecto utiliza inteligencia artificial para detectar la forma de tu rostro y proporcionar recomendaciones personalizadas de cortes de cabello y monturas de lentes que mejor se adapten a ti.

## Características

- Detección de rostro en tiempo real
- Clasificación de la forma del rostro en 5 categorías:
  - Corazón (Heart)
  - Alargado (Oblong)
  - Ovalado (Oval)
  - Redondo (Round)
  - Cuadrado (Square)
- Recomendaciones personalizadas de:
  - Cortes de cabello
  - Monturas de lentes
- Interfaz en tiempo real con la cámara web

## Requisitos Previos

- Python 3.8 o superior
- Cámara web
- Sistema operativo: Windows, macOS, o Linux
- Acceso a internet (para la instalación inicial)

## Instalación

1. Clona el repositorio:

```bash
git clone <url-del-repositorio>
cd <nombre-del-directorio>
```

2. Crea un ambiente virtual:

```bash
# En macOS/Linux
python3 -m venv venv
source venv/bin/activate

# En Windows
python -m venv venv
.\venv\Scripts\activate
```

3. Instala las dependencias:

```bash
pip install -r requirements.txt
```

## Archivos Necesarios

El proyecto requiere dos archivos de modelo pre-entrenado:

- `yolov10n-face.pt` - Modelo YOLO para detección de rostros en tiempo real. Este modelo se encarga de localizar y enmarcar los rostros en el video de la cámara.

- `best.pt` - Modelo de clasificación entrenado específicamente para identificar la forma del rostro. Este modelo analiza las características faciales y clasifica el rostro en una de las cinco categorías (Corazón, Alargado, Ovalado, Redondo o Cuadrado). El modelo ha sido entrenado con un dataset especializado de formas de rostro para proporcionar recomendaciones precisas de cortes de cabello y monturas de lentes.

Estos archivos deben estar en el directorio raíz del proyecto.

## Configuración

### Permisos de Cámara

#### En macOS:

1. Ve a Preferencias del Sistema
2. Selecciona "Privacidad y Seguridad"
3. En el lado izquierdo, busca "Cámara"
4. Asegúrate de que Terminal o Python tenga permiso para acceder a la cámara

#### En Windows:

1. Ve a Configuración
2. Selecciona "Privacidad"
3. Selecciona "Cámara"
4. Activa "Permitir que las aplicaciones accedan a tu cámara"

## Uso

1. Activa el ambiente virtual si no está activado:

```bash
# En macOS/Linux
source venv/bin/activate

# En Windows
.\venv\Scripts\activate
```

2. Ejecuta el programa:

```bash
python main.py
```

3. Posiciónate frente a la cámara a una distancia adecuada
4. El programa mostrará:
   - Un recuadro verde alrededor de tu rostro
   - La forma de rostro detectada
   - Recomendaciones personalizadas de cortes y lentes
5. Presiona 'ESC' para salir del programa

## Solución de Problemas

### Error de Cámara

Si recibes el error "No se pudo abrir la cámara":

1. Verifica que tu cámara esté conectada y funcionando
2. Asegúrate de que otras aplicaciones no estén usando la cámara
3. Verifica los permisos de cámara en tu sistema operativo

### Error de Modelos

Si recibes errores relacionados con los archivos .pt:

1. Verifica que ambos archivos estén en el directorio correcto
2. Verifica que los nombres de los archivos coincidan con los especificados en `main.py`

## Tecnologías Utilizadas

- OpenCV (cv2) - Para captura y procesamiento de video
- PyTorch - Para el procesamiento de modelos de IA
- YOLO - Para detección de objetos y clasificación
- NumPy - Para procesamiento numérico
