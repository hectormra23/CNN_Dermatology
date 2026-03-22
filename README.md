# DermaAI — Clasificacion de Lesiones Cutaneas con IA

> Sistema de diagnostico dermatologico basado en Deep Learning con precision >90%  
> Presentado en **Innovatec 2026**

---

## Descripcion

DermaAI es un sistema de inteligencia artificial para la clasificacion automatica de lesiones cutaneas en 7 categorias diagnosticas usando el dataset HAM10000. Implementa un ensemble de dos redes neuronales convolucionales de ultima generacion (EfficientNetV2-S y ConvNeXt-Tiny) con tecnicas avanzadas de preprocesado, augmentacion, calibracion y Test Time Augmentation.

---

## Resultados

| Metrica | Valor |
|---|---|
| Accuracy | >93% |
| Balanced Accuracy | >91% |
| Cohen Kappa | >0.91 |
| ROC-AUC macro | >0.97 |
| Sensitivity Melanoma | >93% |

---

## Dataset requerido en Kaggle

Antes de ejecutar el notebook, añade el siguiente dataset en la pestaña **Input** de tu notebook:

```
Nombre  : Skin cancer: HAM10000
Autor   : Suraj Ghuwalewala
URL     : https://www.kaggle.com/datasets/surajghuwalewala/ham1000-segmentation-and-classification
Archivos: GroundTruth.csv  +  images/ (10,015 JPGs)
Tamaño  : ~2.5 GB
```

**Pasos para añadirlo:**
1. Abre tu notebook en Kaggle
2. Click en `Input` → `+ Add Input`
3. Busca: `ham1000 segmentation classification surajghuwalewala`
4. Click en el boton `+` para añadirlo

---

## Configuracion del entorno Kaggle

| Parametro | Valor requerido |
|---|---|
| Accelerator | GPU T4 x2 |
| Internet | ON |
| Python | 3.10+ |
| RAM | 16 GB |

**Activar GPU:**
```
Panel derecho → Session Options → Accelerator → GPU T4 x2
```

---

## Arquitectura del Sistema

```
Imagen Dermoscopica
        |
   DullRazor           <- Eliminacion de pelos
        |
      CLAHE             <- Mejora de contraste local (LAB)
        |
   384x384 px           <- Resolucion alta (vs 224 estandar)
        |
   Augmentacion         <- 15 tecnicas especializadas
        |
   ┌────┴────┐
   |         |
EfficientNetV2-S   ConvNeXt-Tiny     <- 2 modelos paralelos
   |         |
   └────┬────┘
        |
  Temperature Scaling  <- Calibracion de probabilidades
        |
   TTA x5 transforms   <- Test Time Augmentation
        |
  Weighted Ensemble    <- Promedio ponderado por val_acc
        |
   7 Clases + Confianza + Riesgo Clinico
```

---

## Clases Diagnosticas

| Codigo | Nombre | Riesgo |
|---|---|---|
| nv | Melanocytic Nevi (Lunar) | BAJO |
| mel | Melanoma | CRITICO |
| bkl | Benign Keratosis | BAJO |
| bcc | Basal Cell Carcinoma | ALTO |
| akiec | Actinic Keratosis | ALTO |
| vasc | Vascular Lesion | MEDIO |
| df | Dermatofibroma | BAJO |

---

## Tecnicas Implementadas

### Preprocesado
- **DullRazor**: Detecta y elimina pelos mediante morfologia matematica e inpainting
- **CLAHE**: Ecualizacion adaptativa de histograma en canal L del espacio LAB

### Balanceo de clases
- `WeightedRandomSampler`: sobremuestra clases minoritarias
- **Effective Number of Samples**: pesos de clase mas precisos que inverso de frecuencia
- **Focal Loss** (gamma=2.0): concentra aprendizaje en ejemplos dificiles
- **Label Smoothing** (0.1): regularizacion de etiquetas

### Entrenamiento
- **3 fases de fine-tuning**: head only → top stages → full model
- **OneCycleLR**: mejor convergencia que CosineAnnealing
- **Gradient Accumulation** (x4): batch efectivo de 64 con GPU limitada
- **Mixed Precision** (AMP): entrena en FP16, 2x mas rapido
- **MixUp + CutMix**: augmentacion avanzada de pares de imagenes
- **Early Stopping** (patience=10)

### Inferencia
- **TTA x5**: 5 transformaciones por imagen, promedio de predicciones
- **Ensemble**: EfficientNetV2-S + ConvNeXt-Tiny con pesos proporcionales a val_acc
- **Temperature Scaling**: calibracion post-entrenamiento con LBFGS

### Interpretabilidad
- **Grad-CAM++**: mapa de calor de regiones diagnosticas
- Distribucion de probabilidades por clase
- Nivel de riesgo clinico automatico

---

## Estructura del Repositorio

```
dermaai/
├── dermatology_kaggle_final.ipynb   <- Notebook principal (ejecutar en Kaggle)
├── README.md
├── requirements.txt
└── outputs/                         <- Generado al ejecutar
    ├── checkpoints/
    │   ├── best_a.pth               <- Pesos EfficientNetV2-S
    │   ├── best_b.pth               <- Pesos ConvNeXt-Tiny
    │   ├── model_a_scripted.pt      <- TorchScript (Raspberry Pi)
    │   ├── model_b_scripted.pt
    │   ├── model_a.onnx             <- ONNX (universal)
    │   └── model_b.onnx
    ├── plots/
    │   ├── preprocessing.png
    │   ├── evaluation.png
    │   ├── dashboard.png
    │   └── gradcam.png
    ├── metrics.json
    └── model_config.json
```

---

## Como Ejecutar

### En Kaggle (recomendado)

1. Crea un nuevo notebook en [kaggle.com/code](https://www.kaggle.com/code)
2. Importa el archivo: `File → Import Notebook → dermatology_kaggle_final.ipynb`
3. Añade el dataset (ver seccion Dataset requerido)
4. Activa GPU T4 x2 e Internet ON
5. Ejecuta: `Run → Run All`
6. Tiempo estimado: **2.5 - 3 horas**

### Ejecucion local

```bash
# Instalar dependencias
pip install torch torchvision timm albumentations
pip install grad-cam captum reportlab scikit-learn
pip install opencv-python-headless pillow pandas numpy
pip install matplotlib seaborn tqdm ipywidgets

# Ejecutar notebook
jupyter notebook dermatology_kaggle_final.ipynb
```

Nota: para ejecucion local necesitas descargar el dataset manualmente desde Kaggle
y ajustar las rutas en la Celda 4.

---

## Dependencias

```
torch>=2.1.0
torchvision>=0.16.0
timm>=1.0.0
albumentations>=1.3.0
opencv-python-headless>=4.8.0
grad-cam>=1.4.8
captum>=0.7.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
pillow>=10.0.0
tqdm>=4.65.0
ipywidgets>=8.0.0
reportlab>=4.0.0
```

---

## Flujo de Despliegue

```
Kaggle (entrena)
      |
      v
Descargar checkpoints/
      |
      v
HuggingFace Hub  <-- subir model_a_scripted.pt + model_b_scripted.pt
      |
      v
Flask API REST   <-- POST /predict  GET /health  GET /metrics
      |
      v
Raspberry Pi 4   <-- cliente Python + camara dermoscopica
```

Para el cliente de Raspberry Pi, usa el modelo ONNX con `onnxruntime`:

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

sess = ort.InferenceSession('model_a.onnx')
img  = np.array(Image.open('lesion.jpg').resize((384,384))).astype(np.float32)
img  = (img/255.0 - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
img  = img.transpose(2,0,1)[None]
logits = sess.run(None, {'image': img})[0]
pred   = logits.argmax()
print(['akiec','bcc','bkl','df','mel','nv','vasc'][pred])
```

---

## Aviso Legal

Este sistema es una herramienta de apoyo diagnostico. **NO reemplaza** la evaluacion clinica de un dermatologo certificado. Toda decision medica debe ser tomada por un profesional de la salud habilitado tras evaluacion presencial del paciente.

---

## Licencia

MIT License — libre para uso academico y de investigacion.

---

## Presentado en

**InnovatecNM 2026**  
Proyecto: Sistema de Diagnostico Dermatologico con Inteligencia Artificial
