# DermaAI — Clasificacion de Lesiones Cutaneas con IA

> Sistema de diagnostico dermatologico basado en Deep Learning con precision >90%
> Presentado en **Innovatec 2026**

---

## Descripcion

DermaAI clasifica automaticamente lesiones cutaneas en 7 categorias diagnosticas usando el dataset HAM10000.
Implementa un ensemble de dos modelos:

- **Modelo A:** ConvNeXtV2-Large preentrenado en HAM10000 (HuggingFace) — 196M parametros, sin necesidad de entrenar
- **Modelo B:** EfficientNetV2-S entrenado desde ImageNet (timm) — 21.5M parametros, 25 epocas

La combinacion con TTA x5 y Temperature Scaling alcanza **>90% de accuracy** en aproximadamente **50 minutos de GPU**.

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

Añade este dataset en la pestana **Input** antes de ejecutar:

```
Nombre  : Skin cancer: HAM10000
Autor   : Suraj Ghuwalewala
URL     : https://www.kaggle.com/datasets/surajghuwalewala/ham1000-segmentation-and-classification
Archivos: GroundTruth.csv + images/ (10,015 JPGs)
Tamano  : ~2.5 GB
```

Pasos:
1. Abre tu notebook en Kaggle
2. Click en `Input` → `+ Add Input`
3. Busca: `ham1000 segmentation classification surajghuwalewala`
4. Click en `+` para añadirlo

---

## Configuracion del entorno Kaggle

| Parametro | Valor requerido |
|---|---|
| Accelerator | GPU T4 x2 |
| Internet | ON |
| Python | 3.10+ |

Activar GPU:
```
Panel derecho → Session Options → Accelerator → GPU T4 x2
```

---

## Modelos utilizados

### Modelo A — ConvNeXtV2-Large (HuggingFace, sin entrenar)
```
ALM-AHME/convnextv2-large-1k-224-finetuned-Lesion-Classification-HAM10000-AH-60-20-20
```
- Ya entrenado en HAM10000 con las 7 clases exactas en el mismo orden
- 196 millones de parametros
- Se descarga automaticamente en la Celda 9
- No requiere tiempo de entrenamiento adicional

### Modelo B — EfficientNetV2-S (timm, 25 epocas)
```
tf_efficientnetv2_s
```
- Parte de pesos ImageNet
- Fine-tuning en 3 fases con OneCycleLR
- 21.5 millones de parametros
- ~40 minutos en GPU T4 x2

---

## Arquitectura del sistema

```
Imagen Dermoscopica
        |
   DullRazor              <- Eliminacion de pelos
        |
     CLAHE                <- Mejora de contraste (canal L, espacio LAB)
        |
   384 x 384 px           <- Alta resolucion (vs 224 estandar)
        |
   Augmentacion           <- 15 tecnicas dermoscopicas especializadas
        |
   +----------+----------+
   |                     |
ConvNeXtV2-Large    EfficientNetV2-S
(HuggingFace)       (timm, ImageNet)
196M params         21.5M params
   |                     |
   +----------+----------+
              |
   Temperature Scaling   <- Calibracion de probabilidades (LBFGS)
              |
          TTA x 5        <- Original + HFlip + VFlip + Rot90 + Rot180
              |
   Ensemble ponderado    <- Peso = val_acc / (val_acc_A + val_acc_B)
              |
   Diagnostico + Riesgo clinico
```

---

## Clases diagnosticas

| Codigo | Nombre | Riesgo |
|---|---|---|
| nv | Melanocytic Nevi | BAJO |
| mel | Melanoma | CRITICO |
| bkl | Benign Keratosis | BAJO |
| bcc | Basal Cell Carcinoma | ALTO |
| akiec | Actinic Keratosis | ALTO |
| vasc | Vascular Lesion | MEDIO |
| df | Dermatofibroma | BAJO |

---

## Tecnicas implementadas

### Preprocesado medico
- **DullRazor**: morfologia matematica blackhat + inpainting TELEA para eliminar pelos
- **CLAHE**: ecualizacion adaptativa en canal L del espacio LAB, preserva color natural

### Balanceo de clases
- `WeightedRandomSampler`: sobremuestra clases minoritarias en cada batch
- **Effective Number of Samples**: pesos mas precisos que inverso de frecuencia simple
- **Focal Loss** gamma=2.0: concentra gradiente en ejemplos dificiles
- **Label Smoothing** 0.1: regularizacion de etiquetas duras

### Entrenamiento EfficientNetV2
- **3 fases**: head only → top 35% → full model
- **OneCycleLR**: sube LR 30%, baja 70%, mejor convergencia que CosineAnnealing
- **Gradient Accumulation x4**: batch efectivo de 64 con batch size de 16
- **Mixed Precision AMP**: 2x mas rapido, misma precision
- **MixUp + CutMix**: augmentacion de pares con probabilidad 50%
- **Early Stopping** patience=8 por fase

### Inferencia
- **TTA x5**: 5 orientaciones por imagen, promedio de predicciones
- **Ensemble ponderado**: peso proporcional a val_accuracy de cada modelo
- **Temperature Scaling**: calibracion post-entrenamiento con LBFGS

---

## Estructura del repositorio

```
dermaai/
├── dermatology_ensemble_final.ipynb   <- Notebook principal (ejecutar en Kaggle)
├── README.md
└── outputs/                           <- Generado al ejecutar
    ├── checkpoints/
    │   ├── best_eff.pth               <- Pesos EfficientNetV2-S
    │   ├── efficientnet_scripted.pt   <- TorchScript para Raspberry Pi
    │   └── efficientnet.onnx          <- ONNX universal
    ├── plots/
    │   ├── preprocessing.png
    │   ├── evaluation.png
    │   └── training.png
    ├── metrics.json
    └── model_config.json
```

---

## Como ejecutar

### En Kaggle (recomendado)

1. Crea un nuevo notebook en [kaggle.com/code](https://www.kaggle.com/code)
2. Importa: `File → Import Notebook → dermatology_ensemble_final.ipynb`
3. Añade el dataset HAM10000 de Suraj Ghuwalewala
4. Activa GPU T4 x2 e Internet ON
5. Ejecuta `Run → Run All`
6. Tiempo estimado: **50-60 minutos**

### Ejecucion local

```bash
pip install torch torchvision timm transformers albumentations
pip install grad-cam captum scikit-learn opencv-python-headless
pip install pillow pandas numpy matplotlib seaborn tqdm ipywidgets

jupyter notebook dermatology_ensemble_final.ipynb
```

---

## Flujo de despliegue

```
Kaggle (~50 min)
      |
      v
Descargar outputs/checkpoints/
      |
      v
HuggingFace Hub
  hf_api.upload('efficientnet_scripted.pt')
  hf_api.upload('efficientnet.onnx')
      |
      v
Flask API REST
  POST /predict   <- imagen -> diagnostico JSON
  GET  /health    <- liveness check
  GET  /metrics   <- accuracy, AUC, kappa
      |
      v
Raspberry Pi 4 + camara
  onnxruntime -> ~100ms sin GPU
```

Cliente minimo Raspberry Pi:

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

sess   = ort.InferenceSession('efficientnet.onnx')
img    = np.array(Image.open('lesion.jpg').resize((384,384))).astype(np.float32)
img    = (img/255.0 - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
img    = img.transpose(2,0,1)[None]
logits = sess.run(None, {'image': img})[0]
clases = ['akiec','bcc','bkl','df','mel','nv','vasc']
print('Diagnostico:', clases[logits.argmax()])
```

---

## Dependencias

```
torch>=2.1.0
torchvision>=0.16.0
timm>=1.0.0
transformers>=4.40.0
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
```

---

## Aviso legal

Este sistema es una herramienta de apoyo diagnostico.
**NO reemplaza** la evaluacion clinica de un dermatologo certificado.
Toda decision medica debe ser tomada por un profesional de la salud habilitado
tras evaluacion presencial del paciente.

---

## Presentado en

**Innovatec 2026** — Feria de Innovacion Tecnologica
Proyecto: Sistema de Diagnostico Dermatologico con Inteligencia Artificial
