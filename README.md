
#  CAS — Call Center Analysis System  
**Un sistema de clasificación y evaluación del desempeño diseñado para mejorar las métricas de calidad en nuestros operadores. Su objetivo es brindar total transparencia tanto al equipo como a los clientes, identificando con precisión los puntos fuertes y las áreas de oportunidad. La plataforma ayuda a detectar fallos que suelen pasar desapercibidos y permite optimizar la atención al cliente mediante retroalimentación clara y accionable.**

---

##  Badges  
![Python](https://img.shields.io/badge/Python-3.10-blue)  
![Status](https://img.shields.io/badge/Status-En%20Desarrollo-yellow)  
![License](https://img.shields.io/badge/License-MIT-green)  
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)  
![AI](https://img.shields.io/badge/AI-Transformers-purple)

---

# 1️ Problema que resuelve  
Los call centers manejan miles de llamadas diarias, pero:  
- No existen sistemas automáticos que identifiquen **venta, soporte, reclamos o consultas**.  
- No se detectan **emociones**, **intenciones** ni **comportamientos del operador**.  
- La información se procesa manualmente, generando retrasos y errores.  

**CAS automatiza todo esto**: transcribe, analiza y clasifica cada llamada.

---

# 2️ Nuestra solución  
CAS es una plataforma modular con IA que:  
- Transcribe audio con Whisper  
- Clasifica intención (venta, soporte, reclamo, consulta, general)  
- Identifica operador vs cliente  
- Detecta emociones (enojo, alegría, frustración, gratitud…)  
- Genera puntaje automático del operador  
- Extrae datos del cliente (ID, nombre, ticket, teléfono)  
- Guarda todo en una base de datos SQLite  
- Muestra una interfaz en Gradio

---

# 3️ Características principales  
✔️ Transcripción automática (Whisper)  
✔️ Análisis semántico de la conversación  
✔️ Clasificación multi-clase  
✔️ Multi-label (emociones múltiples)  
✔️ Identificación de operador vs cliente  
✔️ Scoring del desempeño del operador  
✔️ Integración con DB (SQLite)  
✔️ Visualización de gráficas y reportes  
✔️ Estructura modular tipo Cookiecutter

---

# 4️ Tecnologías utilizadas  
- **Python 3.10+**  
- **Transformers (HuggingFace)**  
- **Torch**  
- **Whisper**  
- **Gradio**  
- **SQLite3**  
- **Pandas / NumPy**  
- **Matplotlib**  
- **Regex / NLP heurístico**

---

# 5️ Estructura del proyecto (CAS/)  
```plaintext
CAS/
├── start.py
├── src/
│   ├── config.py
│   ├── audio/
│   │   └── audio_transcription.py
│   ├── analyzer/
│   │   └── conversation_analyzer.py
│   ├── database/
│   │   └── database.py
│   ├── model/
│   │   ├── data_preparation.py
│   │   ├── model_training.py
│   │   └── model_evaluation.py
│   └── interface/
│       └── gradio_app.py
├── data/
├── models/
├── reports/
│   └── plot_reports.py
├── audio_samples/
└── requirements.txt
```

---

# 6️⃣ Instalación y Uso  
###  Instalar dependencias
```bash
pip install -r requirements.txt
```

### ▶ Ejecutar la app
```bash
export PYTHONPATH=$(pwd)
python src/interface/gradio_app.py
```

###  Procesar una llamada desde Python
```python
from src.interface.gradio_app import CallCenterApp

app = CallCenterApp()
result = app.process_audio_file("audio_samples/llamada1.mp3")
print(result)
```

---

# 7️⃣ Ejemplos de Resultados  
### Ejemplo de datos extraídos:
```json
{
  "transcription": "Hola, tengo un problema con mi internet...",
  "customer_id": "C1234",
  "ticket": "T9988",
  "call_type": "Reclamo",
  "emotion": ["frustration", "anger"],
  "operator_score": 78
}
```

### Ejemplo de clasificación automática:
- Venta → 97%  
- Soporte → 2%  
- Reclamo → 1%

---

# 8️ Roadmap  
### Versión Actual (v1.0)
- [x] Transcripción automática  
- [x] Análisis semántico  
- [x] Clasificación multi-clase  
- [x] DB con historial  

### Próximas Versiones
- [ ] Integración con CRM  
- [ ] Dashboard profesional  
- [ ] Training del modelo propio  
- [ ] API REST  
- [ ] App web completa  
- [ ] Multi-idioma  

---

# 9️ Contribución  
1. Haz fork  
2. Crea una rama  
```bash
git checkout -b feature/nueva-funcionalidad
```
3. Haz commit  
```bash
git.commit -m "feat: agrega clasificación emocional mejorada"
```
4. Haz push  
```bash
git push origin feature/nueva-funcionalidad
```
5. Abre un Pull Request  

---

#  Equipo  
| Nombre | Rol | GitHub |
|--------|------|--------|
| italo | Dev  | *(@italo0072)* |

---

# 1️ Contacto  
 Email: italo.antonio.45@outlook.com  
 LinkedIn: https://www.linkedin.com/in/italo-antonio-9965b026a

---

#  Licencia  
MIT License.
