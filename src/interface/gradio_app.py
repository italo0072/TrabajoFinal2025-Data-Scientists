import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)



import json
import os
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import MODELS_DIR, DEVICE
from src.analyzer.conversation_analyzer import ConversationAnalyzer
from src.audio.audio_transcription import AudioTranscriber, DataExtractor
from src.database.database import CallCenterDatabase


MODEL_DIRS = [Path(MODELS_DIR) / 'call_center_model', Path(MODELS_DIR) / 'finetuned_model']


class CallCenterApp:
    def __init__(self):
        self.analyzer = ConversationAnalyzer()
        self.model = None
        self.tokenizer = None
        self.transcriber = None
        self.db = CallCenterDatabase()
        self.load_model()
        self.load_transcriber()

    def load_model(self):
        for model_dir in MODEL_DIRS:
            try:
                if model_dir.exists():
                    print(f"Cargando modelo desde {model_dir}")
                    self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
                    self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
                    self.model = self.model.to(DEVICE)
                    self.model.eval()
                    print("Modelo ML cargado correctamente.")
                    return
            except Exception as e:
                print("Error cargando modelo:", e)
        print("No se encontro modelo entrenado. Usando heurística únicamente.")

    def load_transcriber(self):
        try:
            print("Cargando modelo de transcripción...")
            # Ajusta el constructor a tu implementación
            self.transcriber = AudioTranscriber(model_size='base')
            print("Transcriptor cargado correctamente.")
        except Exception as e:
            print("Error cargando transcriptor:", e)
            self.transcriber = None

    def predict_sentiment(self, text: str) -> Dict[str, Any]:
        if not self.model or not self.tokenizer:
            return {'label': 'Neutral', 'confidence': 0.5, 'sat_score': 0.5}
        try:
            inputs = self.tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors='pt')
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                prediction = int(torch.argmax(probs, dim=-1).item())
                confidence = float(probs[0][prediction].item())
                sat_score = float(probs[0][1].item()) if probs.shape[-1] > 1 else float(probs[0][prediction].item())
            labels = ['Insatisfecho', 'Satisfecho']
            return {'label': labels[prediction] if prediction < len(labels) else 'Neutral',
                    'confidence': confidence, 'sat_score': sat_score}
        except Exception as e:
            print('Error en predict_sentiment:', e)
            return {'label': 'Neutral', 'confidence': 0.5, 'sat_score': 0.5}

    def _build_satisfaction_html(self, sat_level: str, sat_score: float) -> str:
        """
        Badge prominente sin emojis.
        """
        level = (sat_level or 'Desconocido').lower()
        if 'satis' in level:
            color = '#2ecc71'  # verde
            label_text = 'Satisfecho'
        elif 'neutral' in level or 'neut' in level:
            color = '#f1c40f'  # amarillo
            label_text = 'Neutral'
        else:
            color = '#e74c3c'  # rojo
            label_text = 'Insatisfecho'

        pct = f"{sat_score*100:0.1f}%"
        pct_bar_width = max(2, min(100, int(sat_score * 100)))

        html = f"""
        <div style="display:flex;align-items:center;gap:14px;font-family:Arial, Helvetica, sans-serif;">
          <div style="background:{color};color:white;border-radius:10px;padding:12px 18px;font-size:17px;font-weight:800;min-width:200px;text-align:center;">
            <div style="font-size:15px">{label_text}</div>
            <div style="font-size:30px;margin-top:6px">{pct}</div>
          </div>
          <div style="flex:1">
            <div style="font-size:12px;color:#bdbdbd;margin-bottom:6px">Nivel de satisfacción (combinado)</div>
            <div style="background:#ececec;border-radius:8px;height:14px;overflow:hidden;">
              <div style="width:{pct_bar_width}%;height:100%;background:{color};"></div>
            </div>
          </div>
        </div>
        """
        return html

    def format_summary(self, result: Dict[str, Any], ml_pred: Dict[str, Any]) -> str:
        duration_seconds = result.get('duration_seconds', 0)
        duration_min = duration_seconds // 60
        duration_sec = duration_seconds % 60
        summary = f"""RESUMEN DE LA LLAMADA

Duracion: {duration_min}m {duration_sec}s
Cliente: {result.get('customer_info', {}).get('name') or 'No identificado'}
ID Cliente: {result.get('customer_info', {}).get('id') or 'No proporcionado'}

SATISFACCION DEL CLIENTE
- Nivel heurístico: {result.get('customer_satisfaction', {}).get('satisfaction_level')}
- Score heurístico: {result.get('customer_satisfaction', {}).get('score', 0):.2%}
- Predicción ML: {ml_pred.get('label')} (Confianza: {ml_pred.get('confidence', 0):.2%})

DESEMPENO DEL OPERADOR
- Nivel: {result.get('operator_performance', {}).get('performance_level')}
- Score: {result.get('operator_performance', {}).get('score', 0):.2%}
"""
        return summary

    def format_customer_analysis(self, result: dict, ml_pred: dict) -> str:
        sat = result.get('customer_satisfaction', {})
        analysis = f"""ANALISIS DEL CLIENTE

Nivel de Satisfaccion: {sat.get('satisfaction_level')}
Score: {sat.get('score', 0):.2%}

Indicadores Positivos: {sat.get('positive_indicators')}
Indicadores Negativos: {sat.get('negative_indicators')}

Prediccion del Modelo ML: {ml_pred.get('label')} (Confianza: {ml_pred.get('confidence', 0):.2%})
"""
        return analysis

    def format_operator_analysis(self, result: dict) -> str:
        perf = result.get('operator_performance', {})
        analysis = f"""ANALISIS DEL OPERADOR

Nivel de Desempeno: {perf.get('performance_level')}
Score: {perf.get('score', 0):.2%}

Frases de Cortesia: {perf.get('courtesy_count')}
Intento de Resolucion: {'Si' if perf.get('resolution_attempt') else 'No'}
"""
        return analysis

    def process_audio(self, audio_file: str):
        """
        Procesa el audio y devuelve las salidas + un objeto state (dict) con todos los datos
        State es devuelto para que el botón "Generar reporte" lo use.
        """
        default_badge = self._build_satisfaction_html('Desconocido', 0.0)
        try:
            if audio_file is None:
                return ("Error: No se proporciono archivo de audio", "", "No se pudo procesar", "", "", "", default_badge, None)
            if not self.transcriber:
                return ("Error: Modelo de transcripcion no disponible", "", "No se pudo procesar", "", "", "", default_badge, None)

            # Transcripción (tu implementacion)
            transcription_result = self.transcriber.process_audio_file(audio_file)
            formatted_text = transcription_result.get('formatted_transcription', '') or transcription_result.get('transcription', '')
            duration = transcription_result.get('duration_seconds', 0)

            extracted_data = {}
            try:
                extracted_data = DataExtractor.extract_all_data(formatted_text)
            except Exception:
                # Si DataExtractor no existe o falla, seguimos sin extraer
                extracted_data = {}

            # DB lookups
            ticket_info = ''
            customer_info_text = ''
            if extracted_data.get('tickets'):
                for ticket_id in extracted_data.get('tickets'):
                    try:
                        ticket = self.db.buscar_ticket(ticket_id)
                        if ticket:
                            ticket_info += f"TICKET: {ticket_id} - {ticket.get('estado')} - {ticket.get('producto')}\n"
                    except Exception:
                        pass
            if extracted_data.get('customer_id'):
                try:
                    customer = self.db.buscar_cliente(extracted_data.get('customer_id'))
                    if customer:
                        customer_info_text += f"CLIENTE: {customer.get('nombre')} - {customer.get('email')} - {customer.get('telefono')}\n"
                except Exception:
                    pass

            db_info = (ticket_info + "\n" + customer_info_text).strip() or 'No se encontro informacion adicional en la base de datos'
            extracted_info = f"Duracion: {duration}s\nTipo: {extracted_data.get('call_type')}\n"
            if extracted_data.get('customer_name'):
                extracted_info += f"Nombre cliente: {extracted_data.get('customer_name')}\n"
            if extracted_data.get('tickets'):
                try:
                    extracted_info += f"Tickets: {', '.join(extracted_data.get('tickets'))}\n"
                except Exception:
                    pass

            # Análisis 
            try:
                result = self.analyzer.analyze_conversation(formatted_text)
            except Exception as e:
                print("Error analyzer:", e)
                result = {}

            # ML prediction
            ml_pred = self.predict_sentiment(formatted_text)

            # combined satisfaction: prefer analyzer.get_combined_satisfaction if existe
            try:
                combined_info = self.analyzer.get_combined_satisfaction(result, ml_pred)
                sat_level = combined_info.get('level', ml_pred.get('label', 'Desconocido'))
                combined_score = float(combined_info.get('score', 0.0))
            except Exception:
                heur_score = float(result.get('customer_satisfaction', {}).get('score', 0.0) or 0.0)
                ml_score = float(ml_pred.get('sat_score', 0.0) or 0.0)
                combined_score = (heur_score + ml_score) / 2.0 if heur_score > 0 else ml_score
                sat_level = result.get('customer_satisfaction', {}).get('satisfaction_level') or ml_pred.get('label', 'Desconocido')

            summary = self.format_summary(result, ml_pred)
            customer_analysis = self.format_customer_analysis(result, ml_pred)
            operator_analysis = self.format_operator_analysis(result)

            satisfaction_html = self._build_satisfaction_html(sat_level, combined_score)

            # Preparamos el estado (state) que usará el botón generar reporte
            call_state = {
                "audio_path": audio_file,
                "transcription": formatted_text,
                "duration_seconds": duration,
                "extracted_data": extracted_data,
                "db_info": db_info,
                "analysis_result": result,
                "ml_pred": ml_pred,
                "sat_level": sat_level,
                "sat_score": combined_score,
                "summary": summary,
                "customer_analysis": customer_analysis,
                "operator_analysis": operator_analysis,
                "timestamp": datetime.now().isoformat()
            }

            # Intentar guardar la llamada en la DB (no crítico)
            try:
                llamada_id = f"CALL-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
                llamada_data = {
                    'llamada_id': llamada_id,
                    'cliente_id': extracted_data.get('customer_id'),
                    'ticket_id': extracted_data.get('tickets')[0] if extracted_data.get('tickets') else None,
                    'operador': 'Sistema',
                    'duracion': duration,
                    'satisfaccion': sat_level,
                    'transcripcion': formatted_text,
                    'analisis_completo': result
                }
                self.db.guardar_llamada(llamada_data)
            except Exception as e:
                print("Advertencia: no se pudo guardar llamada en DB:", e)

            return (formatted_text, extracted_info, db_info, summary, customer_analysis, operator_analysis, satisfaction_html, call_state)

        except Exception as e:
            tb = traceback.format_exc()
            print("Error procesando audio:", e, tb)
            return (f"Error procesando audio: {e}", "", "", "", "", "", default_badge, None)

    def save_report(self, call_state: Dict[str, Any], operator_name: str, operator_id: str):
        """
        Guarda reporte JSON en la misma carpeta que el audio:
        {audio_folder}/report_{llamada_id or timestamp}.json
        Devuelve mensaje de estado.
        """
        if not call_state:
            return "No hay datos de llamada procesados. Primero procesa un audio."

        audio_path = call_state.get('audio_path')
        if not audio_path:
            return "No se pudo determinar la ruta del audio en el estado."

        try:
            audio_path = Path(audio_path)
            audio_folder = audio_path.parent if audio_path.parent.exists() else Path.cwd()
            # file name
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            report_name = f"report_{timestamp}_{uuid.uuid4().hex[:6]}.json"
            report_path = audio_folder / report_name

            # contenido del reporte
            report_obj = {
                "report_id": uuid.uuid4().hex,
                "generated_at": datetime.now().isoformat(),
                "operator_name": operator_name or None,
                "operator_id": operator_id or None,
                "audio_file": str(audio_path.name),
                "audio_full_path": str(audio_path.resolve()),
                "duration_seconds": call_state.get("duration_seconds"),
                "sat_level": call_state.get("sat_level"),
                "sat_score": call_state.get("sat_score"),
                "ml_prediction": call_state.get("ml_pred"),
                "extracted_data": call_state.get("extracted_data"),
                "db_info": call_state.get("db_info"),
                "summary": call_state.get("summary"),
                "customer_analysis": call_state.get("customer_analysis"),
                "operator_analysis": call_state.get("operator_analysis"),
                "transcription": call_state.get("transcription")
            }

            # escribir JSON
            with open(report_path, 'w', encoding='utf-8') as fh:
                json.dump(report_obj, fh, ensure_ascii=False, indent=2)

            return f"Reporte guardado: {str(report_path)}"

        except Exception as e:
            tb = traceback.format_exc()
            print("Error guardando reporte:", e, tb)
            return f"Error guardando reporte: {e}"

    def create_interface(self):
        css = """
        <style>
        body, .gradio-container { font-family: Arial, Helvetica, sans-serif !important; }
        .gradio-container .output-textbox { font-family: Arial, sans-serif; }
        </style>
        """
        with gr.Blocks(title='Prueba A43', theme=gr.themes.Soft()) as demo:
            gr.HTML(css)
            gr.Markdown("""
            # Prueba por interfaz 

            """)
            # state: guarda el último funcion call_state retornado por process_audio
            call_state = gr.State(value=None)

            with gr.Tab('Analizar Audio'):
                audio_input = gr.Audio(label='Sube archivo de audio', type='filepath', sources=['upload'])
                process_btn = gr.Button('PROCESAR AUDIO', variant='primary')

                with gr.Row():
                    with gr.Column():
                        transcription_output = gr.Textbox(label='Transcripción', lines=10)
                        extracted_output = gr.Textbox(label='Datos Extraídos', lines=6)
                    with gr.Column():
                        db_info_output = gr.Textbox(label='Información Base de Datos', lines=6)
                        summary_output = gr.Textbox(label='Resumen', lines=8)
                        satisfaction_display = gr.HTML("<div>El nivel de satisfacción aparecerá aquí.</div>")

                with gr.Row():
                    with gr.Column():
                        customer_output = gr.Textbox(label='Análisis Cliente', lines=6)
                        operator_output = gr.Textbox(label='Análisis Operador', lines=6)

                # Bind process button: additionally devuelve el state
                process_btn.click(
                    fn=self.process_audio,
                    inputs=[audio_input],
                    outputs=[transcription_output, extracted_output, db_info_output, summary_output,
                             customer_output, operator_output, satisfaction_display, call_state]
                )

                gr.HTML("<hr/>")
                gr.Markdown("### Generar reporte")
                with gr.Row():
                    operator_name = gr.Textbox(label="Nombre del operador", placeholder="Ej: Juan Perez")
                    operator_id = gr.Textbox(label="ID del operador (opcional)", placeholder="Ej: OP-1234")
                generate_btn = gr.Button("GENERAR REPORTE", variant='secondary')
                report_status = gr.Textbox(label="Estado reporte", lines=2)

                # conectar botón para generar reporte (usa call_state)
                def _save_report_and_return_msg(state_obj, name, op_id):
                    return self.save_report(state_obj, name, op_id)

                generate_btn.click(
                    fn=_save_report_and_return_msg,
                    inputs=[call_state, operator_name, operator_id],
                    outputs=[report_status]
                )

        return demo


def main():
    app = CallCenterApp()
    demo = app.create_interface()
    demo.launch(share=False, server_name='127.0.0.1', server_port=7860, inbrowser=True)


if __name__ == '__main__':
    main()
