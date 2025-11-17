"""
Interfaz Gradio para analisis de llamadas de call center
"""

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from datetime import datetime
import uuid

from config import MODELS_DIR, DEVICE
from conversation_analyzer import ConversationAnalyzer
from audio_transcription import AudioTranscriber, DataExtractor
from database import CallCenterDatabase


class CallCenterApp:
    
    def __init__(self):
        self.analyzer = ConversationAnalyzer()
        self.model = None
        self.tokenizer = None
        self.history = []
        self.transcriber = None
        self.db = CallCenterDatabase()
        self.load_model()
        self.load_transcriber()
    
    def load_model(self):
        model_paths = [
            MODELS_DIR / 'call_center_model',
            MODELS_DIR / 'finetuned_model'
        ]
        
        for model_dir in model_paths:
            if model_dir.exists():
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
                    self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
                    self.model = self.model.to(DEVICE)
                    self.model.eval()
                    print(f"Modelo cargado desde {model_dir}")
                    return
                except Exception as e:
                    print(f"Error cargando modelo desde {model_dir}: {e}")
        
        print("No se encontro modelo entrenado. Usando solo analisis de reglas.")
    
    def load_transcriber(self):
        """Cargar modelo de transcripcion Whisper"""
        try:
            print("Cargando modelo de transcripcion...")
            self.transcriber = AudioTranscriber(model_size='base')
            print("Transcriptor cargado correctamente")
        except Exception as e:
            print(f"Error cargando transcriptor: {e}")
            print("Transcripcion de audio no estara disponible")
    
    def predict_sentiment(self, text: str) -> dict:
        if not self.model or not self.tokenizer:
            return {'label': 'Neutral', 'confidence': 0.5}
        
        try:
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors='pt'
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                prediction = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][prediction].item()
            
            labels = ['Insatisfecho', 'Satisfecho']
            
            return {
                'label': labels[prediction] if prediction < len(labels) else 'Neutral',
                'confidence': confidence
            }
        except Exception as e:
            print(f"Error en prediccion: {e}")
            return {'label': 'Neutral', 'confidence': 0.5}
    
    def analyze_text(self, text: str) -> tuple:
        if not text or len(text.strip()) < 10:
            return (
                "Error: El texto es muy corto. Por favor ingresa una conversacion completa.",
                "", "", "", "", ""
            )
        
        result = self.analyzer.analyze_conversation(text)
        ml_prediction = self.predict_sentiment(text)
        
        self.history.append({
            'timestamp': result['timestamp'],
            'customer_id': result['customer_info']['id'],
            'satisfaction': result['customer_satisfaction']['satisfaction_level'],
            'operator_performance': result['operator_performance']['performance_level'],
            'is_sale': result['sale_info']['is_sale']
        })
        
        summary = self.format_summary(result, ml_prediction)
        customer_analysis = self.format_customer_analysis(result, ml_prediction)
        operator_analysis = self.format_operator_analysis(result)
        sale_analysis = self.format_sale_analysis(result)
        timeline = self.format_timeline(result)
        feedback = result['operator_feedback']
        
        return (
            summary,
            customer_analysis,
            operator_analysis,
            sale_analysis,
            timeline,
            feedback
        )
    
    def format_summary(self, result: dict, ml_pred: dict) -> str:
        duration_min = result['duration_seconds'] // 60
        duration_sec = result['duration_seconds'] % 60
        
        summary = f"""RESUMEN DE LA LLAMADA

Duracion: {duration_min}m {duration_sec}s
Cliente: {result['customer_info']['name'] or 'No identificado'}
ID Cliente: {result['customer_info']['id'] or 'No proporcionado'}

SATISFACCION DEL CLIENTE
- Nivel: {result['customer_satisfaction']['satisfaction_level']}
- Score: {result['customer_satisfaction']['score']:.2%}
- Prediccion ML: {ml_pred['label']} (Confianza: {ml_pred['confidence']:.2%})

DESEMPENO DEL OPERADOR
- Nivel: {result['operator_performance']['performance_level']}
- Score: {result['operator_performance']['score']:.2%}

RESULTADO DE VENTA
- Se concreto venta: {'Si' if result['sale_info']['is_sale'] else 'No'}
- Confianza: {result['sale_info']['confidence']:.1f}%"""
        
        return summary
    
    def format_customer_analysis(self, result: dict, ml_pred: dict) -> str:
        sat = result['customer_satisfaction']
        
        analysis = f"""ANALISIS DEL CLIENTE

Nivel de Satisfaccion: {sat['satisfaction_level']}
Score: {sat['score']:.2%}

Indicadores Positivos: {sat['positive_indicators']}
Indicadores Negativos: {sat['negative_indicators']}

Prediccion del Modelo ML: {ml_pred['label']}
Confianza del Modelo: {ml_pred['confidence']:.2%}

INTERPRETACION:
"""
        
        if sat['satisfaction_level'] == 'Satisfecho':
            analysis += "El cliente mostro una actitud positiva durante la llamada. Se recomienda seguimiento para mantener la relacion."
        elif sat['satisfaction_level'] == 'Neutral':
            analysis += "El cliente no mostro emociones fuertes. Se puede mejorar la experiencia con un trato mas personalizado."
        else:
            analysis += "El cliente mostro insatisfaccion. ACCION REQUERIDA: Contactar al cliente para resolver problemas."
        
        return analysis
    
    def format_operator_analysis(self, result: dict) -> str:
        perf = result['operator_performance']
        
        analysis = f"""ANALISIS DEL OPERADOR

Nivel de Desempeno: {perf['performance_level']}
Score: {perf['score']:.2%}

Frases de Cortesia: {perf['courtesy_count']}
Intento de Resolucion: {'Si' if perf['resolution_attempt'] else 'No'}

EVALUACION:
"""
        
        if perf['score'] >= 0.7:
            analysis += "Excelente desempeno. El operador mantuvo profesionalismo y cortesia."
        elif perf['score'] >= 0.5:
            analysis += "Buen desempeno general. Hay areas de mejora en cortesia y resolucion."
        elif perf['score'] >= 0.3:
            analysis += "Desempeno regular. Se requiere capacitacion adicional."
        else:
            analysis += "Desempeno deficiente. Se requiere intervencion inmediata y capacitacion."
        
        return analysis
    
    def format_sale_analysis(self, result: dict) -> str:
        sale = result['sale_info']
        
        analysis = f"""ANALISIS DE VENTA

Resultado: {'VENTA CONCRETADA' if sale['is_sale'] else 'SIN VENTA'}
Confianza: {sale['confidence']:.1f}%
Indicadores de Venta: {sale['indicators_found']}

"""
        
        if sale['is_sale']:
            analysis += """FELICITACIONES! Se logro concretar la venta.

Acciones de Seguimiento:
- Enviar confirmacion al cliente
- Registrar en sistema CRM
- Programar seguimiento post-venta"""
        else:
            analysis += """No se concreto una venta en esta llamada.

Recomendaciones:
- Analizar objeciones del cliente
- Mejorar tecnicas de cierre
- Programar llamada de seguimiento"""
        
        return analysis
    
    def format_timeline(self, result: dict) -> str:
        turns = result['turns']
        
        timeline = "LINEA DE TIEMPO DE LA CONVERSACION\n\n"
        
        for i, turn in enumerate(turns, 1):
            timeline += f"[{i}] {turn['role']}: {turn['content'][:100]}"
            if len(turn['content']) > 100:
                timeline += "..."
            timeline += "\n\n"
        
        return timeline
    
    def process_audio(self, audio_file) -> tuple:
        """Procesar archivo de audio y analizar"""
        
        if audio_file is None:
            return ("Error: No se proporciono archivo de audio", "", "", "", "", "", "", "")
        
        if not self.transcriber:
            return ("Error: Modelo de transcripcion no disponible", "", "", "", "", "", "", "")
        
        try:
            # Transcribir audio
            print(f"\nProcesando audio: {audio_file}")
            transcription_result = self.transcriber.process_audio_file(audio_file)
            
            formatted_text = transcription_result['formatted_transcription']
            duration = transcription_result['duration_seconds']
            
            # Extraer datos automaticamente
            extracted_data = DataExtractor.extract_all_data(formatted_text)
            
            # Buscar en base de datos
            db_info = ""
            ticket_info = ""
            customer_info_text = ""
            
            # Buscar tickets mencionados
            if extracted_data['tickets']:
                for ticket_id in extracted_data['tickets']:
                    ticket = self.db.buscar_ticket(ticket_id)
                    if ticket:
                        ticket_info += f"\nTICKET ENCONTRADO: {ticket_id}\n"
                        ticket_info += f"  Tipo: {ticket['tipo']}\n"
                        ticket_info += f"  Estado: {ticket['estado']}\n"
                        ticket_info += f"  Producto: {ticket['producto']}\n"
                        ticket_info += f"  Descripcion: {ticket['descripcion']}\n"
                        if ticket['monto']:
                            ticket_info += f"  Monto: ${ticket['monto']:.2f}\n"
                        ticket_info += f"  Cliente: {ticket['cliente_nombre']}\n"
            
            # Buscar cliente
            if extracted_data['customer_id']:
                customer = self.db.buscar_cliente(extracted_data['customer_id'])
                if customer:
                    customer_info_text += f"\nCLIENTE ENCONTRADO: {extracted_data['customer_id']}\n"
                    customer_info_text += f"  Nombre: {customer['nombre']}\n"
                    customer_info_text += f"  Email: {customer['email']}\n"
                    customer_info_text += f"  Telefono: {customer['telefono']}\n"
                    customer_info_text += f"  Tipo: {customer['tipo_cliente']}\n"
            
            # Informacion extraida
            extracted_info = "DATOS EXTRAIDOS AUTOMATICAMENTE\n\n"
            extracted_info += f"Duracion de llamada: {duration} segundos ({duration//60}m {duration%60}s)\n"
            extracted_info += f"Tipo de llamada: {extracted_data['call_type']}\n"
            
            if extracted_data['customer_name']:
                extracted_info += f"Nombre cliente: {extracted_data['customer_name']}\n"
            if extracted_data['customer_id']:
                extracted_info += f"ID cliente: {extracted_data['customer_id']}\n"
            if extracted_data['phone_number']:
                extracted_info += f"Telefono: {extracted_data['phone_number']}\n"
            if extracted_data['tickets']:
                extracted_info += f"Tickets mencionados: {', '.join(extracted_data['tickets'])}\n"
            if extracted_data['products']:
                extracted_info += f"Productos: {', '.join(extracted_data['products'])}\n"
            
            # Analizar conversacion
            summary, customer_analysis, operator_analysis, sale_analysis, timeline, feedback = self.analyze_text(formatted_text)
            
            # Guardar en base de datos
            llamada_id = f"CALL-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
            
            result = self.analyzer.analyze_conversation(formatted_text)
            
            llamada_data = {
                'llamada_id': llamada_id,
                'cliente_id': extracted_data['customer_id'],
                'ticket_id': extracted_data['tickets'][0] if extracted_data['tickets'] else None,
                'operador': 'Sistema',
                'duracion': duration,
                'satisfaccion': result['customer_satisfaction']['satisfaction_level'],
                'desempeno_operador': result['operator_performance']['performance_level'],
                'es_venta': result['sale_info']['is_sale'],
                'transcripcion': formatted_text,
                'analisis_completo': result
            }
            
            self.db.guardar_llamada(llamada_data)
            
            # Combinar informacion de base de datos
            if ticket_info or customer_info_text:
                db_info = "INFORMACION DE BASE DE DATOS\n\n" + ticket_info + customer_info_text
            else:
                db_info = "No se encontro informacion adicional en la base de datos"
            
            return (
                formatted_text,
                extracted_info,
                db_info,
                summary,
                customer_analysis,
                operator_analysis,
                sale_analysis,
                feedback
            )
        
        except Exception as e:
            import traceback
            error_msg = f"Error procesando audio: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return (error_msg, "", "", "", "", "", "", "")
    
    def check_history(self, customer_id: str, customer_name: str) -> str:
        if not customer_id and not customer_name:
            return "Por favor proporciona ID de cliente o nombre para buscar historial."
        
        matches = []
        for record in self.history:
            if customer_id and record.get('customer_id') == customer_id:
                matches.append(record)
            elif customer_name and customer_name.lower() in str(record).lower():
                matches.append(record)
        
        if not matches:
            return f"No se encontro historial para el cliente {customer_name or customer_id}"
        
        history_text = f"HISTORIAL DEL CLIENTE {customer_name or customer_id}\n\n"
        history_text += f"Total de llamadas: {len(matches)}\n\n"
        
        for i, record in enumerate(matches, 1):
            history_text += f"Llamada {i}:\n"
            history_text += f"  Fecha: {record['timestamp']}\n"
            history_text += f"  Satisfaccion: {record['satisfaction']}\n"
            history_text += f"  Desempeno Operador: {record['operator_performance']}\n"
            history_text += f"  Venta: {'Si' if record['is_sale'] else 'No'}\n\n"
        
        return history_text
    
    def create_interface(self):
        
        with gr.Blocks(title="Analisis de Call Center", theme=gr.themes.Soft()) as demo:
            gr.Markdown("""
            # Sistema de Analisis de Llamadas de Call Center
            
            Analiza llamadas automaticamente mediante:
            - Transcripcion automatica de audio a texto
            - Identificacion de Cliente y Operador
            - Extraccion automatica de datos (ticket, cliente, productos)
            - Consulta a base de datos en tiempo real
            - Analisis de satisfaccion y desempeno
            - Feedback automatico para operadores
            """)
            
            with gr.Tab("1. ANALIZAR AUDIO - PRINCIPAL"):
                gr.Markdown("""
                ### SUBIR ARCHIVO DE AUDIO DE LLAMADA
                
                INSTRUCCIONES:
                1. Haz clic en el recuadro de abajo para subir un archivo de audio
                2. Selecciona tu archivo (MP3, WAV, M4A, OGG)
                3. Presiona el boton "PROCESAR AUDIO"
                4. Espera mientras se transcribe y analiza (puede tardar 10-30 segundos)
                
                El sistema automaticamente:
                - Transcribe el audio a texto
                - Identifica quien es Cliente y quien es Operador
                - Extrae numeros de ticket, IDs de cliente, productos mencionados
                - Consulta la base de datos y muestra informacion relevante
                - Analiza satisfaccion del cliente y desempeno del operador
                - Genera feedback automatico para el operador
                
                Ejemplo: Si el cliente dice "mi ticket es 556", el sistema buscara
                ese ticket en la base de datos y mostrara toda la informacion.
                """)
                
                audio_input = gr.Audio(
                    label="HAZ CLIC AQUI PARA SUBIR AUDIO",
                    type="filepath",
                    sources=["upload"],
                    elem_id="audio_upload"
                )
                
                gr.Markdown("**Formatos soportados:** MP3, WAV, M4A, OGG, FLAC")
                
                process_audio_btn = gr.Button(
                    "PROCESAR AUDIO Y ANALIZAR LLAMADA",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown("### Resultados del Analisis")
                
                with gr.Row():
                    with gr.Column():
                        transcription_output = gr.Textbox(
                            label="Transcripcion (con identificacion de hablantes)",
                            lines=10
                        )
                        extracted_data_output = gr.Textbox(
                            label="Datos Extraidos Automaticamente",
                            lines=8
                        )
                    
                    with gr.Column():
                        db_info_output = gr.Textbox(
                            label="Informacion de Base de Datos",
                            lines=10
                        )
                        audio_summary_output = gr.Textbox(
                            label="Resumen del Analisis",
                            lines=8
                        )
                
                with gr.Row():
                    with gr.Column():
                        audio_customer_output = gr.Textbox(
                            label="Analisis del Cliente",
                            lines=6
                        )
                        audio_operator_output = gr.Textbox(
                            label="Analisis del Operador",
                            lines=6
                        )
                    
                    with gr.Column():
                        audio_sale_output = gr.Textbox(
                            label="Analisis de Venta",
                            lines=6
                        )
                        audio_feedback_output = gr.Textbox(
                            label="Feedback para el Operador",
                            lines=6
                        )
                
                process_audio_btn.click(
                    fn=self.process_audio,
                    inputs=[audio_input],
                    outputs=[
                        transcription_output,
                        extracted_data_output,
                        db_info_output,
                        audio_summary_output,
                        audio_customer_output,
                        audio_operator_output,
                        audio_sale_output,
                        audio_feedback_output
                    ]
                )
            
            with gr.Tab("Analizar Conversacion"):
                gr.Markdown("### Ingresa el texto de la conversacion")
                gr.Markdown("Formato: 'Operador: ... Cliente: ...'")
                
                text_input = gr.Textbox(
                    label="Texto de la conversacion",
                    placeholder="Operador: Buenas tardes, en que puedo ayudarle?\nCliente: Hola, quiero hacer una compra...",
                    lines=10
                )
                
                analyze_btn = gr.Button("Analizar Llamada", variant="primary")
                
                with gr.Row():
                    with gr.Column():
                        summary_output = gr.Textbox(label="Resumen General", lines=12)
                        customer_output = gr.Textbox(label="Analisis del Cliente", lines=8)
                    
                    with gr.Column():
                        operator_output = gr.Textbox(label="Analisis del Operador", lines=8)
                        sale_output = gr.Textbox(label="Analisis de Venta", lines=8)
                
                timeline_output = gr.Textbox(label="Linea de Tiempo", lines=10)
                feedback_output = gr.Textbox(label="Feedback para el Operador", lines=5)
                
                analyze_btn.click(
                    fn=self.analyze_text,
                    inputs=[text_input],
                    outputs=[
                        summary_output,
                        customer_output,
                        operator_output,
                        sale_output,
                        timeline_output,
                        feedback_output
                    ]
                )
            
            with gr.Tab("Consultar Historial"):
                gr.Markdown("### Buscar historial de cliente")
                
                with gr.Row():
                    id_input = gr.Textbox(label="ID del Cliente", placeholder="CR-12345")
                    name_input = gr.Textbox(label="Nombre del Cliente", placeholder="Carlos Rodriguez")
                
                search_btn = gr.Button("Buscar Historial", variant="primary")
                history_output = gr.Textbox(label="Historial", lines=15)
                
                search_btn.click(
                    fn=self.check_history,
                    inputs=[id_input, name_input],
                    outputs=[history_output]
                )
            
            with gr.Tab("Consultar Base de Datos"):
                gr.Markdown("### Consultar tickets y clientes")
                
                with gr.Tabs():
                    with gr.Tab("Buscar Ticket"):
                        ticket_id_input = gr.Textbox(label="Numero de Ticket", placeholder="556")
                        search_ticket_btn = gr.Button("Buscar Ticket")
                        ticket_result = gr.Textbox(label="Resultado", lines=10)
                        
                        def search_ticket_fn(ticket_id):
                            if not ticket_id:
                                return "Por favor ingresa un numero de ticket"
                            
                            ticket = self.db.buscar_ticket(ticket_id)
                            if ticket:
                                result = f"TICKET: {ticket['ticket_id']}\n\n"
                                result += f"Cliente: {ticket['cliente_nombre']}\n"
                                result += f"Email: {ticket['cliente_email']}\n"
                                result += f"Tipo: {ticket['tipo']}\n"
                                result += f"Estado: {ticket['estado']}\n"
                                result += f"Producto: {ticket['producto']}\n"
                                if ticket['monto']:
                                    result += f"Monto: ${ticket['monto']:.2f}\n"
                                result += f"Descripcion: {ticket['descripcion']}\n"
                                result += f"Fecha creacion: {ticket['fecha_creacion']}\n"
                                result += f"Ultima actualizacion: {ticket['fecha_actualizacion']}\n"
                                return result
                            else:
                                return f"No se encontro el ticket {ticket_id}"
                        
                        search_ticket_btn.click(
                            fn=search_ticket_fn,
                            inputs=[ticket_id_input],
                            outputs=[ticket_result]
                        )
                    
                    with gr.Tab("Buscar Cliente"):
                        client_id_input = gr.Textbox(label="ID de Cliente", placeholder="CR-12345")
                        search_client_btn = gr.Button("Buscar Cliente")
                        client_result = gr.Textbox(label="Resultado", lines=10)
                        
                        def search_client_fn(client_id):
                            if not client_id:
                                return "Por favor ingresa un ID de cliente"
                            
                            cliente = self.db.buscar_cliente(client_id)
                            if cliente:
                                result = f"CLIENTE: {cliente['cliente_id']}\n\n"
                                result += f"Nombre: {cliente['nombre']}\n"
                                result += f"Email: {cliente['email']}\n"
                                result += f"Telefono: {cliente['telefono']}\n"
                                result += f"Tipo: {cliente['tipo_cliente']}\n"
                                result += f"Fecha registro: {cliente['fecha_registro']}\n"
                                return result
                            else:
                                return f"No se encontro el cliente {client_id}"
                        
                        search_client_btn.click(
                            fn=search_client_fn,
                            inputs=[client_id_input],
                            outputs=[client_result]
                        )
                    
                    with gr.Tab("Tickets Pendientes"):
                        refresh_btn = gr.Button("Actualizar Lista")
                        pending_tickets = gr.Textbox(label="Tickets Pendientes", lines=15)
                        
                        def list_pending_fn():
                            tickets = self.db.listar_tickets_pendientes()
                            if not tickets:
                                return "No hay tickets pendientes"
                            
                            result = f"TICKETS PENDIENTES ({len(tickets)})\n\n"
                            for ticket in tickets:
                                result += f"Ticket: {ticket['ticket_id']}\n"
                                result += f"  Cliente: {ticket['cliente_nombre']}\n"
                                result += f"  Tipo: {ticket['tipo']}\n"
                                result += f"  Estado: {ticket['estado']}\n"
                                result += f"  Descripcion: {ticket['descripcion']}\n\n"
                            return result
                        
                        refresh_btn.click(
                            fn=list_pending_fn,
                            outputs=[pending_tickets]
                        )
                        
                        # Cargar automaticamente al abrir
                        demo.load(fn=list_pending_fn, outputs=[pending_tickets])
            
            with gr.Tab("Ejemplos"):
                gr.Markdown("""
                ### Ejemplos de conversaciones
                
                **Ejemplo 1: Venta Exitosa**
                ```
                Operador: Buenas tardes, gracias por comunicarse. Mi nombre es Ana, en que puedo ayudarle?
                Cliente: Hola, quiero comprar un plan de telefono.
                Operador: Perfecto! Por favor, podria decirme su nombre y numero de cliente?
                Cliente: Soy Carlos Rodriguez, mi ID es CR-12345.
                Operador: Gracias senor Rodriguez. Le ofrezco nuestro plan premium por $50 al mes.
                Cliente: Me parece bien, quiero contratarlo.
                Operador: Excelente! Su pedido ha sido confirmado. Numero de orden: ORD-789.
                Cliente: Muchas gracias por su ayuda!
                ```
                
                **Ejemplo 2: Cliente Insatisfecho**
                ```
                Operador: Buenas tardes, como puedo ayudarle?
                Cliente: Tengo un problema grave con mi servicio. No funciona desde ayer.
                Operador: Entiendo. Cual es su numero de cliente?
                Cliente: Es CLI-555. Esto es terrible, necesito que lo arreglen ya.
                Operador: Voy a revisar su cuenta.
                Cliente: Llevo esperando demasiado. Pesimo servicio.
                Operador: Lamento la situacion. Enviare un tecnico manana.
                Cliente: Manana? No es aceptable. Muy insatisfecho.
                ```
                """)
        
        return demo


def main():
    app = CallCenterApp()
    demo = app.create_interface()
    
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True
    )


if __name__ == "__main__":
    main()