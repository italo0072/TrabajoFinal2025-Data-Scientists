"""
enroll_ui.py
Interfaz Gradio para enrolar operadores desde segmentos de una transcripción.
Compatible con Gradio sin usar source= en Audio.
"""

import gradio as gr
import pandas as pd
import os
from src.audio.audio_transcription import AudioTranscriber
from embedding_utils import save_audio_clip_from_file, compute_embedding_stub, enroll_operator_record, OPERATORS_DIR

os.makedirs("data", exist_ok=True)
os.makedirs(OPERATORS_DIR, exist_ok=True)
OPERATORS_CSV = "data/operators.csv"

# Inicializa transcriber. Si usas Whisper y está instalado, se inicializará automáticamente.
transcriber = AudioTranscriber(role_model_dir=None)  # si quieres role classifier pon path

def process_audio_for_ui(audio_file_path):
    if not audio_file_path:
        return "No hay audio", pd.DataFrame(columns=["index","start","end","text"])
    try:
        result = transcriber.process_audio_file(audio_file_path)
    except Exception as e:
        return f"Error en transcripción: {e}", pd.DataFrame(columns=["index","start","end","text"])
    segments = result.get("asr_result", {}).get("segments", [])
    rows = []
    for i, seg in enumerate(segments):
        rows.append({"index": i, "start": float(seg.get("start", 0.0)), "end": float(seg.get("end", 0.0)), "text": seg.get("text", "").strip()})
    df = pd.DataFrame(rows)
    full_text = " ".join([s.get("text","") for s in segments])
    return full_text, df

def enroll_selected_segment(audio_file_path, segment_index, operator_id, operator_name):
    if not audio_file_path:
        return "No hay audio", load_operators_df()
    try:
        result = transcriber.process_audio_file(audio_file_path)
    except Exception as e:
        return f"Error al procesar audio: {e}", load_operators_df()
    segments = result.get("asr_result", {}).get("segments", [])
    try:
        segment_index = int(segment_index)
    except Exception:
        return "Índice inválido", load_operators_df()
    if segment_index < 0 or segment_index >= len(segments):
        return "Índice fuera de rango", load_operators_df()
    seg = segments[segment_index]
    start, end = float(seg.get("start", 0.0)), float(seg.get("end", 0.0))

    # Guardar clip: usar método del transcriber si existe (save_segment_audio), sino fallback
    clip_path = None
    try:
        if hasattr(transcriber, "save_segment_audio"):
            clip_path = transcriber.save_segment_audio(audio_file_path, start, end, out_path=None)
            if not clip_path:
                raise Exception("save_segment_audio devolvió None")
        else:
            raise AttributeError("No hay save_segment_audio en transcriber")
    except Exception:
        clip_path = save_audio_clip_from_file(audio_file_path, start, end)

    emb_path = compute_embedding_stub(clip_path)
    enroll_operator_record(operator_id.strip(), operator_name.strip(), clip_path, emb_path, csv_path=OPERATORS_CSV)
    msg = f"Operador {operator_name} ({operator_id}) enrolado. Clip: {clip_path}"
    return msg, load_operators_df()

def load_operators_df():
    if not os.path.exists(OPERATORS_CSV):
        return pd.DataFrame(columns=["operator_id","name","clip_path","embedding_path","created_at"])
    return pd.read_csv(OPERATORS_CSV)

def delete_operator(operator_id):
    df = load_operators_df()
    df2 = df[df["operator_id"] != operator_id]
    df2.to_csv(OPERATORS_CSV, index=False)
    return df2

# --- Interfaz Gradio ---
with gr.Blocks() as demo:
    gr.Markdown("## Enrolamiento de Operadores - Call Center")
    with gr.Row():
        audio_in = gr.Audio(type="filepath", label="Sube grabación de llamada")
        btn_process = gr.Button("Procesar y mostrar segmentos")
    out_text = gr.Textbox(label="Transcripción completa", lines=4)
    segments_table = gr.Dataframe(label="Segmentos detectados (index,start,end,text)")
    with gr.Row():
        seg_index = gr.Number(label="Índice segmento a enrolar (columna index)", value=0, precision=0)
        op_id = gr.Textbox(label="Operator ID (ej: emp123)")
        op_name = gr.Textbox(label="Nombre del operador")
        btn_enroll = gr.Button("Enrolar operador desde segmento")
    out_msg = gr.Textbox(label="Mensaje de enrolamiento")
    operators_df = gr.Dataframe(value=load_operators_df(), label="Operadores inscritos")
    delete_id = gr.Textbox(label="Eliminar operator_id")
    btn_delete = gr.Button("Eliminar operador")

    btn_process.click(fn=process_audio_for_ui, inputs=[audio_in], outputs=[out_text, segments_table])
    btn_enroll.click(fn=enroll_selected_segment, inputs=[audio_in, seg_index, op_id, op_name], outputs=[out_msg, operators_df])
    btn_delete.click(fn=lambda x: (f"Operador {x} eliminado", delete_operator(x)), inputs=[delete_id], outputs=[out_msg, operators_df])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
