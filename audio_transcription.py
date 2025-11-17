"""
Modulo de transcripcion de audio usando Whisper
Identifica automaticamente hablantes (Cliente/Operador)
"""

import whisper
import torch
import numpy as np
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional

from config import DEVICE


class AudioTranscriber:
    
    def __init__(self, model_size='base'):
        print(f"Cargando modelo Whisper ({model_size})...")
        self.model = whisper.load_model(model_size)
        print("Modelo Whisper cargado")
        
    def transcribe_audio(self, audio_path: str) -> Dict:
        print(f"\nTranscribiendo audio: {audio_path}")
        
        result = self.model.transcribe(
            audio_path,
            language='es',
            task='transcribe',
            verbose=False
        )
        
        return result
    
    def identify_speakers(self, transcription: str) -> str:
        """
        Identifica y etiqueta hablantes basandose en patrones de conversacion
        """
        
        segments = transcription.split('.')
        formatted_text = ""
        
        # Patrones que indican que es el operador
        operator_patterns = [
            r'buenos? d[ií]as?',
            r'buenas? tardes?',
            r'buenas? noches?',
            r'en qu[eé] puedo ayudarle',
            r'c[oó]mo puedo asistirle',
            r'gracias por comunicarse',
            r'mi nombre es',
            r'soy [\w]+',
            r'puedo ayudarle',
            r'voy a revisar',
            r'un momento por favor',
        ]
        
        # Patrones que indican que es el cliente
        client_patterns = [
            r'quiero [\w]+',
            r'necesito [\w]+',
            r'tengo un problema',
            r'mi (nombre|id) es',
            r'soy [\w]+ [\w]+',
            r'llamo (por|para)',
            r'quisiera',
        ]
        
        is_operator_turn = True  # Normalmente el operador saluda primero
        
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            
            # Verificar patrones
            is_operator = any(re.search(pattern, segment.lower()) for pattern in operator_patterns)
            is_client = any(re.search(pattern, segment.lower()) for pattern in client_patterns)
            
            if is_operator:
                speaker = "Operador"
                is_operator_turn = False
            elif is_client:
                speaker = "Cliente"
                is_operator_turn = True
            else:
                # Alternar si no hay patron claro
                speaker = "Operador" if is_operator_turn else "Cliente"
                is_operator_turn = not is_operator_turn
            
            formatted_text += f"{speaker}: {segment}.\n"
        
        return formatted_text
    
    def process_audio_file(self, audio_path: str) -> Dict:
        """
        Procesa archivo de audio completo y retorna transcripcion formateada
        """
        
        # Transcribir
        result = self.transcribe_audio(audio_path)
        
        # Obtener texto completo
        raw_text = result['text']
        
        # Identificar hablantes
        formatted_text = self.identify_speakers(raw_text)
        
        # Calcular duracion (segundos)
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000)
        duration = len(audio) / sr
        
        return {
            'raw_transcription': raw_text,
            'formatted_transcription': formatted_text,
            'duration_seconds': int(duration),
            'language': result.get('language', 'es'),
            'segments': result.get('segments', [])
        }


class DataExtractor:
    """Extrae informacion especifica de la transcripcion"""
    
    @staticmethod
    def extract_ticket_id(text: str) -> List[str]:
        """Extrae numeros de ticket/orden de la conversacion"""
        
        patterns = [
            r'ticket[:\s]+([0-9]+)',
            r'orden[:\s]+([A-Z0-9\-]+)',
            r'pedido[:\s]+([A-Z0-9\-]+)',
            r'numero\s+(?:de\s+)?(?:ticket|orden|pedido)[:\s]+([A-Z0-9\-]+)',
            r'mi\s+ticket\s+es\s+([0-9]+)',
            r'(?:ticket|orden|pedido)\s+#?\s*([A-Z0-9\-]+)',
        ]
        
        tickets = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                ticket = match.group(1)
                if ticket not in tickets:
                    tickets.append(ticket)
        
        return tickets
    
    @staticmethod
    def extract_customer_id(text: str) -> Optional[str]:
        """Extrae ID de cliente"""
        
        patterns = [
            r'(?:mi\s+)?(?:id|identificacion|numero\s+de\s+cliente)[:\s]+([A-Z0-9\-]+)',
            r'cliente[:\s]+([A-Z]{1,3}\-[0-9]+)',
            r'soy\s+(?:el\s+)?cliente\s+([A-Z0-9\-]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    @staticmethod
    def extract_customer_name(text: str) -> Optional[str]:
        """Extrae nombre del cliente"""
        
        patterns = [
            r'(?:mi\s+nombre\s+es|me\s+llamo|soy)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'nombre[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    @staticmethod
    def extract_phone_number(text: str) -> Optional[str]:
        """Extrae numero de telefono"""
        
        patterns = [
            r'(?:mi\s+)?(?:telefono|numero)[:\s]+([0-9\-]{7,15})',
            r'(\d{3}[\-\s]?\d{3}[\-\s]?\d{4})',
            r'(\d{3}[\-\s]?\d{4})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return None
    
    @staticmethod
    def extract_product_name(text: str) -> List[str]:
        """Extrae nombres de productos mencionados"""
        
        common_products = [
            'plan premium', 'plan basico', 'plan familiar',
            'telefono', 'celular', 'smartphone',
            'router', 'modem', 'internet',
            'tablet', 'laptop', 'computadora'
        ]
        
        products = []
        text_lower = text.lower()
        
        for product in common_products:
            if product in text_lower:
                products.append(product.title())
        
        return products
    
    @staticmethod
    def classify_call_type(text: str) -> str:
        """Clasifica el tipo de llamada"""
        
        text_lower = text.lower()
        
        # Venta
        if any(word in text_lower for word in ['comprar', 'adquirir', 'contratar', 'quiero el plan']):
            return 'Venta'
        
        # Reclamo
        if any(word in text_lower for word in ['problema', 'no funciona', 'reclamo', 'queja']):
            return 'Reclamo'
        
        # Soporte
        if any(word in text_lower for word in ['ayuda', 'como', 'configurar', 'instalar']):
            return 'Soporte'
        
        # Consulta
        if any(word in text_lower for word in ['informacion', 'consulta', 'preguntar', 'saber']):
            return 'Consulta'
        
        return 'General'
    
    @staticmethod
    def extract_all_data(text: str) -> Dict:
        """Extrae todos los datos relevantes de la transcripcion"""
        
        return {
            'tickets': DataExtractor.extract_ticket_id(text),
            'customer_id': DataExtractor.extract_customer_id(text),
            'customer_name': DataExtractor.extract_customer_name(text),
            'phone_number': DataExtractor.extract_phone_number(text),
            'products': DataExtractor.extract_product_name(text),
            'call_type': DataExtractor.classify_call_type(text)
        }


def test_transcriber():
    """Prueba el transcriptor con un archivo de audio"""
    
    transcriber = AudioTranscriber(model_size='base')
    
    # Ejemplo de uso
    audio_file = "audio_samples/sample_call.mp3"
    
    if Path(audio_file).exists():
        result = transcriber.process_audio_file(audio_file)
        
        print("\nTranscripcion:")
        print(result['formatted_transcription'])
        
        print(f"\nDuracion: {result['duration_seconds']} segundos")
        
        # Extraer datos
        data = DataExtractor.extract_all_data(result['formatted_transcription'])
        print("\nDatos extraidos:")
        print(f"  Tickets: {data['tickets']}")
        print(f"  Cliente ID: {data['customer_id']}")
        print(f"  Nombre: {data['customer_name']}")
        print(f"  Tipo de llamada: {data['call_type']}")
    else:
        print(f"Archivo no encontrado: {audio_file}")


if __name__ == "__main__":
    test_transcriber()