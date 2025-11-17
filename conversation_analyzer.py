"""
Modulo de analisis de conversaciones de call center
"""

import re
from datetime import datetime
from typing import Dict, List
from config import CALL_CENTER_CONFIG


class ConversationAnalyzer:
    
    def __init__(self):
        self.config = CALL_CENTER_CONFIG
        self.sale_keywords = self.config['sale_keywords']
        self.positive_keywords = self.config['positive_keywords']
        self.negative_keywords = self.config['negative_keywords']
    
    def separate_speakers(self, text: str) -> List[Dict[str, str]]:
        turns = []
        
        patterns = [
            r'(Cliente|Client|Customer):\s*(.+?)(?=(?:Operador|Agent|Operator):|$)',
            r'(Operador|Agent|Operator):\s*(.+?)(?=(?:Cliente|Client|Customer):|$)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                speaker = match.group(1)
                content = match.group(2).strip()
                
                if speaker.lower() in ['cliente', 'client', 'customer']:
                    role = 'Cliente'
                else:
                    role = 'Operador'
                
                turns.append({
                    'role': role,
                    'content': content,
                    'timestamp': None
                })
        
        return turns
    
    def detect_sale(self, conversation: str) -> Dict[str, any]:
        text_lower = conversation.lower()
        
        sale_indicators = 0
        for keyword in self.sale_keywords:
            if keyword in text_lower:
                sale_indicators += 1
        
        confirmation_patterns = [
            r'(confirmo|confirmed|pedido\s+realizado|order\s+placed)',
            r'(gracias\s+por\s+su\s+compra|thank\s+you\s+for\s+your\s+purchase)',
            r'(numero\s+de\s+orden|order\s+number|pedido\s+#)',
        ]
        
        has_confirmation = any(
            re.search(pattern, text_lower) 
            for pattern in confirmation_patterns
        )
        
        is_sale = sale_indicators >= 2 or has_confirmation
        confidence = min((sale_indicators / 5) * 100, 100)
        
        return {
            'is_sale': is_sale,
            'confidence': confidence,
            'indicators_found': sale_indicators
        }
    
    def calculate_duration(self, turns: List[Dict]) -> int:
        if not turns:
            return 0
        
        avg_turn_duration = 3
        estimated_duration = len(turns) * avg_turn_duration
        
        return estimated_duration
    
    def analyze_customer_satisfaction(self, turns: List[Dict]) -> Dict[str, any]:
        customer_turns = [t for t in turns if t['role'] == 'Cliente']
        
        if not customer_turns:
            return {
                'satisfaction_level': 'Unknown',
                'score': 0.5,
                'positive_indicators': 0,
                'negative_indicators': 0
            }
        
        positive_count = 0
        negative_count = 0
        
        for turn in customer_turns:
            content = turn['content'].lower()
            
            for keyword in self.positive_keywords:
                if keyword in content:
                    positive_count += 1
            
            for keyword in self.negative_keywords:
                if keyword in content:
                    negative_count += 1
        
        total_indicators = positive_count + negative_count
        if total_indicators == 0:
            score = 0.5
        else:
            score = positive_count / total_indicators
        
        if score >= 0.7:
            level = 'Satisfecho'
        elif score >= 0.4:
            level = 'Neutral'
        else:
            level = 'Insatisfecho'
        
        return {
            'satisfaction_level': level,
            'score': score,
            'positive_indicators': positive_count,
            'negative_indicators': negative_count
        }
    
    def analyze_operator_performance(self, turns: List[Dict]) -> Dict[str, any]:
        operator_turns = [t for t in turns if t['role'] == 'Operador']
        
        if not operator_turns:
            return {
                'performance_level': 'Unknown',
                'score': 0.5,
                'courtesy_count': 0,
                'resolution_attempt': False
            }
        
        courtesy_count = 0
        resolution_count = 0
        
        courtesy_keywords = self.config['operator_performance_metrics']['courtesy_keywords']
        resolution_keywords = self.config['operator_performance_metrics']['resolution_keywords']
        
        for turn in operator_turns:
            content = turn['content'].lower()
            
            for keyword in courtesy_keywords:
                if keyword in content:
                    courtesy_count += 1
            
            for keyword in resolution_keywords:
                if keyword in content:
                    resolution_count += 1
        
        total_turns = len(operator_turns)
        courtesy_ratio = min(courtesy_count / total_turns, 1.0) if total_turns > 0 else 0
        has_resolution = resolution_count > 0
        
        score = (courtesy_ratio * 0.6) + (0.4 if has_resolution else 0)
        
        if score >= 0.7:
            level = 'Excelente'
        elif score >= 0.5:
            level = 'Bueno'
        elif score >= 0.3:
            level = 'Regular'
        else:
            level = 'Necesita Mejorar'
        
        return {
            'performance_level': level,
            'score': score,
            'courtesy_count': courtesy_count,
            'resolution_attempt': has_resolution
        }
    
    def extract_customer_info(self, turns: List[Dict]) -> Dict[str, str]:
        info = {
            'name': None,
            'id': None
        }
        
        name_patterns = [
            r'(?:me\s+llamo|my\s+name\s+is|soy)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'(?:nombre|name):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
        ]
        
        id_patterns = [
            r'(?:id|identificacion|numero\s+de\s+cliente):\s*([A-Z0-9\-]+)',
            r'(?:mi\s+id\s+es|my\s+id\s+is)\s+([A-Z0-9\-]+)'
        ]
        
        full_text = ' '.join([t['content'] for t in turns])
        
        for pattern in name_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                info['name'] = match.group(1)
                break
        
        for pattern in id_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                info['id'] = match.group(1)
                break
        
        return info
    
    def generate_operator_feedback(
        self,
        performance: Dict,
        satisfaction: Dict,
        sale: Dict
    ) -> str:
        feedback_parts = []
        
        if performance['score'] >= 0.7:
            feedback_parts.append("Excelente trabajo! Mantuviste un tono profesional y cortez.")
        elif performance['score'] >= 0.5:
            feedback_parts.append("Buen trabajo. Considera usar mas frases de cortesia.")
        else:
            feedback_parts.append("Necesitas mejorar tu trato con el cliente. Se mas amable y profesional.")
        
        if satisfaction['satisfaction_level'] == 'Satisfecho':
            feedback_parts.append("El cliente quedo satisfecho con el servicio.")
        elif satisfaction['satisfaction_level'] == 'Neutral':
            feedback_parts.append("El cliente mostro una actitud neutral. Intenta mejorar su experiencia.")
        else:
            feedback_parts.append("El cliente mostro insatisfaccion. Revisa que salio mal y como mejorar.")
        
        if sale['is_sale']:
            feedback_parts.append("Felicidades! Se concreto una venta.")
        else:
            feedback_parts.append("No se concreto una venta. Considera tecnicas de persuasion etica.")
        
        if performance['courtesy_count'] < 2:
            feedback_parts.append("Consejo: Usa mas frases como 'por favor', 'gracias', 'disculpe'.")
        
        if not performance['resolution_attempt']:
            feedback_parts.append("Consejo: Asegurate de confirmar si se resolvio el problema del cliente.")
        
        return " ".join(feedback_parts)
    
    def analyze_conversation(self, text: str) -> Dict[str, any]:
        turns = self.separate_speakers(text)
        
        if not turns:
            turns = [{'role': 'Unknown', 'content': text, 'timestamp': None}]
        
        sale_info = self.detect_sale(text)
        duration = self.calculate_duration(turns)
        customer_satisfaction = self.analyze_customer_satisfaction(turns)
        operator_performance = self.analyze_operator_performance(turns)
        customer_info = self.extract_customer_info(turns)
        
        feedback = self.generate_operator_feedback(
            operator_performance,
            customer_satisfaction,
            sale_info
        )
        
        return {
            'turns': turns,
            'sale_info': sale_info,
            'duration_seconds': duration,
            'customer_satisfaction': customer_satisfaction,
            'operator_performance': operator_performance,
            'customer_info': customer_info,
            'operator_feedback': feedback,
            'timestamp': datetime.now().isoformat()
        }