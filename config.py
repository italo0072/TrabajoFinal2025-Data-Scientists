"""
Configuracion del Sistema de Analisis de Call Center
"""

import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
AUDIO_DIR = PROJECT_ROOT / "audio_samples"

for directory in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR, AUDIO_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

DATASET_CONFIG = {
    'name': 'amazon_polarity',
    'train_samples': 50000,
    'test_samples': 10000,
    'text_column': 'content',
    'label_column': 'label',
    'num_labels': 2,
    'label_names': ['Insatisfecho', 'Satisfecho']
}

MODEL_CONFIG = {
    'model_name': 'distilbert-base-uncased',
    'max_length': 256,
    'truncation': True,
    'padding': True,
}

TRAINING_CONFIG = {
    'output_dir': str(MODELS_DIR / 'call_center_model'),
    'num_train_epochs': 10,
    'per_device_train_batch_size': 16,
    'per_device_eval_batch_size': 32,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'warmup_steps': 500,
    'logging_steps': 100,
    'eval_strategy': 'epoch',
    'save_strategy': 'epoch',
    'load_best_model_at_end': True,
    'metric_for_best_model': 'f1',
    'greater_is_better': True,
    'save_total_limit': 2,
    'fp16': torch.cuda.is_available(),
    'gradient_accumulation_steps': 2,
    'max_grad_norm': 1.0,
    'seed': 42,
    'report_to': 'none'
}

BASELINE_CONFIG = {
    'architecture': 'simple_lstm',
    'vocab_size': 10000,
    'embedding_dim': 128,
    'hidden_dim': 256,
    'num_layers': 2,
    'dropout': 0.3,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 10,
    'max_length': 200
}

CALL_CENTER_CONFIG = {
    'min_call_duration': 30,
    'max_call_duration': 1800,
    'speaker_labels': ['Cliente', 'Operador'],
    'sale_keywords': [
        'comprar', 'adquirir', 'contratar', 'pedir', 'solicitar',
        'purchase', 'buy', 'order', 'subscribe', 'get', 'confirmo',
        'pedido', 'orden'
    ],
    'positive_keywords': [
        'gracias', 'excelente', 'perfecto', 'bien', 'satisfecho',
        'thank', 'excellent', 'perfect', 'great', 'satisfied',
        'contento', 'feliz', 'bueno', 'genial'
    ],
    'negative_keywords': [
        'problema', 'mal', 'insatisfecho', 'terrible', 'pesimo',
        'problem', 'bad', 'unsatisfied', 'terrible', 'awful',
        'horrible', 'decepcionado', 'molesto', 'enojado'
    ],
    'operator_performance_metrics': {
        'response_time': 5,
        'resolution_keywords': ['solucionado', 'resuelto', 'fixed', 'resolved', 'arreglado'],
        'courtesy_keywords': ['por favor', 'gracias', 'disculpe', 'please', 'sorry', 'con gusto']
    }
}

AUDIO_CONFIG = {
    'model_name': 'openai/whisper-base',
    'language': 'es',
    'sample_rate': 16000,
    'chunk_length': 30,
}

GRADIO_CONFIG = {
    'theme': 'default',
    'max_audio_length': 300,
    'show_history': True,
    'enable_feedback': True
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def print_system_info():
    print("Configuracion del Sistema")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

if __name__ == "__main__":
    print_system_info()