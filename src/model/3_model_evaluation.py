
import os
import sys
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json
import time
from datetime import datetime
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from config import (
    MODEL_CONFIG, TRAINING_CONFIG, DATASET_CONFIG,
    DATA_DIR, MODELS_DIR, DEVICE,
    set_seed, print_system_info
)

set_seed(42)


class SentimentDataset(Dataset):
    
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        self.labels = labels.tolist()
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)


class ModelTrainer:
    
    def __init__(self, model_config, training_config, dataset_config):
        self.model_config = model_config
        self.training_config = training_config
        self.dataset_config = dataset_config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.train_df = None
        self.test_df = None
        self.training_history = {}
    
    def load_processed_data(self):
        print("\nCargando datos procesados")
        
        train_path = DATA_DIR / 'train_processed.csv'
        test_path = DATA_DIR / 'test_processed.csv'
        
        if not train_path.exists() or not test_path.exists():
            print("Error: Datos procesados no encontrados.")
            print("Ejecuta primero: python 1_data_preparation.py")
            return False
        
        print(f"\nCargando desde:")
        print(f"Train: {train_path}")
        print(f"Test: {test_path}")
        
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        
        print(f"\nDatos cargados:")
        print(f"Train: {len(self.train_df):,} samples")
        print(f"Test: {len(self.test_df):,} samples")
        
        return True
    
    def initialize_model(self):
        print("\nInicializando modelo pre-entrenado")
        
        model_name = self.model_config['model_name']
        num_labels = self.dataset_config['num_labels']
        
        print(f"\nModelo: {model_name}")
        print(f"Numero de clases: {num_labels}")
        
        print("\nCargando tokenizer y modelo...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                ignore_mismatched_sizes=True
            )
            
            self.model = self.model.to(DEVICE)
            
            print(f"Modelo cargado en: {DEVICE}")
            
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            print(f"\nInformacion del modelo:")
            print(f"Parametros totales: {total_params:,}")
            print(f"Parametros entrenables: {trainable_params:,}")
            print(f"Porcentaje entrenable: {100 * trainable_params / total_params:.1f}%")
            
            if trainable_params == 0:
                print("ERROR: No hay parametros entrenables!")
                return False
            
            print("Todos los pesos estan descongelados para fine-tuning")
            
            return True
            
        except Exception as e:
            print(f"Error inicializando modelo: {e}")
            return False
    
    def create_datasets(self):
        print("\nCreando datasets tokenizados")
        
        max_length = self.model_config['max_length']
        text_col = 'content_clean'
        label_col = self.dataset_config['label_column']
        
        print(f"\nConfiguracion de tokenizacion:")
        print(f"Max length: {max_length}")
        print(f"Truncation: {self.model_config['truncation']}")
        print(f"Padding: {self.model_config['padding']}")
        
        print("\nTokenizando datos...")
        
        train_dataset = SentimentDataset(
            self.train_df[text_col],
            self.train_df[label_col],
            self.tokenizer,
            max_length
        )
        
        test_dataset = SentimentDataset(
            self.test_df[text_col],
            self.test_df[label_col],
            self.tokenizer,
            max_length
        )
        
        print(f"Datasets creados:")
        print(f"Train dataset: {len(train_dataset):,} samples")
        print(f"Test dataset: {len(test_dataset):,} samples")
        
        return train_dataset, test_dataset
    
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='binary')
        precision = precision_score(labels, predictions, average='binary')
        recall = recall_score(labels, predictions, average='binary')
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train_model(self, train_dataset, test_dataset):
        print("\nEntrenando modelo (Fine-tuning)")
        
        print("\nConfiguracion de entrenamiento:")
        print(f"Epochs: {self.training_config['num_train_epochs']}")
        print(f"Batch size (train): {self.training_config['per_device_train_batch_size']}")
        print(f"Batch size (eval): {self.training_config['per_device_eval_batch_size']}")
        print(f"Learning rate: {self.training_config['learning_rate']}")
        print(f"Weight decay: {self.training_config['weight_decay']}")
        print(f"FP16: {self.training_config['fp16']}")
        print(f"Gradient accumulation: {self.training_config['gradient_accumulation_steps']}")
        
        training_args = TrainingArguments(**self.training_config)
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        print("\nIniciando entrenamiento...")
        
        start_time = time.time()
        
        try:
            train_result = self.trainer.train()
            
            training_time = time.time() - start_time
            
            print("\nEntrenamiento completado")
            
            print(f"\nTiempo de entrenamiento: {training_time/60:.2f} minutos")
            print(f"Training loss: {train_result.training_loss:.4f}")
            
            self.training_history = {
                'training_time_minutes': training_time / 60,
                'training_loss': float(train_result.training_loss),
                'global_steps': train_result.global_step,
                'logs': [log for log in self.trainer.state.log_history]
            }
            
            return True
            
        except Exception as e:
            print(f"\nError durante entrenamiento: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def evaluate_model(self):
        print("\nEvaluando modelo en test set")
        
        print("\nEvaluando...")
        
        eval_results = self.trainer.evaluate()
        
        print("\nResultados en test set:")
        print(f"Accuracy:  {eval_results['eval_accuracy']:.4f}")
        print(f"F1 Score:  {eval_results['eval_f1']:.4f}")
        print(f"Precision: {eval_results['eval_precision']:.4f}")
        print(f"Recall:    {eval_results['eval_recall']:.4f}")
        print(f"Loss:      {eval_results['eval_loss']:.4f}")
        
        self.training_history['test_results'] = {
            'accuracy': float(eval_results['eval_accuracy']),
            'f1': float(eval_results['eval_f1']),
            'precision': float(eval_results['eval_precision']),
            'recall': float(eval_results['eval_recall']),
            'loss': float(eval_results['eval_loss'])
        }
        
        return eval_results
    
    def save_model(self):
        print("\nGuardando modelo")
        
        model_dir = MODELS_DIR / 'call_center_model'
        model_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGuardando en: {model_dir}")
        
        self.trainer.save_model(str(model_dir))
        self.tokenizer.save_pretrained(str(model_dir))
        
        print("Modelo guardado")
        print("Tokenizer guardado")
        
        config_path = model_dir / 'training_config.json'
        with open(config_path, 'w') as f:
            json.dump({
                'model_config': self.model_config,
                'training_config': {k: v for k, v in self.training_config.items() 
                                   if not isinstance(v, (type, Path))},
                'dataset_config': self.dataset_config
            }, f, indent=2)
        
        print(f"Configuracion guardada: {config_path}")
        
        history_path = model_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"Historial guardado: {history_path}")
        
        metadata = {
            'model_name': self.model_config['model_name'],
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'training_date': datetime.now().isoformat(),
            'device': str(DEVICE),
            'train_samples': len(self.train_df),
            'test_samples': len(self.test_df),
            'pytorch_version': torch.__version__
        }
        
        metadata_path = model_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata guardado: {metadata_path}")
        
        print(f"\nModelo completo guardado en: {model_dir}")
    
    def run_pipeline(self):
        print("\nIniciando pipeline de entrenamiento")
        
        if not self.load_processed_data():
            return False
        
        if not self.initialize_model():
            return False
        
        train_dataset, test_dataset = self.create_datasets()
        
        if not self.train_model(train_dataset, test_dataset):
            return False
        
        self.evaluate_model()
        
        self.save_model()
        
        print("\nPipeline de entrenamiento completado")
        print("\nProximo paso: python 3_model_evaluation.py")
        
        return True


def main():
    print_system_info()
    
    trainer = ModelTrainer(MODEL_CONFIG, TRAINING_CONFIG, DATASET_CONFIG)
    success = trainer.run_pipeline()
    
    if success:
        print("\nEntrenamiento completado exitosamente!")
        print("\nResumen:")
        print(f"Tiempo: {trainer.training_history.get('training_time_minutes', 0):.1f} min")
        print(f"F1 Score: {trainer.training_history['test_results']['f1']:.4f}")
        print(f"Accuracy: {trainer.training_history['test_results']['accuracy']:.4f}")
    else:
        print("\nEntrenamiento terminado con errores")
        sys.exit(1)


if __name__ == "__main__":
    main()