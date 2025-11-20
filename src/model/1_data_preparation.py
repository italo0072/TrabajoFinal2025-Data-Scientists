
import os
import sys

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from collections import Counter
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from config import (
    DATASET_CONFIG, DATA_DIR, FIGURES_DIR,
    set_seed, print_system_info
)

set_seed(42)


class CallCenterDataPreparator:
    
    def __init__(self, config):
        self.config = config
        self.train_df = None
        self.test_df = None
    
    def load_data(self):
        print("\nCargando Dataset")
        print("Dataset: amazon_polarity")
        
        try:
            dataset = load_dataset('amazon_polarity')
            
            self.train_dataset = dataset['train']
            self.test_dataset = dataset['test']
            
            print(f"\nDataset cargado")
            print(f"Train samples: {len(self.train_dataset):,}")
            print(f"Test samples: {len(self.test_dataset):,}")
            
            if self.config['train_samples']:
                indices = np.random.choice(
                    len(self.train_dataset),
                    min(self.config['train_samples'], len(self.train_dataset)),
                    replace=False
                )
                self.train_dataset = self.train_dataset.select(indices)
            
            if self.config['test_samples']:
                indices = np.random.choice(
                    len(self.test_dataset),
                    min(self.config['test_samples'], len(self.test_dataset)),
                    replace=False
                )
                self.test_dataset = self.test_dataset.select(indices)
            
            print(f"\nSamples finales:")
            print(f"Train: {len(self.train_dataset):,}")
            print(f"Test: {len(self.test_dataset):,}")
            
            return True
            
        except Exception as e:
            print(f"Error cargando dataset: {e}")
            return False
    
    def adapt_to_call_center(self, text, label):
        # Prefijos y sufijos IDENTICOS para ambas clases
        all_prefixes = [
            "Operador: Buenas tardes, gracias por comunicarse. En que puedo ayudarle?\nCliente: ",
            "Operador: Hola, bienvenido. Como puedo asistirle hoy?\nCliente: ",
            "Operador: Buenos dias. En que podemos ayudarle?\nCliente: ",
            "Operador: Gracias por llamar. Cual es el motivo de su consulta?\nCliente: ",
            "Operador: Bienvenido a nuestro servicio. Como puedo ayudarle?\nCliente: ",
        ]
        
        all_suffixes = [
            "\nOperador: Gracias por comunicarse con nosotros.",
            "\nOperador: Fue un placer atenderle.",
            "\nOperador: Estamos para servirle.",
            "\nOperador: Que tenga un buen dia.",
            "\nOperador: Gracias por su tiempo.",
        ]
        
        # Selección aleatoria INDEPENDIENTE de la etiqueta
        prefix = np.random.choice(all_prefixes)
        suffix = np.random.choice(all_suffixes)
        
        # Limitar texto a 200 caracteres para evitar que el modelo memorice reviews completas
        adapted_text = prefix + text[:200] + suffix
        
        return adapted_text
    
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'([!?.]){2,}', r'\1', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def preprocess_data(self):
        print("\nPreprocesando datos (VERSION MEJORADA - sin data leakage)")
        
        self.train_df = pd.DataFrame(self.train_dataset)
        self.test_df = pd.DataFrame(self.test_dataset)
        
        print("Adaptando formato de call center (prefijos neutros)...")
        self.train_df['content_clean'] = self.train_df.apply(
            lambda row: self.adapt_to_call_center(row['content'], row['label']),
            axis=1
        )
        self.test_df['content_clean'] = self.test_df.apply(
            lambda row: self.adapt_to_call_center(row['content'], row['label']),
            axis=1
        )
        
        self.train_df['content_clean'] = self.train_df['content_clean'].apply(self.clean_text)
        self.test_df['content_clean'] = self.test_df['content_clean'].apply(self.clean_text)
        
        # Eliminar textos muy cortos
        self.train_df = self.train_df[self.train_df['content_clean'].str.len() > 50].reset_index(drop=True)
        self.test_df = self.test_df[self.test_df['content_clean'].str.len() > 50].reset_index(drop=True)
        
        self.train_df['word_count'] = self.train_df['content_clean'].str.split().str.len()
        self.test_df['word_count'] = self.test_df['content_clean'].str.split().str.len()
        
        print(f"\nDataset final:")
        print(f"Train: {len(self.train_df):,} samples")
        print(f"Test: {len(self.test_df):,} samples")
        
        sample_pos = self.train_df[self.train_df['label'] == 1].sample(3)
        sample_neg = self.train_df[self.train_df['label'] == 0].sample(3)
        
        print("\nEjemplo clase SATISFECHO:")
        print(sample_pos['content_clean'].values[0][:300])
        print("\nEjemplo clase INSATISFECHO:")
        print(sample_neg['content_clean'].values[0][:300])
        print("\nSi los prefijos/sufijos son iguales, esta OK!")
    
    def exploratory_analysis(self):
        print("\nAnalisis Exploratorio de Datos")
        
        train_dist = Counter(self.train_df['label'])
        test_dist = Counter(self.test_df['label'])
        
        label_names = ['Insatisfecho', 'Satisfecho']
        
        print("\nDistribucion de clases:")
        print("\nTrain:")
        for label, count in sorted(train_dist.items()):
            pct = (count / len(self.train_df)) * 100
            print(f"  {label_names[label]:15s}: {count:,} ({pct:.1f}%)")
        
        print("\nTest:")
        for label, count in sorted(test_dist.items()):
            pct = (count / len(self.test_df)) * 100
            print(f"  {label_names[label]:15s}: {count:,} ({pct:.1f}%)")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        train_counts = [train_dist[i] for i in sorted(train_dist.keys())]
        axes[0].bar(label_names, train_counts, color=['#e74c3c', '#27ae60'])
        axes[0].set_title('Distribucion de Clases - Train')
        axes[0].set_ylabel('Cantidad')
        
        test_counts = [test_dist[i] for i in sorted(test_dist.keys())]
        axes[1].bar(label_names, test_counts, color=['#e74c3c', '#27ae60'])
        axes[1].set_title('Distribucion de Clases - Test')
        axes[1].set_ylabel('Cantidad')
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / '01_class_distribution.png', dpi=300, bbox_inches='tight')
        print(f"\nGrafico guardado: {FIGURES_DIR / '01_class_distribution.png'}")
        plt.close()
    
    def save_processed_data(self):
        print("\nGuardando datos procesados")
        
        train_path = DATA_DIR / 'train_processed.csv'
        test_path = DATA_DIR / 'test_processed.csv'
        
        self.train_df.to_csv(train_path, index=False)
        self.test_df.to_csv(test_path, index=False)
        
        print(f"\nDatos guardados:")
        print(f"Train: {train_path}")
        print(f"Test: {test_path}")
    
    def run_pipeline(self):
    
        
        if not self.load_data():
            return False
        
        self.preprocess_data()
        self.exploratory_analysis()
        self.save_processed_data()
        
        
        
        return True


def main():
    print_system_info()
    
    preparator = CallCenterDataPreparator(DATASET_CONFIG)
    success = preparator.run_pipeline()
    
    if success:
        print("\nProceso completado")
    else:
        print("\nProceso terminado con errores")
        sys.exit(1)


if __name__ == "__main__":
    main()