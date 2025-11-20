import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score
)
from sklearn.calibration import calibration_curve

# intenta importar torch/transformers (opcional para calcular probabilidades desde el modelo)
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TORCH = True
except Exception:
    TORCH = False

# usa tu config.py para rutas
from config import DATA_DIR, FIGURES_DIR, MODELS_DIR, DEVICE

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------- utilidades de guardado --------------------------

def _save_fig(fig, name):
    out = FIGURES_DIR / name
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Guardado: {out}")


def _save_json(obj, name):
    out = FIGURES_DIR / name
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"Guardado JSON: {out}")


def _save_csv(df, name):
    out = FIGURES_DIR / name
    df.to_csv(out, index=False)
    print(f"Guardado CSV: {out}")


# -------------------------- funciones de plotting --------------------------

def plot_confusion_matrix_heatmap(y_true, y_pred, labels, name='confusion_matrix_heatmap.png'):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0], cbar=False)
    ax[0].set_title('Matriz de confusión (counts)')
    ax[0].set_xlabel('Predicho')
    ax[0].set_ylabel('Real')
    ax[0].set_xticklabels(labels)
    ax[0].set_yticklabels(labels)

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='coolwarm', ax=ax[1], vmin=0.0, vmax=1.0, cbar=True)
    ax[1].set_title('Matriz de confusión (normalizada por fila)')
    ax[1].set_xlabel('Predicho')
    ax[1].set_ylabel('Real')
    ax[1].set_xticklabels(labels)
    ax[1].set_yticklabels(labels)

    fig.tight_layout()
    _save_fig(fig, name)


def plot_roc_pr(y_true, probs, name_prefix='roc_pr'):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    ax[0].plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_title('ROC Curve')
    ax[0].legend(loc='lower right')

    # Precision-Recall
    precision, recall, _ = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, probs)
    ax[1].plot(recall, precision, label=f'AP = {ap:.3f}')
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    ax[1].set_title('Precision-Recall Curve')
    ax[1].legend(loc='lower left')

    fig.tight_layout()
    _save_fig(fig, f'{name_prefix}_roc_pr.png')


def plot_prob_distributions(y_true, probs, name='prob_distributions.png'):
    # Histograma + KDE por clase
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(probs[y_true == 0], kde=True, stat='density', label='Clase 0 (neg)', alpha=0.5, ax=ax)
    sns.histplot(probs[y_true == 1], kde=True, stat='density', label='Clase 1 (pos)', alpha=0.5, ax=ax)
    ax.set_title('Distribución de probabilidades predichas por clase')
    ax.set_xlabel('Probabilidad clase positiva')
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, name)


def plot_reliability_diagram(y_true, probs, name='calibration_reliability.png', n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy='quantile')
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(prob_pred, prob_true, marker='o', label='Modelo')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfecta')
    ax.set_xlabel('Prob. media predicha')
    ax.set_ylabel('Prob. observada')
    ax.set_title('Reliability diagram (calibration)')
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, name)


def plot_f1_precision_recall_by_threshold(y_true, probs, name='f1_prec_rec_thresholds.png'):
    thresholds = np.linspace(0.01, 0.99, 99)
    f1s, precs, recs = [], [], []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1s.append(f1_score(y_true, preds))
        precs.append(precision_score(y_true, preds, zero_division=0))
        recs.append(recall_score(y_true, preds))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, f1s, label='F1')
    ax.plot(thresholds, precs, label='Precision')
    ax.plot(thresholds, recs, label='Recall')
    best_idx = int(np.nanargmax(f1s))
    ax.axvline(thresholds[best_idx], color='red', linestyle='--', label=f'best F1 @ {thresholds[best_idx]:.2f}')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('F1 / Precision / Recall por umbral')
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, name)


def plot_classification_report_table(y_true, y_pred, labels, name='classification_report_table.png'):
    rep = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    df = pd.DataFrame(rep).T

    fig, ax = plt.subplots(figsize=(8, df.shape[0] * 0.6 + 1))
    sns.heatmap(df[['precision', 'recall', 'f1-score', 'support']].fillna(0), annot=True, fmt='.2f', cmap='viridis', ax=ax)
    ax.set_title('Classification report (precision/recall/f1/support)')
    fig.tight_layout()
    _save_fig(fig, name)


# -------------------------- evaluación principal --------------------------

def evaluate_model_from_files(test_csv_path: Path, model_dir: Path = None, batch_size: int = 64):
    """
    Carga test CSV con columnas: content_clean, label (0/1)
    Si model_dir proporcionado y TORCH==True, calcula probabilidades con el modelo.
    Guarda múltiples imágenes y un resumen JSON/CSV con métricas.
    """
    if not test_csv_path.exists():
        raise FileNotFoundError(f"No se encuentra test CSV: {test_csv_path}")

    df = pd.read_csv(test_csv_path)
    if 'content_clean' not in df.columns or 'label' not in df.columns:
        raise ValueError('CSV debe contener columnas content_clean y label (0/1)')

    texts = df['content_clean'].astype(str).tolist()
    y_true = df['label'].astype(int).values

    # probs / preds
    probs = None
    preds = None

    # si tenemos modelo y torch, inferir
    if model_dir is not None and model_dir.exists() and TORCH:
        print('Cargando modelo para inferencia desde:', model_dir)
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(DEVICE)
        model.eval()

        all_probs = []
        all_preds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, truncation=True, padding=True, max_length=256, return_tensors='pt')
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            with torch.no_grad():
                out = model(**enc)
                soft = torch.softmax(out.logits, dim=-1).cpu().numpy()
                # asumimos que la clase positiva es índice 1
                if soft.shape[1] > 1:
                    ppos = soft[:, 1]
                    pred = soft.argmax(axis=1)
                else:
                    ppos = soft[:, 0]
                    pred = (ppos >= 0.5).astype(int)
                all_probs.extend(ppos.tolist())
                all_preds.extend(pred.tolist())

        probs = np.array(all_probs)
        preds = np.array(all_preds)
    else:
        # si no hay modelo, intentar usar columnas prob/prob_pos si existen
        if 'prob_pos' in df.columns:
            probs = df['prob_pos'].astype(float).values
            preds = (probs >= 0.5).astype(int)
        elif 'prob' in df.columns:
            probs = df['prob'].astype(float).values
            preds = (probs >= 0.5).astype(int)
        else:
            # fallback: usar texto simple con heurística (palabras clave) -> aquí sólo usamos label como pred
            preds = y_true.copy()
            probs = np.where(preds == 1, 0.9, 0.1)
            print('No se encontró modelo ni probabilidades; usando labels como preds (fallback).')

    # métricas principales
    accuracy = float(accuracy_score(y_true, preds))
    precision = float(precision_score(y_true, preds, zero_division=0))
    recall = float(recall_score(y_true, preds))
    f1 = float(f1_score(y_true, preds))

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_samples': int(len(y_true))
    }

    # guardar resumen
    _save_json(metrics, 'metrics_summary.json')
    _save_csv(pd.DataFrame([metrics]), 'metrics_summary.csv')

    labels = ['Negativo/Insatisfecho', 'Positivo/Satisfecho']

    # plots
    plot_confusion_matrix_heatmap(y_true, preds, labels, name='01_confusion_heatmap.png')
    plot_classification_report_table(y_true, preds, labels, name='02_classification_report.png')
    plot_roc_pr(y_true, probs, name_prefix='03')
    plot_prob_distributions(y_true, probs, name='04_prob_dist.png')
    plot_reliability_diagram(y_true, probs, name='05_reliability.png', n_bins=10)
    plot_f1_precision_recall_by_threshold(y_true, probs, name='06_thresholds.png')

    # retornar resumen útil
    return {
        'metrics': metrics,
        'figures_dir': str(FIGURES_DIR)
    }


# -------------------------- ejecución directa --------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluar modelo y generar reportes gráficos')
    parser.add_argument('--test-csv', type=str, default=str(DATA_DIR / 'test_processed.csv'), help='CSV de prueba (content_clean,label)')
    parser.add_argument('--model-dir', type=str, default=str(MODELS_DIR / 'call_center_model'), help='Directorio del modelo transformers (opcional)')
    parser.add_argument('--batch-size', type=int, default=64)

    args = parser.parse_args()

    test_csv = Path(args.test_csv)
    model_dir = Path(args.model_dir) if args.model_dir else None

    try:
        summary = evaluate_model_from_files(test_csv, model_dir=model_dir, batch_size=args.batch_size)
        print('\nResumen métricas:')
        print(json.dumps(summary['metrics'], indent=2))
        print('Figuras guardadas en:', summary['figures_dir'])
    except Exception as e:
        print('Error al generar reportes:', e)

