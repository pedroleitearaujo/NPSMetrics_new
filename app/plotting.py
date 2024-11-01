import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from utils import generate_output_path
import os

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

HISTOGRAM_PATH = os.getenv('HISTOGRAM_PATH')
HEATMAP_PATH = os.getenv('HEATMAP_PATH')

def plot_histogram(data_frame, column_name='sentiment', base_path=None):
    file_name = f'{base_path}/{HISTOGRAM_PATH}_{column_name}.png'
    
    """Plots and saves a histogram of the sentiment distribution."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data_frame[column_name], bins=30, kde=True, color='purple')
    plt.title(f'Distribuição dos {column_name}')
    plt.xlabel('Sentimento')
    plt.ylabel('Frequência')
    plt.savefig(file_name)

def plot_heatmap(data_frame, column_name='sentiment', type_column='score', base_path=None):
    file_name = f'{base_path}/{HISTOGRAM_PATH}_{column_name}_{type_column}.png'

    """Plots and saves a heatmap comparing NPS scores and sentiment."""
    data_frame['value_rounded'] = data_frame[column_name].round(1)
    heatmap_data = pd.crosstab(data_frame['value_rounded'], data_frame[type_column])
    heatmap_data = heatmap_data.sort_index(ascending=False)
    nps_totals = heatmap_data.sum(axis=0)
    nps_labels = [f'{nps}\n({total})' for nps, total in nps_totals.items()]

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap='viridis', annot=True, fmt='d')
    plt.title(f'Heatmap de Comparação entre {type_column} e {column_name}')
    plt.xlabel(f'{type_column}')
    plt.ylabel(f'{column_name} (Arredondada)')
    plt.xticks(ticks=np.arange(len(nps_labels))+0.5, labels=nps_labels, rotation=0)
    plt.yticks(rotation=0)
    plt.savefig(file_name)

def plot_sentiment_analysis(data_frame, base_path):
    """Plots and saves a histogram and heatmap of the sentiment analysis."""
    plot_histogram(data_frame, 'sentiment', base_path)
    plot_heatmap(data_frame, 'sentiment', 'score', base_path)

    plot_histogram(data_frame, 'classic_sentiment', base_path)
    plot_heatmap(data_frame, 'classic_sentiment', 'score', base_path)

    plot_histogram(data_frame, 'combined_weight_sentiment', base_path)
    plot_heatmap(data_frame, 'combined_weight_sentiment', 'score', base_path)

    plot_histogram(data_frame, 'combined_abs_sentiment', base_path)
    plot_heatmap(data_frame, 'combined_abs_sentiment', 'score', base_path)