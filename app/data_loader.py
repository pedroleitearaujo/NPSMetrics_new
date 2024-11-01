import pandas as pd
import numpy as np
from dotenv import load_dotenv
from utils import generate_output_path
import os

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

INPUT_CSV = os.getenv('INPUT_CSV')

def load_data():
    """Loads the dataset and initializes necessary columns."""
    # try:
    #     df = pd.read_csv(generate_output_path())
    # except FileNotFoundError:
    df = pd.read_csv(INPUT_CSV)
    df['sentiment'] = np.nan
    df['confidence'] = np.nan
    df['error_sentiment'] = '0'
    df['classic_sentiment'] = np.nan
    df['positive'] = np.nan
    df['neutral'] = np.nan
    df['negative'] = np.nan
    df['time'] = np.nan
    
    return df