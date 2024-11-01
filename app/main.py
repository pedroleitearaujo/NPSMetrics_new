import pandas as pd
import numpy as np
import time
from sentiment_analysis import sentiment_analysis
from class_sentiment_analysis import class_sentiment_analysis
from classic_sentiment_analysis import classic_sentiment_analysis
from combined_sentiment import combined_weight_sentiment_analysis, combined_abs_sentiment_analysis
from plotting import plot_sentiment_analysis
from utils import text_time, generate_output_path
from data_loader import load_data
from dotenv import load_dotenv
import os

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

BASE_PATH = generate_output_path()
OUTPUT_CSV = f"{BASE_PATH}/{os.getenv('OUTPUT_CSV')}"
INPUT_CSV = os.getenv('INPUT_CSV')
COLUM_COMMENT = os.getenv('COLUM_COMMENT')

def execute_analysis_functions(text, index, df, functions):
    """
    Executa uma lista de funções de análise de sentimento.
    
    Args:
        text (str): Texto a ser analisado.
        index (int): Índice da linha no DataFrame.
        df (pd.DataFrame): DataFrame contendo os dados.
        functions (list): Lista de funções de análise de sentimento a serem executadas.
    """
    for func in functions:
        try:
            func(text, index, df)
        except Exception as e:
            error_column = f"error_{func.__name__}"
            error_value_column = f"error_{func.__name__}_value"
            print(f"Erro ao processar {func.__name__} na linha {index}: {e}")
            df.at[index, error_column] = 1  # Marcar a linha como erro
            df.at[index, error_value_column] = f"{e}"

def process_row(index, row, df, output_path):
    """Processa uma única linha para análise de sentimento."""
    text = row[COLUM_COMMENT]
    if not isinstance(text, str):
        text = str(text)
    
    functions = [
        sentiment_analysis,
        class_sentiment_analysis,
        classic_sentiment_analysis,
        lambda text, index, df: combined_weight_sentiment_analysis(df.at[index, 'sentiment'], df.at[index, 'classic_sentiment'], index, df),
        lambda text, index, df: combined_abs_sentiment_analysis(df.at[index, 'sentiment'], df.at[index, 'classic_sentiment'], index, df)
    ]
    
    execute_analysis_functions(text, index, df, functions)
    save_progress(df, index, output_path)

# def process_row(index, row, df):
#     """Processa uma única linha para análise de sentimento."""
#     text = row[COLUM_COMMENT]
#     if not isinstance(text, str):
#         text = str(text)
    
#     try:
#         sentiment_analysis(text, index, df)
#     except Exception as e:
#         print(f"Erro ao processar análise de sentimento na linha {index}: {e}")
#         df.at[index, 'error_sentiment_value'] = f"{e}"
#         df.at[index, 'error_sentiment'] = 1  # Marcar a linha como erro

#     try:
#         class_sentiment_analysis(text, index, df)
#     except Exception as e:
#         print(f"Erro ao processar análise de classificacao sentimento na linha {index}: {e}")
#         df.at[index, 'error_class_sentiment_value'] = f"{e}"
#         df.at[index, 'error_class_sentiment'] = 1  # Marcar a linha como erro

#     try:
#         classic_sentiment_analysis(text, index, df)
#     except Exception as e:
#         print(f"Erro ao processar análise de sentimento clássica na linha {index}: {e}")
#         df.at[index, 'error_classic_sentiment'] = 1  # Marcar a linha como erro

#     try:
#         combined_weight_sentiment_analysis(df.at[index, 'sentiment'], df.at[index, 'classic_sentiment'], index, df)
#     except Exception as e:
#         print(f"Erro ao processar análise de sentimento combinado (peso) na linha {index}: {e}")
#         df.at[index, 'error_combined_weight_sentiment'] = 1  # Marcar a linha como erro

#     try:
#         combined_abs_sentiment_analysis(df.at[index, 'sentiment'], df.at[index, 'classic_sentiment'], index, df)
#     except Exception as e:
#         print(f"Erro ao processar análise de sentimento combinado (abs) na linha {index}: {e}")
#         df.at[index, 'error_combined_abs_sentiment'] = 1  # Marcar a linha como erro

def save_progress(df, index, output_path):
    """
    Salva o progresso no arquivo CSV a cada 2 linhas.
    Cria a pasta se ela não existir.
    
    Args:
        df (pd.DataFrame): DataFrame contendo os dados.
        index (int): Índice da linha no DataFrame.
        output_path (str): Caminho do arquivo de output.
    """
    if index % 2 == 0:
        # Verificar se a pasta existe, se não, criar
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Progresso salvo: {index + 1}/{len(df.index)}")

def main():
    """Função principal para realizar a análise de sentimentos e gerar gráficos."""
    start_global = time.time()
    print(f"Iniciando a análise de sentimentos do {INPUT_CSV}")

    df = load_data()

    # for index, row in df.head(100).iterrows():
    for index, row in df.iterrows():
        if 'sentiment' in df.columns and not np.isnan(row['sentiment']):
            print(f"{index} : {len(df.index)} | Sentimento ja calculado: {row['sentiment']} | {row['time']}s")
        else:
            process_row(index, row, df, OUTPUT_CSV)

    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"Análise de sentimentos concluída. Resultados salvos em '{OUTPUT_CSV}'.")

    plot_sentiment_analysis(df, BASE_PATH)
    print("Gráficos gerados e salvos.")

    total_time_from_rows = df['time'].sum()
    print(f"Tempo Total calculado a partir das linhas: {total_time_from_rows:.03f}s")

    print(f"Tempo da execução: {text_time(start_global, time.time())}")

if __name__ == "__main__":
    main()