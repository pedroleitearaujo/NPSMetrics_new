import time
import ollama
from utils import class_normalize, text_time
from dotenv import load_dotenv
import os

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

MODEL_NAME = os.getenv('MODEL_NAME')
GROUP_PRODUCT = ['cartao credito', 'lentidao', 'PIX', 'cartao de debito', 'maquininha', 'atendimento', 'conta', 'venda', 'taxas', 'acesso', 'outros']

def build_content(text):
    """
    Builds the content string for the sentiment analysis model.
    
    Args:
        text (str): Texto a ser analisado.
    
    Returns:
        str: String formatada para o modelo de análise de sentimentos.
    """
    return f'''
    Você é um especialista em análise de sentimentos. 
    Analise o texto entre colchetes [ ].
    Primeiro classifique ele entre usuários promoter, neutral e detractor.
    Leve em conta o sentimento expressado para classificar o usuário.
    Texto onde o usuário não expressa sentimento, classifique como neutro.
    Texto onde o usuário reclama ou expressa insatisfação, classifique como detractor.
    
    Segundo classifique o conteúdo do texto entre somente uma opção do grupos abaixo:
    {" | ".join(GROUP_PRODUCT)}
    
    Texto: [{text}]

    Formato de retorno em json:
    Não retorne nada a mais além do solicitado, somente o json.

    {{
        "sentiment": "promoter" | "neutral" | "detractor",
        "group": {" | ".join(GROUP_PRODUCT)}
    }}
    '''

def process_response(response, index, df, start):
    """
    Processes the response from the sentiment analysis model.
    
    Args:
        response (dict): Resposta do modelo de análise de sentimentos.
        index (int): Índice da linha no DataFrame.
        df (pd.DataFrame): DataFrame contendo os dados.
        start (float): Tempo de início da análise.
    
    Returns:
        bool: True se o processamento for bem-sucedido, False caso contrário.
    """
    try:
        sentiment, group = class_normalize(response['message']['content'], GROUP_PRODUCT)
    except ValueError:
        return False

    end = time.time()
    df.at[index, 'class_sentiment'] = sentiment
    df.at[index, 'class_group'] = group
    df.at[index, 'time'] = float(f"{(end-start):.03f}")

    prefix = f"{index} : {len(df.index)}"
    print(f"{prefix} | Sentimento: {sentiment} | grupo: {group} | {text_time(start, end)}")
    return True

def class_sentiment_analysis(text, index, df):
    """
    Performs sentiment analysis on the given text.
    
    Args:
        text (str): Texto a ser analisado.
        index (int): Índice da linha no DataFrame.
        df (pd.DataFrame): DataFrame contendo os dados.
    
    Returns:
        str: Sentimento classificado.
    """
    start = time.time()
    content = build_content(text)

    response = ollama.chat(
        MODEL_NAME,
        messages=[{'role': 'user', 'content': content}],
    )

    if not process_response(response, index, df, start):
        # Tentar reprocessar a linha
        response = ollama.chat(
            MODEL_NAME,
            messages=[{'role': 'user', 'content': content + ' retorne somente os numeros solicitados'}],
        )
        if not process_response(response, index, df, start):
            # Marcar a linha como não processada
            df.at[index, 'error_class_sentiment'] = 1  # Converter explicitamente para int
            df.at[index, 'error_class_sentiment_value'] = response['message']['content']
            df.at[index, 'time'] = float(f"{(time.time() - start):.03f}")
            print(f"{index} : {len(df.index)} | Erro ao processar a linha. | {text_time(start, time.time())}")
            return 'unknown'

    return df.at[index, 'class_sentiment']