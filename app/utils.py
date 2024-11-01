import json
import re
import os
from datetime import datetime
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

def normalize(string):
    """Converts a string to two floats (sentiment and confidence). Returns (0, 0) if conversion fails."""
    sentiment, confidence = map(float, string.split())
    return sentiment, confidence

def class_normalize(string, group_product):
    """
    Extrai o sentimento e o grupo de um JSON embutido em um texto e retorna "promoter", "neutral" ou "detractor" e o grupo.
    
    Args:
        string (str): Texto contendo o JSON.
        group_product (list): Lista de grupos válidos.
    
    Returns:
        tuple: Sentimento como "promoter", "neutral" ou "detractor" e o grupo. Retorna ("unknown", "unknown") se a extração falhar.
    """
    try:
        # Extrair a parte JSON do texto
        json_string = re.search(r'\{.*\}', string, re.DOTALL).group()
        data = json.loads(json_string)
        sentimento = data.get("sentiment", "").lower()
        grupo = data.get("group", "").lower()

        # if grupo not in group_product:
        #     grupo = "unknown"
        
        # if sentimento not in ["promoter", "neutral", "detractor"]:
        #     sentimento = "unknown"
        
        return sentimento, grupo
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as e:
        print(f"Erro ao decodificar JSON: {e}")
        return "unknown", "unknown"
    

def text_time(start, end):
    """Returns the elapsed time between start and end in seconds."""
    return f"\t{(end-start):.03f}s"

def generate_output_path():
    """
    Gera o caminho das pastas de output com base nas configurações do arquivo .env.
    
    Returns:
        str: Caminho das pastas de output.
    """
    base = 'output'
    model_name = os.getenv('MODEL_NAME').replace(":", "-")
    use_timestamp = os.getenv('USE_TIMESTAMP', 'false').lower() == 'true'

    if use_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(base, model_name, timestamp)
    else:
        output_path = os.path.join(base, model_name)

    return output_path
