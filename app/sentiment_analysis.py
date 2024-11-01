import time
import ollama
from utils import normalize, text_time
from dotenv import load_dotenv
import os

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

MODEL_NAME = os.getenv('MODEL_NAME')

def build_content(text):
    """Builds the content string for the sentiment analysis model."""
    return f'''
    Você é um especialista em análise de sentimentos. Analise o texto entre colchetes [ ] e atribua uma nota entre -1.0 e 1.0 que represente o sentimento expressado. Use -1.0 para sentimentos muito negativos e 1.0 para sentimentos muito positivos e 0 para sentimentos neutros.
    Também atribua uma nota entre 0.0 e 1.0 o quão confiante você está perante a nota que está dando para o sentimento.
    Atenção: Utilize apenas uma casa decimal.

    Considere os seguintes aspectos linguísticos, com a porcentagem indicando a importância de cada aspecto para a análise de sentimento:

    Léxico Emocional (25%)
    Identifique palavras que expressam sentimentos:
    Positivo: Palavras como "feliz", "excelente", "fantástico".
    Neutro: Palavras informativas, como "adequado", "mediano".
    Negativo: Palavras como "triste", "horrível", "fracasso".
    
    Contexto Linguístico (20%)
    Considere como o contexto altera o sentimento das palavras:
    Positivo: Uso de ironia positiva ("Claro que estou super feliz com isso!" expressando felicidade).
    Neutro: Ironia sem carga emocional significativa ("Muito típico").
    Negativo: Sarcasmo negativo ("Claro, isso foi tão útil" com intenção negativa).
    
    Intensidade Emocional (15%)
    Avalie a força das emoções no texto:
    Positivo: Intensificadores de emoções positivas ("extremamente feliz").
    Neutro: Emoções de pouca intensidade ("levemente contente").
    Negativo: Intensificadores de emoções negativas ("completamente horrível").
   
    Sintaxe e Estrutura da Frase (10%)
    A construção da frase pode influenciar o sentimento:
    Positivo: Frases que transmitem satisfação ("Foi uma experiência incrível").
    Neutro: Frases descritivas e objetivas ("O evento ocorreu às 18 horas").
    Negativo: Frases que expressam frustração ("Isso foi um desperdício de tempo").
    
    Contexto Cultural e Temporal (10%)
    Considere o impacto cultural e temporal nas expressões:
    Positivo: Termos com conotação positiva em contextos atuais ("épico" significando algo muito bom).
    Neutro: Palavras com significado literal e neutro.
    Negativo: Termos com conotação negativa ou desatualizada.
    
    Uso de Emojis e Emoticons (5%)
    Emojis podem modificar o tom emocional:
    Positivo: Emojis de felicidade ou aprovação (😊, 👍).
    Neutro: Emojis sem carga emocional significativa (🔄, 🔍).
    Negativo: Emojis de tristeza ou desaprovação (😞, 👎).
    
    Marcadores Discursivos (5%)
    Identifique palavras que mudam o tom ou introduzem novas ideias:
    Positivo: Marcadores de emoção positiva ("excelente").
    Neutro: Marcadores de transição ("no entanto").
    Negativo: Marcadores de desapontamento ("infelizmente").
    
    Polaridade e Multiplicidade de Sentimentos (5%)
    Detecte a presença de múltiplos sentimentos no texto:
    Positivo: Sentimentos positivos predominantes em contextos mistos.
    Neutro: Equilíbrio ou ambiguidade emocional.
    Negativo: Sentimentos negativos predominantes.
    
    Referências a Entidades Nomeadas e Relacionamentos (5%)
    O modo como entidades são mencionadas pode alterar o sentimento:
    Positivo: Mencionar entidades de forma elogiosa ("A Apple fez um trabalho excelente").
    Neutro: Referências neutras ou informativas.
    Negativo: Mencionar entidades de forma crítica ("O governo falhou miseravelmente").
    
    Estilo de Linguagem e Tom (5%)
    A tonalidade geral do texto pode impactar a percepção emocional:
    Positivo: Linguagem otimista.
    Neutro: Estilo informativo ou descritivo.
    Negativo: Linguagem pessimista ou crítica.
    
    Objetividade e Neutralidade (5%)
    Avalie se o texto é imparcial ou inclinado:
    Positivo: Fatos com leve inclinação positiva.
    Neutro: Fatos estritamente objetivos.
    Negativo: Fatos com leve inclinação negativa.
    
    Entonação e Prosódia (em Texto Falado) (5%)
    Considere como a entonação pode influenciar o sentimento:
    Positivo: Entonação alegre ou excitada.
    Neutro: Entonação neutra e equilibrada.
    Negativo: Entonação triste ou irritada.

    Formato de Retorno: Apenas numero com uma casa decimal, e o número que representa os sentimentos e o numero que representa o quão confiante está perante a nota do sentimento. por exemplo "0.2 0.9". 
    Não forneça explicações adicionais!!!.
    Caso não haja nenhum texto para ser lido não retorne nada!!!.

    Texto: [{text}]
    '''

def process_response(response, index, df, start):
    """Processes the response from the sentiment analysis model."""
    try:
        sentiment, confidence = normalize(response['message']['content'])
    except ValueError:
        return False

    end = time.time()
    df.at[index, 'sentiment'] = sentiment
    df.at[index, 'confidence'] = confidence
    df.at[index, 'time'] = float(f"{(end-start):.03f}")
    prefix = f"{index} : {len(df.index)}"
    print(f"{prefix} | Modelo: {response['message']['content'].replace("\n", "")} | Sentimento: {sentiment} | Confianca: {confidence} | {text_time(start, end)}")
    return True

def sentiment_analysis(text, index, df):
    """Performs sentiment analysis on the given text."""
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
            df.at[index, 'error_sentiment'] = 1  # Converter explicitamente para int
            df.at[index, 'error_sentiment_value'] = response['message']['content']
            df.at[index, 'time'] = float(f"{(time.time() - start):.03f}")
            print(f"{index} : {len(df.index)} | Erro ao processar a linha. | {text_time(start, time.time())}")
            return 'N/A', 'N/A'

    return df.at[index, 'sentiment'], df.at[index, 'confidence']