import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as VaderSentimentIntensityAnalyzer
from LeIA import SentimentIntensityAnalyzer as LeIASentimentIntensityAnalyzer
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Baixar recursos necessários do NLTK
nltk.download('vader_lexicon')

# Selecionar o analisador de sentimento com base na variável de ambiente
SENTIMENT_ANALYZER = os.getenv('SENTIMENT_ANALYZER')

if SENTIMENT_ANALYZER == 'LeIA':
    analyzer = LeIASentimentIntensityAnalyzer()
else:
    analyzer = VaderSentimentIntensityAnalyzer()

def sentiment_analysis(text):
    """Performs sentiment analysis using a classic algorithm (VADER or LeIA)."""
    if not isinstance(text, str):
        raise TypeError("The input text must be a string.")
    scores = analyzer.polarity_scores(text)
    return scores['compound'], scores['pos'], scores['neu'], scores['neg']

def classic_sentiment_analysis(text, index, df):
    """Updates the DataFrame with classic sentiment analysis results."""
    classic_sentiment, pos, neu, neg = sentiment_analysis(text)
    df.at[index, 'classic_sentiment'] = classic_sentiment
    df.at[index, 'positive'] = pos
    df.at[index, 'neutral'] = neu
    df.at[index, 'negative'] = neg
    
    print(f"{index} : {len(df.index)} | Classic: {classic_sentiment} | Positivo: {pos} | Neutro: {neu} | Negativo: {neg}")
    return classic_sentiment, pos, neu, neg