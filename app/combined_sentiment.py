def get_weights(classic_value):
    """Retorna os pesos para o cálculo do sentimento combinado."""
    if classic_value == 0:
        return 0.95, 0.05  # Quando zero, usa 0.95 para classic e 0.05 para sentiment
    else:
        return 0.8, 0.2  # Usa os pesos padrão

def combined_weight_sentiment_analysis(sentiment_ia, classic_sentiment, index, df):
    """Calcula o sentimento combinado e atualiza o DataFrame."""
    sentiment_weight, classic_weight = get_weights(classic_sentiment)
    combined_weight_sentiment = ((sentiment_ia * sentiment_weight) + (classic_sentiment * classic_weight)) / (sentiment_weight + classic_weight)
    combined_weight_sentiment = round(combined_weight_sentiment, 2)  # Arredondar para 2 casas decimais

    df.at[index, 'combined_weight_sentiment'] = combined_weight_sentiment
    print(f"{index} : {len(df.index)} | Combinado pesos: {combined_weight_sentiment}")

    return combined_weight_sentiment

def get_adjusted_weights(value, weight_ia, weight_classic):
    """Ajusta os pesos para garantir que a soma seja 100%."""
    if value == 0:
        return weight_ia, weight_classic
    else:
        adjusted_classic_weight = weight_classic * (1 - abs(value))
        adjusted_ia_weight = 1 - adjusted_classic_weight
        return adjusted_ia_weight, adjusted_classic_weight

def combined_abs_sentiment_analysis(sentiment_ia, classic_sentiment, index, df):
    """Calcula o sentimento combinado com base nos pesos ajustados e atualiza o DataFrame."""
    weight_ia = 0.9  # Peso para 'sentiment'
    weight_classic = 0.1  # Peso para 'classic_sentiment' quando for zero

    adjusted_ia_weight, adjusted_classic_weight = get_adjusted_weights(classic_sentiment, weight_ia, weight_classic)
    combined_abs_sentiment = (sentiment_ia * adjusted_ia_weight + classic_sentiment * adjusted_classic_weight)
    combined_abs_sentiment = round(combined_abs_sentiment, 2)  # Arredondar para 2 casas decimais

    df.at[index, 'combined_abs_sentiment'] = combined_abs_sentiment
    print(f"{index} : {len(df.index)} | Combinado abs: {combined_abs_sentiment}")

    return combined_abs_sentiment