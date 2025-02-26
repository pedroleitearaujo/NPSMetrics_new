import os
import pandas as pd

output_folder = "./output"
models = os.listdir(output_folder)

df_main = pd.DataFrame()
for model in models:
    models_folder = os.listdir(f"{output_folder}/{model}")
    df = pd.read_csv(f'{output_folder}/{model}/{models_folder[-1]}/output.csv')
    if df_main.empty:
        df_main = df
        df_main.rename(columns={'sentiment': f'{model}_sentiment'}, inplace=True)
        df_main.rename(columns={'confidence': f'{model}_confidence'}, inplace=True)
    else:
        df_main[f'{model}_sentiment'] = df['sentiment']
        df_main[f'{model}_confidence'] = df['confidence']

df_main.to_csv('compare-models.csv', index=False, encoding='utf-8')