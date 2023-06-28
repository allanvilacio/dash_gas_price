import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import numpy as np

df = pd.read_csv('data_gas.csv')
df = df[df['PRODUTO']=='GASOLINA COMUM']

df['DATA'] = df[['DATA INICIAL','DATA FINAL']].astype('datetime64[ms]').apply(np.mean, axis=1)
df = df[['DATA','REGIÃO','ESTADO','PREÇO MÉDIO REVENDA' ]]

df.sort_values(by=['DATA','REGIÃO','ESTADO'], ignore_index=True, inplace=True)

{x :str(x) for x in df['DATA'].dt.year.unique()}

df['DATA'].dt.year.unique().min()
df['REGIÃO'].unique()[1]

df_region = df[df['DATA'].dt.year==2021]
df_region = (df_region.groupby('REGIÃO')['PREÇO MÉDIO REVENDA']
                .mean().sort_values(ascending=True))

[f'{i[0]}, R${i[1]:.2f}' for i in df_region.items()]


df_region.max()