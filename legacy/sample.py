# Lets take a sample to run some tests on a workflow 

import pandas as pd 

path_to_the_truth= 'dados/microdados_enem_2019/dados_enem_processados.csv.zip'
df = pd.read_csv(path_to_the_truth)

sample_df = df.sample(n=10000, random_state=69)
sample_df.set_index('NU_INSCRICAO', inplace=True)

sample_df.to_csv('dados/microdados_enem_2019/sample_dados_enem_processados.csv', index=True)
