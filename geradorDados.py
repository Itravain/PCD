import numpy as np
import pandas as pd

# Configurações
np.random.seed(42)  # Para reprodutibilidade
num_clusters = 5
num_pessoas_por_grupo = 100

# Simulação de dados em 1D (altura em cm)
# Grupos: crianças, adolescentes, adultos jovens, adultos, idosos
alturas_criancas = np.random.normal(loc=80, scale=10, size=num_pessoas_por_grupo)  # 3 a 10 anos
alturas_adolescentes = np.random.normal(loc=160, scale=8, size=num_pessoas_por_grupo)  # 11 a 17 anos
alturas_adultos_jovens = np.random.normal(loc=175, scale=7, size=num_pessoas_por_grupo)  # 18 a 30 anos
alturas_adultos = np.random.normal(loc=170, scale=6, size=num_pessoas_por_grupo)  # 31 a 50 anos
alturas_idosos = np.random.normal(loc=165, scale=5, size=num_pessoas_por_grupo)  # 51+ anos

# Combina todos os grupos
dados = np.concatenate([alturas_criancas, alturas_adolescentes, alturas_adultos_jovens, alturas_adultos, alturas_idosos])

# Salva os dados em CSV
df_dados = pd.DataFrame(dados, columns=['altura_cm'])
df_dados.to_csv('dados.csv', index=False)
print("Arquivo 'dados.csv' gerado com sucesso!")

# Gerar centroides iniciais aleatórios dentro do intervalo dos dados
min_val, max_val = dados.min(), dados.max()
centroides_iniciais = np.random.uniform(low=min_val, high=max_val, size=num_clusters)

# Salva os centroides iniciais em CSV
df_centroides = pd.DataFrame(centroides_iniciais, columns=['centroide_inicial'])
df_centroides.to_csv('centroides_iniciais.csv', index=False)
print("Arquivo 'centroides_iniciais.csv' gerado com sucesso!")

