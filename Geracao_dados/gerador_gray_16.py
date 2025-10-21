import numpy as np
import pandas as pd
import os

np.random.seed(42)

# Cria diretórios se não existirem
os.makedirs('./inputs/pequeno', exist_ok=True)
os.makedirs('./inputs/medio', exist_ok=True)
os.makedirs('./inputs/grande', exist_ok=True)

def gerar_dataset(n_total, num_centroides, nome_dir, faixas_base):
    """
    Gera dados com mistura de gaussianas em faixas específicas.
    
    Args:
        n_total: número total de pontos
        num_centroides: número de clusters (K)
        nome_dir: diretório de saída
        faixas_base: lista de valores centrais para as gaussianas
    """
    print(f"\n{'='*60}")
    print(f"Gerando dataset: {nome_dir}")
    print(f"N={n_total:,}, K={num_centroides}")
    print(f"{'='*60}")
    
    # Distribui proporções igualmente entre as faixas
    num_faixas = len(faixas_base)
    proporcoes = [1.0 / num_faixas] * num_faixas
    
    # Define desvios padrão proporcionais à distância entre faixas
    if len(faixas_base) > 1:
        dist_media = np.mean(np.diff(sorted(faixas_base)))
        sigma_base = dist_media / 4  # 1/4 da distância entre centros
    else:
        sigma_base = 2.0
    
    sigmas = [sigma_base * np.random.uniform(0.8, 1.2) for _ in range(num_faixas)]
    
    # Calcula tamanhos inteiros por componente
    ns = [int(p * n_total) for p in proporcoes]
    ns[0] += n_total - sum(ns)  # corrige arredondamento
    
    print(f"Faixas (médias): {faixas_base}")
    print(f"Desvios padrão: {[f'{s:.2f}' for s in sigmas]}")
    print(f"Pontos por faixa: {ns}")
    
    # Gera e concatena blocos de dados
    blocos = []
    for mu, sigma, n in zip(faixas_base, sigmas, ns):
        bloco = np.random.normal(mu, sigma, size=n)
        blocos.append(bloco)
    
    dados = np.concatenate(blocos).astype(np.float64)
    np.random.shuffle(dados)
    
    # Salva dados (1 valor por linha, sem cabeçalho)
    dados_path = f'./inputs/{nome_dir}/dados.csv'
    pd.DataFrame(dados).to_csv(dados_path, index=False, header=False)
    print(f"✓ Salvo: {dados_path}")
    print(f"  Min: {dados.min():.2f}, Max: {dados.max():.2f}, Média: {dados.mean():.2f}")
    
    # Centróides iniciais: distribui uniformemente no intervalo dos dados
    data_min, data_max = dados.min(), dados.max()
    centroides_iniciais = np.linspace(data_min, data_max, num_centroides)
    # Adiciona pequena perturbação aleatória
    perturbacao = (data_max - data_min) * 0.05
    centroides_iniciais += np.random.uniform(-perturbacao, perturbacao, size=num_centroides)
    
    centroides_path = f'./inputs/{nome_dir}/centroides_iniciais.csv'
    pd.DataFrame(centroides_iniciais).to_csv(centroides_path, index=False, header=False)
    print(f"✓ Salvo: {centroides_path}")
    print(f"  Centróides: {[f'{c:.2f}' for c in sorted(centroides_iniciais)]}")

# ========== PEQUENO: N=10^4, K=4 ==========
# Faixas bem separadas para facilitar visualização
gerar_dataset(
    n_total=100_000,
    num_centroides=4,
    nome_dir='pequeno',
    faixas_base=[0, 10, 20, 30]
)

# ========== MÉDIO: N=10^5, K=8 ==========
# Mais faixas, ainda bem visíveis
gerar_dataset(
    n_total=1_000_000,
    num_centroides=8,
    nome_dir='medio',
    faixas_base=[0, 10, 20, 30, 40, 50, 60, 70]
)

# ========== GRANDE: N=10^6, K=16 ==========
# Muitas faixas em escala de tons de cinza
gerar_dataset(
    n_total=10_000_000,
    num_centroides=16,
    nome_dir='grande',
    faixas_base=[i * 15 for i in range(16)]  # 0, 15, 30, 45, ..., 225
)

print(f"\n{'='*60}")
print("✓ Todos os datasets gerados com sucesso!")
