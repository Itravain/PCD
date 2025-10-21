import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

SEQUENCIAL_PATH = "Resultados/sequencial.csv"
OMP_PATH = "Resultados/resultados.csv"
OUT_DIR = "Resultados/Graficos"
INPUTS_DIR = "Geracao_dados/inputs"
CENTROIDS_DIR = "Resultados/Original"

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def add_speedup(seq_path=SEQUENCIAL_PATH, omp_path=OMP_PATH):
    if not os.path.exists(seq_path):
        print(f"Arquivo não encontrado: {seq_path}")
        return 1
    if not os.path.exists(omp_path):
        print(f"Arquivo não encontrado: {omp_path}")
        return 1

    df_seq = pd.read_csv(seq_path)
    df_omp = pd.read_csv(omp_path)

    if 'Tamanho' not in df_seq.columns or 'Tempo(ms)' not in df_seq.columns:
        print("sequencial.csv deve ter colunas: Tempo(ms), Tamanho, SSE_Final, Iteracoes")
        return 1
    if 'Tamanho' not in df_omp.columns or 'Tempo(ms)' not in df_omp.columns:
        print("resultados.csv deve ter colunas: Tempo(ms), Tamanho, SSE_Final, Iteracoes, Threads")
        return 1

    # Calcula média do tempo serial por tamanho (5 repetições)
    df_seq_agg = df_seq.groupby('Tamanho').agg({'Tempo(ms)': 'mean'}).reset_index()
    df_seq_agg = df_seq_agg.rename(columns={'Tempo(ms)': 'Tempo_serial(ms)'})

    df_merged = df_omp.merge(df_seq_agg, on='Tamanho', how='left')
    df_merged['Speedup'] = df_merged['Tempo_serial(ms)'] / df_merged['Tempo(ms)']

    # Nova coluna: pontos por segundo (throughput) usando o tempo paralelo
    tempo_ms = df_merged['Tempo(ms)'].replace(0, np.nan)
    df_merged['pontos/s'] = (df_merged['Tamanho'] * 1000.0) / tempo_ms

    if 'Tempo_serial(ms)' in df_merged.columns:
        df_merged = df_merged.drop(columns=['Tempo_serial(ms)'])

    df_merged.to_csv(omp_path, index=False)
    print(f"✓ Speedup e 'pontos/s' adicionados em {omp_path}")
    return 0

def plot_tempo_vs_threads(df_omp, out_dir=OUT_DIR):
    ensure_dir(out_dir)
    if 'Threads' not in df_omp.columns:
        print("Sem coluna 'Threads' em resultados.csv. Pulando gráfico Tempo vs Threads.")
        return
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_omp['Tamanho'].unique())))

    for i, N in enumerate(sorted(df_omp['Tamanho'].unique())):
        g = df_omp[df_omp['Tamanho'] == N].copy()
        
        # Calcula média e desvio padrão por número de threads
        agg = g.groupby('Threads').agg({
            'Tempo(ms)': ['mean', 'std']
        }).reset_index()
        agg.columns = ['Threads', 'Tempo_mean', 'Tempo_std']
        agg = agg.sort_values('Threads')
        
        if agg.empty:
            continue
        
        x = agg['Threads'].values
        y_mean = agg['Tempo_mean'].values
        y_std = agg['Tempo_std'].values

        plt.errorbar(x, y_mean, 
                     yerr=y_std, 
                     marker='o', capsize=5, capthick=1.5,
                     label=f'N={N:,}',
                     color=colors[i])

    plt.title("Tempo de Execução vs Threads para Diferentes Entradas")
    plt.xlabel("Threads")
    plt.ylabel("Tempo (ms)")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Tamanho (N)")
    out_path = os.path.join(out_dir, "tempo_vs_threads_geral.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✓ Salvo: {out_path}")

def plot_speedup_vs_threads(df_omp, df_seq, out_dir=OUT_DIR):
    ensure_dir(out_dir)
    if 'Threads' not in df_omp.columns:
        print("Sem coluna 'Threads' em resultados.csv. Pulando gráfico Speedup vs Threads.")
        return

    # Recalcula speedup se necessário
    if 'Speedup' not in df_omp.columns:
        df_seq_agg = df_seq.groupby('Tamanho').agg({'Tempo(ms)': 'mean'}).reset_index()
        df_seq_agg = df_seq_agg.rename(columns={'Tempo(ms)': 'Tempo_serial(ms)'})
        df_merged = df_omp.merge(df_seq_agg, on='Tamanho', how='left')
        df_merged['Speedup'] = df_merged['Tempo_serial(ms)'] / df_merged['Tempo(ms)']
    else:
        df_merged = df_omp.copy()

    plt.figure(figsize=(10, 6))
    
    # Cores para as diferentes linhas
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_merged['Tamanho'].unique())))

    for i, N in enumerate(sorted(df_merged['Tamanho'].unique())):
        g = df_merged[df_merged['Tamanho'] == N].copy()
        
        # Calcula média e desvio padrão do speedup
        agg = g.groupby('Threads').agg({
            'Speedup': ['mean', 'std']
        }).reset_index()
        agg.columns = ['Threads', 'Speedup_mean', 'Speedup_std']
        agg = agg.sort_values('Threads')
        
        if agg.empty:
            continue
        
        x = agg['Threads'].values
        y_mean = agg['Speedup_mean'].values
        y_std = agg['Speedup_std'].values

        plt.errorbar(x, y_mean, 
                     yerr=y_std,
                     marker='o', capsize=5, capthick=1.5,
                     label=f'N={N:,}',
                     color=colors[i])
    
    # Linha ideal (plotada uma vez)
    all_threads = sorted(df_merged['Threads'].unique())
    ideal = np.minimum(all_threads, 12)
    plt.plot(all_threads, ideal, '--', color='gray', alpha=0.8, label='Ideal (satura em 12)')
    
    plt.title("Speedup vs Threads para Diferentes Tamanhos de Entrada")
    plt.xlabel("Threads")
    plt.ylabel("Speedup")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Tamanho (N)")
    out_path = os.path.join(out_dir, "speedup_vs_threads_geral.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✓ Salvo: {out_path}")

def _read_sse_series(path):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if df.shape[1] == 1:
            sse = df.iloc[:,0].values.astype(float)
        elif 'SSE' in df.columns:
            sse = df['SSE'].values.astype(float)
        else:
            sse = df.iloc[:,0].values.astype(float)
        return sse
    except Exception as e:
        print(f"Falha ao ler SSE de {path}: {e}")
        return None

def plot_sse_validacao(labels=('pequeno','medio','grande'), threads_ref=4, out_dir=OUT_DIR):
    """
    Plota SSE por iteração comparando serial vs OpenMP para todos os labels em um único gráfico.
    Usa a última repetição disponível de cada configuração.
    """
    ensure_dir(out_dir)
    serial_dir = "Resultados/Original"
    omp_dir = "Resultados/OpenMP"

    plt.figure(figsize=(12, 7))
    any_plotted = False
    
    # Paleta de cores para distinguir os labels
    colors = {'pequeno': ('#E15759', '#F28E2B'), 'medio': ('#4E79A7', '#59A14F'), 'grande': ('#76B7B2', '#B07AA1')}
    
    for lbl in labels:
        # Busca última repetição do serial (sse_<label>_r*.csv)
        serial_pattern = os.path.join(serial_dir, f"sse_{lbl}_r*.csv")
        serial_files = sorted(glob(serial_pattern))
        serial_path = serial_files[-1] if serial_files else None
        
        # Busca última repetição do OpenMP com threads_ref
        omp_pattern = os.path.join(omp_dir, f"sse_{lbl}_t{threads_ref}_r*.csv")
        omp_files = sorted(glob(omp_pattern))
        omp_path = omp_files[-1] if omp_files else None

        if not serial_path and not omp_path:
            print(f"SSE não encontrado para '{lbl}'. Padrões tentados:")
            print(f"  Serial: {serial_pattern}")
            print(f"  OpenMP: {omp_pattern}")
            continue

        sse_serial = _read_sse_series(serial_path) if serial_path else None
        sse_omp = _read_sse_series(omp_path) if omp_path else None

        if sse_serial is None and sse_omp is None:
            print(f"Falha ao ler SSE para '{lbl}'. Pulando.")
            continue

        any_plotted = True
        
        if sse_serial is not None:
            it_serial = np.arange(1, len(sse_serial)+1)
            plt.plot(it_serial, sse_serial, 
                    label=f'Serial ({lbl})', 
                    marker='.', markersize=4,
                    linestyle='-', lw=2,
                    color=colors.get(lbl, ('black', 'gray'))[0],
                    alpha=0.9)
        
        if sse_omp is not None:
            it_omp = np.arange(1, len(sse_omp)+1)
            plt.plot(it_omp, sse_omp, 
                    label=f'OpenMP ({lbl}, {threads_ref}t)', 
                    marker='.', markersize=4,
                    linestyle='--', lw=2,
                    color=colors.get(lbl, ('black', 'gray'))[1],
                    alpha=0.9)
    
    if any_plotted:
        plt.title("Convergência do SSE (Serial vs OpenMP)", fontsize=14, fontweight='bold')
        plt.xlabel("Iteração", fontsize=12)
        plt.ylabel("SSE (escala log)", fontsize=12)
        plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.7)
        plt.legend(loc='best', framealpha=0.95, fontsize=10)
        plt.yscale('log')
        
        out_path = os.path.join(out_dir, "sse_por_iteracao_geral.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"✓ Salvo: {out_path}")
    else:
        plt.close() # Fecha a figura se nada foi plotado
        print("Nenhum gráfico de SSE por iteração gerado (arquivos não encontrados).")

def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def _read_vector_csv_general(path, candidate_cols=('altura_cm', 'x', 'valor', 'value')):
    try:
        df = pd.read_csv(path)
        # Tenta colunas conhecidas
        for c in candidate_cols:
            if c in df.columns:
                return df[c].astype(float).values
        # Senão, pega a primeira coluna numérica
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                return df[c].astype(float).values
        # Fallback: primeira coluna
        return df.iloc[:, 0].astype(float).values
    except Exception as e:
        print(f"Falha ao ler vetor de {path}: {e}")
        return None

def _find_centroids_for_label(label, centroids_dir):
    # Tenta centroids_<label>.* e, se não achar, busca por arquivos que contenham dígitos do label
    candidates = [
        os.path.join(centroids_dir, f"centroids_{label}.csv"),
        os.path.join(centroids_dir, f"centroids_{label}.txt"),
        os.path.join(centroids_dir, f"centroids_{label}.dat"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    try:
        for fname in os.listdir(centroids_dir):
            if fname.startswith(f"centroids_{label}"):
                return os.path.join(centroids_dir, fname)
    except FileNotFoundError:
        return None
    digits = ''.join(ch for ch in label if ch.isdigit())
    if digits:
        try:
            for fname in os.listdir(centroids_dir):
                if fname.startswith("centroids_") and digits in fname:
                    return os.path.join(centroids_dir, fname)
        except FileNotFoundError:
            pass
    return None

def plot_distribuicao_e_centroides(labels=('pequeno','medio','grande'),
                                   data_paths=None,
                                   centroid_paths=None,
                                   out_dir=OUT_DIR,
                                   bins=40):
    """
    Plota, para cada label, o histograma (distribuição) 1D dos dados
    com linhas verticais marcando os centroides finais.

    data_paths: dict opcional {label: caminho_csv_dados}
    centroid_paths: dict opcional {label: caminho_csv_centroides}
    """
    ensure_dir(out_dir)
    any_plotted = False
    figs = []

    for lbl in labels:
        # Descobre caminho dos dados
        if data_paths and lbl in data_paths:
            data_path = data_paths[lbl]
        else:
            candidates_data = [
                os.path.join("Resultados", "Datasets", f"dados_{lbl}.csv"),
                os.path.join("Resultados", lbl, "dados.csv"),
                os.path.join("Resultados", f"{lbl}.csv"),
                f"dados_{lbl}.csv",
                f"{lbl}.csv",
                "dados.csv",  # fallback global (ver gerador: geradorDados.py)
            ]
            data_path = _first_existing(candidates_data)

        if not data_path:
            print(f"Dados não encontrados para '{lbl}'. Informe data_paths['{lbl}'] ou crie um CSV.")
            continue

        dados = _read_vector_csv_general(data_path)
        if dados is None or len(dados) == 0:
            print(f"Falha ao ler dados para '{lbl}' em {data_path}.")
            continue

        # Descobre caminho dos centroides finais
        if centroid_paths and lbl in centroid_paths:
            cent_path = centroid_paths[lbl]
        else:
            candidates_cent = [
                os.path.join("Resultados", "Original", f"centroides_finais_{lbl}.csv"),
                os.path.join("Resultados", "OpenMP", f"centroides_finais_{lbl}.csv"),
                os.path.join("Resultados", f"centroides_finais_{lbl}.csv"),
                f"centroides_finais_{lbl}.csv",
                "Resultados/centroides_finais.csv",
                "centroides_finais.csv",
                # Fallback: usa iniciais se finais não existirem
                "Resultados/centroides_iniciais.csv",
                "centroides_iniciais.csv",
            ]
            cent_path = _first_existing(candidates_cent)

        centroides = None
        if cent_path:
            centroides = _read_vector_csv_general(cent_path)
        else:
            print(f"Centroides não encontrados para '{lbl}'. Vou plotar apenas a distribuição.")

        # Plot individual
        plt.figure(figsize=(8, 4.5))
        plt.hist(dados, bins=bins, alpha=0.45, color='#4C78A8', edgecolor='white')
        # Rug plot simples (amostra para não poluir)
        sample = dados if len(dados) <= 800 else np.random.choice(dados, size=800, replace=False)
        y0 = np.zeros_like(sample)
        plt.plot(sample, y0, '|', color='black', alpha=0.25)

        if centroides is not None and len(centroides) > 0:
            for i, c in enumerate(sorted(centroides)):
                plt.axvline(c, color='#F58518', linestyle='--', linewidth=2, alpha=0.9)
                plt.text(c, plt.ylim()[1]*0.95, f"C{i}", color='#F58518',
                         ha='center', va='top', fontsize=9, rotation=90)

        plt.title(f"Distribuição e centroides – {lbl}")
        plt.xlabel("Valor")
        plt.ylabel("Contagem")
        plt.grid(True, alpha=0.25)
        out_path = os.path.join(out_dir, f"distribuicao_centroides_{lbl}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        any_plotted = True
        print(f"✓ Salvo: {out_path}")

    if not any_plotted:
        print("Nenhum gráfico de distribuição/centroides foi gerado (arquivos não encontrados).")

def plot_distribuicoes_inputs_e_centroides(inputs_dir=INPUTS_DIR,
                                           centroids_dir=CENTROIDS_DIR,
                                           out_dir=OUT_DIR,
                                           bins=40):
    """
    Varre todos os arquivos em Geracao_dados/inputs/ e plota a distribuição (1D)
    com linhas verticais para os centroides finais encontrados em Resultados/Original/centroids_*.
    """
    ensure_dir(out_dir)
    if not os.path.isdir(inputs_dir):
        print(f"Diretório de inputs não encontrado: {inputs_dir}")
        return

    files = sorted([f for f in os.listdir(inputs_dir)
                    if f.lower().endswith(('.csv', '.txt', '.dat'))])
    if not files:
        print(f"Nenhum arquivo de dados encontrado em {inputs_dir}")
        return

    any_plotted = False
    for fname in files:
        label = os.path.splitext(fname)[0]
        data_path = os.path.join(inputs_dir, fname)
        dados = _read_vector_csv_general(data_path)
        if dados is None or len(dados) == 0:
            print(f"Falha ao ler dados de {data_path}.")
            continue

        cent_path = _find_centroids_for_label(label, centroids_dir)
        centroides = _read_vector_csv_general(cent_path) if cent_path else None
        if not cent_path:
            print(f"Centroides não encontrados para '{label}' em {centroids_dir}. Plotando só a distribuição.")

        plt.figure(figsize=(8, 4.5))
        plt.hist(dados, bins=bins, alpha=0.45, color='#4C78A8', edgecolor='white')

        # Rug plot (amostra) para visual de distribuição linear dos pontos
        sample = dados if len(dados) <= 800 else np.random.choice(dados, size=800, replace=False)
        plt.plot(sample, np.zeros_like(sample), '|', color='black', alpha=0.25)

        if centroides is not None and len(centroides) > 0:
            ymax = plt.ylim()[1]
            for i, c in enumerate(sorted(centroides)):
                plt.axvline(c, color='#F58518', linestyle='--', linewidth=2, alpha=0.9)
                plt.text(c, ymax*0.95, f"C{i}", color='#F58518',
                         ha='center', va='top', fontsize=9, rotation=90)

        plt.title(f"Distribuição e centroides – {label}")
        plt.xlabel("Valor")
        plt.ylabel("Contagem")
        plt.grid(True, alpha=0.25)
        out_path = os.path.join(out_dir, f"distribuicao_centroides_{label}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        any_plotted = True
        print(f"✓ Salvo: {out_path}")

    if not any_plotted:
        print("Nenhum gráfico de distribuição/centroides foi gerado.")

def diagnosticar_hierarquia():
    print("\n=== Diagnóstico de hierarquia ===")
    print(f"- INPUTS_DIR: {INPUTS_DIR} -> {'ok' if os.path.isdir(INPUTS_DIR) else 'não encontrado'}")
    print(f"- CENTROIDS_DIR: {CENTROIDS_DIR} -> {'ok' if os.path.isdir(CENTROIDS_DIR) else 'não encontrado'}")
    print(f"- OMP_PATH: {OMP_PATH} -> {'ok' if os.path.exists(OMP_PATH) else 'não encontrado'}")
    print(f"- SEQUENCIAL_PATH: {SEQUENCIAL_PATH} -> {'ok' if os.path.exists(SEQUENCIAL_PATH) else 'não encontrado'}")

    inputs = sorted(glob(os.path.join(INPUTS_DIR, "*.*")))
    cents = sorted(glob(os.path.join(CENTROIDS_DIR, "centroids_*.*")))
    sse_serial = sorted(glob(os.path.join("Resultados", "Original", "sse_*.csv")))
    sse_omp = sorted(glob(os.path.join("Resultados", "OpenMP", "sse_*.csv")))

    def _preview(lst, maxn=8):
        if not lst:
            return "nenhum"
        head = [os.path.relpath(p) for p in lst[:maxn]]
        more = f" (+{len(lst)-maxn})" if len(lst) > maxn else ""
        return ", ".join(head) + more

    print(f"- Arquivos em {INPUTS_DIR}: {_preview(inputs)}")
    print(f"- Centróides em {CENTROIDS_DIR} (centroids_*): {_preview(cents)}")
    print(f"- SSE serial (Resultados/Original): {_preview(sse_serial)}")
    print(f"- SSE OpenMP (Resultados/OpenMP): {_preview(sse_omp)}")
    print("=== Fim do diagnóstico ===\n")

def main():
    ensure_dir(OUT_DIR)
    diagnosticar_hierarquia()

    # 1) Se existirem, adiciona speedup e plota tempo/speedup
    have_results = os.path.exists(OMP_PATH) and os.path.exists(SEQUENCIAL_PATH)
    if have_results:
        print("\n" + "="*60)
        print("Calculando Speedup...")
        print("="*60)
        add_speedup(SEQUENCIAL_PATH, OMP_PATH)

        df_omp = pd.read_csv(OMP_PATH)
        df_seq = pd.read_csv(SEQUENCIAL_PATH)

        print("\n" + "="*60)
        print("Gerando gráficos de performance...")
        print("="*60 + "\n")

        plot_tempo_vs_threads(df_omp, OUT_DIR)
        plot_speedup_vs_threads(df_omp, df_seq, OUT_DIR)
    else:
        print("Resultados não encontrados (Resultados/resultados.csv ou Resultados/sequencial.csv). Pulando gráficos de performance.")

    # 2) SSE por iteração (validação) — só plota se os arquivos existirem
    plot_sse_validacao(labels=('pequeno','medio','grande'), threads_ref=4, out_dir=OUT_DIR)

    # 3) Distribuição (1D) + centroides finais a partir dos inputs e centroids_*
    print("\nGerando distribuições com centróides a partir de Geracao_dados/inputs e Resultados/Original/centroids_* ...\n")
    plot_distribuicoes_inputs_e_centroides(inputs_dir=INPUTS_DIR,
                                           centroids_dir=CENTROIDS_DIR,
                                           out_dir=OUT_DIR)

    print("\n" + "="*60)
    print(f"✓ Todos os gráficos disponíveis foram gerados em {OUT_DIR}")
    print("="*60)
    return 0

if __name__ == "__main__":
    sys.exit(main())