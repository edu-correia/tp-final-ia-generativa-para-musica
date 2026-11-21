
from pathlib import Path
import pandas as pd

# Pasta principal do dataset
PASTA_DATASET = Path("dataset/EMOPIA_1.0")

# Arquivos que queremos analisar
arquivos = [
    "label.csv",
    "metadata_by_song.csv"
]

for nome_arquivo in arquivos:

    caminho = PASTA_DATASET / nome_arquivo

    print("\n" + "=" * 60)
    print(f"ARQUIVO: {nome_arquivo}")
    print("=" * 60)

    if not caminho.exists():
        print("Arquivo não encontrado:", caminho)
        continue

    dados = pd.read_csv(caminho)

    print("\nColunas:")
    print(dados.columns.tolist())

    print("\nPrimeiras 5 linhas:")
    print(dados.head().to_string(index=False))

    print("\nQuantidade de registros:")
    print(len(dados))

    print("\nInformações:")
    dados.info()