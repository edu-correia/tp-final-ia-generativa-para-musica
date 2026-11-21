from pathlib import Path
import pandas as pd

# Caminho principal do projeto
PASTA_PROJETO = Path(__file__).resolve().parents[1]

# Caminhos do dataset
PASTA_DATASET = PASTA_PROJETO / "dataset" / "EMOPIA_1.0"

CAMINHO_METADATA = PASTA_DATASET / "metadata_by_song.csv"
CAMINHO_LABELS = PASTA_DATASET / "label.csv"

# Pasta onde serão salvos os arquivos da divisão
PASTA_SPLITS = PASTA_PROJETO / "splits"


def criar_divisao(dados, quantidade=8, semente=1):
    """
    Seleciona uma quantidade de músicas de cada quadrante dominante.
    """

    partes = []

    for quadrante in range(1, 5):
        musicas_quadrante = dados[
            dados["DominantQ"] == quadrante
        ]

        selecionadas = musicas_quadrante.sample(
            n=quantidade,
            random_state=semente
        )

        partes.append(selecionadas)

    return pd.concat(partes)


def main():

    PASTA_SPLITS.mkdir(parents=True, exist_ok=True)

    # Carrega os arquivos CSV
    metadata = pd.read_csv(CAMINHO_METADATA)
    labels = pd.read_csv(CAMINHO_LABELS)

    # Utiliza songID como índice
    metadata = metadata.set_index("songID")

    # Cria o conjunto de teste
    teste = criar_divisao(metadata)

    # Remove as músicas utilizadas no teste
    treino_validacao = metadata.drop(index=teste.index)

    # Cria o conjunto de validação
    validacao = criar_divisao(treino_validacao)

    # As músicas restantes serão utilizadas no treino
    treino = treino_validacao.drop(index=validacao.index)

    # Verifica se não existe sobreposição entre os conjuntos
    assert len(set(treino.index) & set(validacao.index)) == 0
    assert len(set(treino.index) & set(teste.index)) == 0
    assert len(set(validacao.index) & set(teste.index)) == 0

    # Salva a divisão por música
    treino.to_csv(PASTA_SPLITS / "train_SL.csv")
    validacao.to_csv(PASTA_SPLITS / "val_SL.csv")
    teste.to_csv(PASTA_SPLITS / "test_SL.csv")

    # Extrai o identificador da música a partir do ID do clipe.
    # Exemplo: Q1_0vLPYiPN7qY_0 -> 0vLPYiPN7qY
    labels["songID"] = (
        labels["ID"]
        .str.replace(r"^Q[1-4]_", "", regex=True)
        .str.rsplit("_", n=1)
        .str[0]
    )

    # Relaciona cada música ao seu conjunto
    divisao = {}

    for nome, dados in [
        ("train", treino),
        ("val", validacao),
        ("test", teste)
    ]:
        for song_id in dados.index:
            divisao[song_id] = nome

    labels["split"] = labels["songID"].map(divisao)

    # Verifica se todos os clipes foram associados
    if labels["split"].isna().any():
        faltantes = labels[labels["split"].isna()]

        raise ValueError(
            "Existem clipes sem música correspondente:\n"
            + faltantes["ID"].to_string(index=False)
        )

    # Salva a divisão por clipe
    for nome in ["train", "val", "test"]:

        clipes = labels[labels["split"] == nome]

        clipes[["ID"]].rename(
            columns={"ID": "clip_name"}
        ).to_csv(
            PASTA_SPLITS / f"{nome}_clip.csv",
            index=False
        )

    # Exibe os resultados
    print("\nDIVISÃO DO DATASET")
    print("-----------------------------")

    print("\nMúsicas:")
    print(f"Treino: {len(treino)}")
    print(f"Validação: {len(validacao)}")
    print(f"Teste: {len(teste)}")

    print("\nClipes:")
    print(labels["split"].value_counts())

    print(f"\nArquivos salvos em: {PASTA_SPLITS}")


if __name__ == "__main__":
    main()
