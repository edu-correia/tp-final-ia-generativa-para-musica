from pathlib import Path
import json
import statistics
import pandas as pd
from miditok import REMI, TokenizerConfig


# Caminhos principais do projeto.

PASTA_PROJETO = Path(__file__).resolve().parents[1]
PASTA_PROCESSADOS = PASTA_PROJETO / "processed_midi"
PASTA_TOKENS = PASTA_PROJETO / "processed_tokens"
PASTA_SPLITS = PASTA_PROJETO / "splits"

CONJUNTOS = ["train", "val", "test"]
EMOCOES = ["Q1", "Q2", "Q3", "Q4"]


def criar_tokenizador():

    configuracao = TokenizerConfig(
        pitch_range=(21, 109),
        beat_res={(0, 4): 4, (4, 12): 4},
        num_velocities=32,
        special_tokens=["PAD", "BOS", "EOS", "MASK"],
        use_velocities=True,
        use_chords=False,
        use_rests=False,
        use_tempos=True,
        use_time_signatures=False,
        use_programs=True,
    )

    tokenizador = REMI(configuracao)

    for emocao in EMOCOES:
        tokenizador.add_to_vocab(
            f"Emotion_{emocao}",
            special_token=True
        )

    return tokenizador


def identificar_emocao(nome_clipe):

    emocao = nome_clipe.split("_")[0]

    if emocao not in EMOCOES:
        raise ValueError(
            f"Emoção inválida no nome do clipe: {nome_clipe}"
        )

    return emocao


def tokenizar_midi(caminho_midi, tokenizador):

    resultado = tokenizador(caminho_midi)

    if isinstance(resultado, list):

        if len(resultado) != 1:
            raise ValueError(
                "Era esperada uma sequência única do REMI."
            )

        resultado = resultado[0]

    ids_musicais = resultado.ids

    if not ids_musicais:
        raise ValueError("O MIDI gerou uma sequência vazia.")

    nome_clipe = caminho_midi.stem
    emocao = identificar_emocao(nome_clipe)

    id_emocao = tokenizador[f"Emotion_{emocao}"]

    ids_com_emocao = [id_emocao] + list(ids_musicais)

    return emocao, ids_com_emocao


def obter_arquivos_do_conjunto(nome_conjunto):

    """
    Retorna os arquivos MIDI que devem ser tokenizados.

    Treino:
        - MIDI originais.
        - MIDI transpostos, quando disponíveis.

    Validação e teste:
        - Somente MIDI originais.
    """

    caminho_csv = PASTA_SPLITS / f"{nome_conjunto}_clip.csv"

    if not caminho_csv.exists():
        raise FileNotFoundError(caminho_csv)

    dados_originais = pd.read_csv(caminho_csv)

    arquivos = []

    for _, linha in dados_originais.iterrows():

        nome_clipe = Path(
            str(linha["clip_name"]).strip()
        ).stem

        arquivos.append({
            "clip_name": nome_clipe,
            "pasta": nome_conjunto,
            "origem": "original"
        })

    # As versões transpostas são adicionadas somente ao treino.
    if nome_conjunto == "train":

        caminho_csv_aumentado = (
            PASTA_SPLITS / "train_augmented_clip.csv"
        )

        if caminho_csv_aumentado.exists():

            dados_aumentados = pd.read_csv(
                caminho_csv_aumentado
            )

            for _, linha in dados_aumentados.iterrows():

                nome_clipe = Path(
                    str(linha["clip_name"]).strip()
                ).stem

                arquivos.append({
                    "clip_name": nome_clipe,
                    "pasta": "train_augmented",
                    "origem": "transposto"
                })

    return arquivos


def tokenizar_conjunto(nome_conjunto, tokenizador):

    pasta_saida = PASTA_TOKENS / nome_conjunto
    pasta_saida.mkdir(parents=True, exist_ok=True)

    # Evita manter JSON antigos que não fazem parte da execução atual.
    for arquivo_antigo in pasta_saida.glob("*.json"):
        arquivo_antigo.unlink()

    arquivos = obter_arquivos_do_conjunto(nome_conjunto)

    resultados = []
    erros = []

    total = len(arquivos)

    print(f"\nTOKENIZANDO {nome_conjunto.upper()}")
    print("=" * 55)

    for indice, item in enumerate(arquivos):

        nome_clipe = item["clip_name"]
        origem = item["origem"]

        pasta_midi = PASTA_PROCESSADOS / item["pasta"]
        caminho_midi = pasta_midi / f"{nome_clipe}.mid"

        try:

            if not caminho_midi.exists():
                caminho_midi = pasta_midi / f"{nome_clipe}.midi"

            if not caminho_midi.exists():
                raise FileNotFoundError(caminho_midi)

            emocao, ids = tokenizar_midi(
                caminho_midi,
                tokenizador
            )

            caminho_json = pasta_saida / f"{nome_clipe}.json"

            conteudo = {
                "clip_name": nome_clipe,
                "emotion": emocao,
                "origin": origem,
                "tokens": ids
            }

            with open(
                caminho_json,
                "w",
                encoding="utf-8"
            ) as arquivo:

                json.dump(conteudo, arquivo)

            resultados.append({
                "clip_name": nome_clipe,
                "emotion": emocao,
                "origin": origem,
                "sequence_length": len(ids),
                "arquivo": str(caminho_json)
            })

        except Exception as erro:

            erros.append({
                "clip_name": nome_clipe,
                "origem": origem,
                "erro": str(erro)
            })

        if (indice + 1) % 50 == 0 or (indice + 1) == total:
            print(f"Progresso: {indice + 1}/{total}")

    # Relatórios individuais do conjunto.
    pd.DataFrame(resultados).to_csv(
        PASTA_TOKENS / f"estatisticas_{nome_conjunto}.csv",
        index=False
    )

    pd.DataFrame(
        erros,
        columns=["clip_name", "origem", "erro"]
    ).to_csv(
        PASTA_TOKENS / f"erros_{nome_conjunto}.csv",
        index=False
    )

    quantidade_original = sum(
        1 for item in resultados
        if item["origin"] == "original"
    )

    quantidade_transposta = sum(
        1 for item in resultados
        if item["origin"] == "transposto"
    )

    print("\nRESULTADO")
    print("Esperados:", total)
    print("Tokenizados:", len(resultados))
    print("Originais:", quantidade_original)
    print("Transpostos:", quantidade_transposta)
    print("Erros:", len(erros))

    return resultados, erros


def tokenizar_todos():

    PASTA_TOKENS.mkdir(parents=True, exist_ok=True)

    print("\nCRIANDO TOKENIZADOR REMI")

    tokenizador = criar_tokenizador()

    todos_resultados = []
    total_erros = 0

    for conjunto in CONJUNTOS:

        resultados, erros = tokenizar_conjunto(
            conjunto,
            tokenizador
        )

        todos_resultados.extend(resultados)
        total_erros += len(erros)

    # Salva o tokenizador utilizado nesta execução.
    caminho_tokenizador = (
        PASTA_TOKENS / "tokenizer_remi.json"
    )

    tokenizador.save(caminho_tokenizador)

    # Estatísticas das sequências tokenizadas.
    comprimentos = [
        item["sequence_length"]
        for item in todos_resultados
    ]

    dados_resultados = pd.DataFrame(todos_resultados)

    contagem_emocoes = (
        dados_resultados["emotion"]
        .value_counts()
        .reindex(EMOCOES, fill_value=0)
    )

    contagem_origens = (
        dados_resultados["origin"]
        .value_counts()
    )

    vocabulario = len(tokenizador)

    estatisticas = {
        "total_sequencias": len(todos_resultados),
        "total_erros": total_erros,
        "tamanho_vocabulario": vocabulario,
        "comprimento_medio": (
            statistics.mean(comprimentos) if comprimentos else 0
        ),
        "comprimento_mediano": (
            statistics.median(comprimentos) if comprimentos else 0
        ),
        "comprimento_minimo": (
            min(comprimentos) if comprimentos else 0
        ),
        "comprimento_maximo": (
            max(comprimentos) if comprimentos else 0
        ),
        "sequencias_originais": int(
            contagem_origens.get("original", 0)
        ),
        "sequencias_transpostas": int(
            contagem_origens.get("transposto", 0)
        )
    }

    for emocao in EMOCOES:

        estatisticas[f"clipes_{emocao}"] = int(
            contagem_emocoes[emocao]
        )

    caminho_estatisticas = (
        PASTA_TOKENS / "estatisticas_gerais.json"
    )

    with open(
        caminho_estatisticas,
        "w",
        encoding="utf-8"
    ) as arquivo:

        json.dump(
            estatisticas,
            arquivo,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 55)
    print("TOKENIZAÇÃO FINALIZADA")
    print("=" * 55)

    for chave, valor in estatisticas.items():
        print(f"{chave}: {valor}")

    print("\nTokenizador salvo em:", caminho_tokenizador)
    print("Estatísticas salvas em:", caminho_estatisticas)

    if total_erros == 0:
        print("\nSUCESSO: Todos os clipes foram tokenizados!")
    else:
        print(f"\nATENÇÃO: Foram encontrados {total_erros} erros.")


if __name__ == "__main__":
    tokenizar_todos()
