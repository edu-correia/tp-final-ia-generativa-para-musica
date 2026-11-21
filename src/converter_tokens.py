import json
from pathlib import Path

from miditok import REMI

from src.tokenizacao import (
    PASTA_PROJETO,
    criar_tokenizador,
)

PASTA_ORIGINAL = PASTA_PROJETO / "processed_tokens"
PASTA_DESTINO = PASTA_PROJETO / "processed_tokens_convertidos"

CONJUNTOS = ["train", "val", "test"]


def converter_sequencia(tokens, tokenizador_original, tokenizador_carregado):
    """
    Converte os IDs de uma sequência para o vocabulário
    do tokenizador carregado, preservando os nomes dos tokens.
    """
    tokens_convertidos = []

    for id_original in tokens:
        nome_token = tokenizador_original[id_original]
        id_convertido = tokenizador_carregado[nome_token]
        tokens_convertidos.append(id_convertido)

    return tokens_convertidos


def converter_conjunto(tokenizador_original, tokenizador_carregado, conjunto):
    pasta_entrada = PASTA_ORIGINAL / conjunto
    pasta_saida = PASTA_DESTINO / conjunto

    pasta_saida.mkdir(parents=True, exist_ok=True)

    arquivos = list(pasta_entrada.glob("*.json"))

    total_convertidos = 0
    erros = []

    print(f"\nCONVERTENDO {conjunto.upper()}")
    print("=" * 50)

    for indice, caminho_entrada in enumerate(arquivos):
        try:
            with open(caminho_entrada, encoding="utf-8") as arquivo:
                dados = json.load(arquivo)

            tokens_originais = dados["tokens"]

            tokens_convertidos = converter_sequencia(
                tokens_originais,
                tokenizador_original,
                tokenizador_carregado,
            )

            dados["tokens"] = tokens_convertidos

            caminho_saida = pasta_saida / caminho_entrada.name

            with open(caminho_saida, "w", encoding="utf-8") as arquivo:
                json.dump(dados, arquivo, ensure_ascii=False)

            total_convertidos += 1

        except Exception as erro:
            erros.append({
                "arquivo": caminho_entrada.name,
                "erro": str(erro),
            })

        if (indice + 1) % 100 == 0 or indice + 1 == len(arquivos):
            print(f"Progresso: {indice + 1}/{len(arquivos)}")

    print("\nRESULTADO")
    print("Arquivos encontrados:", len(arquivos))
    print("Convertidos:", total_convertidos)
    print("Erros:", len(erros))

    return total_convertidos, erros


def converter_todos():
    print("CARREGANDO TOKENIZADORES")

    # Recria o vocabulário na ordem usada na tokenização original.
    tokenizador_original = criar_tokenizador()

    # Carrega o tokenizador salvo, com a ordem reorganizada pelo MidiTok.
    caminho_tokenizador = PASTA_ORIGINAL / "tokenizer_remi.json"
    tokenizador_carregado = REMI(params=caminho_tokenizador)

    print("Tamanho do vocabulário original:", len(tokenizador_original))
    print("Tamanho do vocabulário carregado:", len(tokenizador_carregado))

    total_geral = 0
    total_erros = 0

    for conjunto in CONJUNTOS:
        convertidos, erros = converter_conjunto(
            tokenizador_original,
            tokenizador_carregado,
            conjunto,
        )

        total_geral += convertidos
        total_erros += len(erros)

        if erros:
            caminho_erros = PASTA_DESTINO / f"erros_{conjunto}.json"

            with open(caminho_erros, "w", encoding="utf-8") as arquivo:
                json.dump(erros, arquivo, indent=4, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("CONVERSÃO FINALIZADA")
    print("=" * 50)
    print("Total convertido:", total_geral)
    print("Total de erros:", total_erros)
    print("Pasta de saída:", PASTA_DESTINO)

    if total_erros == 0:
        print("\nSUCESSO: Todos os arquivos foram convertidos!")
    else:
        print("\nATENÇÃO: Verifique os arquivos de erros.")


if __name__ == "__main__":
    converter_todos()
