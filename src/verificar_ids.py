from pathlib import Path
import json

from src.dataset_treino import (
    PASTA_PROJETO,
    PASTA_TOKENS_TREINO,
    CONJUNTOS,
    carregar_tokenizador,
)


# Confere se os IDs de processed_tokens_convertidos/ estão na ordem
# do tokenizador carregado, que é a usada no treino e na geração.

PASTA_VERIFICACAO = PASTA_PROJETO / "resultados" / "verificacao"

EMOCOES = ["Q1", "Q2", "Q3", "Q4"]

NOMES_ESPECIAIS = [
    "PAD_None",
    "BOS_None",
    "EOS_None",
    "MASK_None",
] + [f"Emotion_{emocao}" for emocao in EMOCOES]


def obter_ids_especiais(tokenizador):

    return {
        nome: int(tokenizador[nome])
        for nome in NOMES_ESPECIAIS
    }


def verificar_conjunto(conjunto, ids_especiais, tamanho_vocabulario):

    pasta = PASTA_TOKENS_TREINO / conjunto
    arquivos = sorted(pasta.glob("*.json"))

    ids_emocao = {
        emocao: ids_especiais[f"Emotion_{emocao}"]
        for emocao in EMOCOES
    }
    conjunto_especiais = set(ids_especiais.values())

    erros = []
    total_tokens = 0

    for caminho in arquivos:

        with open(caminho, "r", encoding="utf-8") as arquivo:
            dados = json.load(arquivo)

        tokens = dados["tokens"]
        emocao = dados["emotion"]
        total_tokens += len(tokens)

        if tokens[0] != ids_emocao[emocao]:
            erros.append(
                f"{caminho.name}: primeiro token {tokens[0]}, "
                f"esperado {ids_emocao[emocao]} (Emotion_{emocao})"
            )

        especiais_no_meio = [
            token for token in tokens[1:]
            if token in conjunto_especiais
        ]

        if especiais_no_meio:
            erros.append(
                f"{caminho.name}: {len(especiais_no_meio)} tokens especiais "
                f"na parte musical"
            )

        fora_do_vocabulario = [
            token for token in tokens
            if token < 0 or token >= tamanho_vocabulario
        ]

        if fora_do_vocabulario:
            erros.append(
                f"{caminho.name}: {len(fora_do_vocabulario)} IDs fora do vocabulário"
            )

    print(
        f"{conjunto.upper()}: {len(arquivos)} arquivos, "
        f"{total_tokens} tokens, {len(erros)} erros"
    )

    for erro in erros[:10]:
        print("  -", erro)

    return len(arquivos), erros


def decodificar_exemplo(tokenizador):

    arquivos = sorted((PASTA_TOKENS_TREINO / "val").glob("*.json"))
    caminho = arquivos[0]

    with open(caminho, "r", encoding="utf-8") as arquivo:
        dados = json.load(arquivo)

    # Remove o token de emoção antes de decodificar.
    ids_musicais = dados["tokens"][1:]

    score = tokenizador.decode(ids_musicais)

    PASTA_VERIFICACAO.mkdir(parents=True, exist_ok=True)
    caminho_saida = PASTA_VERIFICACAO / f"{dados['clip_name']}_decodificado.mid"
    score.dump_midi(caminho_saida)

    numero_notas = sum(len(trilha.notes) for trilha in score.tracks)

    # Converte a duração de ticks para segundos.
    score_segundos = score.to("second")
    duracao = score_segundos.end()

    print("\nDECODIFICAÇÃO DE EXEMPLO")
    print("Clipe:", dados["clip_name"], f"({dados['emotion']})")
    print("Tokens musicais:", len(ids_musicais))
    print("Notas:", numero_notas)
    print(f"Duração: {duracao:.2f} s")
    print("Arquivo salvo em:", caminho_saida)

    return numero_notas, duracao


def verificar_ids():

    tokenizador, _ = carregar_tokenizador()
    ids_especiais = obter_ids_especiais(tokenizador)

    print("\nIDs especiais do tokenizador carregado:")
    for nome, token_id in ids_especiais.items():
        print(f"  {nome}: {token_id}")

    print("\nVERIFICANDO", PASTA_TOKENS_TREINO)
    print("=" * 50)

    total_erros = 0

    for conjunto in CONJUNTOS:
        _, erros = verificar_conjunto(
            conjunto,
            ids_especiais,
            len(tokenizador)
        )
        total_erros += len(erros)

    decodificar_exemplo(tokenizador)

    print("\n" + "=" * 50)

    if total_erros == 0:
        print("SUCESSO: IDs consistentes com o tokenizador carregado!")
    else:
        print(f"ATENÇÃO: {total_erros} problemas encontrados.")

    return total_erros


if __name__ == "__main__":
    verificar_ids()
