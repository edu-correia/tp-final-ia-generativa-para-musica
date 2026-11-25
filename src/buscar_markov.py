import argparse
import csv
import json
import math
import time

from src.dataset_treino import (
    PASTA_PROJETO,
    PASTA_TOKENS_TREINO,
    carregar_tokenizador,
    dividir_em_janelas,
)
from src.markov import (
    QUADRANTES,
    carregar_sequencias,
    obter_ids_proibidos,
    treinar_cadeias,
)


# Escolhe a ordem n e o peso do recuo (lambda) do baseline pela loss de validação
# e mede loss e perplexidade por quadrante, no mesmo formato de avaliar_modelo.py.
#
# A avaliação usa as mesmas janelas da LSTM, então os números são comparáveis.
# O conjunto de teste só é usado com --incluir_teste, na avaliação final.

PASTA_RESULTADOS = PASTA_PROJETO / "resultados" / "markov"
CAMINHO_CONFIG = PASTA_RESULTADOS / "config.json"

ORDENS = (0, 1, 2, 3, 4, 5, 6)
PESOS_RECUO = (1, 2, 4, 8, 16, 32, 64)

# Entre ordens praticamente empatadas na validação, fica a menor.
TOLERANCIA_LOSS = 0.001


def dividir_sequencias(sequencias):

    return {
        quadrante: [j for tokens in lista for j in dividir_em_janelas(tokens)]
        for quadrante, lista in sequencias.items()
    }


# Loss média por token (nats) no total e em cada quadrante.

def avaliar(cadeias, janelas):

    resultados = {}
    total = {"janelas": 0, "tokens": 0, "soma": 0.0}

    for quadrante in QUADRANTES:

        soma = -sum(
            float(cadeias[quadrante].log_probabilidades(j).sum())
            for j in janelas[quadrante]
        )
        tokens = sum(len(j) - 1 for j in janelas[quadrante])

        resultados[quadrante] = {
            "janelas": len(janelas[quadrante]), "tokens": tokens, "soma": soma
        }
        for chave in total:
            total[chave] += resultados[quadrante][chave]

    resultados = {"total": total, **resultados}

    for valores in resultados.values():
        valores["loss"] = valores.pop("soma") / valores["tokens"]
        valores["perplexidade"] = math.exp(valores["loss"])

    return resultados


def salvar_csv(caminho, linhas):

    caminho.parent.mkdir(parents=True, exist_ok=True)

    with open(caminho, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0].keys()))
        escritor.writeheader()
        escritor.writerows(linhas)


def buscar(treino, janelas_val, tamanho_vocabulario, ids_proibidos):

    linhas = []

    for ordem in ORDENS:

        # As contagens não dependem de lambda: treina uma vez por ordem.
        cadeias = treinar_cadeias(
            treino, ordem, PESOS_RECUO[0], tamanho_vocabulario, ids_proibidos
        )
        ngramas = sum(sum(c.contar_ngramas()) for c in cadeias.values())

        for peso in PESOS_RECUO:

            for cadeia in cadeias.values():
                cadeia.peso_recuo = peso

            total = avaliar(cadeias, janelas_val)["total"]

            linhas.append({
                "ordem": ordem,
                "peso_recuo": peso,
                "ngramas_distintos": ngramas,
                "val_loss": round(total["loss"], 6),
                "val_perplexidade": round(total["perplexidade"], 4),
            })

        melhor = min(linhas[-len(PESOS_RECUO):], key=lambda linha: linha["val_loss"])
        print(
            f"ordem {ordem}: val_loss {melhor['val_loss']:.4f} | "
            f"ppl {melhor['val_perplexidade']:.2f} (lambda {melhor['peso_recuo']})"
        )

    return linhas


# Menor ordem cuja loss fica a até TOLERANCIA_LOSS da melhor, com o seu melhor lambda.

def escolher(linhas):

    limite = min(linha["val_loss"] for linha in linhas) + TOLERANCIA_LOSS
    empatadas = [linha for linha in linhas if linha["val_loss"] <= limite]

    return min(empatadas, key=lambda linha: (linha["ordem"], linha["val_loss"]))


def main():

    parser = argparse.ArgumentParser(
        description="Busca a ordem da cadeia de Markov e avalia o baseline."
    )
    parser.add_argument(
        "--incluir_teste",
        action="store_true",
        help="Inclui o conjunto de teste (apenas na avaliação final)."
    )
    argumentos = parser.parse_args()

    tokenizador, _ = carregar_tokenizador()
    tamanho_vocabulario = len(tokenizador)
    ids_proibidos = obter_ids_proibidos(tokenizador)

    treino = carregar_sequencias("train")
    janelas_val = dividir_sequencias(carregar_sequencias("val"))

    linhas = buscar(treino, janelas_val, tamanho_vocabulario, ids_proibidos)
    salvar_csv(PASTA_RESULTADOS / "busca_ordem.csv", linhas)

    melhor = escolher(linhas)
    ordem, peso = melhor["ordem"], melhor["peso_recuo"]
    print(f"\nEscolhido na validação: ordem {ordem}, lambda {peso}")

    inicio = time.time()
    cadeias = treinar_cadeias(treino, ordem, peso, tamanho_vocabulario, ids_proibidos)
    tempo = time.time() - inicio

    avaliacao = []
    conjuntos = {"val": janelas_val}
    if argumentos.incluir_teste:
        conjuntos["test"] = dividir_sequencias(carregar_sequencias("test"))

    for conjunto, janelas in conjuntos.items():
        print(f"\n{conjunto.upper()}")
        for chave, valores in avaliar(cadeias, janelas).items():
            print(
                f"  {chave:>5}: loss {valores['loss']:.4f} | "
                f"ppl {valores['perplexidade']:.2f} | "
                f"{valores['janelas']} janelas, {valores['tokens']} tokens"
            )
            avaliacao.append({
                "conjunto": conjunto,
                "quadrante": chave,
                "janelas": valores["janelas"],
                "tokens": valores["tokens"],
                "loss": round(valores["loss"], 6),
                "perplexidade": round(valores["perplexidade"], 4),
            })

    salvar_csv(PASTA_RESULTADOS / "avaliacao_markov.csv", avaliacao)

    config = {
        "ordem": ordem,
        "peso_recuo": peso,
        "tamanho_vocabulario": tamanho_vocabulario,
        "pasta_tokens": PASTA_TOKENS_TREINO.name,
        "val_loss": melhor["val_loss"],
        "val_perplexidade": melhor["val_perplexidade"],
        "clipes_treino": {q: len(treino[q]) for q in QUADRANTES},
        "ngramas_distintos_por_ordem": {
            q: cadeias[q].contar_ngramas() for q in QUADRANTES
        },
        "tempo_treino_s": round(tempo, 2),
    }
    CAMINHO_CONFIG.write_text(
        json.dumps(config, indent=4, ensure_ascii=False), encoding="utf-8"
    )

    print("\nResultados salvos em:", PASTA_RESULTADOS)


if __name__ == "__main__":
    main()
