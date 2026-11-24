import argparse
import csv

from src.dataset_treino import carregar_tokenizador
from src.treinar import (
    PASTA_RESULTADOS_TREINO,
    carregar_dados,
    criar_parser,
    treinar,
)


# Busca pequena de hiperparâmetros, pensada para prazo curto.
# Cada configuração treina poucas épocas e é comparada pela val_loss.

CONFIGURACOES = {
    "A": {"dim_embedding": 256, "dim_oculta": 512, "num_camadas": 2, "dropout": 0.3},
    "B": {"dim_embedding": 256, "dim_oculta": 512, "num_camadas": 3, "dropout": 0.3},
    "C": {"dim_embedding": 256, "dim_oculta": 768, "num_camadas": 2, "dropout": 0.3},
    "D": {"dim_embedding": 256, "dim_oculta": 512, "num_camadas": 2, "dropout": 0.5},
}

LR = 1e-3
TAMANHO_LOTE = 32
EPOCAS_BUSCA = 6
PACIENCIA_BUSCA = 2

CAMINHO_SAIDA = PASTA_RESULTADOS_TREINO / "busca_hiperparametros.csv"


def montar_argumentos(nome, config, max_lotes):

    lista = [
        "--dim_embedding", str(config["dim_embedding"]),
        "--dim_oculta", str(config["dim_oculta"]),
        "--num_camadas", str(config["num_camadas"]),
        "--dropout", str(config["dropout"]),
        "--lr", str(LR),
        "--tamanho_lote", str(TAMANHO_LOTE),
        "--epocas", str(EPOCAS_BUSCA),
        "--paciencia", str(PACIENCIA_BUSCA),
        "--nome_experimento", f"busca_{nome}",
    ]

    if max_lotes is not None:
        lista += ["--max_lotes_por_epoca", str(max_lotes)]

    return criar_parser().parse_args(lista)


def comando_treino_final(config):

    return (
        "python -m src.treinar"
        f" --dim_embedding {config['dim_embedding']}"
        f" --dim_oculta {config['dim_oculta']}"
        f" --num_camadas {config['num_camadas']}"
        f" --dropout {config['dropout']}"
        f" --lr {LR} --tamanho_lote {TAMANHO_LOTE}"
        " --epocas 30 --paciencia 4"
        " --nome_experimento lstm_final"
    )


def salvar_resultados(linhas):

    CAMINHO_SAIDA.parent.mkdir(parents=True, exist_ok=True)

    colunas = [
        "config", "dim_embedding", "dim_oculta", "num_camadas", "dropout",
        "lr", "tamanho_lote", "max_lotes_por_epoca", "numero_parametros",
        "melhor_epoca", "melhor_val_loss", "melhor_val_perplexidade",
        "epocas_treinadas", "tempo_total_s",
    ]

    with open(CAMINHO_SAIDA, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=colunas)
        escritor.writeheader()
        escritor.writerows(linhas)


def buscar(max_lotes=None, configuracoes=None):

    nomes = configuracoes or list(CONFIGURACOES)

    _, pad_id = carregar_tokenizador()

    # Os dados são carregados uma vez e reaproveitados em todas as configurações.
    dados = carregar_dados(TAMANHO_LOTE, pad_id)

    linhas = []

    for nome in nomes:

        config = CONFIGURACOES[nome]
        argumentos = montar_argumentos(nome, config, max_lotes)

        resumo = treinar(argumentos, dados=dados)

        linhas.append({
            "config": nome,
            **config,
            "lr": LR,
            "tamanho_lote": TAMANHO_LOTE,
            "max_lotes_por_epoca": max_lotes,
            "numero_parametros": resumo["numero_parametros"],
            "melhor_epoca": resumo["melhor_epoca"],
            "melhor_val_loss": resumo["melhor_val_loss"],
            "melhor_val_perplexidade": resumo["melhor_val_perplexidade"],
            "epocas_treinadas": resumo["epocas_treinadas"],
            "tempo_total_s": resumo["tempo_total_s"],
        })

        # Salva a cada configuração, para não perder resultados parciais.
        salvar_resultados(linhas)

    melhor = min(linhas, key=lambda linha: linha["melhor_val_loss"])

    print("\n" + "=" * 60)
    print("RESULTADO DA BUSCA")
    print("=" * 60)

    for linha in sorted(linhas, key=lambda linha: linha["melhor_val_loss"]):
        print(
            f"{linha['config']}: val_loss {linha['melhor_val_loss']:.4f} | "
            f"ppl {linha['melhor_val_perplexidade']:.2f} | "
            f"{linha['numero_parametros']:,} parâmetros | "
            f"{linha['tempo_total_s'] / 60:.1f} min"
        )

    print(f"\nMelhor configuração: {melhor['config']}")
    print("Resultados salvos em:", CAMINHO_SAIDA)
    print("\nComando para o treino final:")
    print(comando_treino_final(CONFIGURACOES[melhor["config"]]))

    return linhas


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Busca pequena de hiperparâmetros do LSTM."
    )
    parser.add_argument(
        "--max_lotes_por_epoca",
        type=int,
        default=None,
        help="Limita os lotes de treino por época para reduzir o custo."
    )
    parser.add_argument(
        "--configuracoes",
        nargs="+",
        choices=list(CONFIGURACOES),
        default=None,
        help="Subconjunto de configurações a rodar (padrão: todas)."
    )

    argumentos = parser.parse_args()

    buscar(argumentos.max_lotes_por_epoca, argumentos.configuracoes)
