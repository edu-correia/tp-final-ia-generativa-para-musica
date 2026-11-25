from pathlib import Path
import argparse
import csv
import json
import random
import sys

import numpy as np
import torch
from miditok import REMI

PASTA_PROJETO = Path(__file__).resolve().parents[1]
if str(PASTA_PROJETO) not in sys.path:
    sys.path.insert(0, str(PASTA_PROJETO))

from src.amostragem import amostrar_proximo_token  # noqa: E402
from src.gerar import (  # noqa: E402
    COLUNAS_METADADOS, QUADRANTES, carregar_comprimentos_teste,
    carregar_ids_especiais, escolher_pasta_tokens, salvar_midi,
)
from src.markov import carregar_sequencias, obter_ids_proibidos, treinar_cadeias  # noqa: E402


# Gera as amostras do baseline no mesmo formato de gerar.py (LSTM): mesma função
# de amostragem, mesmos nomes de arquivo e mesmo metadados.csv.
#
# Não há checkpoint: o treino é só contagem e leva poucos segundos.

CONFIG_PADRAO = PASTA_PROJETO / "resultados" / "markov" / "config.json"
METADADOS_LSTM = PASTA_PROJETO / "amostras_geradas" / "lstm" / "metadados.csv"


def carregar_comprimentos_pareados(caminho: Path) -> dict[str, list[int]]:
    # Com os comprimentos da LSTM, a música i de cada quadrante tem o mesmo tamanho
    # nos dois sistemas e o tamanho não interfere na comparação.
    comprimentos = {q: [] for q in QUADRANTES}
    with caminho.open(encoding="utf-8") as arquivo:
        for linha in csv.DictReader(arquivo):
            comprimentos[linha["quadrante"]].append(int(linha["max_len"]))
    return comprimentos


def definir_comprimentos(origem: Path | None, n: int, pasta_tokens: Path) -> dict[str, list[int]]:
    if origem is not None and origem.exists():
        comprimentos = carregar_comprimentos_pareados(origem)
        if all(len(valores) >= n for valores in comprimentos.values()):
            print(f"Comprimentos: pareados com {origem}")
            return {q: valores[:n] for q, valores in comprimentos.items()}
        print(f"(aviso) {origem} tem menos de {n} músicas em algum quadrante.")

    print("Comprimentos: sorteados a partir do conjunto de teste")
    # Ordena antes de sortear: a ordem do glob muda entre sistemas operacionais.
    teste = carregar_comprimentos_teste(pasta_tokens)
    return {q: random.choices(sorted(teste[q]), k=n) for q in QUADRANTES}


def gerar_lote(
    cadeia,
    prefixo: list[int],
    max_lens: list[int],
    eos_id: int,
    ids_proibidos: list[int],
    temperatura: float,
    top_k: int | None,
    top_p: float | None,
) -> tuple[list[list[int]], list[bool]]:
    historicos = [list(prefixo) for _ in max_lens]
    terminou_com_eos = [False] * len(max_lens)
    ativas = [i for i, max_len in enumerate(max_lens) if len(prefixo) < max_len]

    while ativas:
        probabilidades = np.stack([cadeia.distribuicao(historicos[i]) for i in ativas])
        with np.errstate(divide="ignore"):
            logits = torch.from_numpy(np.log(probabilidades))

        proximos = amostrar_proximo_token(
            logits, temperatura=temperatura, top_k=top_k, top_p=top_p,
            ids_proibidos=ids_proibidos,
        ).tolist()

        ainda_ativas = []
        for indice, token in zip(ativas, proximos):
            if token == eos_id:
                terminou_com_eos[indice] = True
                continue
            historicos[indice].append(token)
            if len(historicos[indice]) < max_lens[indice]:
                ainda_ativas.append(indice)
        ativas = ainda_ativas

    return [h[len(prefixo):] for h in historicos], terminou_com_eos


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera as amostras do baseline de Markov.")
    parser.add_argument("--config", type=Path, default=CONFIG_PADRAO,
                        help="config.json gerado por buscar_markov.py")
    parser.add_argument("--ordem", type=int, default=None, help="substitui a ordem do config")
    parser.add_argument("--peso-recuo", type=float, default=None, help="substitui o lambda do config")
    parser.add_argument("--quadrantes", nargs="+", default=list(QUADRANTES), choices=QUADRANTES)
    parser.add_argument("--n-por-quadrante", type=int, default=25)
    parser.add_argument("--lote", type=int, default=25, help="músicas geradas em paralelo")
    parser.add_argument("--temperatura", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--semente", type=int, default=42)
    parser.add_argument("--sistema", default="markov", help="prefixo dos nomes dos arquivos")
    parser.add_argument("--comprimentos-de", type=Path, default=METADADOS_LSTM,
                        help="metadados.csv cujos comprimentos serão copiados")
    parser.add_argument("--sortear-comprimentos", action="store_true",
                        help="ignora --comprimentos-de e sorteia a partir do teste")
    parser.add_argument("--saida", type=Path, default=PASTA_PROJETO / "amostras_geradas" / "markov")
    parser.add_argument("--pasta-tokens", type=Path, default=None)
    argumentos = parser.parse_args()

    random.seed(argumentos.semente)
    torch.manual_seed(argumentos.semente)

    if argumentos.config.exists():
        config = json.loads(argumentos.config.read_text(encoding="utf-8"))
    elif argumentos.ordem is not None and argumentos.peso_recuo is not None:
        config = {}
    else:
        raise SystemExit(
            f"{argumentos.config} não encontrado: rode python -m src.buscar_markov "
            "ou informe --ordem e --peso-recuo."
        )
    ordem = argumentos.ordem if argumentos.ordem is not None else config["ordem"]
    peso_recuo = argumentos.peso_recuo if argumentos.peso_recuo is not None else config["peso_recuo"]

    pasta_tokens = escolher_pasta_tokens(argumentos.pasta_tokens)
    tokenizador = REMI(params=pasta_tokens / "tokenizer_remi.json")
    ids = carregar_ids_especiais(tokenizador)
    ids_proibidos = obter_ids_proibidos(tokenizador)

    n = argumentos.n_por_quadrante
    comprimentos = definir_comprimentos(
        None if argumentos.sortear_comprimentos else argumentos.comprimentos_de, n, pasta_tokens,
    )

    print(f"Treinando cadeias de ordem {ordem} (lambda {peso_recuo}) com {pasta_tokens.name}/train")
    cadeias = treinar_cadeias(
        carregar_sequencias("train", pasta_tokens), ordem, peso_recuo,
        len(tokenizador), ids_proibidos,
    )
    print(f"Amostragem: T={argumentos.temperatura} top_k={argumentos.top_k} top_p={argumentos.top_p}")

    argumentos.saida.mkdir(parents=True, exist_ok=True)
    linhas = []

    for quadrante in argumentos.quadrantes:
        print(f"\n{quadrante}: gerando {n} músicas")
        prefixo = [ids[f"Emotion_{quadrante}"]]
        max_lens = comprimentos[quadrante]

        for inicio in range(0, n, argumentos.lote):
            fim = min(inicio + argumentos.lote, n)
            sequencias, terminou_eos = gerar_lote(
                cadeias[quadrante], prefixo, max_lens[inicio:fim], ids["EOS_None"],
                ids_proibidos, argumentos.temperatura, argumentos.top_k, argumentos.top_p,
            )

            for numero, (sequencia, eos) in enumerate(zip(sequencias, terminou_eos), start=inicio):
                nome = f"{argumentos.sistema}_{quadrante}_{numero:03d}.mid"
                num_notas = salvar_midi(tokenizador, sequencia, argumentos.saida / quadrante / nome)
                linhas.append({
                    "arquivo": f"{quadrante}/{nome}",
                    "sistema": argumentos.sistema,
                    "quadrante": quadrante,
                    "semente": argumentos.semente,
                    "temperatura": argumentos.temperatura,
                    "top_k": argumentos.top_k if argumentos.top_k is not None else "",
                    "top_p": argumentos.top_p if argumentos.top_p is not None else "",
                    "max_len": max_lens[numero],
                    "num_tokens": len(prefixo) + len(sequencia),
                    "terminou_com_eos": int(eos),
                    "num_notas": num_notas,
                })
            print(f"  {fim}/{n}")

    caminho_metadados = argumentos.saida / "metadados.csv"
    with caminho_metadados.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=COLUNAS_METADADOS)
        escritor.writeheader()
        escritor.writerows(linhas)

    vazias = sum(1 for linha in linhas if linha["num_notas"] == 0)
    com_eos = sum(linha["terminou_com_eos"] for linha in linhas)
    print(f"\nConcluído: {len(linhas)} músicas | {com_eos} terminaram com EOS | {vazias} vazias")
    print(f"Metadados: {caminho_metadados}")


if __name__ == "__main__":
    main()
