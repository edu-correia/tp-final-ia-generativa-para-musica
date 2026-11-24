from collections import defaultdict
from pathlib import Path
import argparse
import csv
import statistics
import sys

PASTA_PROJETO = Path(__file__).resolve().parents[1]
if str(PASTA_PROJETO) not in sys.path:
    sys.path.insert(0, str(PASTA_PROJETO))

from src.estatisticas_teste import (  # noqa: E402
    QUADRANTES, comprimentos_teste, encontrar_pasta_tokens, formatar, imprimir_tabela, salvar_csv,
)

POUCAS_NOTAS = 20


def tabela_sistema(pasta_sistema: Path, medias_teste: dict[str, float], pasta_saida: Path) -> None:
    caminho_metadados = pasta_sistema / "metadados.csv"
    if not caminho_metadados.exists():
        print(f"\n(aviso) {caminho_metadados} não encontrado; sistema ignorado.")
        return

    with caminho_metadados.open(encoding="utf-8") as arquivo:
        registros = list(csv.DictReader(arquivo))
    if not registros:
        print(f"\n(aviso) {caminho_metadados} está vazio; sistema ignorado.")
        return

    sistema = registros[0].get("sistema") or pasta_sistema.name
    grupos = defaultdict(list)
    for registro in registros:
        grupos[registro["quadrante"]].append(registro)
        grupos["Total"].append(registro)

    colunas = [
        "Quadrante", "n", "Com EOS", "Atingiu max_len", "Vazias", f"< {POUCAS_NOTAS} notas",
        "Notas (média)", "Tokens (média)", "Tokens no teste (média)", "Notas por token",
    ]
    linhas_md, linhas_csv = [], []
    for grupo in (*QUADRANTES, "Total"):
        itens = grupos.get(grupo, [])
        if not itens:
            continue
        n = len(itens)
        notas = [int(i["num_notas"]) for i in itens]
        tokens = [int(i["num_tokens"]) for i in itens]
        com_eos = sum(int(i["terminou_com_eos"]) for i in itens)
        vazias = sum(x == 0 for x in notas)
        poucas = sum(x < POUCAS_NOTAS for x in notas)
        media_notas = statistics.mean(notas)
        media_tokens = statistics.mean(tokens)
        notas_por_token = sum(notas) / sum(tokens) if sum(tokens) else 0.0

        # A média geral do teste já aparece na tabela do teste.
        media_teste = None if grupo == "Total" else medias_teste.get(grupo)

        linhas_csv.append([
            grupo, n, com_eos, n - com_eos, vazias, poucas, round(media_notas, 1),
            round(media_tokens, 1), round(media_teste, 1) if media_teste else "", round(notas_por_token, 4),
        ])
        rotulo = f"**{grupo}**" if grupo == "Total" else grupo
        linhas_md.append([
            rotulo, n, com_eos, n - com_eos, vazias, poucas, formatar(media_notas),
            formatar(media_tokens), formatar(media_teste) if media_teste else "—",
            formatar(notas_por_token, 3),
        ])

    imprimir_tabela(f"Músicas geradas: {sistema}", colunas, linhas_md)
    salvar_csv(pasta_saida / f"geracao_{sistema}.csv", colunas, linhas_csv)

    configuracoes = {(r["temperatura"], r["top_k"], r["top_p"], r["semente"]) for r in registros}
    for temperatura, top_k, top_p, semente in sorted(configuracoes):
        print(f"\nConfiguração: T={temperatura} top_k={top_k or '—'} top_p={top_p or '—'} semente={semente}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Estatísticas por quadrante das músicas geradas.")
    parser.add_argument("--sistemas", nargs="+", type=Path,
                        default=[PASTA_PROJETO / "amostras_geradas" / "lstm",
                                 PASTA_PROJETO / "amostras_geradas" / "markov"],
                        help="pastas com metadados.csv (uma por sistema)")
    parser.add_argument("--pasta-tokens", type=Path, default=None,
                        help="tokens de teste, para a coluna de comparação")
    parser.add_argument("--saida", type=Path, default=PASTA_PROJETO / "saidas" / "estatisticas")
    argumentos = parser.parse_args()

    pasta_tokens = encontrar_pasta_tokens(argumentos.pasta_tokens)
    medias_teste = {q: statistics.mean(v) for q, v in comprimentos_teste(pasta_tokens).items() if v}

    for pasta_sistema in argumentos.sistemas:
        tabela_sistema(pasta_sistema.resolve(), medias_teste, argumentos.saida)

    print(f"\nCSVs salvos em: {argumentos.saida}")


if __name__ == "__main__":
    main()