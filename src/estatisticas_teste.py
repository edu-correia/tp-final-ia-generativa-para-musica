from collections import defaultdict
from pathlib import Path
import argparse
import csv
import json
import statistics

PASTA_PROJETO = Path(__file__).resolve().parents[1]
QUADRANTES = ("Q1", "Q2", "Q3", "Q4")


def formatar(valor, casas=0):
    if isinstance(valor, float):
        texto = f"{valor:,.{casas}f}"
    else:
        texto = f"{valor:,}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def imprimir_tabela(titulo: str, colunas: list[str], linhas: list[list]) -> None:
    print(f"\n### {titulo}\n")
    print("| " + " | ".join(colunas) + " |")
    print("|" + "|".join("---" for _ in colunas) + "|")
    for linha in linhas:
        print("| " + " | ".join(str(c) for c in linha) + " |")


def salvar_csv(caminho: Path, colunas: list[str], linhas: list[list]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(colunas)
        escritor.writerows(linhas)


def encontrar_pasta_tokens(pasta_informada: Path | None) -> Path:
    if pasta_informada is not None:
        pasta = pasta_informada
    else:
        pasta = next(
            (PASTA_PROJETO / nome for nome in ("processed_tokens_convertidos", "processed_tokens")
             if (PASTA_PROJETO / nome / "test").is_dir()),
            None,
        )
    if pasta is None or not (pasta / "test").is_dir():
        raise SystemExit("Pasta de tokens de teste não encontrada; use --pasta-tokens.")
    return pasta


# --------------------------------------------------------------------------- #
# Conjunto de teste
# --------------------------------------------------------------------------- #

def comprimentos_teste(pasta_tokens: Path) -> dict[str, list[int]]:
    comprimentos = defaultdict(list)
    for caminho in (pasta_tokens / "test").glob("*.json"):
        dados = json.loads(caminho.read_text(encoding="utf-8"))
        quadrante = dados["clip_name"][:2]
        if quadrante in QUADRANTES:
            comprimentos[quadrante].append(len(dados["tokens"]))
    return comprimentos


def tabela_teste(comprimentos: dict[str, list[int]], pasta_saida: Path) -> dict[str, float]:
    colunas = ["Quadrante", "Clipes", "Média", "Mediana", "Desvio", "Mín", "Máx"]
    linhas_md, linhas_csv = [], []
    todos = []
    for quadrante in QUADRANTES:
        valores = comprimentos.get(quadrante, [])
        if not valores:
            continue
        todos += valores
        media = statistics.mean(valores)
        mediana = statistics.median(valores)
        desvio = statistics.stdev(valores) if len(valores) > 1 else 0.0
        linhas_csv.append([quadrante, len(valores), round(media, 1), mediana, round(desvio, 1), min(valores), max(valores)])
        linhas_md.append([quadrante, len(valores), formatar(media), formatar(float(mediana)),
                          formatar(desvio), formatar(min(valores)), formatar(max(valores))])
    if todos:
        linhas_csv.append(["Total", len(todos), round(statistics.mean(todos), 1), statistics.median(todos),
                           round(statistics.stdev(todos), 1), min(todos), max(todos)])
        linhas_md.append(["**Total**", f"**{len(todos)}**", formatar(statistics.mean(todos)),
                          formatar(float(statistics.median(todos))), formatar(statistics.stdev(todos)),
                          formatar(min(todos)), formatar(max(todos))])

    imprimir_tabela("Comprimento das sequências do conjunto de teste (tokens)", colunas, linhas_md)
    salvar_csv(pasta_saida / "comprimentos_teste.csv", colunas, linhas_csv)
    return {q: statistics.mean(v) for q, v in comprimentos.items() if v}


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprimento das sequências do teste por quadrante.")
    parser.add_argument("--pasta-tokens", type=Path, default=None,
                        help="padrão: processed_tokens_convertidos, se existir, senão processed_tokens")
    parser.add_argument("--saida", type=Path, default=PASTA_PROJETO / "saidas" / "estatisticas")
    argumentos = parser.parse_args()

    pasta_tokens = encontrar_pasta_tokens(argumentos.pasta_tokens)
    print(f"Tokens de teste: {pasta_tokens}")
    tabela_teste(comprimentos_teste(pasta_tokens), argumentos.saida)
    print(f"\nCSV salvo em: {argumentos.saida / 'comprimentos_teste.csv'}")


if __name__ == "__main__":
    main()