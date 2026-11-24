from pathlib import Path
import argparse
import csv
import json
import random
import sys

import torch
from miditok import REMI
from symusic import Score

PASTA_PROJETO = Path(__file__).resolve().parents[1]
if str(PASTA_PROJETO) not in sys.path:
    sys.path.insert(0, str(PASTA_PROJETO))

from src.amostragem import amostrar_proximo_token  # noqa: E402
from src.modelo import carregar_modelo  # noqa: E402


QUADRANTES = ("Q1", "Q2", "Q3", "Q4")
NOMES_ESPECIAIS = ("PAD_None", "BOS_None", "EOS_None", "MASK_None") + tuple(
    f"Emotion_{q}" for q in QUADRANTES
)
COLUNAS_METADADOS = (
    "arquivo", "sistema", "quadrante", "semente", "temperatura", "top_k", "top_p",
    "max_len", "num_tokens", "terminou_com_eos", "num_notas",
)


def escolher_pasta_tokens(pasta_informada: Path | None) -> Path:
    if pasta_informada is not None:
        return pasta_informada
    for nome in ("processed_tokens_convertidos", "processed_tokens"):
        candidata = PASTA_PROJETO / nome
        if (candidata / "tokenizer_remi.json").exists():
            return candidata
    raise SystemExit("Não encontrei tokenizer_remi.json em processed_tokens(_convertidos)/")


def carregar_ids_especiais(tokenizador) -> dict[str, int]:
    return {nome: int(tokenizador.vocab[nome]) for nome in NOMES_ESPECIAIS}


def detectar_usa_bos(pasta_tokens: Path, ids: dict[str, int]) -> bool:
    #olha um arquivo de treino para saber se as sequências começam com BOS
    arquivos = sorted((pasta_tokens / "train").glob("*.json"))
    if not arquivos:
        raise SystemExit(f"Nenhum .json em {pasta_tokens / 'train'} para detectar o prefixo")
    primeiro = int(json.loads(arquivos[0].read_text(encoding="utf-8"))["tokens"][0])
    if primeiro == ids["BOS_None"]:
        return True
    if primeiro in {ids[f"Emotion_{q}"] for q in QUADRANTES}:
        return False
    raise SystemExit(f"Primeiro token inesperado em {arquivos[0].name}: {primeiro}")


def carregar_comprimentos_teste(pasta_tokens: Path) -> dict[str, list[int]]:
    comprimentos = {q: [] for q in QUADRANTES}
    for caminho in (pasta_tokens / "test").glob("*.json"):
        dados = json.loads(caminho.read_text(encoding="utf-8"))
        quadrante = dados["clip_name"][:2]
        if quadrante in comprimentos:
            comprimentos[quadrante].append(len(dados["tokens"]))
    for quadrante, valores in comprimentos.items():
        if not valores:
            raise SystemExit(f"Sem sequências de teste para {quadrante}")
    return comprimentos


@torch.no_grad()
def gerar_lote(
    modelo,
    prefixo: torch.Tensor,
    emocao_idx: torch.Tensor,
    max_lens: list[int],
    eos_id: int,
    ids_proibidos: list[int],
    temperatura: float,
    top_k: int | None,
    top_p: float | None,
) -> tuple[list[list[int]], list[bool]]:
    if prefixo.ndim != 2:
        raise ValueError("prefixo deve ter formato (lote, comprimento_do_prefixo)")

    tamanho_lote = prefixo.shape[0]
    if tamanho_lote == 0:
        return [], []
    if len(max_lens) != tamanho_lote:
        raise ValueError("max_lens deve ter um comprimento por música do lote")

    comprimento_prefixo = prefixo.shape[1]
    if comprimento_prefixo == 0:
        raise ValueError("prefixo não pode estar vazio")

    for max_len in max_lens:
        if not isinstance(max_len, int):
            raise TypeError("Comprimentos máximos devem ser inteiros")
        if max_len < comprimento_prefixo:
            raise ValueError("max_len não pode ser menor que o comprimento do prefixo")

    sequencias: list[list[int]] = [[] for _ in range(tamanho_lote)]
    terminou_com_eos = [False] * tamanho_lote

    ativa = torch.tensor(
        [comprimento_prefixo < max_len for max_len in max_lens],
        dtype=torch.bool,
        device=prefixo.device,
    )

    logits, estado = modelo(prefixo, emocao_idx=emocao_idx)

    while bool(ativa.any().item()):
        indices_ativos = torch.nonzero(ativa, as_tuple=False).flatten()
        proximos = amostrar_proximo_token(
            logits[indices_ativos, -1, :],
            temperatura=temperatura,
            top_k=top_k,
            top_p=top_p,
            ids_proibidos=ids_proibidos,
        )

        entrada_seguinte = torch.full(
            (tamanho_lote,),
            eos_id,
            dtype=torch.long,
            device=prefixo.device,
        )

        for indice_tensor, proximo_tensor in zip(indices_ativos.tolist(), proximos.tolist()):
            entrada_seguinte[indice_tensor] = proximo_tensor

            if proximo_tensor == eos_id:
                terminou_com_eos[indice_tensor] = True
                ativa[indice_tensor] = False
                continue

            sequencias[indice_tensor].append(proximo_tensor)

            comprimento_atual = comprimento_prefixo + len(sequencias[indice_tensor])
            if comprimento_atual >= max_lens[indice_tensor]:
                ativa[indice_tensor] = False

        if bool(ativa.any().item()):
            logits, estado = modelo(
                entrada_seguinte.unsqueeze(1),
                estado,
                emocao_idx=emocao_idx,
            )

    return sequencias, terminou_com_eos

def salvar_midi(tokenizador, ids_musicais: list[int], caminho: Path) -> int:
    #decodifica e salva o MIDI retorna o número de notas
    caminho.parent.mkdir(parents=True, exist_ok=True)
    # Sequência vazia: salva um MIDI válido sem notas, para não quebrar as etapas seguintes.
    score = tokenizador.decode(ids_musicais) if ids_musicais else Score(tokenizador.time_division)
    caminho.write_bytes(score.dumps_midi())
    return sum(len(trilha.notes) for trilha in score.tracks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera músicas condicionadas por emoção.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--quadrantes", nargs="+", default=list(QUADRANTES), choices=QUADRANTES)
    parser.add_argument("--n-por-quadrante", type=int, default=25)
    parser.add_argument("--lote", type=int, default=25, help="músicas geradas em paralelo")
    parser.add_argument("--temperatura", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--semente", type=int, default=42)
    parser.add_argument("--sistema", default="lstm", help="prefixo dos nomes dos arquivos")
    parser.add_argument("--saida", type=Path, default=PASTA_PROJETO / "amostras_geradas" / "lstm")
    parser.add_argument("--pasta-tokens", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    argumentos = parser.parse_args()

    random.seed(argumentos.semente)
    torch.manual_seed(argumentos.semente)

    pasta_tokens = escolher_pasta_tokens(argumentos.pasta_tokens)
    tokenizador = REMI(params=pasta_tokens / "tokenizer_remi.json")
    ids = carregar_ids_especiais(tokenizador)
    usa_bos = detectar_usa_bos(pasta_tokens, ids)
    comprimentos_teste = carregar_comprimentos_teste(pasta_tokens)

    modelo, config = carregar_modelo(argumentos.checkpoint, argumentos.device)
    if modelo.tamanho_vocabulario != len(tokenizador):
        raise SystemExit(
            f"Vocabulário do modelo ({modelo.tamanho_vocabulario}) diferente do "
            f"tokenizer ({len(tokenizador)}): confira a pasta de tokens."
        )

    ids_proibidos = [ids[n] for n in ("PAD_None", "BOS_None", "MASK_None")] + [
        ids[f"Emotion_{q}"] for q in QUADRANTES
    ]
    ids_a_remover = set(ids.values())

    print(f"Tokenizer: {pasta_tokens.name} | prefixo: {'[BOS, Emotion]' if usa_bos else '[Emotion]'}")
    print(f"Modelo: condicionamento={config['modelo']['condicionamento']} | device={argumentos.device}")
    print(f"Amostragem: T={argumentos.temperatura} top_k={argumentos.top_k} top_p={argumentos.top_p}")

    argumentos.saida.mkdir(parents=True, exist_ok=True)
    caminho_metadados = argumentos.saida / "metadados.csv"
    linhas = []

    for quadrante in argumentos.quadrantes:
        indice_emocao = QUADRANTES.index(quadrante)
        id_emocao = ids[f"Emotion_{quadrante}"]
        prefixo_lista = ([ids["BOS_None"]] if usa_bos else []) + [id_emocao]

        max_lens = [random.choice(comprimentos_teste[quadrante]) for _ in range(argumentos.n_por_quadrante)]
        print(f"\n{quadrante}: gerando {argumentos.n_por_quadrante} músicas")

        for inicio in range(0, argumentos.n_por_quadrante, argumentos.lote):
            fim = min(inicio + argumentos.lote, argumentos.n_por_quadrante)
            tamanho = fim - inicio

            prefixo = torch.tensor([prefixo_lista] * tamanho, dtype=torch.long, device=argumentos.device)
            emocao_idx = torch.full((tamanho,), indice_emocao, dtype=torch.long, device=argumentos.device)

            sequencias, terminou_eos = gerar_lote(
                modelo, prefixo, emocao_idx, max_lens[inicio:fim], ids["EOS_None"],
                ids_proibidos, argumentos.temperatura, argumentos.top_k, argumentos.top_p,
            )

            for deslocamento, (sequencia, eos) in enumerate(zip(sequencias, terminou_eos)):
                numero = inicio + deslocamento
                ids_musicais = [int(t) for t in sequencia if int(t) not in ids_a_remover]
                nome = f"{argumentos.sistema}_{quadrante}_{numero:03d}.mid"
                caminho = argumentos.saida / quadrante / nome
                num_notas = salvar_midi(tokenizador, ids_musicais, caminho)

                linhas.append({
                    "arquivo": f"{quadrante}/{nome}",
                    "sistema": argumentos.sistema,
                    "quadrante": quadrante,
                    "semente": argumentos.semente,
                    "temperatura": argumentos.temperatura,
                    "top_k": argumentos.top_k if argumentos.top_k is not None else "",
                    "top_p": argumentos.top_p if argumentos.top_p is not None else "",
                    "max_len": max_lens[numero],
                    "num_tokens": len(prefixo_lista) + len(sequencia),
                    "terminou_com_eos": int(eos),
                    "num_notas": num_notas,
                })
            print(f"  {fim}/{argumentos.n_por_quadrante}")

    with caminho_metadados.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=COLUNAS_METADADOS)
        escritor.writeheader()
        escritor.writerows(linhas)

    total = len(linhas)
    vazias = sum(1 for l in linhas if l["num_notas"] == 0)
    com_eos = sum(l["terminou_com_eos"] for l in linhas)
    print(f"\nConcluído: {total} músicas | {com_eos} terminaram com EOS | {vazias} vazias")
    print(f"Metadados: {caminho_metadados}")


if __name__ == "__main__":
    main()