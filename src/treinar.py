from pathlib import Path
import argparse
import csv
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.dataset_treino import (
    PASTA_PROJETO,
    PASTA_TOKENS_TREINO,
    MusicaDataset,
    carregar_tokenizador,
    criar_collate_fn,
    dividir_em_janelas,
)
from src.modelo import (
    CAMINHO_TOKENIZADOR,
    CONDICIONAMENTOS,
    ModeloLSTM,
    carregar_checkpoint,
    criar_modelo_da_config,
    salvar_checkpoint,
)


PASTA_CHECKPOINTS = PASTA_PROJETO / "checkpoints"
PASTA_RESULTADOS_TREINO = PASTA_PROJETO / "resultados" / "treino"

INTERVALO_PROGRESSO = 100

CORES_CURVAS = {
    "train": "#2a78d6",
    "val": "#eb6834",
}

COLUNAS_HISTORICO = [
    "epoca",
    "train_loss",
    "val_loss",
    "val_perplexidade",
    "lr",
    "tempo_epoca_s",
    "lotes_treino",
    "tokens_treino",
]


def criar_parser():

    parser = argparse.ArgumentParser(
        description="Treina o modelo LSTM condicionado por emoção."
    )

    # Modelo
    parser.add_argument("--dim_embedding", type=int, default=256)
    parser.add_argument("--dim_oculta", type=int, default=512)
    parser.add_argument("--num_camadas", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--condicionamento",
        choices=CONDICIONAMENTOS,
        default="token"
    )
    parser.add_argument("--amarrar_pesos", action="store_true")

    # Treino
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--tamanho_lote", type=int, default=32)
    parser.add_argument("--epocas", type=int, default=30)
    parser.add_argument("--paciencia", type=int, default=4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--semente", type=int, default=42)

    # Execução
    parser.add_argument("--nome_experimento", default="lstm_base")
    parser.add_argument(
        "--max_lotes_por_epoca",
        type=int,
        default=None,
        help="Limita os lotes de treino por época (para testes)."
    )
    parser.add_argument(
        "--teste_rapido",
        action="store_true",
        help="Teste de sanidade com dados aleatórios e overfit de um lote."
    )
    parser.add_argument(
        "--retomar",
        default=None,
        help="Caminho de um checkpoint ultimo.pt para continuar o treino."
    )

    return parser


def fixar_semente(semente):

    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semente)


def obter_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def obter_ids_emocao(tokenizador):

    return [
        int(tokenizador[f"Emotion_Q{numero}"])
        for numero in range(1, 5)
    ]


# Loss média por token de um lote e número de tokens válidos (labels != -100).

def calcular_loss_lote(modelo, lote, criterio, device, usar_amp):

    entrada = lote["input_ids"].to(device)
    alvo = lote["labels"].to(device)

    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=usar_amp
    ):
        logits, _ = modelo(entrada)

    # A loss é calculada em float32 para estabilidade numérica.
    logits = logits.float()
    tamanho_vocabulario = logits.shape[-1]

    loss = criterio(
        logits.reshape(-1, tamanho_vocabulario),
        alvo.reshape(-1)
    )

    tokens_validos = int((alvo != -100).sum().item())

    return loss, tokens_validos


def treinar_epoca(
    modelo,
    loader,
    otimizador,
    criterio,
    device,
    scaler,
    clip_grad,
    max_lotes=None,
    epoca=None,
):
    modelo.train()

    soma_loss = 0.0
    total_tokens = 0
    lotes = 0

    total_lotes = len(loader)
    if max_lotes is not None:
        total_lotes = min(total_lotes, max_lotes)

    usar_amp = scaler is not None and scaler.is_enabled()
    inicio = time.time()

    for indice, lote in enumerate(loader):

        if max_lotes is not None and indice >= max_lotes:
            break

        otimizador.zero_grad(set_to_none=True)

        loss, tokens_validos = calcular_loss_lote(
            modelo, lote, criterio, device, usar_amp
        )

        if usar_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(otimizador)
            nn.utils.clip_grad_norm_(modelo.parameters(), clip_grad)
            scaler.step(otimizador)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), clip_grad)
            otimizador.step()

        soma_loss += loss.item() * tokens_validos
        total_tokens += tokens_validos
        lotes += 1

        if lotes % INTERVALO_PROGRESSO == 0:
            decorrido = time.time() - inicio
            print(
                f"  época {epoca} | lote {lotes}/{total_lotes} | "
                f"loss {soma_loss / total_tokens:.4f} | "
                f"{decorrido / lotes:.2f} s/lote"
            )

    return soma_loss / max(total_tokens, 1), lotes, total_tokens


# Loss média por token (ponderada pelos tokens válidos, não pela média dos lotes).

@torch.no_grad()
def avaliar(modelo, loader, device, usar_amp=False):

    modelo.eval()

    criterio_soma = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

    soma_loss = 0.0
    total_tokens = 0

    for lote in loader:

        loss, tokens_validos = calcular_loss_lote(
            modelo, lote, criterio_soma, device, usar_amp
        )

        soma_loss += loss.item()
        total_tokens += tokens_validos

    loss_media = soma_loss / max(total_tokens, 1)

    return loss_media, total_tokens


def perplexidade(loss):

    # Evita overflow quando a loss é muito alta.
    return math.exp(min(loss, 50.0))


def contar_tokens(dataset):

    # Cada janela de n tokens gera n - 1 alvos.
    return sum(len(item["tokens"]) - 1 for item in dataset.janelas)


# Arquivos de resultado.

def salvar_historico_csv(caminho, historico):

    with open(caminho, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=COLUNAS_HISTORICO)
        escritor.writeheader()
        escritor.writerows(historico)


def salvar_json(caminho, dados):

    with open(caminho, "w", encoding="utf-8") as arquivo:
        json.dump(dados, arquivo, indent=4, ensure_ascii=False)


def plotar_curvas(historico, pasta_saida, nome_experimento):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epocas = [linha["epoca"] for linha in historico]
    train_loss = [linha["train_loss"] for linha in historico]
    val_loss = [linha["val_loss"] for linha in historico]

    indice_melhor = int(np.argmin(val_loss))
    melhor_epoca = epocas[indice_melhor]
    melhor_val = val_loss[indice_melhor]

    figura, eixo = plt.subplots(figsize=(6.4, 4.0))

    eixo.plot(
        epocas, train_loss,
        color=CORES_CURVAS["train"], linewidth=2,
        marker="o", markersize=4, label="Treino"
    )
    eixo.plot(
        epocas, val_loss,
        color=CORES_CURVAS["val"], linewidth=2, linestyle="--",
        marker="s", markersize=4, label="Validação"
    )

    eixo.axvline(melhor_epoca, color="#8a8984", linewidth=1, linestyle=":")
    eixo.scatter(
        [melhor_epoca], [melhor_val],
        s=70, facecolor="none", edgecolor="#0b0b0b", linewidth=1.5,
        zorder=5, label=f"Melhor época ({melhor_epoca})"
    )
    eixo.annotate(
        f"val = {melhor_val:.3f}",
        (melhor_epoca, melhor_val),
        textcoords="offset points", xytext=(8, 10),
        fontsize=9, color="#52514e"
    )

    eixo.set_xlabel("Época")
    eixo.set_ylabel("Loss (entropia cruzada por token)")
    eixo.set_title(f"Curvas de loss: {nome_experimento}")
    eixo.set_xticks(epocas if len(epocas) <= 15 else epocas[::2])
    eixo.grid(True, color="#e6e5e0", linewidth=0.8)
    eixo.set_axisbelow(True)

    for lado in ("top", "right"):
        eixo.spines[lado].set_visible(False)

    eixo.legend(frameon=False)
    figura.tight_layout()

    figura.savefig(pasta_saida / "curvas_loss.png", dpi=150)
    figura.savefig(pasta_saida / "curvas_loss.pdf")
    plt.close(figura)


def carregar_dados(tamanho_lote, pad_id):

    collate_fn = criar_collate_fn(pad_id)

    # O conjunto de teste nunca é carregado no treino.
    dataset_treino = MusicaDataset("train", PASTA_TOKENS_TREINO)
    dataset_val = MusicaDataset("val", PASTA_TOKENS_TREINO)

    gerador = torch.Generator()

    loader_treino = DataLoader(
        dataset_treino,
        batch_size=tamanho_lote,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        generator=gerador
    )

    loader_val = DataLoader(
        dataset_val,
        batch_size=tamanho_lote,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    return {
        "dataset_treino": dataset_treino,
        "dataset_val": dataset_val,
        "loader_treino": loader_treino,
        "loader_val": loader_val,
        "gerador": gerador,
    }


# Treino completo. Retorna o resumo do experimento.
# "dados" permite reaproveitar os datasets já carregados (usado na busca).

def treinar(argumentos, dados=None):

    if argumentos.teste_rapido:
        return executar_teste_rapido()

    fixar_semente(argumentos.semente)

    device = obter_device()
    usar_amp = device.type == "cuda"

    nome_gpu = torch.cuda.get_device_name(0) if usar_amp else None

    print("\n" + "=" * 60)
    print(f"EXPERIMENTO: {argumentos.nome_experimento}")
    print("=" * 60)
    print("Device:", device, f"({nome_gpu})" if nome_gpu else "")

    tokenizador, pad_id = carregar_tokenizador()
    ids_emocao = obter_ids_emocao(tokenizador)

    if dados is None:
        dados = carregar_dados(argumentos.tamanho_lote, pad_id)

    dados["gerador"].manual_seed(argumentos.semente)

    loader_treino = dados["loader_treino"]
    loader_val = dados["loader_val"]

    # Modelo, otimizador e agendador.
    checkpoint = None

    if argumentos.retomar:

        print("Retomando de:", argumentos.retomar)
        checkpoint = carregar_checkpoint(argumentos.retomar, device)

        # A arquitetura vem do checkpoint, não da linha de comando.
        modelo = criar_modelo_da_config(checkpoint["config"]["modelo"])
        modelo.load_state_dict(checkpoint["estado_modelo"])

    else:

        modelo = ModeloLSTM(
            tamanho_vocabulario=len(tokenizador),
            dim_embedding=argumentos.dim_embedding,
            dim_oculta=argumentos.dim_oculta,
            num_camadas=argumentos.num_camadas,
            dropout=argumentos.dropout,
            pad_id=pad_id,
            condicionamento=argumentos.condicionamento,
            ids_emocao=ids_emocao,
            amarrar_pesos=argumentos.amarrar_pesos
        )

    modelo.to(device)

    otimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=argumentos.lr,
        weight_decay=argumentos.weight_decay
    )

    agendador = torch.optim.lr_scheduler.ReduceLROnPlateau(
        otimizador,
        mode="min",
        factor=0.5,
        patience=1
    )

    scaler = torch.amp.GradScaler(device.type, enabled=usar_amp)

    criterio = nn.CrossEntropyLoss(ignore_index=-100)

    historico = []
    melhor_val_loss = float("inf")
    epoca_inicial = 1

    if checkpoint is not None:

        otimizador.load_state_dict(checkpoint["estado_otimizador"])

        if checkpoint.get("estado_agendador"):
            agendador.load_state_dict(checkpoint["estado_agendador"])

        if checkpoint.get("estado_scaler") and usar_amp:
            scaler.load_state_dict(checkpoint["estado_scaler"])

        historico = checkpoint["historico"]
        melhor_val_loss = checkpoint["melhor_val_loss"]
        epoca_inicial = checkpoint["epoca"] + 1

        # Avança o gerador para não repetir a ordem das épocas já vistas.
        dados["gerador"].manual_seed(argumentos.semente + epoca_inicial)

    # Pastas de saída.
    pasta_checkpoints = PASTA_CHECKPOINTS / argumentos.nome_experimento
    pasta_resultados = PASTA_RESULTADOS_TREINO / argumentos.nome_experimento

    pasta_checkpoints.mkdir(parents=True, exist_ok=True)
    pasta_resultados.mkdir(parents=True, exist_ok=True)

    tokens_treino = contar_tokens(dados["dataset_treino"])
    tokens_val = contar_tokens(dados["dataset_val"])

    numero_parametros = modelo.contar_parametros()

    config = {
        "hiperparametros": vars(argumentos),
        "modelo": modelo.obter_config(),
        "numero_parametros": numero_parametros,
        "device": str(device),
        "gpu": nome_gpu,
        "versao_torch": str(torch.__version__),
        "janelas_treino": len(dados["dataset_treino"]),
        "janelas_val": len(dados["dataset_val"]),
        "lotes_treino_por_epoca": len(loader_treino),
        "tokens_treino_por_epoca": tokens_treino,
        "tokens_val": tokens_val,
        "pasta_tokens": str(PASTA_TOKENS_TREINO.relative_to(PASTA_PROJETO)),
        "caminho_tokenizador": CAMINHO_TOKENIZADOR,
    }

    salvar_json(pasta_resultados / "config.json", config)

    print("Parâmetros:", f"{numero_parametros:,}")
    print(
        f"Janelas: {config['janelas_treino']} treino, "
        f"{config['janelas_val']} validação"
    )
    print(
        f"Lotes por época: {len(loader_treino)}"
        + (
            f" (limitado a {argumentos.max_lotes_por_epoca})"
            if argumentos.max_lotes_por_epoca else ""
        )
    )
    print(f"Tokens de treino por época: {tokens_treino:,}")

    # Quantas épocas seguidas já passaram sem melhora (relevante ao retomar).
    epocas_sem_melhora = 0
    if historico:
        perdas = [linha["val_loss"] for linha in historico]
        epocas_sem_melhora = len(perdas) - 1 - int(np.argmin(perdas))

    parou_antecipadamente = False

    for epoca in range(epoca_inicial, argumentos.epocas + 1):

        inicio_epoca = time.time()

        train_loss, lotes, tokens = treinar_epoca(
            modelo,
            loader_treino,
            otimizador,
            criterio,
            device,
            scaler,
            argumentos.clip_grad,
            max_lotes=argumentos.max_lotes_por_epoca,
            epoca=epoca
        )

        val_loss, _ = avaliar(modelo, loader_val, device, usar_amp)
        val_perplexidade = perplexidade(val_loss)

        lr_atual = otimizador.param_groups[0]["lr"]
        agendador.step(val_loss)

        tempo_epoca = time.time() - inicio_epoca

        historico.append({
            "epoca": epoca,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_perplexidade": round(val_perplexidade, 4),
            "lr": lr_atual,
            "tempo_epoca_s": round(tempo_epoca, 2),
            "lotes_treino": lotes,
            "tokens_treino": tokens,
        })

        melhorou = val_loss < melhor_val_loss

        if melhorou:
            melhor_val_loss = val_loss
            epocas_sem_melhora = 0

            # melhor.pt guarda só o modelo (sem otimizador), para ficar pequeno.
            salvar_checkpoint(
                pasta_checkpoints / "melhor.pt",
                modelo, None, epoca, config, historico, melhor_val_loss
            )
        else:
            epocas_sem_melhora += 1

        salvar_checkpoint(
            pasta_checkpoints / "ultimo.pt",
            modelo, otimizador, epoca, config, historico, melhor_val_loss,
            agendador=agendador,
            scaler=scaler
        )

        print(
            f"Época {epoca:02d} | train {train_loss:.4f} | "
            f"val {val_loss:.4f} | ppl {val_perplexidade:.2f} | "
            f"lr {lr_atual:.1e} | {tempo_epoca:.1f} s"
            + (" | melhor" if melhorou else "")
        )

        # Os arquivos são atualizados a cada época, para sobreviver a interrupções.
        salvar_historico_csv(pasta_resultados / "historico.csv", historico)
        plotar_curvas(historico, pasta_resultados, argumentos.nome_experimento)

        if epocas_sem_melhora >= argumentos.paciencia:
            print(
                f"Parada antecipada: {epocas_sem_melhora} épocas sem melhora "
                f"na validação."
            )
            parou_antecipadamente = True
            break

    # Resumo do experimento.
    perdas = [linha["val_loss"] for linha in historico]
    indice_melhor = int(np.argmin(perdas))
    melhor = historico[indice_melhor]

    resumo = {
        "nome_experimento": argumentos.nome_experimento,
        "melhor_epoca": melhor["epoca"],
        "melhor_val_loss": melhor["val_loss"],
        "melhor_val_perplexidade": melhor["val_perplexidade"],
        "train_loss_na_melhor_epoca": melhor["train_loss"],
        "epocas_treinadas": len(historico),
        "parou_antecipadamente": parou_antecipadamente,
        "tempo_total_s": round(sum(l["tempo_epoca_s"] for l in historico), 2),
        "numero_parametros": numero_parametros,
        "device": str(device),
    }

    salvar_json(pasta_resultados / "resumo.json", resumo)

    print("\n" + "=" * 60)
    print("TREINO FINALIZADO")
    print("=" * 60)
    print(
        f"Melhor época: {resumo['melhor_epoca']} | "
        f"val_loss {resumo['melhor_val_loss']:.4f} | "
        f"perplexidade {resumo['melhor_val_perplexidade']:.2f}"
    )
    print(f"Tempo total: {resumo['tempo_total_s'] / 60:.1f} min")
    print("Checkpoints em:", pasta_checkpoints)
    print("Resultados em:", pasta_resultados)

    return resumo


# Teste rápido de sanidade do pipeline.

class DatasetAleatorio(Dataset):

    """
    Sequências de tokens sorteados a partir de uma cadeia de Markov aleatória
    (cada token tem poucos sucessores possíveis). Os tokens são aleatórios,
    mas existe padrão para aprender, então a loss deve cair.
    """

    def __init__(
        self,
        ids_emocao,
        tamanho_vocabulario=417,
        comprimento=128,
        quantidade=256,
        sucessores=4,
        semente=0,
    ):
        gerador = np.random.default_rng(semente)

        primeiro_musical = max(ids_emocao) + 1
        ids_musicais = np.arange(primeiro_musical, tamanho_vocabulario)

        transicoes = gerador.choice(
            ids_musicais,
            size=(tamanho_vocabulario, sucessores)
        )

        self.sequencias = []

        for _ in range(quantidade):

            sequencia = [int(gerador.choice(ids_emocao))]
            atual = int(gerador.choice(ids_musicais))

            for _ in range(comprimento - 1):
                sequencia.append(atual)
                atual = int(gerador.choice(transicoes[atual]))

            self.sequencias.append(sequencia)

    def __len__(self):

        return len(self.sequencias)

    def __getitem__(self, indice):

        tokens = torch.tensor(self.sequencias[indice], dtype=torch.long)

        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
            "emotion": None,
            "clip_name": f"aleatorio_{indice}",
            "window_index": 0,
        }


def carregar_lote_real(pad_id, quantidade=4):

    pasta = PASTA_TOKENS_TREINO / "train"
    arquivos = sorted(pasta.glob("*.json"))[:quantidade]

    itens = []

    for caminho in arquivos:

        with open(caminho, "r", encoding="utf-8") as arquivo:
            dados = json.load(arquivo)

        janela = dividir_em_janelas(dados["tokens"])[0]
        tokens = torch.tensor(janela, dtype=torch.long)

        itens.append({
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
            "emotion": dados["emotion"],
            "clip_name": dados["clip_name"],
        })

    return criar_collate_fn(pad_id)(itens)


def executar_teste_rapido():

    fixar_semente(42)
    device = obter_device()
    usar_amp = device.type == "cuda"

    print("\nTESTE RÁPIDO")
    print("=" * 60)
    print("Device:", device)

    pad_id = 0
    ids_emocao = [4, 5, 6, 7]
    tamanho_vocabulario = 417

    caminho_tokenizador = PASTA_PROJETO / CAMINHO_TOKENIZADOR

    if caminho_tokenizador.exists():
        tokenizador, pad_id = carregar_tokenizador()
        ids_emocao = obter_ids_emocao(tokenizador)
        tamanho_vocabulario = len(tokenizador)

    # 1. Dados aleatórios: a loss deve cair em 2 épocas.
    print("\n1) Dados aleatórios com padrão (2 épocas, modelo pequeno)")

    collate_fn = criar_collate_fn(pad_id)

    loader_treino = DataLoader(
        DatasetAleatorio(ids_emocao, tamanho_vocabulario, semente=0),
        batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    loader_val = DataLoader(
        DatasetAleatorio(ids_emocao, tamanho_vocabulario, quantidade=64, semente=0),
        batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    modelo = ModeloLSTM(
        tamanho_vocabulario,
        dim_embedding=64,
        dim_oculta=128,
        num_camadas=1,
        dropout=0.0,
        pad_id=pad_id,
        ids_emocao=ids_emocao
    ).to(device)

    otimizador = torch.optim.AdamW(modelo.parameters(), lr=1e-2)
    scaler = torch.amp.GradScaler(device.type, enabled=usar_amp)
    criterio = nn.CrossEntropyLoss(ignore_index=-100)

    val_inicial, _ = avaliar(modelo, loader_val, device, usar_amp)
    print(f"  val loss inicial: {val_inicial:.4f} (ln V = {math.log(tamanho_vocabulario):.4f})")

    perdas_val = [val_inicial]

    for epoca in (1, 2):

        train_loss, _, _ = treinar_epoca(
            modelo, loader_treino, otimizador, criterio,
            device, scaler, clip_grad=1.0, epoca=epoca
        )
        val_loss, _ = avaliar(modelo, loader_val, device, usar_amp)
        perdas_val.append(val_loss)

        print(f"  época {epoca} | train {train_loss:.4f} | val {val_loss:.4f}")

    assert perdas_val[-1] < perdas_val[0] - 0.5, perdas_val
    print("  OK: a loss caiu.")

    # 2. Overfit em um único lote real.
    if not (PASTA_TOKENS_TREINO / "train").exists():
        print("\n2) Dados reais não encontrados; overfit pulado.")
        print("\nSUCESSO: Teste rápido concluído!")
        return None

    print("\n2) Overfit em um único lote real (100 passos)")

    lote = carregar_lote_real(pad_id)
    print("  lote:", tuple(lote["input_ids"].shape), lote["clip_names"])

    modelo = ModeloLSTM(
        tamanho_vocabulario,
        dim_embedding=128,
        dim_oculta=256,
        num_camadas=1,
        dropout=0.0,
        pad_id=pad_id,
        ids_emocao=ids_emocao
    ).to(device)

    otimizador = torch.optim.AdamW(modelo.parameters(), lr=1e-2)
    inicio = time.time()

    for passo in range(1, 101):

        modelo.train()
        otimizador.zero_grad(set_to_none=True)

        loss, _ = calcular_loss_lote(modelo, lote, criterio, device, False)
        loss.backward()
        nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        otimizador.step()

        if passo == 1 or passo % 20 == 0:
            print(f"  passo {passo:3d} | loss {loss.item():.4f}")

    loss_final = loss.item()
    print(f"  tempo: {time.time() - inicio:.1f} s")

    assert loss_final < 0.5, loss_final
    print("  OK: o modelo decorou o lote.")

    print("\nSUCESSO: Teste rápido concluído!")
    return None


if __name__ == "__main__":

    argumentos = criar_parser().parse_args()
    treinar(argumentos)
