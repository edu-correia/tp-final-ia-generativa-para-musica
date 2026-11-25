from pathlib import Path
from collections import Counter
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import version
import argparse
import csv
import json
import math
import platform
import sys

import torch
from miditok import REMI


# Permite executar o script diretamente a partir da raiz do projeto.
PASTA_PROJETO = Path(__file__).resolve().parents[1]

if str(PASTA_PROJETO) not in sys.path:
    sys.path.insert(0, str(PASTA_PROJETO))

from src.baseline_markov import MarkovPorQuadrante, QUADRANTES
from src.dataset_treino import (
    dividir_em_janelas,
    TAMANHO_JANELA,
    SOBREPOSICAO,
)
from src.gerar import (
    carregar_ids_especiais,
    gerar_lote,
    salvar_midi,
    COLUNAS_METADADOS,
)


# Caminhos principais do projeto.
PASTA_TOKENS = PASTA_PROJETO / "processed_tokens_convertidos"
PASTA_SPLITS = PASTA_PROJETO / "splits"
PASTA_AMOSTRAS = PASTA_PROJETO / "amostras_geradas"

CAMINHO_CHECKPOINT = (
    PASTA_PROJETO / "checkpoints" / "markov" / "modelo.json.gz"
)

CAMINHO_METADADOS_P3 = (
    PASTA_AMOSTRAS / "lstm" / "metadados.csv"
)

PASTA_SAIDA = PASTA_AMOSTRAS / "markov"
AMOSTRAS_POR_QUADRANTE = 25


# Funções auxiliares para os arquivos de saída.
# Identifica a ordem dos IDs utilizada pelo tokenizador.
def assinatura_vocabulario(tokenizador):

    conteudo = json.dumps(
        tokenizador.vocab,
        sort_keys=True
    ).encode("utf-8")

    return sha256(conteudo).hexdigest()


# Salva os dados em um arquivo JSON.
def salvar_json(caminho, dados):

    caminho.parent.mkdir(parents=True, exist_ok=True)

    with open(caminho, "w", encoding="utf-8") as arquivo:
        json.dump(
            dados,
            arquivo,
            indent=4,
            ensure_ascii=False
        )


# Salva as linhas em um arquivo CSV com cabeçalho.
def salvar_csv(caminho, linhas, colunas=None):

    caminho.parent.mkdir(parents=True, exist_ok=True)

    if colunas is None:
        colunas = list(linhas[0])

    with open(
        caminho,
        "w",
        newline="",
        encoding="utf-8"
    ) as arquivo:

        escritor = csv.DictWriter(
            arquivo,
            fieldnames=colunas
        )

        escritor.writeheader()
        escritor.writerows(linhas)


# Treinamento das cadeias de Markov.
def treinar(argumentos):

    if argumentos.checkpoint.exists() and not argumentos.sobrescrever:

        print("\nCHECKPOINT JÁ EXISTENTE")
        print("Arquivo:", argumentos.checkpoint)
        print("Treinamento não executado. O checkpoint foi preservado.")
        print("Use --sobrescrever para treinar novamente nesse arquivo.")
        return

    modelo = MarkovPorQuadrante(argumentos.ordem)

    caminho_tokenizador = (
        argumentos.pasta_tokens / "tokenizer_remi.json"
    )

    caminho_tokenizador_lstm = (
        PASTA_PROJETO / "processed_tokens" / "tokenizer_remi.json"
    )

    tokenizador = REMI(params=caminho_tokenizador)
    tokenizador_lstm = REMI(params=caminho_tokenizador_lstm)

    if tokenizador.vocab != tokenizador_lstm.vocab:
        raise ValueError(
            "O vocabulário do baseline difere daquele usado pela LSTM."
        )

    ids_especiais = carregar_ids_especiais(tokenizador)

    with open(
        PASTA_SPLITS / "train_clip.csv",
        encoding="utf-8"
    ) as arquivo:

        originais_esperados = set()

        for linha in csv.DictReader(arquivo):
            originais_esperados.add(linha["clip_name"])

    pasta_treino = argumentos.pasta_tokens / "train"
    arquivos = sorted(pasta_treino.glob("*.json"))

    if not arquivos:
        raise ValueError(
            "Não há arquivos na pasta de tokens de treino."
        )

    assinatura_treino = sha256()
    clipes = Counter()
    origens = Counter()
    originais_presentes = set()

    print("\nTREINANDO O BASELINE MARKOV")
    print("Ordem:", modelo.ordem)
    print("Clipes encontrados:", len(arquivos))

    for indice, caminho in enumerate(arquivos, 1):

        conteudo = caminho.read_bytes()
        dados = json.loads(conteudo)

        quadrante = dados["emotion"]
        tokens = dados["tokens"]
        nome_clipe = dados["clip_name"]
        original = nome_clipe.split("__transpose_", 1)[0]

        # Confere se o clipe e suas transposições pertencem ao treino.
        if original not in originais_esperados or caminho.stem != nome_clipe:
            raise ValueError(
                f"{caminho.name}: clipe fora da partição de treino."
            )

        if quadrante not in QUADRANTES or len(tokens) < 2:
            raise ValueError(
                f"{caminho.name}: prefixo incompatível com o tokenizer."
            )

        id_emocao = ids_especiais[f"Emotion_{quadrante}"]

        if tokens[0] != id_emocao:
            raise ValueError(
                f"{caminho.name}: prefixo incompatível com o tokenizer."
            )

        for token in tokens:
            if type(token) is not int or not 0 <= token < len(tokenizador):
                raise ValueError(
                    f"{caminho.name}: ID inválido."
                )

        for token in tokens[1:]:
            if token in ids_especiais.values():
                raise ValueError(
                    f"{caminho.name}: token especial no conteúdo; "
                    "confira o contrato de P1."
                )

        # Reutiliza exatamente o recorte em janelas feito por P1.
        janelas = dividir_em_janelas(tokens)

        for janela in janelas:
            modelo.observar(quadrante, janela)

        clipes[quadrante] += 1
        origens[dados["origin"]] += 1

        if dados["origin"] == "original":
            originais_presentes.add(original)

        assinatura_treino.update(
            caminho.name.encode("utf-8") + b"\0" + conteudo + b"\0"
        )

        if indice % 500 == 0 or indice == len(arquivos):
            print(f"Progresso: {indice}/{len(arquivos)}", flush=True)

    if originais_presentes != originais_esperados or set(clipes) != set(QUADRANTES):
        raise ValueError(
            "Faltam clipes originais ou quadrantes na entrada de treino."
        )

    metadados = {
        "sistema": "markov",
        "ordem": modelo.ordem,
        "criterio_ordem": "configuracao_fixa; sem busca de hiperparametros",
        "particao": "train",
        "treino_sha256": assinatura_treino.hexdigest(),
        "vocabulario_sha256": assinatura_vocabulario(tokenizador),
        "clipes_por_quadrante": dict(clipes),
        "origens": dict(origens),
        "janelas_por_quadrante": dict(modelo.janelas),
        "tamanho_janela": TAMANHO_JANELA,
        "sobreposicao": SOBREPOSICAO,
        "eos_no_treino": False,
        "python": platform.python_version(),
        "versoes": {
            "torch": version("torch"),
            "miditok": version("miditok"),
            "symusic": version("symusic")
        }
    }

    modelo.salvar(argumentos.checkpoint, metadados)

    print("\nTREINAMENTO FINALIZADO")
    print("Checkpoint salvo em:", argumentos.checkpoint)


# Interface para utilizar o procedimento de geração já escrito em P3.
class AdaptadorMarkov:

    # Guarda o modelo e o tamanho do vocabulário para a geração.
    def __init__(self, modelo, tamanho_vocabulario):

        self.modelo = modelo
        self.tamanho_vocabulario = tamanho_vocabulario

    # Converte as contagens em logits e reaproveita contextos repetidos.
    @lru_cache(maxsize=4096)
    def logits_contexto(self, quadrante, historico):

        contagens = self.modelo.distribuicao(quadrante, historico)

        logits = torch.full(
            (self.tamanho_vocabulario,),
            float("-inf")
        )

        for token, contagem in contagens.items():
            logits[token] = math.log(contagem)

        return logits

    # Recebe tokens e devolve logits e histórico, como esperado por P3.
    def __call__(self, input_ids, estado=None, emocao_idx=None):

        if input_ids.device.type != "cpu" or emocao_idx is None:
            raise ValueError(
                "O baseline requer CPU e os índices dos quadrantes."
            )

        saidas = []
        novos_estados = []

        for indice, entrada in enumerate(input_ids.tolist()):

            if estado is None:
                historico = []
            else:
                historico = list(estado[indice])

            indice_quadrante = int(emocao_idx[indice])
            quadrante = QUADRANTES[indice_quadrante]
            passos = []

            for token in entrada:

                historico.append(token)
                historico = historico[-self.modelo.ordem:]

                logits = self.logits_contexto(
                    quadrante,
                    tuple(historico)
                )

                passos.append(logits)

            saidas.append(torch.stack(passos))
            novos_estados.append(tuple(historico))

        return torch.stack(saidas), novos_estados


# Leitura dos limites de comprimento fornecidos por P3.
def ler_limites(caminho):

    with open(caminho, encoding="utf-8") as arquivo:
        linhas = list(csv.DictReader(arquivo))

    nomes = set()
    quantidades = Counter()

    for linha in linhas:

        quadrante = linha["quadrante"]
        comprimento = int(linha["max_len"])

        if quadrante not in QUADRANTES or comprimento < 2:
            raise ValueError(
                "Quadrante ou comprimento de P3 inválido."
            )

        if linha["arquivo"] in nomes:
            raise ValueError(
                "Há arquivos repetidos nos metadados de P3."
            )

        nomes.add(linha["arquivo"])
        quantidades[quadrante] += 1

    for quadrante in QUADRANTES:
        if quantidades[quadrante] < AMOSTRAS_POR_QUADRANTE:
            raise ValueError(
                "São necessárias pelo menos 25 referências de P3 por quadrante."
            )

    return linhas


# Geração das amostras do baseline.
def gerar(argumentos):

    if argumentos.lote < 1 or argumentos.semente < 0:
        raise ValueError(
            "Lote deve ser positivo e semente não negativa."
        )

    if not math.isfinite(argumentos.temperatura) or argumentos.temperatura < 0:
        raise ValueError("Temperatura ou top-p inválido.")

    if not 0 < argumentos.top_p <= 1:
        raise ValueError("Temperatura ou top-p inválido.")

    if argumentos.top_k is not None and argumentos.top_k < 0:
        raise ValueError("top-k deve ser não negativo.")

    if argumentos.saida.exists() and any(argumentos.saida.iterdir()):

        print("\nPASTA DE SAÍDA JÁ PREENCHIDA")
        print("Pasta:", argumentos.saida)
        print("Geração não executada. Os arquivos foram preservados.")
        print("Use --saida para indicar outra pasta, vazia.")
        return

    modelo, metadados_treino = MarkovPorQuadrante.carregar(
        argumentos.checkpoint
    )

    tokenizador = REMI(
        params=argumentos.pasta_tokens / "tokenizer_remi.json"
    )

    assinatura = assinatura_vocabulario(tokenizador)

    if assinatura != metadados_treino["vocabulario_sha256"]:
        raise ValueError(
            "O tokenizer difere daquele usado no treino."
        )

    referencias = ler_limites(argumentos.metadados_p3)
    ids_especiais = carregar_ids_especiais(tokenizador)
    ids_proibidos = []

    for nome, token_id in ids_especiais.items():
        if nome != "EOS_None":
            ids_proibidos.append(token_id)

    adaptador = AdaptadorMarkov(modelo, len(tokenizador))

    torch.set_num_threads(1)
    torch.manual_seed(argumentos.semente)

    linhas = []
    origens = []

    print("\nGERANDO AMOSTRAS DO BASELINE")

    for indice_quadrante, quadrante in enumerate(QUADRANTES):

        referencias_quadrante = []

        for referencia in referencias:
            if referencia["quadrante"] == quadrante:
                referencias_quadrante.append(referencia)

        referencias_quadrante = referencias_quadrante[:AMOSTRAS_POR_QUADRANTE]

        for inicio in range(0, len(referencias_quadrante), argumentos.lote):

            fim = inicio + argumentos.lote
            lote = referencias_quadrante[inicio:fim]
            id_emocao = ids_especiais[f"Emotion_{quadrante}"]

            prefixo = torch.tensor([[id_emocao]] * len(lote))

            indices = torch.full(
                (len(lote),),
                indice_quadrante,
                dtype=torch.long
            )

            comprimentos = []

            for referencia in lote:
                comprimentos.append(int(referencia["max_len"]))

            sequencias, terminou = gerar_lote(
                adaptador,
                prefixo,
                indices,
                comprimentos,
                ids_especiais["EOS_None"],
                ids_proibidos,
                argumentos.temperatura,
                argumentos.top_k,
                argumentos.top_p
            )

            for deslocamento, tokens in enumerate(sequencias):

                numero = inicio + deslocamento
                referencia = lote[deslocamento]
                eos = terminou[deslocamento]
                nome = f"markov_{quadrante}_{numero:03d}"
                relativo = f"{quadrante}/{nome}.mid"

                # O MIDI recebe somente os tokens musicais.
                tokens_musicais = []

                for token in tokens:
                    if token not in ids_especiais.values():
                        tokens_musicais.append(int(token))

                notas = salvar_midi(
                    tokenizador,
                    tokens_musicais,
                    argumentos.saida / relativo
                )

                caminho_tokens = (
                    argumentos.saida / "tokens" / quadrante / f"{nome}.json"
                )

                salvar_json(caminho_tokens, {
                    "quadrante": quadrante,
                    "tokens": [id_emocao] + tokens
                })

                top_k = argumentos.top_k

                if top_k is None:
                    top_k = ""

                linhas.append({
                    "arquivo": relativo,
                    "sistema": "markov",
                    "quadrante": quadrante,
                    "semente": argumentos.semente,
                    "temperatura": argumentos.temperatura,
                    "top_k": top_k,
                    "top_p": argumentos.top_p,
                    "max_len": int(referencia["max_len"]),
                    "num_tokens": len(tokens) + 1,
                    "terminou_com_eos": int(eos),
                    "num_notas": notas
                })

                origens.append({
                    "arquivo_markov": relativo,
                    "referencia_p3": referencia["arquivo"],
                    "quadrante": quadrante,
                    "max_len": int(referencia["max_len"])
                })

        print(f"{quadrante}: {AMOSTRAS_POR_QUADRANTE} MIDIs gerados", flush=True)

    salvar_csv(
        argumentos.saida / "metadados.csv",
        linhas,
        colunas=COLUNAS_METADADOS
    )

    salvar_csv(
        argumentos.saida / "origem_limites.csv",
        origens
    )

    configuracao = {
        "treinamento": metadados_treino,
        "checkpoint_sha256": sha256(argumentos.checkpoint.read_bytes()).hexdigest(),
        "metadados_p3_sha256": sha256(argumentos.metadados_p3.read_bytes()).hexdigest(),
        "origem_limites": "campo max_len dos metadados fornecidos por P3",
        "amostragem": {
            "temperatura": argumentos.temperatura,
            "top_k": argumentos.top_k,
            "top_p": argumentos.top_p,
            "semente": argumentos.semente,
            "lote": argumentos.lote
        },
        "funcoes_reutilizadas": [
            "src.gerar.gerar_lote",
            "src.gerar.salvar_midi"
        ],
        "total_amostras": len(linhas),
        "device": "cpu",
        "versoes": {
            "torch": version("torch"),
            "miditok": version("miditok"),
            "symusic": version("symusic")
        }
    }

    salvar_json(
        argumentos.saida / "configuracao.json",
        configuracao
    )

    print("\nGERAÇÃO FINALIZADA")
    print("Total de amostras:", len(linhas))
    print("Metadados:", argumentos.saida / "metadados.csv")


# Argumentos para executar o treinamento ou a geração.
def main():

    parser = argparse.ArgumentParser(
        description="Treinamento e geração do baseline Markov."
    )

    parser.add_argument(
        "comando",
        choices=["treinar", "gerar"],
        help="Escolhe entre treinar o baseline ou gerar as músicas."
    )
    parser.add_argument("--pasta-tokens", type=Path, default=PASTA_TOKENS)
    parser.add_argument("--checkpoint", type=Path, default=CAMINHO_CHECKPOINT)
    parser.add_argument("--ordem", type=int, default=1)
    parser.add_argument(
        "--sobrescrever",
        action="store_true",
        help="Treina novamente e substitui o checkpoint existente."
    )
    parser.add_argument("--metadados-p3", type=Path, default=CAMINHO_METADADOS_P3)
    parser.add_argument("--saida", type=Path, default=PASTA_SAIDA)
    parser.add_argument("--lote", type=int, default=25)
    parser.add_argument("--temperatura", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--semente", type=int, default=42)

    argumentos = parser.parse_args()

    if argumentos.comando == "treinar":
        treinar(argumentos)
    elif argumentos.comando == "gerar":
        gerar(argumentos)


if __name__ == "__main__":
    main()
