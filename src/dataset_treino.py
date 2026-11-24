from pathlib import Path
import json
import statistics
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from miditok import REMI


# Configurações do dataset

PASTA_PROJETO = Path(__file__).resolve().parents[1]
PASTA_TOKENS = PASTA_PROJETO / "processed_tokens"

# IDs já convertidos para a ordem do tokenizador carregado (usada na geração).
PASTA_TOKENS_TREINO = PASTA_PROJETO / "processed_tokens_convertidos"

CONJUNTOS = ["train", "val", "test"]

TAMANHO_JANELA = 512
SOBREPOSICAO = 128

TAMANHO_LOTE = 8

# Quantidade máxima de tokens musicais por janela.
TAMANHO_CONTEUDO = TAMANHO_JANELA - 1

# Passo entre o início de duas janelas.
PASSO = TAMANHO_CONTEUDO - SOBREPOSICAO

# Carrega o tokenizador REMI e o ID do token PAD.

def carregar_tokenizador():

    caminho = PASTA_TOKENS / "tokenizer_remi.json"

    if not caminho.exists():
        raise FileNotFoundError(
            f"Tokenizador não encontrado: {caminho}"
        )

    tokenizador = REMI(params=caminho)

    pad_id = tokenizador["PAD_None"]

    print("Tokenizador carregado.")
    print(f"Vocabulário: {len(tokenizador)} tokens")
    print(f"ID do PAD: {pad_id}")

    return tokenizador, pad_id

# Dividir tokens em janelas de tamanho fixo, com sobreposição. Cada janela começa com o token de emoção.

def dividir_em_janelas(tokens):

    if len(tokens) < 2:
        return []

    # O primeiro token é sempre o token de emoção.
    token_emocao = tokens[0]
    tokens_musicais = tokens[1:]

    janelas = []

    if not tokens_musicais:
        return []

    # Uma janela pode conter até 511 tokens musicais.
    if len(tokens_musicais) <= TAMANHO_CONTEUDO:

        return [[token_emocao] + tokens_musicais]

    inicios = list(range(0, len(tokens_musicais), PASSO))

    # Evita criar uma última janela muito pequena.
    ultimo_inicio = inicios[-1]

    if len(tokens_musicais) - ultimo_inicio < SOBREPOSICAO:
        novo_inicio = max(
            0,
            len(tokens_musicais) - TAMANHO_CONTEUDO
        )

        inicios[-1] = novo_inicio

    # Remove possíveis posições repetidas e mantém a ordem.
    inicios = sorted(set(inicios))

    for inicio in inicios:

        fim = inicio + TAMANHO_CONTEUDO

        trecho = tokens_musicais[inicio:fim]

        if len(trecho) == 0:
            continue

        # Cada janela começa com a emoção original.
        janela = [token_emocao] + trecho

        janelas.append(janela)

    return janelas

# Dataset e DataLoader para o treinamento do modelo pytorch.

class MusicaDataset(Dataset):

    def __init__(self, nome_conjunto, pasta_tokens=PASTA_TOKENS_TREINO):

        self.nome_conjunto = nome_conjunto
        self.janelas = []

        pasta = Path(pasta_tokens) / nome_conjunto

        if not pasta.exists():
            raise FileNotFoundError(
                f"Pasta não encontrada: {pasta}"
            )

        arquivos = sorted(pasta.glob("*.json"))

        if not arquivos:
            raise ValueError(
                f"Nenhum JSON encontrado em {pasta}"
            )

        self.total_clipes = 0

        for caminho in arquivos:

            with open(caminho, "r", encoding="utf-8") as arquivo:
                dados = json.load(arquivo)

            tokens = dados["tokens"]
            emocao = dados["emotion"]

            janelas_clipe = dividir_em_janelas(tokens)

            for indice, janela in enumerate(janelas_clipe):

                self.janelas.append({
                    "tokens": janela,
                    "emotion": emocao,
                    "clip_name": dados["clip_name"],
                    "window_index": indice
                })

            self.total_clipes += 1

        print(
            f"{nome_conjunto.upper()}: "
            f"{self.total_clipes} clipes, "
            f"{len(self.janelas)} janelas"
        )

    def __len__(self):

        return len(self.janelas)

    def __getitem__(self, indice):

        item = self.janelas[indice]

        tokens = torch.tensor(
            item["tokens"],
            dtype=torch.long
        )

        # Previsão do próximo token:
        # entrada: tokens até a penúltima posição.
        # alvo: tokens a partir da segunda posição.
        entrada = tokens[:-1]
        alvo = tokens[1:]

        return {
            "input_ids": entrada,
            "labels": alvo,
            "emotion": item["emotion"],
            "clip_name": item["clip_name"],
            "window_index": item["window_index"]
        }

# Collate function para o DataLoader, que padroniza o tamanho das sequências em cada lote.

def criar_collate_fn(pad_id):

    def collate_fn(lote):

        entradas = [item["input_ids"] for item in lote]
        alvos = [item["labels"] for item in lote]

        entradas_padded = pad_sequence(
            entradas,
            batch_first=True,
            padding_value=pad_id
        )

        alvos_padded = pad_sequence(
            alvos,
            batch_first=True,
            padding_value=-100
        )

        return {
            "input_ids": entradas_padded,
            "labels": alvos_padded,
            "emotions": [item["emotion"] for item in lote],
            "clip_names": [item["clip_name"] for item in lote]
        }

    return collate_fn

# Cria os DataLoaders para os conjuntos de treino, validação e teste.

def criar_dataloaders(
    tamanho_lote=TAMANHO_LOTE,
    pasta_tokens=PASTA_TOKENS_TREINO,
    conjuntos=CONJUNTOS
):

    _, pad_id = carregar_tokenizador()

    datasets = {}
    dataloaders = {}

    collate_fn = criar_collate_fn(pad_id)

    for conjunto in conjuntos:

        dataset = MusicaDataset(conjunto, pasta_tokens)

        # Embaralha apenas o treinamento.
        embaralhar = conjunto == "train"

        loader = DataLoader(
            dataset,
            batch_size=tamanho_lote,
            shuffle=embaralhar,
            collate_fn=collate_fn,
            num_workers=0
        )

        datasets[conjunto] = dataset
        dataloaders[conjunto] = loader

    return datasets, dataloaders

# Testa os DataLoaders, exibindo estatísticas e o formato do primeiro lote de cada conjunto.

def testar_dataloaders(datasets, dataloaders):

    print("\n" + "=" * 60)
    print("ESTATÍSTICAS DOS DATASETS")
    print("=" * 60)

    for conjunto in CONJUNTOS:

        dataset = datasets[conjunto]
        loader = dataloaders[conjunto]

        comprimentos = [
            len(item["tokens"])
            for item in dataset.janelas
        ]

        print(f"\n{conjunto.upper()}")
        print(f"Clipes: {dataset.total_clipes}")
        print(f"Janelas: {len(dataset)}")

        if comprimentos:
            print(
                "Comprimento médio das janelas:",
                round(statistics.mean(comprimentos), 2)
            )

            print(
                "Comprimento máximo:",
                max(comprimentos)
            )

        # Testa o primeiro lote.
        lote = next(iter(loader))

        print("\nFormato do primeiro lote:")
        print("Input:", tuple(lote["input_ids"].shape))
        print("Labels:", tuple(lote["labels"].shape))
        print("Emoções:", lote["emotions"])

        assert (
            lote["input_ids"].shape ==
            lote["labels"].shape
        )

        assert lote["input_ids"].dtype == torch.long
        assert lote["labels"].dtype == torch.long

    print("\nSUCESSO: Dataset e DataLoaders funcionando!")

#Executa o teste dos DataLoaders se o script for executado diretamente.

if __name__ == "__main__":

    datasets, dataloaders = criar_dataloaders()

    testar_dataloaders(
        datasets,
        dataloaders
    )
