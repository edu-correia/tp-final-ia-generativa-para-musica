from pathlib import Path

import torch
import torch.nn as nn


PASTA_PROJETO = Path(__file__).resolve().parents[1]

# Caminho do tokenizador, relativo à raiz do projeto, guardado no checkpoint.
CAMINHO_TOKENIZADOR = "processed_tokens/tokenizer_remi.json"

CONDICIONAMENTOS = ("token", "token_embedding")

NUMERO_EMOCOES = 4


# Modelo de linguagem LSTM sobre tokens REMI, condicionado por emoção.
#
# condicionamento="token": a emoção entra apenas como primeiro token da sequência.
# condicionamento="token_embedding": além do token, um embedding da emoção
# é concatenado ao embedding de todos os passos.

class ModeloLSTM(nn.Module):

    def __init__(
        self,
        tamanho_vocabulario,
        dim_embedding=256,
        dim_oculta=512,
        num_camadas=2,
        dropout=0.3,
        pad_id=0,
        condicionamento="token",
        ids_emocao=None,
        dim_emocao=32,
        amarrar_pesos=False,
    ):
        super().__init__()

        if condicionamento not in CONDICIONAMENTOS:
            raise ValueError(
                f"condicionamento deve ser um de {CONDICIONAMENTOS}, "
                f"recebido: {condicionamento}"
            )

        if condicionamento == "token_embedding":
            if ids_emocao is None or len(ids_emocao) != NUMERO_EMOCOES:
                raise ValueError(
                    "condicionamento='token_embedding' exige ids_emocao "
                    "com os 4 IDs de Emotion_Q1..Q4"
                )

        if amarrar_pesos and dim_embedding != dim_oculta:
            raise ValueError(
                "amarrar_pesos exige dim_embedding == dim_oculta "
                f"({dim_embedding} != {dim_oculta})"
            )

        self.tamanho_vocabulario = tamanho_vocabulario
        self.dim_embedding = dim_embedding
        self.dim_oculta = dim_oculta
        self.num_camadas = num_camadas
        self.taxa_dropout = dropout
        self.pad_id = pad_id
        self.condicionamento = condicionamento
        self.ids_emocao = [int(token_id) for token_id in ids_emocao] if ids_emocao else None
        self.dim_emocao = dim_emocao
        self.amarrar_pesos = amarrar_pesos

        self.embedding = nn.Embedding(
            tamanho_vocabulario,
            dim_embedding,
            padding_idx=pad_id
        )

        dim_entrada_lstm = dim_embedding

        if condicionamento == "token_embedding":

            self.embedding_emocao = nn.Embedding(NUMERO_EMOCOES, dim_emocao)
            dim_entrada_lstm += dim_emocao

            # Mapeia o ID de cada token para o índice da emoção (0..3), ou -1.
            mapa = torch.full((tamanho_vocabulario,), -1, dtype=torch.long)
            for indice, token_id in enumerate(self.ids_emocao):
                mapa[token_id] = indice

            self.register_buffer("mapa_emocao", mapa, persistent=False)

        self.dropout_entrada = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=dim_entrada_lstm,
            hidden_size=dim_oculta,
            num_layers=num_camadas,
            batch_first=True,
            # O PyTorch só aplica dropout entre camadas quando há mais de uma.
            dropout=dropout if num_camadas > 1 else 0.0
        )

        self.dropout_saida = nn.Dropout(dropout)

        self.saida = nn.Linear(dim_oculta, tamanho_vocabulario)

        if amarrar_pesos:
            self.saida.weight = self.embedding.weight

    # Obtém o índice da emoção (0..3) a partir do primeiro token da sequência.

    def obter_emocao(self, input_ids):

        emocao_idx = self.mapa_emocao[input_ids[:, 0]]

        if (emocao_idx < 0).any():
            raise ValueError(
                "O primeiro token de cada sequência deve ser um token "
                "Emotion_Qx quando condicionamento='token_embedding'"
            )

        return emocao_idx

    # input_ids: [B, T]. Retorna logits [B, T, V] e o novo estado (h, c).

    def forward(self, input_ids, estado=None, emocao_idx=None):

        x = self.embedding(input_ids)

        if self.condicionamento == "token_embedding":

            if emocao_idx is None:
                if estado is not None:
                    raise ValueError(
                        "Na geração passo a passo com "
                        "condicionamento='token_embedding', passe emocao_idx "
                        "(tensor [B] com valores 0..3): com estado não nulo o "
                        "primeiro token da entrada não é mais o token de emoção."
                    )
                emocao_idx = self.obter_emocao(input_ids)

            emocao = self.embedding_emocao(emocao_idx.to(input_ids.device))
            emocao = emocao.unsqueeze(1).expand(-1, input_ids.shape[1], -1)

            x = torch.cat([x, emocao.to(x.dtype)], dim=-1)

        x = self.dropout_entrada(x)

        saida_lstm, novo_estado = self.lstm(x, estado)

        saida_lstm = self.dropout_saida(saida_lstm)

        logits = self.saida(saida_lstm)

        return logits, novo_estado

    def inicializar_estado(self, tamanho_lote, device):

        formato = (self.num_camadas, tamanho_lote, self.dim_oculta)

        h = torch.zeros(formato, device=device)
        c = torch.zeros(formato, device=device)

        return (h, c)

    def contar_parametros(self):

        return sum(
            parametro.numel()
            for parametro in self.parameters()
            if parametro.requires_grad
        )

    # Hiperparâmetros necessários para reconstruir o modelo.

    def obter_config(self):

        return {
            "tamanho_vocabulario": self.tamanho_vocabulario,
            "dim_embedding": self.dim_embedding,
            "dim_oculta": self.dim_oculta,
            "num_camadas": self.num_camadas,
            "dropout": self.taxa_dropout,
            "pad_id": self.pad_id,
            "condicionamento": self.condicionamento,
            "ids_emocao": self.ids_emocao,
            "dim_emocao": self.dim_emocao,
            "amarrar_pesos": self.amarrar_pesos,
        }


# Salva um checkpoint com tudo que é preciso para reconstruir o modelo.

def salvar_checkpoint(
    caminho,
    modelo,
    otimizador,
    epoca,
    config,
    historico,
    melhor_val_loss,
    agendador=None,
    scaler=None,
):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    config_completa = dict(config)
    config_completa["modelo"] = modelo.obter_config()
    config_completa.setdefault("caminho_tokenizador", CAMINHO_TOKENIZADOR)

    checkpoint = {
        "estado_modelo": modelo.state_dict(),
        "estado_otimizador": otimizador.state_dict() if otimizador else None,
        "estado_agendador": agendador.state_dict() if agendador else None,
        "estado_scaler": scaler.state_dict() if scaler else None,
        "epoca": epoca,
        "config": config_completa,
        "historico": historico,
        "melhor_val_loss": melhor_val_loss,
    }

    # Salva em arquivo temporário e renomeia, para não corromper o
    # checkpoint se a execução for interrompida no meio da escrita.
    caminho_temporario = caminho.with_suffix(".tmp")
    torch.save(checkpoint, caminho_temporario)
    caminho_temporario.replace(caminho)


def carregar_checkpoint(caminho, device="cpu"):

    return torch.load(caminho, map_location=device, weights_only=True)


def criar_modelo_da_config(config_modelo):

    return ModeloLSTM(**config_modelo)


# Carrega um checkpoint e devolve o modelo em modo eval() e a configuração.

def carregar_modelo(caminho, device="cpu"):

    checkpoint = carregar_checkpoint(caminho, device)
    config = checkpoint["config"]

    modelo = criar_modelo_da_config(config["modelo"])
    modelo.load_state_dict(checkpoint["estado_modelo"])
    modelo.to(device)
    modelo.eval()

    return modelo, config


# Testes rápidos do modelo.

def testar_forward_passo_a_passo(modelo, input_ids, emocao_idx=None):

    with torch.no_grad():

        logits_completos, _ = modelo(input_ids)

        estado = None
        logits_passos = []

        for posicao in range(input_ids.shape[1]):

            token = input_ids[:, posicao:posicao + 1]

            if estado is None:
                logits, estado = modelo(token)
            else:
                logits, estado = modelo(token, estado, emocao_idx=emocao_idx)

            logits_passos.append(logits)

        logits_passos = torch.cat(logits_passos, dim=1)

    diferenca = (logits_completos - logits_passos).abs().max().item()

    assert torch.allclose(logits_completos, logits_passos, atol=1e-5), diferenca

    return diferenca


if __name__ == "__main__":

    import tempfile

    from src.amostragem import amostrar_proximo_token
    from src.dataset_treino import carregar_tokenizador

    torch.manual_seed(42)

    tokenizador, pad_id = carregar_tokenizador()
    tamanho_vocabulario = len(tokenizador)

    ids_emocao = [tokenizador[f"Emotion_Q{numero}"] for numero in range(1, 5)]
    ids_proibidos = [
        tokenizador[nome]
        for nome in ("PAD_None", "BOS_None", "MASK_None")
    ] + ids_emocao

    tamanho_lote = 3
    comprimento = 40

    for condicionamento in CONDICIONAMENTOS:

        print(f"\nCONDICIONAMENTO: {condicionamento}")
        print("=" * 50)

        modelo = ModeloLSTM(
            tamanho_vocabulario,
            pad_id=pad_id,
            condicionamento=condicionamento,
            ids_emocao=ids_emocao
        )
        modelo.eval()

        print("Parâmetros:", f"{modelo.contar_parametros():,}")

        # 1. Forward com tokens aleatórios, começando por um token de emoção.
        input_ids = torch.randint(8, tamanho_vocabulario, (tamanho_lote, comprimento))
        input_ids[:, 0] = torch.tensor(ids_emocao[:tamanho_lote])

        logits, (h, c) = modelo(input_ids)

        assert logits.shape == (tamanho_lote, comprimento, tamanho_vocabulario)
        assert h.shape == (modelo.num_camadas, tamanho_lote, modelo.dim_oculta)
        print("Forward completo:", tuple(logits.shape))

        # 2. Passo a passo com estado deve reproduzir o forward completo.
        emocao_idx = None
        if condicionamento == "token_embedding":
            emocao_idx = modelo.obter_emocao(input_ids)

        diferenca = testar_forward_passo_a_passo(modelo, input_ids, emocao_idx)
        print(f"Passo a passo == completo: OK (diferença máx. {diferenca:.2e})")

        if condicionamento == "token_embedding":
            try:
                modelo(input_ids[:, 1:2], estado=(h, c))
                raise AssertionError("Deveria exigir emocao_idx")
            except ValueError:
                print("Erro explicativo sem emocao_idx: OK")

        # 3. Compatibilidade com src.amostragem: 20 tokens a partir de Emotion_Q1.
        with torch.no_grad():

            token = torch.tensor([[tokenizador["Emotion_Q1"]]])
            emocao_geracao = torch.tensor([0])

            logits, estado = modelo(token)
            gerados = []

            for _ in range(20):

                proximo = amostrar_proximo_token(
                    logits[:, -1, :],
                    temperatura=1.0,
                    top_k=50,
                    ids_proibidos=ids_proibidos
                )

                gerados.append(int(proximo.item()))

                logits, estado = modelo(
                    proximo.unsqueeze(1),
                    estado,
                    emocao_idx=emocao_geracao
                )

        assert len(gerados) == 20
        assert not set(gerados) & set(ids_proibidos)
        print("Amostragem de 20 tokens: OK", gerados[:8], "...")

        # 4. Salvar e carregar checkpoint reconstrói o mesmo modelo.
        with tempfile.TemporaryDirectory() as pasta_temporaria:

            caminho = Path(pasta_temporaria) / "teste.pt"
            otimizador = torch.optim.AdamW(modelo.parameters())

            salvar_checkpoint(
                caminho, modelo, otimizador, 0, {}, [], float("inf")
            )

            modelo_carregado, config = carregar_modelo(caminho)

            with torch.no_grad():
                logits_original, _ = modelo(input_ids)
                logits_carregado, _ = modelo_carregado(input_ids)

            assert torch.allclose(logits_original, logits_carregado)
            assert config["modelo"]["condicionamento"] == condicionamento
            print("Salvar/carregar checkpoint: OK")

    # Weight tying só com dim_embedding == dim_oculta.
    modelo_amarrado = ModeloLSTM(
        tamanho_vocabulario,
        dim_embedding=512,
        dim_oculta=512,
        amarrar_pesos=True
    )
    assert modelo_amarrado.saida.weight is modelo_amarrado.embedding.weight
    print("\nWeight tying: OK")

    print("\nSUCESSO: Testes do modelo concluídos!")
