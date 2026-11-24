# Modelo LSTM: arquitetura, treino e resultados

Modelo autorregressivo que gera tokens REMI condicionados por emoção (quadrantes Q1–Q4 do modelo de Russell). A condição é o token `Emotion_Qx`, colocado no início de cada sequência.

## Arquivos

| Arquivo | Função |
|---|---|
| `src/modelo.py` | Classe `ModeloLSTM` e as funções `salvar_checkpoint` e `carregar_modelo` |
| `src/treinar.py` | Laço de treino, checkpoints, histórico e curvas de loss |
| `src/buscar_hiperparametros.py` | Busca entre as configurações A–D |
| `src/avaliar_modelo.py` | Loss e perplexidade, no total e por quadrante |
| `src/verificar_ids.py` | Confere a consistência dos IDs dos tokens com o tokenizador |
| `checkpoints/lstm_final/melhor.pt` | Modelo final (melhor época na validação) |

Os dados vêm de `processed_tokens_convertidos/`, cujos IDs seguem a ordem do tokenizador `processed_tokens/tokenizer_remi.json`: PAD=0, BOS=1, EOS=2, MASK=3, `Emotion_Q1`..`Emotion_Q4` = 4..7 e os tokens musicais de 8 a 416.

## Arquitetura

`nn.Embedding` (256) → dropout → `nn.LSTM` (2 camadas, 512 unidades, dropout entre camadas) → dropout → `nn.Linear` (512 → 417).

- Vocabulário: 417 tokens.
- Dropout: 0,3.
- Parâmetros: 3.998.881.
- Condicionamento: `token` (a emoção é só o primeiro token da sequência). O modelo também aceita `token_embedding`, que concatena um embedding da emoção (dimensão 32) em todos os passos; essa opção não foi usada no modelo final.

## Treino

- Objetivo: entropia cruzada da previsão do próximo token, ignorando as posições de preenchimento.
- Dados: janelas de até 512 tokens com sobreposição de 128. Cada janela começa com o token de emoção.
- Volume: 15.896 janelas de treino (497 lotes, cerca de 7,3 milhões de tokens por época) e 423 janelas de validação.
- Otimização: AdamW (lr inicial de 10⁻³, weight decay de 0,01), lotes de 32 janelas e recorte da norma do gradiente em 1,0.
- Taxa de aprendizado: `ReduceLROnPlateau` (fator 0,5, paciência de 1 época) sobre a loss de validação.
- Parada antecipada: paciência de 4 épocas, no máximo 30.
- Precisão mista (float16) em GPU. Semente 42.
- Hardware: GPU NVIDIA Tesla T4 (Google Colab), PyTorch 2.14.1. Em CPU, uma época completa leva cerca de 1 hora.

## Como rodar

Todos os comandos são executados na raiz do projeto.

```bash
# Conferência dos IDs e testes do modelo
python -m src.verificar_ids
python -m src.modelo

# Teste de sanidade do pipeline (dados aleatórios e overfit de um lote real)
python -m src.treinar --teste_rapido

# Busca de hiperparâmetros (configurações A–D, até 6 épocas e paciência de 2 cada)
python -m src.buscar_hiperparametros

# Treino final (configuração A)
python -m src.treinar --dim_embedding 256 --dim_oculta 512 --num_camadas 2 --dropout 0.3 --lr 0.001 --tamanho_lote 32 --epocas 30 --paciencia 4 --nome_experimento lstm_final

# Retomar um treino interrompido (a arquitetura é lida do checkpoint; repita os demais argumentos)
python -m src.treinar --epocas 30 --paciencia 4 --nome_experimento lstm_final --retomar checkpoints/lstm_final/ultimo.pt

# Avaliação na validação (o conjunto de teste só entra com --incluir_teste)
python -m src.avaliar_modelo
python -m src.avaliar_modelo --incluir_teste
```

## Busca de hiperparâmetros

Todas as configurações usam embedding de 256, lr de 10⁻³ e lotes de 32. Cada uma rodou até 6 épocas, com seleção pela loss de validação.

| Config. | Camadas | Dim. oculta | Dropout | Parâmetros | Val loss | Perplexidade | Tempo |
|---|---|---|---|---|---|---|---|
| **A** | 2 | 512 | 0,3 | 4,0 M | **1,758** | **5,80** | 3,2 min |
| D | 2 | 512 | 0,5 | 4,0 M | 1,763 | 5,83 | 3,3 min |
| C | 2 | 768 | 0,3 | 8,3 M | 1,769 | 5,87 | 6,3 min |
| B | 3 | 512 | 0,3 | 6,1 M | 1,830 | 6,23 | 4,7 min |

A configuração A foi a escolhida.

## Resultado do treino final

- Épocas treinadas: 11, com parada antecipada.
- Melhor época: 7.
- Loss de treino na melhor época: 1,602.
- Loss de validação: 1,756.
- Perplexidade de validação: 5,79.
- Tempo total: 6,0 min.

Loss e perplexidade na validação, por quadrante:

| Quadrante | Janelas | Tokens | Loss | Perplexidade |
|---|---|---|---|---|
| Q1 | 103 | 48.460 | 1,750 | 5,76 |
| Q2 | 163 | 77.402 | 1,633 | 5,12 |
| Q3 | 87 | 38.170 | 1,947 | 7,01 |
| Q4 | 70 | 30.146 | 1,839 | 6,29 |
| **Total** | **423** | **194.178** | **1,756** | **5,79** |

A perplexidade é `exp` da loss média por token, ponderada pelo número de tokens válidos.

## Interface do modelo

```python
import torch
from src.modelo import carregar_modelo

device = "cuda" if torch.cuda.is_available() else "cpu"
modelo, config = carregar_modelo("checkpoints/lstm_final/melhor.pt", device)  # já em modo eval()
```

- `modelo(input_ids, estado=None)` devolve `(logits, novo_estado)`.
  - `input_ids`: `LongTensor [B, T]`; `logits`: `[B, T, 417]`.
  - `estado`: tupla `(h, c)` da LSTM, cada um `[num_camadas, B, dim_oculta]`. Passar o estado devolvido na chamada seguinte permite processar um token por vez, sem reprocessar a sequência.
- `modelo.inicializar_estado(B, device)` cria um estado zerado (equivale a `estado=None`).
- Com `condicionamento="token_embedding"`, as chamadas que recebem estado também precisam de `emocao_idx` (tensor `[B]` com valores 0..3 para Q1..Q4).
- O checkpoint guarda os hiperparâmetros, o tamanho do vocabulário, o `pad_id`, os IDs de emoção e o caminho do tokenizador (`config["caminho_tokenizador"]`), então o modelo é reconstruído sem argumentos extras.
- Os dados de treino não têm BOS nem EOS: cada janela é `Emotion_Qx` seguido de tokens musicais.
- O modelo foi treinado com janelas de até 512 tokens.

## Arquivos de saída

| Arquivo | Conteúdo |
|---|---|
| `resultados/treino/lstm_final/curvas_loss.png` e `.pdf` | Curvas de loss de treino e validação por época |
| `resultados/treino/lstm_final/historico.csv` | train_loss, val_loss, val_perplexidade, lr e tempo por época |
| `resultados/treino/lstm_final/config.json` | Hiperparâmetros, número de parâmetros, hardware e volume de dados |
| `resultados/treino/lstm_final/resumo.json` | Melhor época, melhor val loss, perplexidade e tempo total |
| `resultados/treino/busca_hiperparametros.csv` | Comparação das configurações A–D |
| `resultados/treino/busca_A` … `busca_D/` | Histórico e curvas de cada configuração da busca |
| `resultados/treino/avaliacao_lstm_final_melhor.csv` | Loss e perplexidade por quadrante |
