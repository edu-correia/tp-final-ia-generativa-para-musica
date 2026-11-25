# Execução com Make

Execute na raiz do projeto. O Make usa `venv/bin/python3` automaticamente.

```bash
make all     # Instalar, preparar dados, treinar, gerar, sintetizar e avaliar
make -n all  # Mostrar os comandos sem executá-los
```

`make` sem argumentos também executa `all`. É necessário ter o EMOPIA em `dataset/EMOPIA_1.0/` e instalar as ferramentas de áudio indicadas no README. O fluxo completo inclui treinamento e pode recriar saídas.

## Executar por etapa

Os alvos são independentes. Execute na ordem necessária e pule as etapas já prontas.

| Comando | Finalidade |
|---|---|
| `make ambiente` | Criar o ambiente e instalar as dependências Python |
| `make dados` | Separar, quantizar, aumentar e tokenizar os dados |
| `make treinar-lstm` | Treinar a LSTM |
| `make gerar-lstm` | Gerar 100 MIDIs da LSTM |
| `make markov` | Treinar o Markov e gerar seus 100 MIDIs |
| `make audio` | Sintetizar LSTM e Markov e separar cinco exemplos do baseline |
| `make avaliar` | Executar a avaliação musical de P4 |

## Comandos opcionais

| Comando | Finalidade |
|---|---|
| `make treinar-markov` / `make gerar-markov` | Executar separadamente as operações do baseline |
| `make exemplos-markov` | Preparar os cinco exemplos MIDI/WAV |
| `make audio-lstm` / `make audio-markov` | Renderizar todas as músicas de um sistema |
| `make busca-lstm` | Buscar hiperparâmetros |
| `make retomar-lstm` | Continuar pelo checkpoint `ultimo.pt` |
| `make retreinar-markov` | Substituir o checkpoint do baseline |
| `make verificar` / `make teste-rapido` | Executar verificações e teste rápido |
| `make estatisticas` | Gerar estatísticas de teste e geração |
| `make avaliar-lstm` | Calcular loss e perplexidade na validação |
| `make avaliar-lstm-teste` | Incluir o teste na avaliação final da LSTM |

## Configurar uma execução

```bash
make treinar-lstm EXPERIMENTO_LSTM=lstm_nova
make gerar-lstm EXPERIMENTO_LSTM=lstm_nova
make audio-lstm FORMATO_AUDIO=mp3
```

Repita a configuração do experimento nas etapas seguintes. Também podem ser alterados `PYTHON`, `VENV`, `CHECKPOINT_LSTM`, `CHECKPOINT_MARKOV`, `SAIDA_LSTM`, `SAIDA_MARKOV`, `SAIDA_EXEMPLOS` e `SAIDA_AVALIACAO`.
