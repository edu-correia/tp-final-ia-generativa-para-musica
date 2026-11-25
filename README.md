# Geração Musical Condicionada por Emoções

Geração de músicas de piano condicionadas por emoção, usando o EMOPIA, tokens REMI, uma LSTM e um baseline de Markov.

## Executar pelo Make

Execute na raiz do repositório, com o EMOPIA e as ferramentas de áudio abaixo disponíveis:

```bash
make all     # Instalar dependências Python, preparar dados, treinar, gerar áudio e avaliar
make -n all  # Mostrar os comandos sem executá-los
```

O [Makefile](Makefile) utiliza o ambiente `venv` automaticamente. Para executar etapas separadas, como `make dados`, `make treinar-lstm` e `make markov`, consulte [docs/make.md](docs/make.md). O fluxo completo inclui treinamento e pode substituir saídas existentes; execute somente as etapas necessárias se já tiver dados e modelos prontos.

## Pré-requisitos

Tenha Python 3 e o EMOPIA em `dataset/EMOPIA_1.0/`, contendo `midis/`, `label.csv` e `metadata_by_song.csv`.

No Ubuntu/Debian, instale o Make e a síntese de áudio:

```bash
sudo apt install make fluidsynth fluid-soundfont-gm
sudo apt install ffmpeg  # Apenas para exportar MP3
```

## Comandos diretos, sem Make

Execute na raiz e aguarde cada comando terminar. Pule as etapas cujos arquivos já estiverem prontos.

### Ambiente

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

Nas próximas sessões, basta executar `source venv/bin/activate`.

### Dados

```bash
# Separar por música e conferir os MIDIs
python3 -m src.separar_dados
python3 -m src.verificar_midis

# Quantizar e aplicar transposições somente ao treino
python3 -m src.preprocessamento --quantizar-todos
python3 -m src.preprocessamento --aumentar-treino

# Tokenizar, converter IDs e conferir
python3 -m src.tokenizacao
python3 -m src.converter_tokens
cp processed_tokens/tokenizer_remi.json processed_tokens_convertidos/tokenizer_remi.json
python3 -m src.verificar_ids
```

### LSTM

```bash
# Treinar; checkpoints em checkpoints/lstm_final/
python3 -m src.treinar --nome_experimento lstm_final

# Gerar 100 MIDIs em amostras_geradas/lstm/
python3 src/gerar.py --checkpoint checkpoints/lstm_final/melhor.pt --pasta-tokens processed_tokens_convertidos
```

### Markov

```bash
# Treinar uma cadeia de ordem 1 por quadrante
python3 src/executar_markov.py treinar

# Gerar 100 MIDIs em amostras_geradas/markov/
python3 src/executar_markov.py gerar

# Separar cinco pares MIDI/WAV em amostras_geradas/exemplos_markov/
python3 src/preparar_exemplos_markov.py --renderizar
```

A geração do Markov depende de `amostras_geradas/lstm/metadados.csv`. Aguarde a conclusão da geração do Markov e a criação de seu `metadados.csv` antes de preparar os exemplos.

### Áudio e avaliação

```bash
# Sintetizar todas as músicas de cada sistema em WAV
python3 src/renderizar_audio.py amostras_geradas/lstm --formato wav
python3 src/renderizar_audio.py amostras_geradas/markov --formato wav

# Comparar humanos, LSTM e Markov; salvar resultados em saidas/avaliacao/
python3 src/evaluate.py --test_dir processed_midi/test --lstm_dir amostras_geradas/lstm --baseline_dir amostras_geradas/markov --out_dir saidas/avaliacao
```

## Comandos opcionais

| Finalidade | Comando |
|---|---|
| Conferir o DataLoader | `python3 -m src.dataset_treino` |
| Conferir a reconstrução dos MIDIs | `python3 -m src.verificar_roundtrip` |
| Testar o treinamento | `python3 -m src.treinar --teste_rapido` |
| Buscar hiperparâmetros | `python3 -m src.buscar_hiperparametros` |
| Retomar a LSTM | `python3 -m src.treinar --nome_experimento lstm_final --retomar checkpoints/lstm_final/ultimo.pt` |
| Avaliar loss e perplexidade | `python3 -m src.avaliar_modelo` |
| Refazer e substituir o checkpoint Markov | `python3 src/executar_markov.py treinar --sobrescrever` |
| Conferir estatísticas | `python3 src/estatisticas_teste.py` e `python3 src/estatisticas_geracao.py` |

## Documentação

[Make](docs/make.md) · [LSTM](docs/modeloLSTM.md) · [Geração](docs/amostragem_geracao.md) · [Markov](docs/markov.md)
