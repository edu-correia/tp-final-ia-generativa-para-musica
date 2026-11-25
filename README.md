# Geração Musical Condicionada por Emoções

Geração de piano MIDI por quadrante emocional do EMOPIA, usando tokens REMI, uma LSTM e cadeias de Markov por quadrante.

## Executar pelo Make

Execute na raiz do repositório, com o dataset e as ferramentas de áudio disponíveis:

```bash
make all     # Instalar, preparar dados, treinar, gerar músicas, sintetizar e avaliar
make -n all  # Mostrar os comandos sem executá-los
```

O Make usa `venv/bin/python3` automaticamente. O fluxo completo refaz preparação e treinamento e pode substituir saídas. Se os arquivos já estiverem prontos, execute somente as etapas necessárias:

| Comando | O que faz |
|---|---|
| `make ambiente` | Criar o ambiente e instalar as dependências Python |
| `make dados` | Separar, quantizar, transpor o treino e tokenizar |
| `make treinar-lstm` | Treinar a LSTM |
| `make gerar-lstm` | Gerar 100 MIDIs, 25 por quadrante |
| `make markov` | Selecionar a configuração do baseline e gerar 100 MIDIs |
| `make audio` | Sintetizar todos os MIDIs de LSTM e Markov em WAV |
| `make avaliar` | Comparar humanos, LSTM e Markov por métricas, OA e KL |

As etapas individuais pressupõem os arquivos das anteriores. Para separar as operações do baseline, use `make treinar-markov` e `make gerar-markov`. Para MP3: `make audio FORMATO_AUDIO=mp3`.

## Pré-requisitos

Python 3 e o EMOPIA em `dataset/EMOPIA_1.0/`, contendo `midis/`, `label.csv` e `metadata_by_song.csv`.

No Ubuntu/Debian:

```bash
sudo apt install python3-venv make fluidsynth fluid-soundfont-gm
sudo apt install ffmpeg  # Apenas para MP3
```

## Comandos diretos, sem Make

Execute na ordem abaixo, pulando as etapas já concluídas.

```bash
# Criar e ativar o ambiente; nas próximas sessões, basta ativá-lo
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt

# Preparar os dados e conferir os IDs
python3 -m src.separar_dados
python3 -m src.verificar_midis
python3 -m src.preprocessamento --quantizar-todos
python3 -m src.preprocessamento --aumentar-treino
python3 -m src.tokenizacao
python3 -m src.converter_tokens
cp processed_tokens/tokenizer_remi.json processed_tokens_convertidos/tokenizer_remi.json
python3 -m src.verificar_ids

# Treinar e gerar com a LSTM
python3 -m src.treinar --nome_experimento lstm_final
python3 src/gerar.py --checkpoint checkpoints/lstm_final/melhor.pt --pasta-tokens processed_tokens_convertidos

# Selecionar ordem e suavização do Markov na validação e gerar
python3 -m src.buscar_markov
python3 src/gerar_markov.py

# Sintetizar áudio e comparar os sistemas
python3 src/renderizar_audio.py amostras_geradas/lstm --formato wav
python3 src/renderizar_audio.py amostras_geradas/markov --formato wav
python3 src/evaluate.py --test_dir processed_midi/test --lstm_dir amostras_geradas/lstm --baseline_dir amostras_geradas/markov --out_dir saidas/avaliacao_com_baseline
```

O Markov salva a configuração em `resultados/markov/config.json` e refaz as contagens ao gerar, sem checkpoint. Seus comprimentos são pareados com `amostras_geradas/lstm/metadados.csv`; aguarde a geração da LSTM terminar.

As músicas ficam em `amostras_geradas/{lstm,markov}/`, os WAVs em `amostras_geradas/{lstm_audio,markov_audio}/` e os resultados em `saidas/avaliacao_com_baseline/`.

## Comandos opcionais

| Comando | O que faz |
|---|---|
| `make busca-lstm` | Comparar configurações da LSTM na validação |
| `make retomar-lstm` | Continuar pelo checkpoint `ultimo.pt`, se disponível |
| `make verificar` | Conferir IDs, DataLoader, modelo e amostragem |
| `make teste-rapido` | Executar um treinamento pequeno de verificação |
| `make estatisticas` | Resumir comprimentos de teste e músicas geradas |
| `make avaliar-lstm` | Calcular loss e perplexidade na validação |

Para outro experimento, repita `EXPERIMENTO_LSTM=outro_nome` nos comandos Make de treino, geração e avaliação da LSTM. Também é possível alterar `CHECKPOINT_LSTM`, `SAIDA_LSTM`, `SAIDA_MARKOV`, `SAIDA_AVALIACAO`, `VENV` e `PYTHON`.

Detalhes: [LSTM](docs/modeloLSTM.md) · [Amostragem e geração](docs/amostragem_geracao.md).
