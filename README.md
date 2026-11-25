# Geração Musical Condicionada por Emoções

Geração de músicas MIDI de piano por quadrante emocional do EMOPIA, usando LSTM e Markov. As músicas são representadas por tokens REMI e comparadas com músicas humanas.

Para executar o fluxo completo pelo Make:

```bash
make all
```

O comando instala as dependências Python, prepara os dados, treina, gera músicas, sintetiza áudio e executa a avaliação. Requer o EMOPIA e as ferramentas de áudio disponíveis e pode recriar saídas. Para executar etapas como `make dados`, `make treinar-lstm` e `make gerar-markov`, consulte [docs/make.md](docs/make.md).

## Comandos diretos

Execute na raiz do projeto e aguarde cada comando terminar. Pule preparação e treinamento quando os arquivos necessários já estiverem prontos.

### Ambiente

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt

# Áudio no Ubuntu/Debian; ffmpeg é necessário para MP3
sudo apt install fluidsynth fluid-soundfont-gm ffmpeg
```

### Dados

Requer `dataset/EMOPIA_1.0/` com `midis/`, `label.csv` e `metadata_by_song.csv`.

```bash
# Separar por música e conferir os MIDIs
python3 -m src.separar_dados
python3 -m src.verificar_midis

# Quantizar e criar transposições somente no treino
python3 -m src.preprocessamento --quantizar-todos
python3 -m src.preprocessamento --aumentar-treino

# Tokenizar, converter os IDs e conferir
python3 -m src.tokenizacao
python3 -m src.converter_tokens
cp processed_tokens/tokenizer_remi.json processed_tokens_convertidos/tokenizer_remi.json
python3 -m src.verificar_ids
```

### LSTM

```bash
# Treinar e salvar checkpoints em checkpoints/lstm_final/
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

# Separar cinco exemplos MIDI/WAV em amostras_geradas/exemplos_markov/
python3 src/preparar_exemplos_markov.py --renderizar
```

O Markov utiliza `amostras_geradas/lstm/metadados.csv`. Aguarde a criação de `metadados.csv` ao final de cada geração antes de preparar os exemplos.

### Áudio e avaliação

```bash
# Sintetizar os sistemas completos; saídas nas pastas lstm_audio/ e markov_audio/
python3 src/renderizar_audio.py amostras_geradas/lstm --formato wav
python3 src/renderizar_audio.py amostras_geradas/markov --formato wav

# Comparar humanos, LSTM e Markov; salvar gráficos e tabelas em saidas/avaliacao/
python3 src/evaluate.py --test_dir processed_midi/test --lstm_dir amostras_geradas/lstm --baseline_dir amostras_geradas/markov --out_dir saidas/avaliacao
```

## Comandos opcionais

| Finalidade | Comando |
|---|---|
| Analisar os metadados | `python3 -m src.analisar_dataset` |
| Analisar notas e durações do treino | `python3 -m src.preprocessamento --analisar-treino` |
| Conferir o DataLoader | `python3 -m src.dataset_treino` |
| Conferir a reconstrução dos MIDIs | `python3 -m src.verificar_roundtrip` |
| Testar modelo e amostragem | `python3 -m src.modelo` e `python3 -m src.amostragem` |
| Executar um treino rápido | `python3 -m src.treinar --teste_rapido` |
| Buscar hiperparâmetros | `python3 -m src.buscar_hiperparametros` |
| Retomar a LSTM | `python3 -m src.treinar --nome_experimento lstm_final --retomar checkpoints/lstm_final/ultimo.pt` |
| Loss/perplexidade na validação | `python3 -m src.avaliar_modelo` |
| Incluir o teste na avaliação final | `python3 -m src.avaliar_modelo --incluir_teste` |
| Refazer o Markov e substituir seu checkpoint | `python3 src/executar_markov.py treinar --sobrescrever` |
| Separar exemplos sem áudio | `python3 src/preparar_exemplos_markov.py` |
| Estatísticas de teste e geração | `python3 src/estatisticas_teste.py` e `python3 src/estatisticas_geracao.py` |

Para outros argumentos, acrescente `--help` aos scripts que recebem opções, por exemplo `python3 src/executar_markov.py --help`. Para MP3, use `--formato mp3` no renderizador.

O auxiliar `python3 -m src.preparar_dataset` baixa e converte o EMOPIA, mas sua impressão final de `musica.notes` precisa de revisão para o MusPy instalado. Com os dados presentes, não é necessário executá-lo.

## Documentação

[Make](docs/make.md) · [LSTM](docs/modeloLSTM.md) · [Geração](docs/amostragem_geracao.md) · [Markov](docs/markov.md)
