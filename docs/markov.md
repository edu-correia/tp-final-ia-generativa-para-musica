# Markov — fluxo de execução

Execute os comandos na raiz do projeto.

**Ordem:** ativar o ambiente → treinar → gerar 100 MIDIs → preparar cinco exemplos com áudio.

Neste fluxo, usamos sempre estas pastas:

| Etapa | Caminho |
|---|---|
| Modelo treinado | `checkpoints/markov/modelo.json.gz` |
| Lote de 100 MIDIs | `amostras_geradas/markov/` |
| Cinco exemplos MIDI/WAV | `amostras_geradas/exemplos_markov/` |

## 1. Ativar o ambiente

```bash
source venv/bin/activate
```

## 2. Treinar o baseline

```bash
python3 src/executar_markov.py treinar
```

O comando treina uma cadeia de ordem 1 por quadrante e salva o checkpoint em `checkpoints/markov/modelo.json.gz`.

**Quando avançar:** após aparecer `TREINAMENTO FINALIZADO`. Se aparecer `CHECKPOINT JÁ EXISTENTE`, o modelo salvo foi preservado e você pode seguir para a etapa 3.

## 3. Gerar o lote de 100 MIDIs

```bash
python3 src/executar_markov.py gerar
```

São geradas 25 músicas por quadrante. Os MIDIs, tokens e metadados ficam em `amostras_geradas/markov/`.

**Quando avançar:** aguarde aparecer `GERAÇÃO FINALIZADA` e confirme que existe `amostras_geradas/markov/metadados.csv`. Esse arquivo é salvo somente ao final; ter alguns MIDIs na pasta não significa que a geração terminou.

Se aparecer `PASTA DE SAÍDA JÁ PREENCHIDA` e o lote já estiver completo, siga para a etapa 4. Se não houver `metadados.csv`, a pasta está incompleta: gere em outra pasta vazia e use esse mesmo nome na etapa 4.

## 4. Preparar cinco exemplos e gerar os áudios

```bash
python3 src/preparar_exemplos_markov.py --renderizar
```

O comando seleciona o primeiro MIDI de cada quadrante e o segundo de Q1, depois sintetiza os WAVs. Os cinco pares MIDI/WAV e o arquivo de proveniência ficam em `amostras_geradas/exemplos_markov/`.

**Quando terminar:** deve aparecer `EXEMPLOS PREPARADOS` e `Total de exemplos: 5`.

As duas etapas utilizam `amostras_geradas/markov/` por padrão. Se indicar outra pasta com `--saida` na geração, informe a mesma pasta com `--entrada` na preparação dos exemplos.

## Comandos opcionais

### Treinar novamente

Para refazer o treinamento e substituir o checkpoint:

```bash
python3 src/executar_markov.py treinar --sobrescrever
```

Isso substitui somente o modelo. Para gerar um novo lote depois, escolha uma pasta de saída vazia e informe a mesma pasta como entrada na preparação dos exemplos. Use também outro destino em `--saida` do preparador se quiser preservar os exemplos anteriores.

### Separar apenas os MIDIs, sem áudio

```bash
python3 src/preparar_exemplos_markov.py
```

### Consultar os argumentos

O primeiro argumento de `executar_markov.py` escolhe a operação: `treinar` ou `gerar`. Todas as opções são definidas no mesmo parser, e os comandos do fluxo permanecem iguais.

```bash
python3 src/executar_markov.py --help
python3 src/preparar_exemplos_markov.py --help
```

## Preparação inicial — somente se ainda não foi feita

O treino precisa dos tokens de P1 em `processed_tokens_convertidos/`, do tokenizer em `processed_tokens/tokenizer_remi.json` e de `splits/train_clip.csv`. A geração também precisa de `amostras_geradas/lstm/metadados.csv`, fornecido por P3.

Para criar o ambiente e instalar as dependências:

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

O `requirements.txt` inclui as dependências Python necessárias à P5, como `torch`, `miditok` e `symusic`. Na conferência do ambiente local, todos os pacotes listados estavam instalados no `venv`.

As versões dos pacotes não estão fixadas com `==`. Por isso, uma nova instalação pode utilizar versões diferentes das presentes neste ambiente.

Para instalar a síntese de áudio no Ubuntu/Debian:

```bash
sudo apt install fluidsynth fluid-soundfont-gm
```

O FluidSynth e o soundfont são dependências do sistema: não entram no `requirements.txt` e não são instalados pelo `pip`. Ambos já foram encontrados no ambiente local. O caminho padrão do soundfont é `/usr/share/sounds/sf2/FluidR3_GM.sf2`.

Depois dessa preparação, siga o fluxo a partir da etapa 1.
