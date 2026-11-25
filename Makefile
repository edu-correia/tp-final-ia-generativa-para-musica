SHELL := /bin/bash
.DEFAULT_GOAL := all

# Mantém a ordem das etapas, mesmo com make -j.
.NOTPARALLEL:

VENV ?= venv
PYTHON_SISTEMA ?= python3
PYTHON ?= $(VENV)/bin/python3

EXPERIMENTO_LSTM ?= lstm_final
CHECKPOINT_LSTM ?= checkpoints/$(EXPERIMENTO_LSTM)/melhor.pt
CHECKPOINT_MARKOV ?= checkpoints/markov/modelo.json.gz
SAIDA_LSTM ?= amostras_geradas/lstm
SAIDA_MARKOV ?= amostras_geradas/markov
SAIDA_EXEMPLOS ?= amostras_geradas/exemplos_markov
SAIDA_AVALIACAO ?= saidas/avaliacao
FORMATO_AUDIO ?= wav

.PHONY: all verificar-ambiente \
	ambiente dados treinar-lstm gerar-lstm markov treinar-markov gerar-markov \
	audio audio-lstm audio-markov exemplos-markov avaliar \
	busca-lstm retomar-lstm retreinar-markov verificar teste-rapido \
	estatisticas avaliar-lstm avaliar-lstm-teste

all: ambiente dados treinar-lstm gerar-lstm markov audio avaliar

verificar-ambiente:
	@command -v "$(PYTHON)" >/dev/null || { \
		printf '%s\n' 'Python do ambiente não encontrado. Execute make ambiente.'; \
		exit 1; \
	}

# Ambiente Python.
ambiente:
	$(PYTHON_SISTEMA) -m venv "$(VENV)"
	"$(PYTHON)" -m pip install -r requirements.txt

# Preparação dos dados de P1.
dados: verificar-ambiente
	"$(PYTHON)" -m src.separar_dados
	"$(PYTHON)" -m src.verificar_midis
	"$(PYTHON)" -m src.preprocessamento --quantizar-todos
	"$(PYTHON)" -m src.preprocessamento --aumentar-treino
	"$(PYTHON)" -m src.tokenizacao
	"$(PYTHON)" -m src.converter_tokens
	cp processed_tokens/tokenizer_remi.json processed_tokens_convertidos/tokenizer_remi.json
	"$(PYTHON)" -m src.verificar_ids

# LSTM: treinamento e geração.
treinar-lstm: verificar-ambiente
	"$(PYTHON)" -m src.treinar --nome_experimento "$(EXPERIMENTO_LSTM)"

gerar-lstm: verificar-ambiente
	"$(PYTHON)" src/gerar.py --checkpoint "$(CHECKPOINT_LSTM)" --pasta-tokens processed_tokens_convertidos --saida "$(SAIDA_LSTM)"

# Markov: treinamento e geração usando os metadados de P3.
markov: treinar-markov gerar-markov

treinar-markov: verificar-ambiente
	"$(PYTHON)" src/executar_markov.py treinar --checkpoint "$(CHECKPOINT_MARKOV)"

gerar-markov: verificar-ambiente
	"$(PYTHON)" src/executar_markov.py gerar --checkpoint "$(CHECKPOINT_MARKOV)" --metadados-p3 "$(SAIDA_LSTM)/metadados.csv" --saida "$(SAIDA_MARKOV)"
	@test -f "$(SAIDA_MARKOV)/metadados.csv" || { \
		printf '%s\n' 'Geração Markov incompleta: falta $(SAIDA_MARKOV)/metadados.csv.'; \
		exit 1; \
	}

# Síntese de áudio e seleção de exemplos.
audio: audio-lstm audio-markov exemplos-markov

audio-lstm: verificar-ambiente
	"$(PYTHON)" src/renderizar_audio.py "$(SAIDA_LSTM)" --formato "$(FORMATO_AUDIO)"

audio-markov: verificar-ambiente
	"$(PYTHON)" src/renderizar_audio.py "$(SAIDA_MARKOV)" --formato "$(FORMATO_AUDIO)"

exemplos-markov: verificar-ambiente
	@test -f "$(SAIDA_MARKOV)/metadados.csv" || { \
		printf '%s\n' 'Conclua a geração do Markov antes de preparar os exemplos.'; \
		exit 1; \
	}
	"$(PYTHON)" src/preparar_exemplos_markov.py --entrada "$(SAIDA_MARKOV)" --saida "$(SAIDA_EXEMPLOS)" --renderizar

# Avaliação já implementada por P4.
avaliar: verificar-ambiente
	@test -f "$(SAIDA_LSTM)/metadados.csv" -a -f "$(SAIDA_MARKOV)/metadados.csv" || { \
		printf '%s\n' 'Conclua a geração da LSTM e do Markov antes da avaliação musical.'; \
		exit 1; \
	}
	"$(PYTHON)" src/evaluate.py --test_dir processed_midi/test --lstm_dir "$(SAIDA_LSTM)" --baseline_dir "$(SAIDA_MARKOV)" --out_dir "$(SAIDA_AVALIACAO)"

# Comandos opcionais, executados somente quando solicitados.
busca-lstm: verificar-ambiente
	"$(PYTHON)" -m src.buscar_hiperparametros

retomar-lstm: verificar-ambiente
	"$(PYTHON)" -m src.treinar --nome_experimento "$(EXPERIMENTO_LSTM)" --retomar "checkpoints/$(EXPERIMENTO_LSTM)/ultimo.pt"

retreinar-markov: verificar-ambiente
	"$(PYTHON)" src/executar_markov.py treinar --checkpoint "$(CHECKPOINT_MARKOV)" --sobrescrever

verificar: verificar-ambiente
	"$(PYTHON)" -m src.verificar_ids
	"$(PYTHON)" -m src.dataset_treino
	"$(PYTHON)" -m src.modelo
	"$(PYTHON)" -m src.amostragem

teste-rapido: verificar-ambiente
	"$(PYTHON)" -m src.treinar --teste_rapido

estatisticas: verificar-ambiente
	"$(PYTHON)" src/estatisticas_teste.py
	"$(PYTHON)" src/estatisticas_geracao.py --sistemas "$(SAIDA_LSTM)" "$(SAIDA_MARKOV)"

avaliar-lstm: verificar-ambiente
	"$(PYTHON)" -m src.avaliar_modelo --checkpoint "$(CHECKPOINT_LSTM)"

avaliar-lstm-teste: verificar-ambiente
	"$(PYTHON)" -m src.avaliar_modelo --checkpoint "$(CHECKPOINT_LSTM)" --incluir_teste
