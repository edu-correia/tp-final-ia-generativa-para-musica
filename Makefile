SHELL := /bin/bash
.DEFAULT_GOAL := all
.NOTPARALLEL:

VENV ?= venv
PYTHON_SISTEMA ?= python3
PYTHON ?= $(VENV)/bin/python3
EXPERIMENTO_LSTM ?= lstm_final
CHECKPOINT_LSTM ?= checkpoints/$(EXPERIMENTO_LSTM)/melhor.pt
SAIDA_LSTM ?= amostras_geradas/lstm
SAIDA_MARKOV ?= amostras_geradas/markov
SAIDA_AVALIACAO ?= saidas/avaliacao_com_baseline
FORMATO_AUDIO ?= wav

.PHONY: all ambiente verificar-ambiente dados treinar-lstm gerar-lstm \
	markov treinar-markov gerar-markov audio audio-lstm audio-markov avaliar \
	busca-lstm retomar-lstm verificar teste-rapido estatisticas avaliar-lstm

all: ambiente dados treinar-lstm gerar-lstm markov audio avaliar

ambiente:
	$(PYTHON_SISTEMA) -m venv "$(VENV)"
	"$(PYTHON)" -m pip install -r requirements.txt

verificar-ambiente:
	@command -v "$(PYTHON)" >/dev/null || { \
		printf '%s\n' 'Ambiente não encontrado. Execute make ambiente.'; exit 1; \
	}

dados: verificar-ambiente
	"$(PYTHON)" -m src.separar_dados
	"$(PYTHON)" -m src.verificar_midis
	"$(PYTHON)" -m src.preprocessamento --quantizar-todos
	"$(PYTHON)" -m src.preprocessamento --aumentar-treino
	"$(PYTHON)" -m src.tokenizacao
	"$(PYTHON)" -m src.converter_tokens
	cp processed_tokens/tokenizer_remi.json processed_tokens_convertidos/tokenizer_remi.json
	"$(PYTHON)" -m src.verificar_ids

treinar-lstm: verificar-ambiente
	"$(PYTHON)" -m src.treinar --nome_experimento "$(EXPERIMENTO_LSTM)"

gerar-lstm: verificar-ambiente
	"$(PYTHON)" src/gerar.py --checkpoint "$(CHECKPOINT_LSTM)" --pasta-tokens processed_tokens_convertidos --saida "$(SAIDA_LSTM)"

markov: treinar-markov gerar-markov

# Seleciona ordem e suavização na validação e salva resultados/markov/config.json.
treinar-markov: verificar-ambiente
	"$(PYTHON)" -m src.buscar_markov

# Reconta as transições do treino e gera usando a configuração selecionada.
gerar-markov: verificar-ambiente
	@test -f "$(SAIDA_LSTM)/metadados.csv" || { \
		printf '%s\n' 'Conclua a geração da LSTM antes de gerar o Markov.'; exit 1; \
	}
	"$(PYTHON)" src/gerar_markov.py --pasta-tokens processed_tokens_convertidos --comprimentos-de "$(SAIDA_LSTM)/metadados.csv" --saida "$(SAIDA_MARKOV)"

audio: audio-lstm audio-markov

audio-lstm: verificar-ambiente
	"$(PYTHON)" src/renderizar_audio.py "$(SAIDA_LSTM)" --formato "$(FORMATO_AUDIO)"

audio-markov: verificar-ambiente
	"$(PYTHON)" src/renderizar_audio.py "$(SAIDA_MARKOV)" --formato "$(FORMATO_AUDIO)"

avaliar: verificar-ambiente
	@test -d processed_midi/test -a -f "$(SAIDA_LSTM)/metadados.csv" -a -f "$(SAIDA_MARKOV)/metadados.csv" || { \
		printf '%s\n' 'A avaliação requer os MIDIs de teste e as gerações completas de LSTM e Markov.'; exit 1; \
	}
	"$(PYTHON)" src/evaluate.py --test_dir processed_midi/test --lstm_dir "$(SAIDA_LSTM)" --baseline_dir "$(SAIDA_MARKOV)" --out_dir "$(SAIDA_AVALIACAO)"

busca-lstm: verificar-ambiente
	"$(PYTHON)" -m src.buscar_hiperparametros

retomar-lstm: verificar-ambiente
	"$(PYTHON)" -m src.treinar --nome_experimento "$(EXPERIMENTO_LSTM)" --retomar "checkpoints/$(EXPERIMENTO_LSTM)/ultimo.pt"

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
