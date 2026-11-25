from pathlib import Path
import json

import numpy as np

from src.dataset_treino import PASTA_TOKENS_TREINO


# Baseline: cadeia de Markov de ordem n sobre tokens REMI, uma por quadrante.
#
# Cada cadeia é treinada só com os clipes do seu quadrante, então a condição de
# emoção é a própria escolha da cadeia. Os dados são os mesmos da LSTM
# (processed_tokens_convertidos/, com as transposições no treino).
#
# A probabilidade interpola todas as ordens de 0 a n, pela suavização de
# Witten-Bell com um peso lambda ajustado na validação:
#
#   P_k(w | h) = (c(h, w) + lambda * t(h) * P_{k-1}(w | h')) / (c(h) + lambda * t(h))
#
# em que c(h, w) é a contagem de w depois do contexto h, c(h) é o total do
# contexto, t(h) é o número de tokens distintos vistos depois de h e h' é h sem o
# token mais antigo. Contextos nunca vistos caem para a ordem imediatamente menor.
# Abaixo da ordem 0 a distribuição é uniforme sobre os tokens permitidos.

QUADRANTES = ("Q1", "Q2", "Q3", "Q4")

# Um n-grama vira um inteiro na base do vocabulário. Com 417 tokens, 7 tokens
# (contexto de 6 mais o próximo) é o máximo que cabe em 64 bits sem colisão.
ORDEM_MAXIMA = 6


class CadeiaMarkov:

    def __init__(self, ordem, tamanho_vocabulario, ids_proibidos=(), peso_recuo=1.0):

        if not 0 <= ordem <= ORDEM_MAXIMA:
            raise ValueError(f"ordem deve estar entre 0 e {ORDEM_MAXIMA}: {ordem}")

        if tamanho_vocabulario ** (ordem + 1) > np.iinfo(np.int64).max:
            raise ValueError("n-grama não cabe em 64 bits com este vocabulário")

        if peso_recuo <= 0:
            raise ValueError("peso_recuo deve ser maior que zero")

        self.ordem = ordem
        self.tamanho_vocabulario = tamanho_vocabulario
        self.peso_recuo = peso_recuo

        self.base = np.ones(tamanho_vocabulario, dtype=np.float64)
        self.base[list(ids_proibidos)] = 0.0
        self.base /= self.base.sum()

        # Uma tabela por ordem (0..ordem), preenchida em treinar().
        self.tabelas = []

    # Código do contexto de k tokens que antecede cada posição alvo (a partir de
    # max(k, 1)). A posição 0 é o token de emoção: é contexto, nunca alvo.

    def _codificar_contextos(self, tokens, k):

        inicio = max(k, 1)
        codigos = np.zeros(len(tokens) - inicio, dtype=np.int64)

        for atraso in range(k, 0, -1):
            codigos = (
                codigos * self.tamanho_vocabulario
                + tokens[inicio - atraso:len(tokens) - atraso]
            )

        return codigos

    def treinar(self, sequencias):

        sequencias = [np.asarray(s, dtype=np.int64) for s in sequencias]
        self.tabelas = []

        for k in range(self.ordem + 1):

            inicio = max(k, 1)

            ngramas = np.concatenate([
                self._codificar_contextos(s, k) * self.tamanho_vocabulario + s[inicio:]
                for s in sequencias
                if len(s) > inicio
            ])

            # Os n-gramas ficam ordenados, então as continuações de um mesmo
            # contexto são vizinhas e podem ser achadas por busca binária.
            ngramas, contagens = np.unique(ngramas, return_counts=True)

            contextos, inicios = np.unique(
                ngramas // self.tamanho_vocabulario, return_index=True
            )

            limites = np.append(inicios, len(ngramas))

            self.tabelas.append({
                "ngramas": ngramas,
                "contagens": contagens,
                "contextos": contextos,
                "limites": limites,
                "totais": np.add.reduceat(contagens, inicios),
                "distintos": np.diff(limites),
            })

        return self

    def contar_ngramas(self):

        return [len(tabela["ngramas"]) for tabela in self.tabelas]

    # Distribuição do próximo token dado o histórico (que começa pela emoção).

    def distribuicao(self, historico):

        probabilidades = self.base.copy()

        for k in range(min(self.ordem, len(historico)) + 1):

            codigo = 0
            for token in historico[len(historico) - k:]:
                codigo = codigo * self.tamanho_vocabulario + int(token)

            tabela = self.tabelas[k]
            posicao = np.searchsorted(tabela["contextos"], codigo)

            # Se o contexto não foi visto, os mais longos também não foram.
            if (
                posicao == len(tabela["contextos"])
                or tabela["contextos"][posicao] != codigo
            ):
                break

            fatia = slice(tabela["limites"][posicao], tabela["limites"][posicao + 1])
            proximos = tabela["ngramas"][fatia] % self.tamanho_vocabulario

            massa = self.peso_recuo * tabela["distintos"][posicao]

            probabilidades = probabilidades * massa
            probabilidades[proximos] += tabela["contagens"][fatia]
            probabilidades /= tabela["totais"][posicao] + massa

        return probabilidades

    # Log-probabilidade de cada token de uma sequência, exceto o primeiro (emoção).
    # Equivale a distribuicao() em cada posição, mas vetorizado.

    def log_probabilidades(self, sequencia):

        tokens = np.asarray(sequencia, dtype=np.int64)
        probabilidades = self.base[tokens[1:]].copy()

        for k in range(self.ordem + 1):

            inicio = max(k, 1)
            if len(tokens) <= inicio:
                break

            tabela = self.tabelas[k]
            codigos = self._codificar_contextos(tokens, k)
            ngramas = codigos * self.tamanho_vocabulario + tokens[inicio:]

            contagens = _buscar(tabela["ngramas"], tabela["contagens"], ngramas)
            totais = _buscar(tabela["contextos"], tabela["totais"], codigos)
            distintos = _buscar(tabela["contextos"], tabela["distintos"], codigos)

            massa = self.peso_recuo * distintos
            anteriores = probabilidades[inicio - 1:]

            probabilidades[inicio - 1:] = np.where(
                totais > 0,
                (contagens + massa * anteriores) / np.maximum(totais + massa, 1),
                anteriores
            )

        return np.log(probabilidades)


def _buscar(chaves, valores, consultas):

    posicoes = np.minimum(np.searchsorted(chaves, consultas), len(chaves) - 1)

    return np.where(chaves[posicoes] == consultas, valores[posicoes], 0)


# Sequências de tokens de um conjunto, agrupadas por quadrante.

def carregar_sequencias(conjunto, pasta_tokens=PASTA_TOKENS_TREINO):

    sequencias = {quadrante: [] for quadrante in QUADRANTES}

    for caminho in sorted((Path(pasta_tokens) / conjunto).glob("*.json")):
        dados = json.loads(caminho.read_text(encoding="utf-8"))
        sequencias[dados["emotion"]].append(dados["tokens"])

    vazios = [q for q, lista in sequencias.items() if not lista]
    if vazios:
        raise ValueError(f"Sem sequências de {conjunto} para: {', '.join(vazios)}")

    return sequencias


# PAD, BOS, MASK e emoções nunca são gerados (mesma regra da LSTM em gerar.py).

def obter_ids_proibidos(tokenizador):

    nomes = ["PAD_None", "BOS_None", "MASK_None"]
    nomes += [f"Emotion_{quadrante}" for quadrante in QUADRANTES]

    return [int(tokenizador[nome]) for nome in nomes]


def treinar_cadeias(sequencias, ordem, peso_recuo, tamanho_vocabulario, ids_proibidos):

    return {
        quadrante: CadeiaMarkov(
            ordem, tamanho_vocabulario, ids_proibidos, peso_recuo
        ).treinar(sequencias[quadrante])
        for quadrante in QUADRANTES
    }


# Testes rápidos da cadeia.

if __name__ == "__main__":

    from collections import Counter

    gerador = np.random.default_rng(42)

    # 1. Exemplo pequeno com conta feita à mão (emoção = 0, tokens 1 e 2).
    #    Contexto [1]: seguido de 2, 1, 2 -> c=3, t=2. Ordem 0: 1 aparece 3x e 2, 2x.
    cadeia = CadeiaMarkov(1, 3, ids_proibidos=[0], peso_recuo=1.0)
    cadeia.treinar([[0, 1, 2, 1, 1, 2]])
    p0 = (2 + 2 * 0.5) / (5 + 2)
    esperado = (2 + 2 * p0) / (3 + 2)
    assert np.isclose(cadeia.distribuicao([0, 1])[2], esperado)
    print("Suavização conferida à mão: OK")

    # 2. Dados aleatórios: distribuições somam 1 e nunca dão massa a proibidos.
    vocabulario = 12
    proibidos = [0, 1, 3]
    dados = [
        [0] + gerador.choice([2, 4, 5, 6, 7, 8], size=80).tolist()
        for _ in range(20)
    ]
    cadeia = CadeiaMarkov(4, vocabulario, proibidos, peso_recuo=4.0).treinar(dados)

    for historico in ([0], [0, 2], dados[0][:10], [0, 11, 11, 11, 11, 11]):
        distribuicao = cadeia.distribuicao(historico)
        assert np.isclose(distribuicao.sum(), 1.0)
        assert np.all(distribuicao[proibidos] == 0)
    print("Distribuições normalizadas e sem tokens proibidos: OK")

    # 3. As contagens batem com uma contagem direta em Python.
    contagem = Counter(
        tuple(s[i - 4:i + 1]) for s in dados for i in range(4, len(s))
    )
    assert cadeia.contar_ngramas()[4] == len(contagem)
    assert cadeia.tabelas[4]["contagens"].sum() == sum(contagem.values())
    print("Contagens de n-gramas: OK")

    # 4. A versão vetorizada é igual à passo a passo.
    teste = [0] + gerador.choice([2, 4, 5, 6, 7, 8, 9], size=60).tolist()
    passo_a_passo = [
        np.log(cadeia.distribuicao(teste[:i])[teste[i]])
        for i in range(1, len(teste))
    ]
    assert np.allclose(cadeia.log_probabilidades(teste), passo_a_passo)
    print("Vetorizado == passo a passo: OK")

    print("\nSUCESSO: Testes da cadeia de Markov concluídos!")
