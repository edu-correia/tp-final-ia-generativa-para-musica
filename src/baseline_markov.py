from pathlib import Path
from collections import Counter
import gzip
import json


# Quadrantes emocionais utilizados pelo EMOPIA.
QUADRANTES = ("Q1", "Q2", "Q3", "Q4")


# Cada quadrante possui sua própria cadeia de Markov.
class MarkovPorQuadrante:

    # Inicializa as contagens de cada quadrante para a ordem escolhida.
    def __init__(self, ordem=1):

        if type(ordem) is not int or ordem < 1:
            raise ValueError("A ordem deve ser um inteiro positivo.")

        self.ordem = ordem
        self.contagens = {}
        self.janelas = Counter()

        for quadrante in QUADRANTES:
            self.contagens[quadrante] = {}

    # Conta as transições de uma janela de treinamento.
    # O token de emoção é utilizado somente como contexto.
    def observar(self, quadrante, tokens):

        if quadrante not in self.contagens or len(tokens) < 2:
            raise ValueError(
                "Informe um quadrante válido e uma janela com conteúdo."
            )

        tabela = self.contagens[quadrante]

        # O histórico começa novamente em cada janela.
        for posicao in range(1, len(tokens)):

            proximo_token = int(tokens[posicao])
            maior_contexto = min(self.ordem, posicao)

            for tamanho in range(maior_contexto + 1):

                if tamanho == 0:
                    contexto = ()
                else:
                    inicio = posicao - tamanho
                    contexto = tuple(tokens[inicio:posicao])

                if contexto not in tabela:
                    tabela[contexto] = Counter()

                tabela[contexto][proximo_token] += 1

        self.janelas[quadrante] += 1

    # Busca as contagens do maior contexto disponível.
    # Se necessário, reduz o contexto até chegar ao unigrama.
    def distribuicao(self, quadrante, historico):

        tabela = self.contagens[quadrante]
        maior_contexto = min(self.ordem, len(historico))

        for tamanho in range(maior_contexto, -1, -1):

            if tamanho == 0:
                contexto = ()
            else:
                contexto = tuple(historico[-tamanho:])

            if contexto in tabela:
                return tabela[contexto]

        raise ValueError(
            f"Não há dados de treino para {quadrante}."
        )

    # Salva as contagens e a configuração em JSON comprimido.
    def salvar(self, caminho, metadados):

        contagens_salvas = {}

        for quadrante, tabela in self.contagens.items():

            linhas = []

            # Tuplas são convertidas para listas para salvar no JSON.
            for contexto, contagens in sorted(tabela.items()):
                linhas.append([
                    list(contexto),
                    dict(contagens)
                ])

            contagens_salvas[quadrante] = linhas

        dados = {
            "formato": 1,
            "ordem": self.ordem,
            "janelas": dict(self.janelas),
            "metadados": metadados,
            "contagens": contagens_salvas
        }

        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)

        conteudo = json.dumps(
            dados,
            sort_keys=True,
            separators=(",", ":")
        ).encode("utf-8")

        # O tempo fixo evita diferenças na compressão do mesmo conteúdo.
        comprimido = gzip.compress(conteudo, mtime=0)
        caminho.write_bytes(comprimido)

    # Carrega um checkpoint salvo anteriormente.
    @classmethod
    def carregar(cls, caminho):

        comprimido = Path(caminho).read_bytes()
        conteudo = gzip.decompress(comprimido)
        dados = json.loads(conteudo)

        if dados["formato"] != 1:
            raise ValueError("Formato de checkpoint desconhecido.")

        modelo = cls(dados["ordem"])
        modelo.janelas.update(dados["janelas"])

        for quadrante, linhas in dados["contagens"].items():

            tabela = {}

            for contexto, contagens in linhas:

                contagens_convertidas = Counter()

                # Chaves de objetos JSON são lidas como texto.
                for token, quantidade in contagens.items():
                    contagens_convertidas[int(token)] = quantidade

                tabela[tuple(contexto)] = contagens_convertidas

            modelo.contagens[quadrante] = tabela

        return modelo, dados["metadados"]
