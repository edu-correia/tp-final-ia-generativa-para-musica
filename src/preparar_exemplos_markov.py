from pathlib import Path
from hashlib import sha256
import argparse
import csv
import json
import shutil
import sys


# Permite executar o script diretamente a partir da raiz do projeto.
PASTA_PROJETO = Path(__file__).resolve().parents[1]

if str(PASTA_PROJETO) not in sys.path:
    sys.path.insert(0, str(PASTA_PROJETO))

from src.renderizar_audio import (
    renderizar_wav,
    SOUNDFONT_PADRAO,
    verificar_dependencias,
)


# Caminhos das amostras do baseline e dos exemplos separados.
PASTA_ENTRADA = PASTA_PROJETO / "amostras_geradas" / "markov"
PASTA_SAIDA = PASTA_PROJETO / "amostras_geradas" / "exemplos_markov"

QUADRANTES = ("Q1", "Q2", "Q3", "Q4")
TAXA_AMOSTRAGEM = 22050
GANHO = 1.0


# Seleciona os exemplos pela ordem dos metadados.
# Seleciona o primeiro de cada quadrante e o próximo ainda não escolhido.
def selecionar_exemplos(linhas):

    for linha in linhas:
        if linha["sistema"] != "markov":
            raise ValueError(
                "Este comando prepara somente exemplos do baseline Markov."
            )

    escolhidos = []

    for quadrante in QUADRANTES:

        candidatos = []

        for linha in linhas:
            if linha["quadrante"] == quadrante:
                candidatos.append(linha)

        if not candidatos:
            raise ValueError(f"Sem amostras para {quadrante}.")

        escolhidos.append(candidatos[0])

    for linha in linhas:
        if linha not in escolhidos:
            escolhidos.append(linha)
            break

    if len(escolhidos) != 5:
        raise ValueError("São necessários cinco exemplos distintos.")

    return escolhidos


# Prepara os arquivos antes de copiar ou renderizar.
def preparar_arquivos(escolhidos, pasta_entrada, pasta_saida):

    arquivos = []
    pasta_entrada = pasta_entrada.resolve()

    for registro in escolhidos:

        origem = (pasta_entrada / registro["arquivo"]).resolve()

        if not origem.is_relative_to(pasta_entrada):
            raise ValueError(
                "O caminho do MIDI está fora da pasta de entrada."
            )

        conteudo = origem.read_bytes()
        destino = pasta_saida / origem.name

        if destino.exists() and destino.read_bytes() != conteudo:
            raise FileExistsError(
                f"O exemplo existente difere da entrada: {destino}"
            )

        arquivos.append((origem, destino, registro))

    return arquivos


# Copia os cinco MIDI e, quando solicitado, sintetiza seus áudios.
def preparar_exemplos(argumentos):

    caminho_metadados = argumentos.entrada / "metadados.csv"

    if not caminho_metadados.is_file():

        print("\nMETADADOS NÃO ENCONTRADOS")
        print("Arquivo:", caminho_metadados)
        print("Esta etapa precisa de um lote de MIDIs com a geração concluída.")
        print("Se usou --saida na geração, informe a mesma pasta com --entrada.")
        print("Nenhum exemplo foi criado.")
        return

    if argumentos.renderizar:

        verificar_dependencias(precisa_ffmpeg=False)

        if not argumentos.soundfont.is_file():
            raise FileNotFoundError(argumentos.soundfont)

    with open(caminho_metadados, encoding="utf-8") as arquivo:
        linhas = list(csv.DictReader(arquivo))

    escolhidos = selecionar_exemplos(linhas)

    arquivos = preparar_arquivos(
        escolhidos,
        argumentos.entrada,
        argumentos.saida
    )

    registros = []

    print("\nPREPARANDO EXEMPLOS DO BASELINE")

    for origem, destino, registro in arquivos:

        destino.parent.mkdir(parents=True, exist_ok=True)

        if not destino.exists():
            shutil.copy2(origem, destino)

        caminho_wav = destino.with_suffix(".wav")

        if argumentos.renderizar and not caminho_wav.exists():
            renderizar_wav(
                destino,
                caminho_wav,
                argumentos.soundfont,
                TAXA_AMOSTRAGEM,
                GANHO
            )

        if caminho_wav.exists():
            nome_audio = caminho_wav.name
        else:
            nome_audio = None

        registros.append({
            "quadrante": registro["quadrante"],
            "origem": registro["arquivo"],
            "midi": destino.name,
            "midi_sha256": sha256(destino.read_bytes()).hexdigest(),
            "audio": nome_audio
        })

        print(f"Exemplo preparado: {destino.name}", flush=True)

    metadados = {
        "criterio": "primeiro de cada quadrante e próximo na ordem dos metadados",
        "exemplos": registros
    }

    if argumentos.renderizar:
        metadados["audio"] = {
            "soundfont": argumentos.soundfont.name,
            "soundfont_sha256": sha256(argumentos.soundfont.read_bytes()).hexdigest(),
            "taxa_amostragem": TAXA_AMOSTRAGEM,
            "ganho": GANHO
        }

    caminho_proveniencia = argumentos.saida / "proveniencia.json"

    with open(caminho_proveniencia, "w", encoding="utf-8") as arquivo:
        json.dump(
            metadados,
            arquivo,
            indent=4,
            ensure_ascii=False
        )

    print("\nEXEMPLOS PREPARADOS")
    print("Total de exemplos:", len(registros))
    print("Pasta de saída:", argumentos.saida)


# Argumentos de entrada e saída.
def main():

    parser = argparse.ArgumentParser(
        description="Separa cinco exemplos MIDI do Markov e sintetiza seus áudios."
    )

    parser.add_argument("--entrada", type=Path, default=PASTA_ENTRADA)
    parser.add_argument("--saida", type=Path, default=PASTA_SAIDA)
    parser.add_argument("--renderizar", action="store_true")
    parser.add_argument("--soundfont", type=Path, default=SOUNDFONT_PADRAO)

    argumentos = parser.parse_args()
    preparar_exemplos(argumentos)


if __name__ == "__main__":
    main()
