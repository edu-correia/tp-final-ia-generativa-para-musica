from pathlib import Path
import pandas as pd

# Caminho principal do projeto
PASTA_PROJETO = Path(__file__).resolve().parents[1]

# Caminhos dos arquivos
PASTA_MIDIS = (
    PASTA_PROJETO
    / "dataset"
    / "EMOPIA_1.0"
    / "midis"
)

PASTA_SPLITS = PASTA_PROJETO / "splits"


def verificar_clipes(nome_arquivo):

    # Carrega a lista de clipes
    dados = pd.read_csv(
        PASTA_SPLITS / f"{nome_arquivo}_clip.csv"
    )

    encontrados = 0
    faltantes = []

    # Verifica se cada MIDI existe
    for clip in dados["clip_name"]:

        caminho_midi = PASTA_MIDIS / f"{clip}.mid"

        if caminho_midi.exists():
            encontrados += 1
        else:
            faltantes.append(clip)

    # Exibe os resultados
    print(f"\nConjunto: {nome_arquivo}")
    print(f"Clipes no CSV: {len(dados)}")
    print(f"MIDIs encontrados: {encontrados}")
    print(f"MIDIs não encontrados: {len(faltantes)}")

    if faltantes:

        print("\nExemplos de arquivos não encontrados:")

        for nome in faltantes[:10]:
            print(nome)


def main():

    # Verifica os três conjuntos
    for nome in ["train", "val", "test"]:
        verificar_clipes(nome)


if __name__ == "__main__":
    main()
