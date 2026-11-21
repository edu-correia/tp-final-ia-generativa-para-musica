from pathlib import Path
from collections import Counter
import argparse
import json

from miditoolkit import MidiFile
from miditok import REMI


PASTA_PROJETO = Path(__file__).resolve().parents[1]
PASTA_TOKENS = PASTA_PROJETO / "processed_tokens_convertidos"
PASTA_PROCESSADOS = PASTA_PROJETO / "processed_midi"
PASTA_SAIDA = PASTA_PROJETO / "processed_midi" / "roundtrip_test"


def carregar_ids_proibidos(tokenizador):
    nomes_especiais = (
        "PAD_None",
        "BOS_None",
        "EOS_None",
        "MASK_None",
        "Emotion_Q1",
        "Emotion_Q2",
        "Emotion_Q3",
        "Emotion_Q4",
    )

    ids = {}
    for nome in nomes_especiais:
        ids[nome] = int(tokenizador.vocab[nome])

    return ids


def observar_notas(caminho_midi: Path):
    midi = MidiFile(str(caminho_midi))
    notas = []

    for instrumento in midi.instruments:
        for nota in instrumento.notes:
            notas.append((int(nota.pitch), int(nota.start)))

    notas.sort()
    return midi.ticks_per_beat, notas


def roundtrip_um_arquivo(caminho_json: Path, tokenizador, ids: dict):
    dados = json.loads(caminho_json.read_text(encoding="utf-8"))
    nome = dados["clip_name"]

    ids_a_remover = {
        ids["PAD_None"],
        ids["BOS_None"],
        ids["EOS_None"],
        ids["MASK_None"],
        ids["Emotion_Q1"],
        ids["Emotion_Q2"],
        ids["Emotion_Q3"],
        ids["Emotion_Q4"],
    }

    ids_musicais = [
        int(token_id)
        for token_id in dados["tokens"]
        if int(token_id) not in ids_a_remover
    ]

    caminho_saida = PASTA_SAIDA / f"{nome}_roundtrip.mid"
    score = tokenizador.decode(ids_musicais)
    caminho_saida.write_bytes(score.dumps_midi())

    caminho_original = PASTA_PROCESSADOS / "test" / f"{nome}.mid"
    if not caminho_original.exists():
        raise FileNotFoundError(f"MIDI original não encontrado: {caminho_original}")

    ppq_original, notas_originais = observar_notas(caminho_original)
    ppq_decodificado, notas_decodificadas = observar_notas(caminho_saida)

    # Normaliza posições de início em quartos de nota para comparar resoluções diferentes.
    inicios_originais = [(pitch, start / ppq_original) for pitch, start in notas_originais]
    inicios_decodificados = [(pitch, start / ppq_decodificado) for pitch, start in notas_decodificadas]

    orig = Counter((p, round(s, 4)) for p, s in inicios_originais)
    deco = Counter((p, round(s, 4)) for p, s in inicios_decodificados)
    comuns = sum((orig & deco).values())
    proporcao = comuns / max(len(inicios_originais), 1)

    so_original = list((orig - deco).elements())[:5]
    so_roundtrip = list((deco - orig).elements())[:5]

    iguais = proporcao

    return {
        "nome": nome,
        "original_notas": len(notas_originais),
        "roundtrip_notas": len(notas_decodificadas),
        "batem_notas_pitche_inicio": iguais,
        "so_original": so_original,
        "so_roundtrip": so_roundtrip,
        "saida": str(caminho_saida),
    }


def selecionar_amostras():
    selecionadas = []
    for quadrante in ("Q1", "Q2", "Q3", "Q4"):
        arquivos = sorted((PASTA_TOKENS / "test").glob(f"{quadrante}_*.json"))
        selecionadas.extend(arquivos)
    return selecionadas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arquivos", nargs="*")
    argumentos = parser.parse_args()

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    caminho_tokenizer_convertido = PASTA_TOKENS / "tokenizer_remi.json"
    caminho_tokenizer = (
        caminho_tokenizer_convertido
        if caminho_tokenizer_convertido.is_file()
        else PASTA_PROJETO / "processed_tokens" / "tokenizer_remi.json"
    )
    tokenizador = REMI(params=caminho_tokenizer)
    ids = carregar_ids_proibidos(tokenizador)

    if argumentos.arquivos:
        caminhos = [Path(nome) for nome in argumentos.arquivos]
    else:
        caminhos = selecionar_amostras()

    print("IDs especiais:")
    for nome in ("PAD_None", "BOS_None", "EOS_None", "MASK_None"):
        print(f"  {nome}: {ids[nome]}")
    for nome in ("Emotion_Q1", "Emotion_Q2", "Emotion_Q3", "Emotion_Q4"):
        print(f"  {nome}: {ids[nome]}")

    print("\nResultados:")
    proporcoes = []
    for caminho in caminhos:
        if not caminho.is_absolute():
            caminho = PASTA_TOKENS / "test" / caminho
        resultado = roundtrip_um_arquivo(caminho, tokenizador, ids)
        proporcao = resultado["batem_notas_pitche_inicio"]
        proporcoes.append(proporcao)
        print(
            f"  {resultado['nome']}: "
            f"{resultado['original_notas']} -> {resultado['roundtrip_notas']} notas"
        )
        print(f"    {proporcao:.1%} das notas batem")
        print("    só no original:", resultado["so_original"])
        print("    só no roundtrip:", resultado["so_roundtrip"])

    if proporcoes:
        print(f"\nResumo: {len(proporcoes)} clipes")
        print(f"  média: {sum(proporcoes) / len(proporcoes):.2%}")
        print(f"  mínimo: {min(proporcoes):.2%}")
        print(f"  abaixo de 95%: {sum(p < 0.95 for p in proporcoes)}")
    else:
        print("\nResumo: 0 clipes")
        

if __name__ == "__main__":
    main()
