from pathlib import Path
import argparse
import shutil
import subprocess
import sys


SOUNDFONT_PADRAO = Path("/usr/share/sounds/sf2/FluidR3_GM.sf2")


def verificar_dependencias(precisa_ffmpeg: bool) -> None:
    faltando = []
    if shutil.which("fluidsynth") is None:
        faltando.append("fluidsynth")
    if precisa_ffmpeg and shutil.which("ffmpeg") is None:
        faltando.append("ffmpeg")
    if faltando:
        sys.exit(
            f"Programas não encontrados: {', '.join(faltando)}. "
            "Instale com: sudo apt install fluidsynth fluid-soundfont-gm ffmpeg"
        )


def renderizar_wav(
    caminho_midi: Path,
    caminho_wav: Path,
    soundfont: Path,
    taxa_amostragem: int,
    ganho: float,
) -> None:
    comando = [
        "fluidsynth",
        "-ni",                      # sem shell interativo, sem entrada MIDI ao vivo
        "-g", str(ganho),           # ganho (o padrão do FluidSynth, 0.2, sai baixo)
        "-r", str(taxa_amostragem),
        "-F", str(caminho_wav),     # renderiza para arquivo em vez de tocar
        str(soundfont),
        str(caminho_midi),
    ]
    subprocess.run(comando, check=True, capture_output=True)


def converter_para_mp3(caminho_wav: Path, caminho_mp3: Path) -> None:
    comando = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(caminho_wav),
        "-codec:a", "libmp3lame",
        "-q:a", "2",                # VBR de alta qualidade (~190 kbps)
        str(caminho_mp3),
    ]
    subprocess.run(comando, check=True, capture_output=True)


def renderizar_arquivo(
    caminho_midi: Path,
    pasta_entrada: Path,
    pasta_saida: Path,
    soundfont: Path,
    formato: str,
    taxa_amostragem: int,
    ganho: float,
    sobrescrever: bool,
) -> str:
    """Renderiza um MIDI. Retorna 'ok', 'pulado' ou lança exceção em caso de erro."""
    relativo = caminho_midi.relative_to(pasta_entrada).with_suffix("")
    base_saida = pasta_saida / relativo
    base_saida.parent.mkdir(parents=True, exist_ok=True)

    caminho_wav = base_saida.with_suffix(".wav")
    caminho_mp3 = base_saida.with_suffix(".mp3")

    alvos = {
        "wav": [caminho_wav],
        "mp3": [caminho_mp3],
        "ambos": [caminho_wav, caminho_mp3],
    }[formato]
    if not sobrescrever and all(alvo.exists() for alvo in alvos):
        return "pulado"

    renderizar_wav(caminho_midi, caminho_wav, soundfont, taxa_amostragem, ganho)

    if formato in ("mp3", "ambos"):
        converter_para_mp3(caminho_wav, caminho_mp3)
        if formato == "mp3":
            caminho_wav.unlink()

    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Renderiza MIDIs em áudio com FluidSynth.")
    parser.add_argument("entrada", type=Path, help="pasta com arquivos .mid (busca recursiva)")
    parser.add_argument(
        "--saida", type=Path, default=None,
        help="pasta de saída (padrão: <entrada>_audio ao lado da entrada)",
    )
    parser.add_argument("--soundfont", type=Path, default=SOUNDFONT_PADRAO)
    parser.add_argument("--formato", choices=("wav", "mp3", "ambos"), default="mp3")
    parser.add_argument("--taxa-amostragem", type=int, default=44100)
    parser.add_argument("--ganho", type=float, default=1.0)
    parser.add_argument("--sobrescrever", action="store_true", help="refaz áudios já existentes")
    argumentos = parser.parse_args()

    pasta_entrada = argumentos.entrada.resolve()
    if not pasta_entrada.is_dir():
        sys.exit(f"Pasta de entrada não encontrada: {pasta_entrada}")
    if not argumentos.soundfont.is_file():
        sys.exit(
            f"Soundfont não encontrado: {argumentos.soundfont}. "
            "Instale com: sudo apt install fluid-soundfont-gm, ou passe --soundfont."
        )

    verificar_dependencias(precisa_ffmpeg=argumentos.formato in ("mp3", "ambos"))

    pasta_saida = (
        argumentos.saida.resolve()
        if argumentos.saida is not None
        else pasta_entrada.with_name(pasta_entrada.name + "_audio")
    )

    arquivos_midi = sorted(
        p for p in pasta_entrada.rglob("*")
        if p.suffix.lower() in (".mid", ".midi") and p.is_file()
    )
    if not arquivos_midi:
        sys.exit(f"Nenhum arquivo .mid encontrado em {pasta_entrada}")

    print(f"Renderizando {len(arquivos_midi)} arquivos de {pasta_entrada}")
    print(f"Saída: {pasta_saida} | formato: {argumentos.formato} | soundfont: {argumentos.soundfont.name}")

    contagem = {"ok": 0, "pulado": 0, "erro": 0}
    erros = []

    for indice, caminho_midi in enumerate(arquivos_midi, start=1):
        try:
            status = renderizar_arquivo(
                caminho_midi,
                pasta_entrada,
                pasta_saida,
                argumentos.soundfont,
                argumentos.formato,
                argumentos.taxa_amostragem,
                argumentos.ganho,
                argumentos.sobrescrever,
            )
        except subprocess.CalledProcessError as erro:
            status = "erro"
            mensagem = (erro.stderr or b"").decode(errors="replace").strip()
            erros.append((caminho_midi, mensagem))

        contagem[status] += 1
        print(f"  [{indice}/{len(arquivos_midi)}] {status:6s} {caminho_midi.relative_to(pasta_entrada)}")

    print(f"\nConcluído: {contagem['ok']} renderizados, {contagem['pulado']} pulados, {contagem['erro']} com erro.")
    for caminho, mensagem in erros:
        print(f"  ERRO em {caminho}: {mensagem[:200]}")

    if contagem["erro"]:
        sys.exit(1)


if __name__ == "__main__":
    main()