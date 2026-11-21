from pathlib import Path
import argparse
import statistics
import copy
import pandas as pd
import miditoolkit


# Caminhos principais do projeto e das pastas de dados.

PASTA_PROJETO = Path(__file__).resolve().parents[1]
PASTA_DATASET = PASTA_PROJETO / "dataset" / "EMOPIA_1.0"
PASTA_MIDIS = PASTA_DATASET / "midis"
PASTA_SPLITS = PASTA_PROJETO / "splits"
PASTA_SAIDA = PASTA_PROJETO / "processed_midi"

CONJUNTOS = ["train", "val", "test"]

# Quatro subdivisões por semínima = semicolcheias.
RESOLUCAO = 4

# Transposições aplicadas somente ao conjunto de treinamento.
TRANSPOSTOES = [-2, -1, 1, 2]

# Intervalo de alturas definido na configuração do REMI.
PITCH_MINIMO = 21
PITCH_MAXIMO = 108


def obter_nome_arquivo(clip):
    """Garante que o nome do clipe tenha uma extensão MIDI."""

    clip = str(clip).strip()

    if clip.lower().endswith((".mid", ".midi")):
        return clip

    return clip + ".mid"


def quantizar_midi(caminho_entrada, caminho_saida):
    """
    Ajusta o início e o fim das notas para a grade rítmica definida.
    O arquivo MIDI original não é modificado.
    """

    midi = miditoolkit.MidiFile(str(caminho_entrada))

    ticks_por_subdivisao = max(
        1,
        round(midi.ticks_per_beat / RESOLUCAO)
    )

    total_notas = 0

    for instrumento in midi.instruments:
        for nota in instrumento.notes:

            inicio = round(
                nota.start / ticks_por_subdivisao
            ) * ticks_por_subdivisao

            fim = round(
                nota.end / ticks_por_subdivisao
            ) * ticks_por_subdivisao

            if fim <= inicio:
                fim = inicio + ticks_por_subdivisao

            nota.start = max(0, inicio)
            nota.end = fim

            total_notas += 1

    caminho_saida.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    midi.dump(str(caminho_saida))

    if not caminho_saida.exists():
        raise IOError("O arquivo MIDI não foi salvo.")

    # Confirma que o arquivo salvo pode ser aberto.
    miditoolkit.MidiFile(str(caminho_saida))

    return total_notas, ticks_por_subdivisao


def quantizar_conjunto(nome_conjunto):

    caminho_csv = PASTA_SPLITS / f"{nome_conjunto}_clip.csv"

    if not caminho_csv.exists():
        raise FileNotFoundError(
            f"CSV não encontrado: {caminho_csv}"
        )

    dados = pd.read_csv(caminho_csv)

    if "clip_name" not in dados.columns:
        raise ValueError(
            f"O arquivo {caminho_csv.name} "
            "não possui a coluna 'clip_name'."
        )

    pasta_saida = PASTA_SAIDA / nome_conjunto
    pasta_saida.mkdir(parents=True, exist_ok=True)

    resultados = []
    erros = []

    print("\n" + "=" * 60)
    print(f"QUANTIZANDO: {nome_conjunto.upper()}")
    print("=" * 60)

    total = len(dados)

    for indice, linha in dados.iterrows():

        clip = str(linha["clip_name"]).strip()
        nome_arquivo = obter_nome_arquivo(clip)

        caminho_entrada = PASTA_MIDIS / nome_arquivo
        caminho_saida = pasta_saida / nome_arquivo

        try:
            if not caminho_entrada.exists():
                raise FileNotFoundError(
                    f"MIDI original não encontrado: {caminho_entrada}"
                )

            quantidade_notas, subdivisao = quantizar_midi(
                caminho_entrada,
                caminho_saida
            )

            resultados.append({
                "conjunto": nome_conjunto,
                "clip_name": Path(nome_arquivo).stem,
                "quantidade_notas": quantidade_notas,
                "ticks_por_subdivisao": subdivisao,
                "arquivo_processado": str(caminho_saida)
            })

        except Exception as erro:
            erros.append({
                "conjunto": nome_conjunto,
                "clip_name": clip,
                "erro": str(erro)
            })

        if (indice + 1) % 50 == 0 or (indice + 1) == total:
            print(f"Progresso: {indice + 1}/{total}")

    print("\nRESULTADO DO CONJUNTO")
    print("-" * 40)
    print("Esperados:", total)
    print("Processados:", len(resultados))
    print("Erros:", len(erros))

    return resultados, erros


def quantizar_todos_conjuntos():

    todos_resultados = []
    todos_erros = []

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    print("\nINICIANDO QUANTIZAÇÃO DO EMOPIA")

    for nome_conjunto in CONJUNTOS:

        resultados, erros = quantizar_conjunto(nome_conjunto)

        todos_resultados.extend(resultados)
        todos_erros.extend(erros)

    caminho_estatisticas = (
        PASTA_SAIDA / "estatisticas_quantizacao.csv"
    )

    pd.DataFrame(todos_resultados).to_csv(
        caminho_estatisticas,
        index=False
    )

    caminho_erros = PASTA_SAIDA / "erros_quantizacao.csv"

    pd.DataFrame(
        todos_erros,
        columns=["conjunto", "clip_name", "erro"]
    ).to_csv(caminho_erros, index=False)

    print("\n" + "=" * 60)
    print("QUANTIZAÇÃO FINALIZADA")
    print("=" * 60)

    print("Total esperado:", 1078)
    print("Total processado:", len(todos_resultados))
    print("Total com erro:", len(todos_erros))

    for nome_conjunto in CONJUNTOS:
        quantidade = sum(
            1 for item in todos_resultados
            if item["conjunto"] == nome_conjunto
        )
        print(f"{nome_conjunto}: {quantidade}")

    print("\nArquivos gerados:")
    print(caminho_estatisticas)
    print(caminho_erros)

    if len(todos_resultados) == 1078 and len(todos_erros) == 0:
        print("\nSUCESSO: Todos os MIDI foram processados!")
    else:
        print("\nATENÇÃO: Verifique erros_quantizacao.csv.")


def transpor_midi(caminho_entrada, caminho_saida, semitons):
    """
    Cria uma cópia transposta de um MIDI.

    Instrumentos de percussão não são transpostos.
    A transposição é rejeitada se alguma nota ficar fora
    do intervalo de alturas configurado para o REMI.
    """

    midi_original = miditoolkit.MidiFile(str(caminho_entrada))
    midi = copy.deepcopy(midi_original)

    notas_encontradas = 0

    for instrumento in midi.instruments:

        # Não altera instrumentos de percussão.
        if instrumento.is_drum:
            continue

        for nota in instrumento.notes:

            nova_altura = nota.pitch + semitons

            if not PITCH_MINIMO <= nova_altura <= PITCH_MAXIMO:
                return False

            notas_encontradas += 1

    if notas_encontradas == 0:
        return False

    # Aplica a transposição depois de verificar todas as notas.
    for instrumento in midi.instruments:

        if instrumento.is_drum:
            continue

        for nota in instrumento.notes:
            nota.pitch += semitons

    caminho_saida.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    midi.dump(str(caminho_saida))

    # Verifica se o arquivo gerado pode ser aberto.
    miditoolkit.MidiFile(str(caminho_saida))

    return True


def aumentar_treino():

    """
    Cria versões transpostas dos MIDI quantizados de treinamento.

    Os arquivos originais permanecem intactos.
    Validação e teste não são modificados.
    """

    caminho_csv = PASTA_SPLITS / "train_clip.csv"

    if not caminho_csv.exists():
        raise FileNotFoundError(caminho_csv)

    dados = pd.read_csv(caminho_csv)

    pasta_origem = PASTA_SAIDA / "train"
    pasta_saida = PASTA_SAIDA / "train_augmented"

    if not pasta_origem.exists():
        raise FileNotFoundError(
            "A pasta de MIDI quantizados de treino não existe. "
            "Execute primeiro --quantizar-todos."
        )

    pasta_saida.mkdir(parents=True, exist_ok=True)

    # Remove versões antigas para evitar duplicações em novas execuções.
    for arquivo_antigo in pasta_saida.glob("*.mid"):
        arquivo_antigo.unlink()

    resultados = []
    erros = []

    total_gerado = 0
    total_ignorado = 0

    print("\n" + "=" * 60)
    print("AUMENTO DE DADOS: TREINAMENTO")
    print("=" * 60)

    for indice, linha in dados.iterrows():

        clip = str(linha["clip_name"]).strip()
        nome_original = Path(obter_nome_arquivo(clip)).name

        caminho_entrada = pasta_origem / nome_original

        if not caminho_entrada.exists():
            erros.append({
                "clip_name": clip,
                "erro": f"MIDI quantizado não encontrado: {caminho_entrada}"
            })
            continue

        for semitons in TRANSPOSTOES:

            sinal = "m" if semitons < 0 else "p"
            nome_saida = (
                f"{Path(nome_original).stem}"
                f"__transpose_{sinal}{abs(semitons)}.mid"
            )

            caminho_saida = pasta_saida / nome_saida

            try:
                gerado = transpor_midi(
                    caminho_entrada,
                    caminho_saida,
                    semitons
                )

                if not gerado:
                    total_ignorado += 1
                    continue

                resultados.append({
                    "clip_name": Path(nome_saida).stem,
                    "original_clip_name": Path(nome_original).stem,
                    "semitons": semitons,
                    "arquivo_processado": str(caminho_saida)
                })

                total_gerado += 1

            except Exception as erro:
                erros.append({
                    "clip_name": clip,
                    "erro": str(erro)
                })

        if (indice + 1) % 50 == 0 or (indice + 1) == len(dados):
            print(f"Progresso: {indice + 1}/{len(dados)}")

    # Salva a lista de MIDI aumentados.
    caminho_csv_saida = PASTA_SPLITS / "train_augmented_clip.csv"

    pd.DataFrame(
        resultados,
        columns=[
            "clip_name",
            "original_clip_name",
            "semitons",
            "arquivo_processado"
        ]
    ).to_csv(caminho_csv_saida, index=False)

    caminho_erros = PASTA_SAIDA / "erros_aumento_treino.csv"

    pd.DataFrame(
        erros,
        columns=["clip_name", "erro"]
    ).to_csv(caminho_erros, index=False)

    print("\nRESULTADO DO AUMENTO")
    print("-" * 40)
    print("Clipes originais de treino:", len(dados))
    print("Versões transpostas criadas:", total_gerado)
    print("Transposições ignoradas:", total_ignorado)
    print("Erros:", len(erros))

    print("\nArquivos gerados:")
    print(pasta_saida)
    print(caminho_csv_saida)
    print(caminho_erros)


def converter_ticks_para_segundos(midi, ultimo_tick):
    """Calcula a duração considerando as mudanças de andamento."""

    ticks_por_beat = midi.ticks_per_beat
    mudancas = sorted(
        midi.tempo_changes,
        key=lambda item: item.time
    )

    tempo_atual = 500000  # 120 BPM, em microssegundos por semínima.
    tick_anterior = 0
    segundos = 0.0

    for mudanca in mudancas:

        if mudanca.time > ultimo_tick:
            break

        if mudanca.time > tick_anterior:
            diferenca = mudanca.time - tick_anterior

            segundos += (
                diferenca * tempo_atual
                / (ticks_por_beat * 1_000_000)
            )

            tick_anterior = mudanca.time

        tempo_atual = mudanca.tempo

    if ultimo_tick > tick_anterior:
        diferenca = ultimo_tick - tick_anterior

        segundos += (
            diferenca * tempo_atual
            / (ticks_por_beat * 1_000_000)
        )

    return segundos


def analisar_treino():

    caminho_csv = PASTA_SPLITS / "train_clip.csv"

    if not caminho_csv.exists():
        raise FileNotFoundError(caminho_csv)

    dados = pd.read_csv(caminho_csv)

    quantidades_notas = []
    duracoes = []
    resultados = []
    arquivos_invalidos = []

    print("\nANÁLISE DOS MIDI DE TREINAMENTO")
    print("-" * 50)
    print("Clipes encontrados:", len(dados))

    for clip in dados["clip_name"]:

        clip = str(clip).strip()
        nome_arquivo = obter_nome_arquivo(clip)
        caminho_midi = PASTA_MIDIS / nome_arquivo

        try:
            midi = miditoolkit.MidiFile(str(caminho_midi))

            quantidade_notas = sum(
                len(instrumento.notes)
                for instrumento in midi.instruments
            )

            duracao = converter_ticks_para_segundos(
                midi,
                midi.max_tick
            )

            quantidades_notas.append(quantidade_notas)
            duracoes.append(duracao)

            resultados.append({
                "clip_name": clip,
                "quantidade_notas": quantidade_notas,
                "duracao_segundos": duracao
            })

        except Exception as erro:
            arquivos_invalidos.append({
                "clip_name": clip,
                "erro": str(erro)
            })

    if not quantidades_notas:
        raise ValueError("Nenhum MIDI válido foi analisado.")

    print("\nESTATÍSTICAS DAS NOTAS")
    print("-" * 40)
    print("Total:", sum(quantidades_notas))
    print("Média:", round(statistics.mean(quantidades_notas), 2))
    print("Mediana:", round(statistics.median(quantidades_notas), 2))
    print("Mínimo:", min(quantidades_notas))
    print("Máximo:", max(quantidades_notas))

    print("\nDURAÇÃO DOS CLIPES")
    print("-" * 40)
    print("Média:", round(statistics.mean(duracoes), 2))
    print("Mediana:", round(statistics.median(duracoes), 2))
    print("Mínima:", round(min(duracoes), 2))
    print("Máxima:", round(max(duracoes), 2))

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(resultados).to_csv(
        PASTA_SAIDA / "estatisticas_treino.csv",
        index=False
    )

    pd.DataFrame(
        arquivos_invalidos,
        columns=["clip_name", "erro"]
    ).to_csv(
        PASTA_SAIDA / "erros_analise_treino.csv",
        index=False
    )

    print("\nAnalisados:", len(resultados))
    print("Com erro:", len(arquivos_invalidos))


def testar_um_midi(nome_clipe):

    nome_arquivo = obter_nome_arquivo(nome_clipe)
    caminho_entrada = PASTA_MIDIS / nome_arquivo

    caminho_saida = (
        PASTA_SAIDA / "teste_individual" / nome_arquivo
    )

    if not caminho_entrada.exists():
        raise FileNotFoundError(
            f"MIDI não encontrado: {caminho_entrada}"
        )

    print("\nTESTE DE QUANTIZAÇÃO")
    print("-" * 40)

    quantidade_notas, subdivisao = quantizar_midi(
        caminho_entrada,
        caminho_saida
    )

    print("Arquivo:", nome_arquivo)
    print("Notas processadas:", quantidade_notas)
    print("Ticks por subdivisão:", subdivisao)
    print("Arquivo salvo em:", caminho_saida)
    print("MIDI processado e verificado com sucesso!")


def main():

    parser = argparse.ArgumentParser(
        description="Pré-processamento dos arquivos MIDI do EMOPIA."
    )

    grupo = parser.add_mutually_exclusive_group(required=True)

    grupo.add_argument(
        "--clip",
        help="Processa um único clipe MIDI."
    )

    grupo.add_argument(
        "--quantizar-todos",
        action="store_true",
        help="Quantiza todos os MIDI de treino, validação e teste."
    )

    grupo.add_argument(
        "--analisar-treino",
        action="store_true",
        help="Analisa os MIDI originais de treinamento."
    )

    grupo.add_argument(
        "--aumentar-treino",
        action="store_true",
        help="Cria versões transpostas somente dos MIDI de treino."
    )

    argumentos = parser.parse_args()

    if argumentos.quantizar_todos:
        quantizar_todos_conjuntos()

    elif argumentos.analisar_treino:
        analisar_treino()

    elif argumentos.aumentar_treino:
        aumentar_treino()

    elif argumentos.clip:
        testar_um_midi(argumentos.clip)


if __name__ == "__main__":
    main()
