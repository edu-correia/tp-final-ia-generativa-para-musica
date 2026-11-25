#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DCC831 - IA Generativa para Música (UFMG)
Parte 4: Pipeline de Avaliação Objetiva de Geração Simbólica
Baseado no protocolo de Yang & Lerch (2020) com MusPy.
"""

import copy
import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import muspy
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from pathlib import Path

# Configuração visual estilo artigo acadêmico / ISMIR
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

# Grade de semicolcheias usada na quantização (P1): posições por tempo.
RESOLUCAO_GROOVE = 4


# ==============================================================================
# 1. Extração de Métricas com MusPy
# ==============================================================================

def extract_music_metrics(midi_path: str) -> dict:
    """Extrai métricas musicais objetivas de um arquivo MIDI com MusPy."""
    try:
        music = muspy.read_midi(midi_path)
    except Exception as e:
        print(f"[Aviso] Falha ao ler {midi_path}: {e}")
        return None

    if len(music.tracks) == 0 or sum(len(t.notes) for t in music.tracks) == 0:
        return None

    try:
        # Métricas tonais e estruturais
        p_range = float(muspy.pitch_range(music))
        p_entropy = float(muspy.pitch_entropy(music))
        pc_entropy = float(muspy.pitch_class_entropy(music))
        scale_cons = float(muspy.scale_consistency(music))
        poly = float(muspy.polyphony(music))
        empty_beats = float(muspy.empty_beat_rate(music))

        # O groove consistency depende da resolução: com 384 ticks por tempo (humanos)
        # quase todas as posições do compasso são vazias e o valor vai a ~1, enquanto os
        # MIDIs gerados têm 4. Iguala a resolução (sem perda: as notas já estão na grade).
        music_grade = copy.deepcopy(music).adjust_resolution(RESOLUCAO_GROOVE)
        groove_cons = float(muspy.groove_consistency(
            music_grade, measure_resolution=4 * music_grade.resolution
        ))

        # CORREÇÃO: Usar get_real_end_time() para tempo real em segundos
        duration_sec = muspy.get_real_end_time(music)
        total_notes = sum(len(t.notes) for t in music.tracks)
        
        # Se por algum motivo o tempo em segundos não estiver disponível, calcula por tempos (beats)
        if duration_sec <= 0:
            total_beats = muspy.get_end_time(music) / (music.resolution if music.resolution > 0 else 1)
            duration_sec = total_beats * 0.5  # Aproximação padrão de 120 bpm (0.5s por beat)

        note_density = total_notes / duration_sec if duration_sec > 0 else 0.0

        # Altura média das notas (útil para analisar Valence Q1 vs Q3)
        all_pitches = [n.pitch for t in music.tracks for n in t.notes]
        pitch_mean = float(np.mean(all_pitches)) if all_pitches else 0.0

        return {
            "pitch_range": p_range,
            "pitch_entropy": p_entropy,
            "pitch_class_entropy": pc_entropy,
            "scale_consistency": scale_cons,
            "groove_consistency": groove_cons,
            "polyphony": poly,
            "empty_beat_rate": empty_beats,
            "note_density": note_density,
            "pitch_mean": pitch_mean
        }
    except Exception as e:
        print(f"[Aviso] Erro calculando métricas de {midi_path}: {e}")
        return None


def detect_quadrant(filename: str) -> str:
    """Identifica o quadrante de emoção (Q1, Q2, Q3, Q4) a partir do nome do arquivo."""
    match = re.search(r'(Q[1-4])', os.path.basename(filename), re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return "Unknown"


def process_directory(dir_path: str, dataset_label: str) -> pd.DataFrame:
    """Processa todos os arquivos MIDI de um diretório e retorna um DataFrame com as métricas."""
    midi_files = glob.glob(os.path.join(dir_path, "**/*.mid"), recursive=True) + \
                 glob.glob(os.path.join(dir_path, "**/*.midi"), recursive=True)

    records = []
    print(f"Processando '{dataset_label}' em {dir_path} ({len(midi_files)} arquivos)...")

    for f in midi_files:
        m = extract_music_metrics(f)
        if m is not None:
            m["filename"] = os.path.basename(f)
            m["dataset"] = dataset_label
            m["quadrant"] = detect_quadrant(f)
            records.append(m)

    df = pd.DataFrame(records)
    print(f"-> Sucesso em {len(df)}/{len(midi_files)} arquivos.")
    return df


# ==============================================================================
# 2. Métricas de Distribuição (Yang & Lerch, 2020)
# ==============================================================================

def compute_overlap_and_kl(dist_ref: np.ndarray, dist_eval: np.ndarray, bins: int = 50, eps: float = 1e-6):
    """
    Calcula a Overlap Area (OA) e a Divergência KL (KL) entre uma distribuição
    de referência (Humano Teste) e uma distribuição de avaliação (Modelo / Baseline).
    """
    ref = dist_ref[~np.isnan(dist_ref)]
    eval_data = dist_eval[~np.isnan(dist_eval)]

    if len(ref) == 0 or len(eval_data) == 0:
        return np.nan, np.nan

    min_val = min(np.min(ref), np.min(eval_data))
    max_val = max(np.max(ref), np.max(eval_data))

    if min_val == max_val:
        return 1.0, 0.0

    # Histogramas discretizados no mesmo suporte
    h_ref, bin_edges = np.histogram(ref, bins=bins, range=(min_val, max_val), density=True)
    h_eval, _ = np.histogram(eval_data, bins=bin_edges, density=True)

    # Converter densidade contínua para probabilidades discretas
    bin_widths = np.diff(bin_edges)
    p_ref = h_ref * bin_widths
    p_eval = h_eval * bin_widths

    # Normalização
    p_ref = p_ref / (np.sum(p_ref) + eps)
    p_eval = p_eval / (np.sum(p_eval) + eps)

    # Overlap Area (OA) = soma dos mínimos entre as probabilidades
    oa = float(np.sum(np.minimum(p_ref, p_eval)))

    # Divergência de Kullback-Leibler KL(Ref || Eval) com regularização de Laplace
    p_ref_smooth = p_ref + eps
    p_eval_smooth = p_eval + eps
    p_ref_smooth /= np.sum(p_ref_smooth)
    p_eval_smooth /= np.sum(p_eval_smooth)

    kl_div = float(entropy(p_ref_smooth, p_eval_smooth))

    return oa, kl_div


# ==============================================================================
# 3. Geração de Tabelas e Gráficos
# ==============================================================================

def generate_distribution_plots(df_all: pd.DataFrame, output_dir: str, metrics: list):
    """Gera gráficos KDE de densidade comparativos entre os sistemas."""
    os.makedirs(output_dir, exist_ok=True)
    n_metrics = len(metrics)
    cols = 3
    rows = int(np.ceil(n_metrics / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = np.array(axes).reshape(-1)

    palette = {"Humano (Teste)": "#2b5c8f", "LSTM": "#d95f02", "Baseline (Markov)": "#7570b3"}

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        sns.kdeplot(
            data=df_all,
            x=metric,
            hue="dataset",
            common_norm=False,
            palette=palette,
            fill=True,
            alpha=0.3,
            linewidth=2,
            ax=ax
        )
        ax.set_title(f"Distribuição: {metric.replace('_', ' ').title()}", fontweight='bold')
        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_ylabel("Densidade")

    # Ocultar eixos vazios excedentes
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    save_path = os.path.join(output_dir, "distribuicoes_metricas.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"-> Gráficos de distribuição salvos em: {save_path}")


def generate_quadrant_condition_plot(df_all: pd.DataFrame, output_dir: str):
    """Avalia o impacto dos quadrantes Q1-Q4 no Arousal (Note Density) e Valence (Pitch Mean)."""
    df_filtered = df_all[df_all["quadrant"].isin(["Q1", "Q2", "Q3", "Q4"])].copy()
    if df_filtered.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    quadrant_order = ["Q1", "Q2", "Q3", "Q4"]

    sns.barplot(
        data=df_filtered,
        x="quadrant",
        y="note_density",
        hue="dataset",
        order=quadrant_order,
        ax=axes[0],
        capsize=0.1,
        err_kws={'linewidth': 1.5}
    )
    axes[0].set_title("Efeito da Emoção no Arousal (Notas / Segundo)", fontweight='bold')
    axes[0].set_xlabel("Quadrante Emocional")
    axes[0].set_ylabel("Notas / Segundo (NPS)")

    sns.barplot(
        data=df_filtered,
        x="quadrant",
        y="pitch_mean",
        hue="dataset",
        order=quadrant_order,
        ax=axes[1],
        capsize=0.1,
        err_kws={'linewidth': 1.5}
    )
    axes[1].set_title("Efeito da Emoção no Registro Tonal (Pitch Médio)", fontweight='bold')
    axes[1].set_xlabel("Quadrante Emocional")
    axes[1].set_ylabel("MIDI Pitch Médio (0–127)")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "analise_condicionamento_quadrantes.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"-> Gráfico de condicionamento atualizado salvo em: {save_path}")


def generate_latex_and_csv_tables(df_all: pd.DataFrame, output_dir: str, metrics: list):
    """Gera tabelas com Médias, Desvios, Overlap Area e Divergência KL em CSV e LaTeX."""
    ref_df = df_all[df_all["dataset"] == "Humano (Teste)"]
    eval_datasets = [d for d in df_all["dataset"].unique() if d != "Humano (Teste)"]

    table_data = []

    for metric in metrics:
        row = {"Métrica": metric.replace("_", " ").title()}

        # Média e desvio da referência
        ref_vals = ref_df[metric].values
        row["Humano (Teste)"] = f"{np.mean(ref_vals):.2f} ± {np.std(ref_vals):.2f}"

        for eval_name in eval_datasets:
            eval_vals = df_all[df_all["dataset"] == eval_name][metric].values
            mean_std_str = f"{np.mean(eval_vals):.2f} ± {np.std(eval_vals):.2f}"
            oa, kl = compute_overlap_and_kl(ref_vals, eval_vals)
            
            row[f"{eval_name} (Média±DP)"] = mean_std_str
            row[f"{eval_name} (OA ↑)"] = f"{oa:.3f}"
            row[f"{eval_name} (KL ↓)"] = f"{kl:.3f}"

        table_data.append(row)

    res_df = pd.DataFrame(table_data)

    # Exportar CSV
    csv_path = os.path.join(output_dir, "tabela_metricas_objetivas.csv")
    res_df.to_csv(csv_path, index=False)

    # Exportar LaTeX (Template ISMIR)
    latex_path = os.path.join(output_dir, "tabela_metricas_ismir.tex")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("% Tabela gerada automaticamente para o artigo ISMIR\n")
        f.write(res_df.to_latex(index=False, escape=False))

    print(f"-> Tabelas salvas em:\n   {csv_path}\n   {latex_path}")
    print("\n" + "="*80)
    print(res_df.to_string())
    print("="*80 + "\n")


# ==============================================================================
# 4. Execução Principal
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="P4: Avaliação Objetiva de Modelos Generativos de Música")
    parser.add_argument("--test_dir", type=str, default="splits/test",
                        help="Diretório com os MIDIs humanos do conjunto de teste")
    parser.add_argument("--lstm_dir", type=str, default="amostras_geradas",
                        help="Diretório com os MIDIs gerados pelo modelo LSTM")
    parser.add_argument("--baseline_dir", type=str, default=None,
                        help="Diretório com os MIDIs do baseline Markov (opcional, pode rodar depois)")
    parser.add_argument("--out_dir", type=str, default="saidas/avaliacao",
                        help="Diretório de saída para salvar figuras e tabelas")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    dfs = []

    # 1. Carregar conjunto de teste humano
    if os.path.exists(args.test_dir):
        df_human = process_directory(args.test_dir, "Humano (Teste)")
        if not df_human.empty:
            dfs.append(df_human)
    else:
        print(f"[Erro] Diretório de teste não encontrado: {args.test_dir}")
        return

    # 2. Carregar amostras da LSTM
    if os.path.exists(args.lstm_dir):
        df_lstm = process_directory(args.lstm_dir, "LSTM")
        if not df_lstm.empty:
            dfs.append(df_lstm)
    else:
        print(f"[Erro] Diretório da LSTM não encontrado: {args.lstm_dir}")
        return

    # 3. Carregar baseline Markov (se existir)
    if args.baseline_dir and os.path.exists(args.baseline_dir):
        df_baseline = process_directory(args.baseline_dir, "Baseline (Markov)")
        if not df_baseline.empty:
            dfs.append(df_baseline)
    else:
        print("[Info] Baseline Markov não fornecido ou pasta inexistente. Executando avaliação parcial Humano vs LSTM.")

    df_all = pd.concat(dfs, ignore_index=True)

    metrics = [
        "pitch_range",
        "pitch_entropy",
        "pitch_class_entropy",
        "scale_consistency",
        "groove_consistency",
        "polyphony",
        "empty_beat_rate",
        "note_density"
    ]

    print("\nGerando figuras e tabelas...")
    generate_distribution_plots(df_all, args.out_dir, metrics)
    generate_quadrant_condition_plot(df_all, args.out_dir)
    generate_latex_and_csv_tables(df_all, args.out_dir, metrics)
    print("Processamento concluído com sucesso!")


if __name__ == "__main__":
    main()