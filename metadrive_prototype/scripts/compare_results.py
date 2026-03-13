"""
Confronto Risultati: Curriculum vs Batch Training
==================================================
Legge i file results.json prodotti da train_experiment.py
e genera grafici comparativi salvati in PNG.

Questo script e' progettato con un formato JSON stabile
che verra' riusato anche per la fase CARLA (cambia solo
meta.simulator e meta.algorithm nei JSON).

Esegui con:
    python ./metadrive_prototype/scripts/compare_results.py --batch metadrive_prototype/experiments/<batch_dir>/results.json --curriculum metadrive_prototype/experiments/<curriculum_dir>/results.json

    python ./metadrive_prototype/scripts/compare_results.py --batch metadrive_prototype/experiments/<batch_dir>/results.json --curriculum metadrive_prototype/experiments/<curriculum_dir>/results.json --output metadrive_prototype/results/plots/my_comparison

Output:
    - <output_dir>/01_success_rate_over_time.png
    - <output_dir>/02_collision_rate_over_time.png
    - <output_dir>/03_window_success_rate_over_time.png
    - <output_dir>/04_reward_over_time.png
    - <output_dir>/05_episode_length_over_time.png
    - <output_dir>/06_evaluation_comparison.png
    - <output_dir>/07_summary_table.png
    - <output_dir>/comparison_summary.txt
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend non-interattivo per salvare PNG senza display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "plots")


# ============================================================
# COSTANTI
# ============================================================

# Colori consistenti per tutta la tesi
COLOR_BATCH = "#2196F3"       # Blu — Batch
COLOR_CURRICULUM = "#FF9800"  # Arancione — Curriculum
COLOR_PROMOTION = "#E91E63"   # Rosa — Linee di promozione

# Stile globale
FIGURE_DPI = 150
FIGURE_SIZE = (10, 6)
FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 12
FONT_SIZE_TICK = 10
FONT_SIZE_LEGEND = 11


# ============================================================
# CARICAMENTO DATI
# ============================================================

def load_results(json_path, strict_status=False):
    """
    Carica un file results.json e valida la struttura.
    
    Args:
        json_path: percorso al file JSON
    
    Returns:
        dizionario con i risultati
    
    Raises:
        FileNotFoundError: se il file non esiste
        KeyError: se mancano campi obbligatori
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File non trovato: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    # Validazione campi obbligatori
    required_keys = ["meta", "timeseries", "evaluation", "training_summary"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Campo obbligatorio '{key}' mancante in {json_path}")

    if "mode" not in data["meta"]:
        raise KeyError(f"Campo 'meta.mode' mancante in {json_path}")

    status = data["meta"].get("status")
    if status is None:
        msg = f"Campo 'meta.status' mancante in {json_path} (schema legacy?)"
        if strict_status:
            raise ValueError(msg)
        print(f"ATTENZIONE: {msg}")
    elif status != "COMPLETATO":
        msg = f"Run non completata in {json_path}: status={status}"
        if strict_status:
            raise ValueError(msg)
        print(f"ATTENZIONE: {msg}")

    if len(data["timeseries"]) == 0:
        print(f"ATTENZIONE: timeseries vuota in {json_path}")

    return data


def extract_timeseries(data, field):
    """
    Estrae una serie temporale dal JSON.
    
    Args:
        data: risultati JSON caricati
        field: nome del campo da estrarre (es. "success_rate")
    
    Returns:
        tuple (timesteps, values) — numpy array
        I valori None vengono convertiti in NaN
    """
    ts = data["timeseries"]
    timesteps = np.array([p["timestep"] for p in ts])
    values = np.array([
        p[field] if p.get(field) is not None else np.nan
        for p in ts
    ])
    return timesteps, values


# ============================================================
# GRAFICI — Funzioni individuali
# ============================================================

def setup_plot(title, xlabel, ylabel):
    """Configura un grafico con stile consistente."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Formatta asse X con migliaia (es. 500K, 1.0M)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, p: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"
    ))

    return fig, ax


def add_promotion_lines(ax, curriculum_data, ymin=None, ymax=None):
    """
    Aggiunge linee verticali tratteggiate per le promozioni del curriculum.
    
    Args:
        ax: matplotlib axes
        curriculum_data: risultati JSON del curriculum
        ymin, ymax: limiti Y per le linee (opzionali)
    """
    for promo in curriculum_data.get("curriculum_history", []):
        timestep = promo["timestep_at_promotion"]
        label_text = f"{promo['from']}→{promo['to']}"
        ax.axvline(x=timestep, color=COLOR_PROMOTION, linestyle="--",
                   alpha=0.7, linewidth=1.5)
        # Posiziona etichetta in alto
        y_pos = ymax if ymax is not None else ax.get_ylim()[1]
        ax.text(timestep, y_pos * 0.95, label_text,
                fontsize=9, color=COLOR_PROMOTION, ha="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=COLOR_PROMOTION, alpha=0.8))


def plot_metric_over_time(batch_data, curriculum_data, field, ylabel, title,
                          output_path, ylim=None, percentage=False):
    """
    Grafico generico: metrica nel tempo, Batch vs Curriculum.
    
    Args:
        batch_data: JSON batch
        curriculum_data: JSON curriculum
        field: campo della timeseries da plottare
        ylabel: etichetta asse Y
        title: titolo del grafico
        output_path: percorso PNG di output
        ylim: tuple (ymin, ymax) opzionale
        percentage: se True, formatta asse Y come percentuale
    """
    fig, ax = setup_plot(title, "Timesteps", ylabel)

    # Batch
    ts_b, vals_b = extract_timeseries(batch_data, field)
    ax.plot(ts_b, vals_b, color=COLOR_BATCH, linewidth=2, label="Batch", alpha=0.85)
    if np.all(np.isnan(vals_b)):
        print(f"ATTENZIONE: metrica '{field}' tutta NaN per Batch.")

    # Curriculum
    ts_c, vals_c = extract_timeseries(curriculum_data, field)
    ax.plot(ts_c, vals_c, color=COLOR_CURRICULUM, linewidth=2,
            label="Curriculum", alpha=0.85)
    if np.all(np.isnan(vals_c)):
        print(f"ATTENZIONE: metrica '{field}' tutta NaN per Curriculum.")

    # Promozioni
    add_promotion_lines(ax, curriculum_data,
                        ymax=ylim[1] if ylim else None)

    if ylim:
        ax.set_ylim(ylim)

    if percentage:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    ax.legend(fontsize=FONT_SIZE_LEGEND, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvato: {output_path}")


def plot_reward_over_time(batch_data, curriculum_data, output_path):
    """
    Reward medio per blocco con banda di deviazione standard.
    """
    fig, ax = setup_plot(
        "Mean Reward per Block — Batch vs Curriculum",
        "Timesteps", "Mean Reward"
    )

    for data, color, label in [
        (batch_data, COLOR_BATCH, "Batch"),
        (curriculum_data, COLOR_CURRICULUM, "Curriculum"),
    ]:
        ts, means = extract_timeseries(data, "reward_mean")
        _, stds = extract_timeseries(data, "reward_std")

        ax.plot(ts, means, color=color, linewidth=2, label=label, alpha=0.85)
        if np.all(np.isnan(means)):
            print(f"ATTENZIONE: reward_mean tutta NaN per {label}.")

        # Banda std (solo dove i dati sono validi)
        valid = ~np.isnan(means) & ~np.isnan(stds)
        if np.any(valid):
            ax.fill_between(
                ts[valid],
                means[valid] - stds[valid],
                means[valid] + stds[valid],
                color=color, alpha=0.15,
            )

    add_promotion_lines(ax, curriculum_data)

    ax.legend(fontsize=FONT_SIZE_LEGEND, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvato: {output_path}")


def plot_evaluation_comparison(batch_data, curriculum_data, output_path):
    """
    Bar chart della valutazione finale su Easy/Medium/Hard/Test.
    Due metriche affiancate: Success Rate e Collision Rate.
    """
    eval_b = batch_data["evaluation"]
    eval_c = curriculum_data["evaluation"]
    levels = ["easy", "medium", "hard", "test"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def extract_eval_values(eval_data, metric, label):
        values = []
        missing_levels = []
        for level in levels:
            value = eval_data.get(level, {}).get(metric)
            if value is None:
                values.append(np.nan)
                missing_levels.append(level)
            else:
                values.append(value)
        if missing_levels:
            missing = ", ".join(missing_levels)
            print(
                f"ATTENZIONE: {label} non ha dati '{metric}' per livelli: "
                f"{missing}. Mostrati come N/A."
            )
        return values

    for ax, metric, title in [
        (axes[0], "success_rate", "Success Rate - Valutazione Finale"),
        (axes[1], "collision_rate", "Collision Rate - Valutazione Finale"),
    ]:
        batch_vals = extract_eval_values(eval_b, metric, "Batch")
        curric_vals = extract_eval_values(eval_c, metric, "Curriculum")

        x = np.arange(len(levels))
        width = 0.35

        bars_b = ax.bar(x - width / 2, batch_vals, width,
                        label="Batch", color=COLOR_BATCH, alpha=0.85)
        bars_c = ax.bar(x + width / 2, curric_vals, width,
                        label="Curriculum", color=COLOR_CURRICULUM, alpha=0.85)

        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([lv.capitalize() for lv in levels],
                           fontsize=FONT_SIZE_TICK)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=FONT_SIZE_LEGEND)
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")

        # Etichette sulle barre
        for bars in [bars_b, bars_c]:
            for bar in bars:
                height = bar.get_height()
                if np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width() / 2., 0.02,
                            "N/A", ha="center", va="bottom",
                            fontsize=8, fontweight="bold")
                elif height > 0.02:  # Non mostrare etichette per valori troppo piccoli
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.015,
                            f"{height:.0%}", ha="center", va="bottom",
                            fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvato: {output_path}")

def plot_summary_table(batch_data, curriculum_data, output_path):
    """
    Genera un'immagine con la tabella riassuntiva delle metriche.
    Utile per inserimento diretto nella tesi.
    """
    meta_b = batch_data["meta"]
    meta_c = curriculum_data["meta"]
    summ_b = batch_data["training_summary"]
    summ_c = curriculum_data["training_summary"]
    eval_b = batch_data["evaluation"]
    eval_c = curriculum_data["evaluation"]

    # Costruisci righe della tabella
    rows = [
        ["Simulator", meta_b["simulator"], meta_c["simulator"]],
        ["Algorithm", meta_b["algorithm"], meta_c["algorithm"]],
        ["Seed", str(meta_b["seed"]), str(meta_c["seed"])],
        ["Total Timesteps", f"{meta_b['total_timesteps_actual']:,}",
         f"{meta_c['total_timesteps_actual']:,}"],
        ["Total Episodes", f"{meta_b['total_episodes']:,}",
         f"{meta_c['total_episodes']:,}"],
        ["Wall-Clock Time", f"{meta_b['wall_clock_seconds']:.0f}s",
         f"{meta_c['wall_clock_seconds']:.0f}s"],
        ["Train Success Rate", f"{summ_b['cumulative_success_rate']:.1%}",
         f"{summ_c['cumulative_success_rate']:.1%}"],
        ["Train Collision Rate", f"{summ_b['cumulative_collision_rate']:.1%}",
         f"{summ_c['cumulative_collision_rate']:.1%}"],
    ]

    # Aggiungi eval per livello
    for level in ["easy", "medium", "hard", "test"]:
        if level in eval_b and level in eval_c:
            sr_b = eval_b[level]["success_rate"]
            sr_c = eval_c[level]["success_rate"]
            cr_b = eval_b[level]["collision_rate"]
            cr_c = eval_c[level]["collision_rate"]

            # Evidenzia il vincitore con freccia
            sr_winner = " >" if sr_b > sr_c else " <" if sr_b < sr_c else " ="
            cr_winner = " <" if cr_b < cr_c else " >" if cr_b > cr_c else " ="

            rows.append([
                f"Eval {level.capitalize()} SR",
                f"{sr_b:.1%}{sr_winner}", f"{sr_c:.1%}"
            ])
            rows.append([
                f"Eval {level.capitalize()} CR",
                f"{cr_b:.1%}{cr_winner}", f"{cr_c:.1%}"
            ])

    # Crea tabella come immagine
    fig, ax = plt.subplots(figsize=(10, 0.4 * len(rows) + 1.5))
    ax.axis("off")
    ax.set_title("Summary: Batch vs Curriculum",
                 fontsize=FONT_SIZE_TITLE, fontweight="bold", pad=20)

    col_labels = ["Metric", "Batch", "Curriculum"]
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    # Stile header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#333333")
        cell.set_text_props(color="white", fontweight="bold")

    # Colori alternati per le righe
    for i in range(1, len(rows) + 1):
        color = "#F5F5F5" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(color)

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvato: {output_path}")


# ============================================================
# REPORT TESTUALE
# ============================================================

def save_comparison_txt(batch_data, curriculum_data, output_path):
    """Salva un confronto testuale sintetico."""
    meta_b = batch_data["meta"]
    meta_c = curriculum_data["meta"]
    eval_b = batch_data["evaluation"]
    eval_c = curriculum_data["evaluation"]

    with open(output_path, "w") as f:
        f.write("CONFRONTO BATCH vs CURRICULUM\n")
        f.write(f"{'=' * 60}\n\n")

        f.write(f"Simulator: {meta_b['simulator']}\n")
        f.write(f"Batch seed: {meta_b['seed']} | "
                f"Curriculum seed: {meta_c['seed']}\n")
        f.write(f"Budget per esperimento: "
                f"{meta_b['total_timesteps_budget']:,} step\n\n")

        f.write(f"{'Livello':<10} {'Batch SR':<12} {'Curric SR':<12} "
                f"{'Batch CR':<12} {'Curric CR':<12} {'Winner SR':<12}\n")
        f.write(f"{'-' * 72}\n")

        for level in ["easy", "medium", "hard", "test"]:
            if level in eval_b and level in eval_c:
                b_sr = eval_b[level]["success_rate"]
                c_sr = eval_c[level]["success_rate"]
                b_cr = eval_b[level]["collision_rate"]
                c_cr = eval_c[level]["collision_rate"]

                if b_sr > c_sr:
                    winner = "Batch"
                elif c_sr > b_sr:
                    winner = "Curriculum"
                else:
                    winner = "Pari"

                f.write(f"{level:<10} {b_sr:<12.1%} {c_sr:<12.1%} "
                        f"{b_cr:<12.1%} {c_cr:<12.1%} {winner:<12}\n")

        # Curriculum promotions
        f.write(f"\nCurriculum Promotions:\n")
        for promo in curriculum_data.get("curriculum_history", []):
            f.write(f"  {promo['from']} -> {promo['to']} at "
                    f"timestep {promo['timestep_at_promotion']:,} "
                    f"(SR: {promo['success_rate_at_promotion']:.1%})\n")

        # Wall-clock
        f.write(f"\nWall-Clock Time:\n")
        f.write(f"  Batch: {meta_b['wall_clock_seconds']:.0f}s\n")
        f.write(f"  Curriculum: {meta_c['wall_clock_seconds']:.0f}s\n")

    print(f"Salvato: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Confronta risultati Batch vs Curriculum",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempio:
    python metadrive_prototype/scripts/compare_results.py \\
        --batch metadrive_prototype/experiments/batch/batch_run_20260311/results.json \\
        --curriculum metadrive_prototype/experiments/curriculum/curriculum_run_20260311/results.json \\
        --output metadrive_prototype/results/plots/comparison_v1
        """
    )
    parser.add_argument("--batch", type=str, required=True,
                        help="Path al results.json del training Batch")
    parser.add_argument("--curriculum", type=str, required=True,
                        help="Path al results.json del training Curriculum")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory di output per i grafici (default: metadrive_prototype/results/plots)")
    parser.add_argument("--strict-status", action="store_true",
                        help="Se attivo, fallisce quando meta.status e' mancante o != COMPLETATO")
    args = parser.parse_args()

    # Carica dati
    print("=" * 60)
    print("CONFRONTO RISULTATI: BATCH vs CURRICULUM")
    print("=" * 60)

    print(f"\nCaricamento Batch: {args.batch}")
    batch_data = load_results(args.batch, strict_status=args.strict_status)
    print(f"  Mode: {batch_data['meta']['mode']}, "
          f"Timesteps: {batch_data['meta']['total_timesteps_actual']:,}, "
          f"Episodes: {batch_data['meta']['total_episodes']:,}")

    print(f"\nCaricamento Curriculum: {args.curriculum}")
    curriculum_data = load_results(args.curriculum, strict_status=args.strict_status)
    print(f"  Mode: {curriculum_data['meta']['mode']}, "
          f"Timesteps: {curriculum_data['meta']['total_timesteps_actual']:,}, "
          f"Episodes: {curriculum_data['meta']['total_episodes']:,}")

    # Validazione: conferma che stiamo confrontando batch vs curriculum
    if batch_data["meta"]["mode"] != "batch":
        print(f"ATTENZIONE: il file --batch ha mode='{batch_data['meta']['mode']}', "
              f"non 'batch'. Procedo comunque.")
    if curriculum_data["meta"]["mode"] != "curriculum":
        print(f"ATTENZIONE: il file --curriculum ha mode='"
              f"{curriculum_data['meta']['mode']}', non 'curriculum'. "
              f"Procedo comunque.")

    # Crea directory output
    os.makedirs(args.output, exist_ok=True)
    print(f"\nOutput directory: {args.output}")

    # Genera grafici
    print("\nGenerazione grafici...")
    plot_paths = {
        "success_rate": os.path.join(args.output, "01_success_rate_over_time.png"),
        "collision_rate": os.path.join(args.output, "02_collision_rate_over_time.png"),
        "window_success_rate": os.path.join(args.output, "03_window_success_rate_over_time.png"),
        "reward": os.path.join(args.output, "04_reward_over_time.png"),
        "episode_length": os.path.join(args.output, "05_episode_length_over_time.png"),
        "evaluation": os.path.join(args.output, "06_evaluation_comparison.png"),
        "summary_table": os.path.join(args.output, "07_summary_table.png"),
    }

    # 1. Success Rate nel tempo (cumulativa)
    plot_metric_over_time(
        batch_data, curriculum_data,
        field="success_rate",
        ylabel="Cumulative Success Rate",
        title="Cumulative Success Rate — Batch vs Curriculum",
        output_path=plot_paths["success_rate"],
        ylim=(0, 1.0),
        percentage=True,
    )

    # 2. Collision Rate nel tempo (cumulativa)
    plot_metric_over_time(
        batch_data, curriculum_data,
        field="collision_rate",
        ylabel="Cumulative Collision Rate",
        title="Cumulative Collision Rate — Batch vs Curriculum",
        output_path=plot_paths["collision_rate"],
        ylim=(0, 1.0),
        percentage=True,
    )

    # 3. Window Success Rate nel tempo (finestra mobile — piu reattiva)
    plot_metric_over_time(
        batch_data, curriculum_data,
        field="window_success_rate",
        ylabel="Window Success Rate (50 episodes)",
        title="Window Success Rate — Batch vs Curriculum",
        output_path=plot_paths["window_success_rate"],
        ylim=(0, 1.0),
        percentage=True,
    )

    # 4. Reward medio per blocco con banda std
    plot_reward_over_time(
        batch_data, curriculum_data,
        output_path=plot_paths["reward"],
    )

    # 5. Episode length nel tempo
    plot_metric_over_time(
        batch_data, curriculum_data,
        field="episode_length_mean",
        ylabel="Mean Episode Length (steps)",
        title="Mean Episode Length — Batch vs Curriculum",
        output_path=plot_paths["episode_length"],
    )

    # 6. Bar chart valutazione finale
    plot_evaluation_comparison(
        batch_data, curriculum_data,
        output_path=plot_paths["evaluation"],
    )

    # 7. Tabella riassuntiva
    plot_summary_table(
        batch_data, curriculum_data,
        output_path=plot_paths["summary_table"],
    )

    # 8. Report testuale
    save_comparison_txt(
        batch_data, curriculum_data,
        output_path=os.path.join(args.output, "comparison_summary.txt"),
    )

    print(f"\n{'=' * 60}")
    print(f"COMPLETATO — {len(plot_paths)} grafici + 1 report salvati in: {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

