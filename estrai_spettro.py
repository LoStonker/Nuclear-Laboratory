"""
sottrai_spettri.py
==================
Isola le misure orarie da uno stack di spettri cumulativi MCA.

Logica:
    file[k] contiene la somma di TUTTE le ore 1..k
    => misura_k = file[k] - file[k-1]   (per k >= 1)
    => misura_0 = file[0]                (prima ora, già isolata)

Output:
    - Un file .dat per ogni misura differenziale nella cartella `output_dir`
    - Un summary plot con il confronto tra cumulativo e differenziale
    - Stampa a schermo di statistiche (counts totali, negativi, ecc.)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
import pandas as pd


# =============================================================================
# CONFIGURAZIONE  –  modifica solo questa sezione
# =============================================================================

INPUT_DIR   = "misure_stabilita"            # cartella con i file .dat cumulativi
OUTPUT_DIR  = "lab nucleare"  # cartella di destinazione dei differenziali
FILE_PATTERN = "*.dat"       # pattern per trovare i file (es. "na22_*.dat")

# Se i file hanno un timestamp nel nome usabile per l'ordinamento, mettilo qui.
# Altrimenti l'ordinamento avviene per data di modifica del file (mtime).
# Opzioni: "mtime" | "name"
SORT_BY = "mtime"

# Numero di canali atteso (8192 per ADC a 13 bit). Usato solo per sanity check.
N_CHANNELS_EXPECTED = 8192

# Se True, i bin con Δ < 0 vengono azzerati (conversione "fisica").
# Se False, vengono mantenuti (utile per diagnostica statistica).
CLIP_NEGATIVE = False

# Quanti spettri differenziali plottare in anteprima (0 = nessuno)
N_PREVIEW_PLOTS = 3

# =============================================================================


def load_spectrum(filepath):
    return pd.read_csv(filepath, header=None, comment='#').to_numpy().ravel()


def sort_files(file_list: list, method: str) -> list:
    """Ordina i file per mtime o per nome."""
    if method == "mtime":
        file_list.sort(key=lambda f: os.path.getmtime(f))
    elif method == "name":
        file_list.sort()
    else:
        raise ValueError(f"SORT_BY deve essere 'mtime' o 'name', non '{method}'")
    return file_list


def compute_differentials(cumulative_spectra: list[np.ndarray],
                           clip_negative: bool = False
                           ) -> list[np.ndarray]:
    """
    Calcola gli spettri differenziali da una lista di spettri cumulativi.

    Parameters
    ----------
    cumulative_spectra : lista ordinata di array 1-D (un array per file)
    clip_negative      : se True, azzera i bin negativi

    Returns
    -------
    Lista di array differenziali (len = len(cumulative_spectra))
    Il primo elemento è uguale al primo cumulativo (misura 0 già isolata).
    """
    differentials = []
    for k, spectrum in enumerate(cumulative_spectra):
        if k == 0:
            diff = spectrum.copy()
        else:
            diff = spectrum - cumulative_spectra[k - 1]

        if clip_negative:
            n_neg = np.sum(diff < 0)
            if n_neg > 0:
                print(f"  [ora {k:03d}] {n_neg} bin negativi azzerati "
                      f"(max deficit = {diff.min():.0f} counts)")
            diff = np.clip(diff, 0, None)

        differentials.append(diff)
    return differentials


def print_statistics(differentials: list[np.ndarray],
                     filenames: list[str]) -> None:
    """Stampa una tabella riassuntiva con le statistiche per ogni ora."""
    print("\n" + "="*70)
    print(f"{'Ora':>4}  {'File sorgente':<35} {'Totale':>9} "
          f"{'Neg_bin':>7} {'Max_ch':>7}")
    print("-"*70)
    for k, (diff, fname) in enumerate(zip(differentials, filenames)):
        total     = int(diff.sum())
        n_neg     = int(np.sum(diff < 0))
        max_ch    = int(np.argmax(diff))
        shortname = Path(fname).name
        print(f"{k:>4}  {shortname:<35} {total:>9,d} {n_neg:>7d} {max_ch:>7d}")
    print("="*70)

    all_totals = [int(d.sum()) for d in differentials]
    print(f"\nConteggi per misura: media = {np.mean(all_totals):,.0f}  "
          f"±  {np.std(all_totals):,.0f}  "
          f"(min={min(all_totals):,d}, max={max(all_totals):,d})")
    print(f"Variazione relativa std/media = "
          f"{np.std(all_totals)/np.mean(all_totals)*100:.1f}%  "
          f"(attesa ~0% se il rate è stabile)\n")


def save_differentials(differentials: list[np.ndarray],
                       output_dir: str,
                       filenames: list[str]) -> None:
    """Salva ogni spettro differenziale come file .dat nella cartella output."""
    os.makedirs(output_dir, exist_ok=True)
    for k, diff in enumerate(differentials):
        # Usa lo stesso nome base del file sorgente, prefissato con "diff_ora_NNN_"
        stem = Path(filenames[k]).stem
        out_path = os.path.join(output_dir, f"diff_ora_{k:03d}_{stem}.dat")
        np.savetxt(out_path, diff, fmt="%d",
                   header=f"Spettro differenziale ora {k} | sorgente: {filenames[k]}")
    print(f"Salvati {len(differentials)} spettri differenziali in '{output_dir}/'")


def plot_summary(cumulative_spectra: list[np.ndarray],
                 differentials: list[np.ndarray]) -> None:
    """Confronto visivo: ultimo cumulativo vs. somma ricostruita dei differenziali."""
    ricostruito = np.sum(differentials, axis=0)
    cumulativo_finale = cumulative_spectra[-1]
    discrepanza = cumulativo_finale - ricostruito

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ch = np.arange(len(cumulativo_finale))

    axes[0].plot(ch, cumulativo_finale, drawstyle='steps-mid',
                 color='steelblue', lw=1, label='Cumulativo finale (file[-1])')
    axes[0].plot(ch, ricostruito, drawstyle='steps-mid',
                 color='tomato', lw=1, ls='--', label='Σ differenziali (controllo)')
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Conteggi')
    axes[0].set_title('Verifica ricostruzione: cumulativo finale vs. somma differenziali')
    axes[0].legend(fontsize=9)

    axes[1].plot(ch, differentials[0], drawstyle='steps-mid',
                 color='green', lw=1, label='Ora 0 (= file[0], già isolata)')
    axes[1].plot(ch, differentials[-1], drawstyle='steps-mid',
                 color='purple', lw=1, label=f'Ora {len(differentials)-1} (ultima)')
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Conteggi')
    axes[1].set_title('Prima e ultima misura differenziale')
    axes[1].legend(fontsize=9)

    axes[2].plot(ch, discrepanza, drawstyle='steps-mid',
                 color='gray', lw=0.8)
    axes[2].axhline(0, color='red', lw=1, ls='--')
    axes[2].set_ylabel('Residuo (counts)')
    axes[2].set_xlabel('Canale ADC')
    axes[2].set_title('Discrepanza bin-per-bin (deve essere ≡ 0)')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "summary_check.png"), dpi=150)
    plt.show()
    print("Plot di controllo salvato in 'summary_check.png'")


def plot_preview(differentials: list[np.ndarray],
                 filenames: list[str],
                 n: int) -> None:
    """Mostra n spettri differenziali a scelta (inizio, metà, fine)."""
    indices = np.linspace(0, len(differentials) - 1, min(n, len(differentials)),
                          dtype=int)
    ch = np.arange(len(differentials[0]))
    fig, axes = plt.subplots(len(indices), 1,
                              figsize=(12, 4 * len(indices)), sharex=True)
    if len(indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        ax.plot(ch, differentials[idx], drawstyle='steps-mid',
                color='royalblue', lw=1)
        ax.set_yscale('log')
        ax.set_ylabel('Conteggi')
        ax.set_title(f'Ora {idx:03d}  –  {Path(filenames[idx]).name}  '
                     f'(tot = {int(differentials[idx].sum()):,d} counts)')

    axes[-1].set_xlabel('Canale ADC')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "preview_differenziali.png"), dpi=150)
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # 1. Trova e ordina i file
    file_list = glob.glob(os.path.join(INPUT_DIR, FILE_PATTERN))
    if not file_list:
        raise FileNotFoundError(
            f"Nessun file trovato con pattern '{FILE_PATTERN}' in '{INPUT_DIR}'")

    file_list = sort_files(file_list, SORT_BY)
    print(f"Trovati {len(file_list)} file, ordinati per '{SORT_BY}'.")
    print(f"Primo: {Path(file_list[0]).name}")
    print(f"Ultimo: {Path(file_list[-1]).name}")

    # 2. Carica tutti gli spettri
    spectra = []
    for f in file_list:
        s = load_spectrum(f)
        if len(s) != N_CHANNELS_EXPECTED:
            print(f"  WARN: {Path(f).name} ha {len(s)} canali "
                  f"(attesi {N_CHANNELS_EXPECTED})")
        spectra.append(s)

    # 3. Sanity check: il cumulativo deve essere monotono non decrescente
    n_violations = 0
    for k in range(1, len(spectra)):
        neg_bins = np.sum(spectra[k] < spectra[k - 1])
        if neg_bins > 0:
            print(f"  WARN: file[{k}] ha {neg_bins} bin con conteggi "
                  f"INFERIORI a file[{k-1}] → possibile errore di ordinamento!")
            n_violations += 1
    if n_violations == 0:
        print("Sanity check OK: tutti i file sono monotonamente non decrescenti.")

    # 4. Calcola differenziali
    differentials = compute_differentials(spectra, clip_negative=CLIP_NEGATIVE)

    # 5. Statistiche
    print_statistics(differentials, file_list)

    # 6. Salva
    save_differentials(differentials, OUTPUT_DIR, file_list)

    # 7. Plot di controllo e anteprima
    plot_summary(spectra, differentials)
    if N_PREVIEW_PLOTS > 0:
        plot_preview(differentials, file_list, N_PREVIEW_PLOTS)