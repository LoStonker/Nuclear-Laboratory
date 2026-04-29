"""
mc_coincidenze.py
=================
Simulazione Monte Carlo dell'efficienza di coincidenza per l'esperimento
di scattering Compton con sorgente Na-22.



Il rivelatore A è FISSO (asse +X). Il rivelatore B RUOTA nel piano XY
attorno alla sorgente (angolo α). Si contano le coincidenze in funzione di α.

GEOMETRIA:
  - Sorgente nell'origine (0,0,0)
  - Entrambi i rivelatori alla stessa distanza D dalla sorgente
  - Rivelatori cilindrici con faccia circolare di raggio R
  - θ_max = arctan(R/D) è il semi-angolo del cono di accettanza
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


# ──────────────────────────────────────────────────────────────────────────────
# PARAMETRI GEOMETRICI  ← MODIFICA CON I VALORI REALI DEL TUO SETUP
# ──────────────────────────────────────────────────────────────────────────────
D_A = 20.0          # [cm] distanza sorgente → rivelatore A (fisso)
D_B = 20.0          # [cm] distanza sorgente → rivelatore B (mobile)
R_A = 2.54          # [cm] raggio faccia cristallo NaI del rivelatore A
R_B = 2.54          # [cm] raggio faccia cristallo NaI del rivelatore B

# Angoli da simulare [gradi] — modifica il range secondo il tuo setup
ANGOLI_DEG = np.arange(160, 181, 2)   # da 0° a 180° a passi di 5°

# Numero di eventi MC per angolo — 5×10^6 è un buon compromesso velocità/statistica
# A 10^7 eventi la fluttuazione statistica è ~0.03%  (σ/N ≈ 1/√N_coincidenze)
N_EVENTS = 50_000_000


# ──────────────────────────────────────────────────────────────────────────────
# FUNZIONI CORE
# ──────────────────────────────────────────────────────────────────────────────

def genera_fotoni_isotropi(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera n coppie di fotoni back-to-back con distribuzione isotropa corretta.

    ATTENZIONE all'errore classico: NON si estrae θ uniforme in [0, π].
    Farlo addenserebbe i fotoni ai "poli" della sfera, perché l'elemento di
    angolo solido è dΩ = sinθ dθ dφ, non dθ dφ.

    La strategia corretta è campionare u = cos(θ) uniformemente in [-1, 1],
    così ogni direzione occupa lo stesso angolo solido ΔΩ.

    Returns
    -------
    v1 : (n, 3)  vettori unitari del fotone 1
    v2 : (n, 3)  vettori unitari del fotone 2  (= -v1, back-to-back)
    """
    # φ ∈ [0, 2π) uniforme
    phi = np.random.uniform(0.0, 2.0 * np.pi, size=n)

    # u = cos(θ) ∈ [-1, 1] uniforme  →  distribuzione isotropa garantita
    u = np.random.uniform(-1.0, 1.0, size=n)
    sin_theta = np.sqrt(1.0 - u**2)   # sin(θ) ≥ 0 perché θ ∈ [0, π]

    # Componenti cartesiane del fotone 1
    v1 = np.column_stack([
        sin_theta * np.cos(phi),   # x
        sin_theta * np.sin(phi),   # y
        u                          # z = cos(θ)
    ])

    # Il fotone 2 va esattamente nella direzione opposta (annichilazione)
    v2 = -v1

    return v1, v2


def cono_accettanza(R: float, D: float) -> float:
    """
    Calcola il coseno del semi-angolo di apertura θ_max del cono di accettanza.

    Un fotone "colpisce" il rivelatore se l'angolo tra la sua direzione e
    l'asse del rivelatore è ≤ θ_max = arctan(R/D).

    Usare il coseno (e non l'angolo) è computazionalmente più efficiente
    perché il test diventa semplicemente:  dot_product > cos_theta_max

    Parameters
    ----------
    R : raggio della faccia del cristallo  [cm]
    D : distanza sorgente-rivelatore        [cm]

    Returns
    -------
    cos_theta_max : float  (valore tra 0 e 1)
    """
    theta_max = np.arctan(R / D)
    return np.cos(theta_max)


def simula_angolo(alpha_deg: float,
                  v1: np.ndarray,
                  v2: np.ndarray,
                  cos_max_A: float,
                  cos_max_B: float) -> dict:
    """
    Conta le coincidenze per un dato angolo α del rivelatore B.

    Schema trigger (replica l'elettronica reale):
      1. Fotone 1 → Rivelatore A?  (trigger)
         → NO:  evento scartato
         → SÌ:  l'elettronica apre la finestra di coincidenza
      2. Fotone 2 → Rivelatore B?
         → SÌ:  COINCIDENZA registrata

    Il test geometrico usa il prodotto scalare:
      cos(angolo_impatto) = v · asse_rivelatore
      Se questo cos > cos_theta_max → il fotone entra nel cono di accettanza.

    Parameters
    ----------
    alpha_deg    : angolo del rivelatore B rispetto all'asse X [gradi]
    v1, v2       : array (N,3) delle direzioni dei fotoni
    cos_max_A/B  : soglie dei coni di accettanza

    Returns
    -------
    dict con conteggi e efficienze
    """
    alpha_rad = np.deg2rad(alpha_deg)

    # Asse del rivelatore A: fisso sull'asse +X
    asse_A = np.array([1.0, 0.0, 0.0])

    # Asse del rivelatore B: ruota nel piano XY
    asse_B = np.array([np.cos(alpha_rad), np.sin(alpha_rad), 0.0])

    # ── Prodotti scalari (broadcasting: shape N) ──────────────────────────────
    # dot(v1, asse_A) = componente x di v1 (semplificazione geometrica qui)
    dot_v1_A = v1 @ asse_A      # fotone 1 vs rivelatore A
    dot_v2_B = v2 @ asse_B      # fotone 2 vs rivelatore B

    # ── Selezioni (maschere booleane) ─────────────────────────────────────────
    # Il rivelatore A "vede" il fotone 1 se il suo angolo di impatto < θ_max_A
    hit_A = dot_v1_A > cos_max_A                     # trigger

    # Tra gli eventi con trigger, quanti coincidono su B?
    hit_B_dato_A = hit_A & (dot_v2_B > cos_max_B)   # coincidenza

    n_trigger     = int(np.sum(hit_A))
    n_coincidenze = int(np.sum(hit_B_dato_A))
    n_totale      = len(v1)

    # Efficienza assoluta: P(coincidenza | evento emesso)
    efficienza = n_coincidenze / n_totale if n_totale > 0 else 0.0

    # Efficienza relativa: P(coincidenza | trigger su A)
    eff_relativa = n_coincidenze / n_trigger if n_trigger > 0 else 0.0

    # Incertezza statistica poissoniana su N_coincidenze: σ = √N
    sigma_coinc = np.sqrt(n_coincidenze) if n_coincidenze > 0 else 1.0
    sigma_eff   = sigma_coinc / n_totale

    return {
        'alpha_deg':    alpha_deg,
        'n_totale':     n_totale,
        'n_trigger':    n_trigger,
        'n_coincidenze': n_coincidenze,
        'efficienza':   efficienza,
        'sigma_eff':    sigma_eff,
        'eff_relativa': eff_relativa,
    }


# ──────────────────────────────────────────────────────────────────────────────
# SIMULAZIONE PRINCIPALE
# ──────────────────────────────────────────────────────────────────────────────

def run_simulazione(n_events:   int   = N_EVENTS,
                    angoli_deg: np.ndarray = ANGOLI_DEG,
                    D_a: float = D_A, R_a: float = R_A,
                    D_b: float = D_B, R_b: float = R_B,
                    seed: int = 42) -> dict:
    """
    Esegue la simulazione MC per tutti gli angoli richiesti.

    Strategia di efficienza:
      I fotoni vengono generati UNA SOLA VOLTA per tutti gli angoli.
      Per ogni angolo si ricalcola solo il prodotto scalare v2·asse_B,
      che è un'operazione O(N) quasi istantanea con NumPy.
      Così evitiamo di rigenerare N_events × len(angoli) coppie di fotoni.

    Parameters
    ----------
    n_events   : numero di coppie di fotoni da simulare
    angoli_deg : array degli angoli di B da testare
    seed       : seme del generatore random per riproducibilità

    Returns
    -------
    Dizionario con array risultati per ogni angolo
    """
    np.random.seed(seed)

    # Pre-calcolo soglie (costanti durante la simulazione)
    cos_max_A = cono_accettanza(R_a, D_a)
    cos_max_B = cono_accettanza(R_b, D_b)

    print(f"{'='*60}")
    print(f"  SIMULAZIONE MC - COINCIDENZE Na-22")
    print(f"{'='*60}")
    print(f"  N eventi:          {n_events:,}")
    print(f"  Rivelatore A:      R={R_a} cm, D={D_a} cm")
    print(f"  Rivelatore B:      R={R_b} cm, D={D_b} cm")
    print(f"  θ_max A:           {np.rad2deg(np.arctan(R_a/D_a)):.2f}°  "
          f"(cos = {cos_max_A:.5f})")
    print(f"  θ_max B:           {np.rad2deg(np.arctan(R_b/D_b)):.2f}°  "
          f"(cos = {cos_max_B:.5f})")
    print(f"  Angoli simulati:   {len(angoli_deg)} "
          f"({angoli_deg[0]:.0f}°–{angoli_deg[-1]:.0f}°)")
    print(f"{'='*60}")

    # ── Generazione fotoni (una tantum) ───────────────────────────────────────
    print("  Generazione fotoni isotropi... ", end='', flush=True)
    v1, v2 = genera_fotoni_isotropi(n_events)
    print("fatto.")

    # Pre-calcola il prodotto scalare fisso v1·asse_A = x-component di v1
    dot_v1_A = v1[:, 0]   # asse_A = (1,0,0) → prodotto scalare = x di v1
    hit_A    = dot_v1_A > cos_max_A
    n_trigger_globale = int(np.sum(hit_A))
    print(f"  Trigger su A:      {n_trigger_globale:,} / {n_events:,}  "
          f"({100*n_trigger_globale/n_events:.3f}%)")
    print()

    # ── Loop sugli angoli ─────────────────────────────────────────────────────
    risultati = {
        'angoli':      [],
        'coincidenze': [],
        'sigma':       [],
        'efficienza':  [],
        'sigma_eff':   [],
        'eff_rel':     [],
    }

    print(f"  {'Angolo':>7}  {'N_coinc':>10}  {'Efficienza':>12}  {'Eff. rel.':>10}")
    print(f"  {'-'*7}  {'-'*10}  {'-'*12}  {'-'*10}")

    for alpha in angoli_deg:
        res = simula_angolo(alpha, v1, v2, cos_max_A, cos_max_B)

        risultati['angoli'].append(res['alpha_deg'])
        risultati['coincidenze'].append(res['n_coincidenze'])
        risultati['sigma'].append(np.sqrt(res['n_coincidenze']))
        risultati['efficienza'].append(res['efficienza'])
        risultati['sigma_eff'].append(res['sigma_eff'])
        risultati['eff_rel'].append(res['eff_relativa'])

        print(f"  {alpha:>6.1f}°  {res['n_coincidenze']:>10,}  "
              f"{res['efficienza']:>11.6f}  {res['eff_relativa']:>10.6f}")

    # Converte in array NumPy per comodità
    for k in risultati:
        risultati[k] = np.array(risultati[k])

    print(f"\n  Angolo solido A = 2π(1 - cos θ_max) = "
          f"{2*np.pi*(1 - cos_max_A):.5f} sr")
    print(f"  Frazione sferica A = {(1 - cos_max_A)/2:.5f}")

    return risultati


# ──────────────────────────────────────────────────────────────────────────────
# PLOT
# ──────────────────────────────────────────────────────────────────────────────

def plot_coincidenze(risultati: dict, normalizza: bool = True) -> None:
    """
    Plotta N_coincidenze vs angolo con barre d'errore poissoniane.

    Parameters
    ----------
    normalizza : se True, normalizza al massimo (tipicamente a 180° per
                 la geometria back-to-back). Utile per confrontare con i
                 dati sperimentali in unità relative.
    """
    angoli  = risultati['angoli']
    coinc   = risultati['coincidenze'].astype(float)
    sigma   = risultati['sigma']

    if normalizza:
        # Trova il massimo per normalizzare
        idx_max = np.argmax(coinc)
        norm    = coinc[idx_max]
        coinc   = coinc / norm
        sigma   = sigma / norm
        ylabel  = 'Coincidenze normalizzate'
        title   = 'MC Na-22 — Coincidenze normalizzate vs angolo'
    else:
        ylabel = 'Conteggi coincidenze'
        title  = 'MC Na-22 — Coincidenze vs angolo'

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(angoli, coinc, yerr=sigma,
                fmt='o-', color='royalblue',
                markersize=5, linewidth=1.5,
                capsize=3, capthick=1,
                elinewidth=1, label='MC (isotropo)')

    # Linea teorica per geometria ideale (asse back-to-back a 180°)
    # Per due rivelatori puntiformi, la coincidenza è possibile solo a 180°;
    # con estensione finita forma un picco attorno a 180°.
    ax.axvline(180, color='gray', linestyle='--', linewidth=1,
               label='Atteso back-to-back (180°)')

    ax.set_xlabel('Angolo α [°]', fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlim(160, 181, 1)
    ax.set_xticks(np.arange(160, 181, 1))
    ax.legend(fontsize=11)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    #plt.savefig('mc_coincidenze.png', dpi=150)
    plt.show()


def plot_diagnostico_isotropy(n_check: int = 100_000) -> None:
    """
    Verifica diagnostica: controlla che la distribuzione in cos(θ) sia
    davvero uniforme (deve esserlo per costruzione). Se vedessi un picco
    ai poli, vorresti dire che hai campionato θ in modo sbagliato.
    """
    v1, _ = genera_fotoni_isotropi(n_check)

    cos_theta = v1[:, 2]   # z-component = cos(θ)
    phi = np.arctan2(v1[:, 1], v1[:, 0])

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Diagnostica isotropia — distribuzione fotoni MC', fontsize=13)

    # cos(θ): deve essere FLAT in [-1, 1]
    ax1.hist(cos_theta, bins=50, color='steelblue', edgecolor='white',
             density=True)
    ax1.axhline(0.5, color='red', linestyle='--', label='Atteso (piatto, 0.5)')
    ax1.set_xlabel('cos θ', fontsize=12)
    ax1.set_ylabel('Densità', fontsize=12)
    ax1.set_title('Distribuzione in cos θ\n(deve essere uniforme)', fontsize=11)
    ax1.legend()

    # φ: deve essere FLAT in [-π, π]
    ax2.hist(phi, bins=50, color='coral', edgecolor='white', density=True)
    ax2.axhline(1/(2*np.pi), color='red', linestyle='--',
                label=f'Atteso (piatto, {1/(2*np.pi):.3f})')
    ax2.set_xlabel('φ [rad]', fontsize=12)
    ax2.set_ylabel('Densità', fontsize=12)
    ax2.set_title('Distribuzione in φ\n(deve essere uniforme)', fontsize=11)
    ax2.legend()

    plt.tight_layout()
    #plt.savefig('mc_diagnostico_isotropy.png', dpi=150)
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# ESEMPIO D'USO
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # 1. Controlla che l'isotropia sia corretta (opzionale ma consigliato)
    #print("Verifica isotropia generatore...")
    #plot_diagnostico_isotropy(n_check=200_000)

    # 2. Esegui la simulazione principale
    risultati = run_simulazione(
        n_events   = N_EVENTS,
        angoli_deg = ANGOLI_DEG,
        D_a = D_A, R_a = R_A,
        D_b = D_B, R_b = R_B,
        seed = 42,
    )

    # 3. Plot
    plot_coincidenze(risultati, normalizza=True)
'''
    # 4. Salva i risultati su file per analisi successive
    np.savetxt(
        'mc_coincidenze_risultati.txt',
        np.column_stack([
            risultati['angoli'],
            risultati['coincidenze'],
            risultati['sigma'],
            risultati['efficienza'],
            risultati['sigma_eff'],
        ]),
        header='angolo[deg]  N_coincidenze  sigma_poisson  efficienza  sigma_efficienza',
        fmt=['%.1f', '%d', '%.1f', '%.8f', '%.8f']
    )
    print("\nRisultati salvati in mc_coincidenze_risultati.txt")'''
