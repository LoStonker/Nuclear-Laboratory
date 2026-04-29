
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
from scipy.stats import norm, t, chi2
import scipy.stats as stats # Aggiunto per le chiamate stats.*
from iminuit import Minuit
from iminuit.cost import LeastSquares, ExtendedBinnedNLL, UnbinnedNLL, BinnedNLL
from scipy.integrate import quad
import sys
import os
import inspect
from scipy.optimize import curve_fit
from scipy.odr import Model, RealData, ODR
from matplotlib.ticker import AutoMinorLocator
from typing import Any
from math import floor, ceil


def calcola_medie_finali(risultati_grezzi, nome_sorgente):
    """
    Raggruppa i risultati per energia e fa la media pesata tra i vari file.
    """
    print(f"\n{'='*40}")
    print(f" RISULTATI FINALI: {nome_sorgente}")
    print(f"{'='*40}")
    
    # 1. Creiamo un dizionario per raggruppare i dati: {energia: {'mu': [...], 'sigma': [...]}}
    dati_raggruppati = {}
    
    for energia, mu, sigma in risultati_grezzi:
        if energia not in dati_raggruppati:
            dati_raggruppati[energia] = {'mu': [], 'sigma': []}
            
        dati_raggruppati[energia]['mu'].append(mu)
        dati_raggruppati[energia]['sigma'].append(sigma)
        
    # 2. Calcoliamo la media pesata per ogni energia trovata
    punti_calibrazione = [] # Salveremo qui le coppie (Energia, Canale_Medio, Errore_Canale)
    
    for energia, valori in dati_raggruppati.items():
        mu_array = np.array(valori['mu'])
        sigma_array = np.array(valori['sigma'])
        
        # Facciamo la media pesata su tutti i file per questo specifico picco
        mu_medio, sigma_medio = media_pesata(mu_array, sigma_array)
        
        punti_calibrazione.append((energia, mu_medio, sigma_medio))
        print(f"Picco {energia:>6.1f} keV  ->  Canale: {mu_medio:.2f} ± {sigma_medio:.2f}")
        
    return punti_calibrazione




def trova_finestra(canali, conteggi, canale_atteso, semi_ampiezza_ricerca=200, semi_ampiezza_finestra=300):
    """
    Trova c_min e c_max automaticamente cercando il massimo locale
    vicino al canale atteso.
    
    Args:
        canale_atteso:        posizione approssimativa del picco in canali
        semi_ampiezza_ricerca: range entro cui cercare il massimo
        semi_ampiezza_finestra: semi-ampiezza della finestra di fit attorno al massimo trovato
    """
    mask_ricerca = (
        (canali >= canale_atteso - semi_ampiezza_ricerca) & 
        (canali <= canale_atteso + semi_ampiezza_ricerca)
    )
    canali_reg  = canali[mask_ricerca]
    conteggi_reg = conteggi[mask_ricerca]
    
    mu_stimato = canali_reg[np.argmax(conteggi_reg)]
    
    c_min = mu_stimato - semi_ampiezza_finestra
    c_max = mu_stimato + semi_ampiezza_finestra
    
    return c_min, c_max, mu_stimato



def canali_to_energia(canali, p0, p1):
    """
    Converte canali ADC in energia [keV] usando la retta di calibrazione.
    
    Ch = p0 + p1 * E  =>  E = (Ch - p0) / p1
    
    Args:
        canali  : float o array, posizione in canali
        p0      : intercetta della retta di calibrazione [Ch]
        p1      : pendenza della retta di calibrazione [Ch/keV]
    
    Returns:
        energia : float o array, energia in keV
    """
    return (np.asarray(canali) - p0) / p1


def energia_to_canali(energie, p0, p1):
    """
    Converte energia [keV] in canali ADC (utile per plottare la curva di fit
    sovrapposta allo spettro in canali).
    
    Args:
        energie : float o array, energia in keV
        p0      : intercetta [Ch]
        p1      : pendenza [Ch/keV]
    
    Returns:
        canali  : float o array, posizione in canali
    """
    return p0 + p1 * np.asarray(energie)


def sigma_canali_to_energia(sigma_canali, p1):
    return np.asarray(sigma_canali) / p1




def picco_su_gaussiana_larga_pdf(x, Area, mu, sigma, A_largo, sigma_largo, **kwargs):
    gauss_stretta = Area * sc.norm.pdf(x, mu, sigma)
    gauss_larga = A_largo * sc.norm.pdf(x, mu, sigma_largo)
    return gauss_stretta + gauss_larga

def picco_doppia_gaussiana_pdf(x, Area, mu, sigma, 
                                  A_largo, mu_largo, sigma_largo, 
                                  A_fondo, tau, c, x0, 
                                  **kwargs):
    arg = np.clip(-(x - x0) / tau, -700, 700)
    fondo = A_fondo * np.exp(arg) + c
    return Area * sc.norm.pdf(x, mu, sigma) + A_largo * sc.norm.pdf(x, mu_largo, sigma_largo) + fondo

def picco_doppia_gaussiana_cdf(x, Area, mu, sigma, 
                                  A_largo, mu_largo, sigma_largo, 
                                  A_fondo, tau, c, x0, 
                                  **kwargs):
    arg = np.clip(-(x - x0) / tau, -700, 700)
    fondo_int = -A_fondo * tau * np.exp(arg) + c * x
    return Area * sc.norm.cdf(x, mu, sigma) + A_largo * sc.norm.cdf(x, mu_largo, sigma_largo) + fondo_int


def picco_parabola_pdf(x, a, b, c, mu, sigma, Area):
    parabola = a*(x**2) + b*x + c
    gaussiana = Area * sc.norm.pdf(x, mu, sigma)
    return parabola + gaussiana

def picco_parabola_cdf(x, Area, a, b, c, mu, sigma):
    parabola_int = (a/3.0)*(x**3) + (b/2.0)*(x**2) + c*x
    gaussiana_int = Area * sc.norm.cdf(x, mu, sigma)
    return parabola_int + gaussiana_int


def picco_esponenziale_pdf(x, Area, mu, sigma, A_fondo, tau, c, x0):
    """
    Modello: Gaussiana + Esponenziale decrescente + Costante piatta
    - A_fondo: Altezza dell'esponenziale nel punto x0
    - tau: 'Decadimento' (canali/energia necessari per ridurre il fondo di 1/e)
    - c: Rumore costante di base
    """
    # 1. Componente Gaussiana
    gaussiana = Area * sc.norm.pdf(x, loc=mu, scale=sigma)
    
    # 2. Componente Esponenziale + Costante
    arg = np.clip(-(x - x0) / tau, -700, 700)
    fondo = A_fondo * np.exp(arg) + c
    
    return gaussiana + fondo

def picco_esponenziale_cdf(x, Area, mu, sigma, A_fondo, tau, c, x0):
    """
    Integrale indefinito del modello (CDF per Binned Likelihood)
    """
    # 1. Integrale Gaussiana
    gaussiana_int = Area * sc.norm.cdf(x, loc=mu, scale=sigma)
    
    # 2. Integrale Esponenziale + Costante
    arg = np.clip(-(x - x0) / tau, -700, 700)
    # L'integrale di A*e^(-x/tau) è -A*tau*e^(-x/tau)
    fondo_int = -A_fondo * tau * np.exp(arg) + c * x
    
    return gaussiana_int + fondo_int

def funzione_esponenziale_pdf(x, x0, tau, A_fondo, c, **kwargs):
    arg = np.clip(-(x - x0) / tau, -700, 700)
    fondo = A_fondo * np.exp(arg) + c
    return fondo

def funzione_gaussiana_pdf(x, mu, sigma, Area, **kwargs):
    """Componente gaussiana del modello dei picchi (stessa firma di funzione_picchi_pdf)."""
    return Area * sc.norm.pdf(x, mu, sigma)

def funzione_parabola_pdf(x, a, b, c, mu, sigma, Area, **kwargs):
    """Componente parabola (fondo) del modello dei picchi (stessa firma di funzione_picchi_pdf)."""
    return a*(x**2) + b*x + c


def picco_lineare_pdf(x, Area, mu, sigma, a, b):
    gaussiana = Area * sc.norm.pdf(x, mu, sigma)
    lineare = a *x + b
    return gaussiana + lineare

def picco_lineare_cdf(x, Area, mu, sigma, a, b):
    gaussiana_int = Area * sc.norm.cdf(x, mu, sigma)
    lin_int = (a/2)*(x**2) + b *x
    return gaussiana_int + lin_int

def funzione_linare_pdf(x, a, b, **kwargs):
    lin = x*a + b 
    return lin


def rebin(canali, conteggi, n=4):
    """Raggruppa n canali consecutivi sommando i conteggi.
    Taglia la coda se len non è multiplo di n."""
    n_bins = len(canali) // n
    canali_rb  = canali[:n_bins * n].reshape(n_bins, n).mean(axis=1)
    conteggi_rb = conteggi[:n_bins * n].reshape(n_bins, n).sum(axis=1)
    return canali_rb, conteggi_rb




def plot_histogram_from_dat(filepath: str, title: str = "Istogramma", 
                             xlabel: str = "Canale", ylabel: str = "Conteggi"):
    """
    Per file .dat già istogrammati (un conteggio per canale).
    """
    counts = np.loadtxt(filepath, comments="#")
    channels = np.arange(len(counts))  # 0, 1, 2, ..., 8191

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(channels, counts, width=1, color="steelblue", edgecolor="none")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.show()
    return fig, ax




def plot_histogram_from_dat2(
    filepath: str,
    column: int = 0,
    bins: int = 30,
    title: str = "Istogramma",
    xlabel: str = "Valori",
    ylabel: str = "Frequenza",
    color: str = "steelblue",
    show: bool = True,
    save_path: str = None
):
    """
    Legge un file .dat e genera un istogramma.

    Args:
        filepath   : percorso al file .dat
        column     : indice della colonna da usare (default 0)
        bins       : numero di barre dell'istogramma (default 30)
        title      : titolo del grafico
        xlabel     : etichetta asse x
        ylabel     : etichetta asse y
        color      : colore delle barre
        show       : se True, mostra il grafico a schermo
        save_path  : se specificato, salva il grafico in quel percorso (es. "output.png")

    Returns:
        fig, ax    : oggetti Matplotlib per personalizzazioni ulteriori
    """
    # Lettura del file (salta righe che iniziano con # o che non sono numeriche)
    data = np.loadtxt(filepath, comments="#")
    values = data if data.ndim == 1 else data[:, column]

    # Calcolo bin con la regola di Sturges
    if bins == "sturges":
        bins = int(np.ceil(1 + np.log2(len(values))))

    # Gestione sia di array 1D che 2D
    if data.ndim == 1:
        values = data
    else:
        values = data[:, column]

    # Creazione dell'istogramma
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=bins, color=color, edgecolor="white", linewidth=0.5)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    # Statistiche descrittive nel grafico
    stats_text = (
        f"N = {len(values)}\n"
        f"Media = {values.mean():.3f}\n"
        f"Std = {values.std():.3f}"
    )
    ax.text(
        0.97, 0.95, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.7)
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Grafico salvato in: {save_path}")

    if show:
        plt.show()

    return fig, ax




# Alias per Tstudent come richiedevano alcune chiamate
def test_compatibilita(*args, **kwargs):
    return Tstudent(*args, **kwargs)


def retta_calibrazione(x, p0, p1):
    """ Modello lineare per la calibrazione energetica (E = p0 * CH + p1) """
    return p1 * x + p0


def cinematica_compton(theta_rad, mc2):
    """ 
    Cinematica Compton. 
    L'energia incidente (E_inc) è fissata a 511 keV.
    mc2 è il parametro libero da fittare (dovrebbe venire ~511 keV).
    """
    E_inc = 511.0
    return E_inc / (1 + (E_inc / mc2) * (1 - np.cos(theta_rad)))




def klein_nishina_solo_re(theta_rad, r_e):
    """
    Formula di Klein-Nishina per la sezione d'urto differenziale.
    r_e è il parametro libero da fittare (il raggio classico dell'elettrone).
    """
    E_inc = 511.0 # Energia incidente in keV
    mc2 = 511.0   # Massa elettrone assunta nota per questo calcolo
    
    # Calcolo il rapporto E'/E teorico
    rapporto = 1.0 / (1 + (E_inc / mc2) * (1 - np.cos(theta_rad)))
    
    # Formula teorica
    sezione_urto = (r_e**2 / 2.0) * (rapporto**2) * (1/rapporto + rapporto - np.sin(theta_rad)**2)
    return sezione_urto



def klein_nishina_completa(theta_rad, r_e, E_inc, mc2):
    """
    Variabile indipendente:
    - theta_rad: angolo di scattering in radianti (Array o float)
    
    Parametri liberi:
    - r_e: raggio classico dell'elettrone (parametro da fittare, teorico ~2.818e-15 m)
    - E_inc: energia del fotone incidente (es. 511.0 keV per Na-22)
    - mc2: energia a riposo dell'elettrone (m_e * c^2, teorica ~511.0 keV)
    """
    
    # 1. Calcolo del rapporto E'/E dalla cinematica Compton
    # E_diffuso = E_inc / (1 + (E_inc / mc2) * (1 - cos(theta)))
    rapporto_E = 1.0 / (1.0 + (E_inc / mc2) * (1.0 - np.cos(theta_rad)))
    
    # 2. Calcolo della sezione d'urto differenziale (Formula di Klein-Nishina)
    # d_sigma/d_omega = (r_e^2 / 2) * (E'/E)^2 * [ E'/E + E/E' - sin^2(theta) ]
    termine_parentesi = rapporto_E + (1.0 / rapporto_E) - np.sin(theta_rad)**2
    
    sezione_urto = (r_e**2 / 2.0) * (rapporto_E**2) * termine_parentesi
    
    return sezione_urto



def estrai_spettro(filename='histo.dat', log_scale=True, show_plot=True):
    """
    Legge il file di output dell'MCA.
    Restituisce gli array dei canali (x) e dei conteggi (y).
    Se show_plot=True, mostra anche il grafico a schermo.
    """
    try:
        # Carica i dati (la singola colonna di conteggi)
        conteggi = np.loadtxt(filename)
    except FileNotFoundError:
        print(f"Errore: Il file '{filename}' non è stato trovato nella cartella corrente.")
        return None, None
        
    # Crea l'asse X (i canali da 0 fino alla fine dell'array)
    canali = np.arange(len(conteggi))
    
    # --- Interruttore per il grafico ---
    if show_plot:
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(10, 6))
        
        # 'steps-mid' disegna il grafico "a istogramma", perfetto per i bin dell'MCA
        plt.plot(canali, conteggi, drawstyle='steps-mid', color='royalblue', linewidth=1.5)
        
        plt.xlabel('Canali ADC (CH)', fontsize=12)
        plt.ylabel('Conteggi', fontsize=12)
        plt.title(f'Spettro di acquisizione ({filename})', fontsize=14, pad=15)
        
        # Scala logaritmica (fondamentale per vedere i picchi piccoli)
        if log_scale:
            plt.yscale('log')
            plt.ylim(bottom=0.5)
            
        plt.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray')
        plt.grid(True, which='minor', linestyle=':', linewidth=0.3, color='lightgray')
        plt.minorticks_on()
        
        plt.tight_layout()
        plt.show()
    
    # Restituisce SEMPRE i dati, a prescindere dal fatto che il grafico sia stato disegnato o no
    return canali, conteggi



import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL
from scipy.stats import norm
from scipy.stats import chi2 as chi2_dist

# ==========================================
# MODELLI PER LA LIKELIHOOD BINNATA
# Per usare la BinnedNLL servono sia la CDF (per il fit) che la PDF (per il grafico)
# ==========================================

def GaussianaFondo_CDF(x, Area, mu, sigma, B):
    """ Funzione Cumulativa: Integrale della Gaussiana + Fondo Costante (B) """
    # norm.cdf è l'integrale della gaussiana standard
    # B * x è l'integrale di una costante B
    return Area * norm.cdf(x, loc=mu, scale=sigma) + B * x

def GaussianaFondo_PDF(x, Area, mu, sigma, B):
    """ Funzione di Densità: Gaussiana normale + Fondo Costante (B) """
    return Area * norm.pdf(x, loc=mu, scale=sigma) + B


# ==========================================
# CLASSE DI FIT LIKELIHOOD COMPLETA
# ==========================================

class FitLikelihoodBomberone:
    def __init__(self, canali, conteggi, modello_cdf, modello_pdf, initial_params, 
                 xlabel='Canali ADC', ylabel='Conteggi', title='Fit Likelihood Binnata'):
        
        self.x = np.array(canali, dtype=float)
        self.y = np.array(conteggi, dtype=float)
        
        self.modello_cdf = modello_cdf
        self.modello_pdf = modello_pdf
        self.initial_params = initial_params
        
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

        # Creazione dei Bin Edges (i margini dei canali per l'integrale)
        step = self.x[1] - self.x[0]
        self.bin_edges = np.append(self.x - step/2, self.x[-1] + step/2)

        self.minuit: Any = None
        self.fit_result: Any = None
        self.is_fit_valid: bool = False

        # Variabili Statistiche
        self.chi2_val: Any = None
        self.ndof: Any = None
        self.chi2_reduced: Any = None
        self.p_value: Any = None
        self.expected_counts: Any = None

    def perform_fit(self, silent=False):
        if silent:
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
    
        try:
            cost = ExtendedBinnedNLL(self.y, self.bin_edges, self.modello_cdf)
            self.minuit = Minuit(cost, **self.initial_params)
        
            if 'x0' in self.initial_params: self.minuit.fixed['x0'] = True
            if 'sigma' in self.initial_params: self.minuit.limits['sigma'] = (0.1, None)
            if 'tau' in self.initial_params: self.minuit.limits['tau'] = (1.0, None)
            if 'Area' in self.initial_params: self.minuit.limits['Area'] = (0.0, None)
            if 'A_fondo' in self.initial_params: self.minuit.limits['A_fondo'] = (0.0, None)
        
            self.minuit.migrad()
            migrad_valid = self.minuit.valid
            self.minuit.hesse()
            self.is_fit_valid = migrad_valid
        
            if self.is_fit_valid:
                self.fit_result = {p: (self.minuit.values[p], self.minuit.errors[p]) 
                                for p in self.minuit.parameters}
                self.calculate_fit_statistics()
                if not silent:
                    self.print_results()
            else:
                if not silent:
                    print("ATTENZIONE: Il fit non ha raggiunto un minimo valido.")
    
        finally:
            if silent:
                sys.stdout.close()
                sys.stdout = old_stdout
    
        return self
    
    def calculate_fit_statistics(self):
        # Calcolo dei conteggi attesi (Expected) integrando la CDF tra i margini
        kwargs = {p: self.minuit.values[p] for p in self.minuit.parameters}
        cdf_edges = self.modello_cdf(self.bin_edges, **kwargs)
        self.expected_counts = np.diff(cdf_edges)

        # Calcolo del Chi-Quadro di Pearson
        # Escludiamo i bin dove l'atteso è zero per evitare divisioni impossibili
        mask = self.expected_counts > 0
        O = self.y[mask]      # Osservati
        E = self.expected_counts[mask] # Attesi

        self.chi2_val = np.sum((O - E)**2 / E)
        self.ndof = len(O) - len(self.initial_params)
        self.chi2_reduced = self.chi2_val / self.ndof if self.ndof > 0 else np.inf
        self.p_value = chi2_dist.sf(self.chi2_val, self.ndof)

    def print_results(self):
        print("="*50)
        print(f" RISULTATI FIT LIKELIHOOD: {self.title}")
        print("="*50)
        for p, (v, err) in self.fit_result.items():
            print(f"{p:>10} = {v:10.4g} ± {err:.4g}")
        print("-" * 50)
        print(f"Chi2 / ndof = {self.chi2_val:.2f} / {self.ndof} = {self.chi2_reduced:.2f}")
        print(f"p-value     = {self.p_value:.4g}")
        print("="*50)

    def _get_info_box_coords(self, position='upper right', pad=0.05):
        positions = {
            'upper left':   {'xy': (pad, 1 - pad), 'ha': 'left', 'va': 'top'},
            'upper right':  {'xy': (1 - pad, 1 - pad), 'ha': 'right', 'va': 'top'},
            'lower left':   {'xy': (pad, pad), 'ha': 'left', 'va': 'bottom'},
            'lower right':  {'xy': (1 - pad, pad), 'ha': 'right', 'va': 'bottom'},
            'center':       {'xy': (0.5, 0.5), 'ha': 'center', 'va': 'center'},
        }
        if isinstance(position, (tuple, list)) and len(position) == 2:
             return {'xy': tuple(position), 'ha': 'center', 'va': 'center'}
        return positions.get(position.lower().replace("_", " "), positions['upper right'])

    def plot_results(self, show_hist=True, info_box_pos='upper right', componenti=None):
        """
        Plotta i risultati del fit.

        Args:
            show_hist: se True, mostra anche i dati con barre d'errore.
            info_box_pos: posizione del box riassuntivo ('upper right', 'upper left', ecc.).
            componenti: lista opzionale di tuple (label, funzione) per plottare
                        le singole componenti del modello separatamente.
                        Ogni funzione deve avere la stessa firma di modello_pdf.
                        Esempio: [('Gaussiana', gauss_pdf), ('Fondo', fondo_pdf)]
        """
        if not self.is_fit_valid:
            print("ATTENZIONE: Esegui prima un fit valido!")
            return

        # Palette professionale: dati grigi, fit rosso scuro, componenti distinte
        _COLORI_COMP = [
            '#2ca02c',   # verde scuro
            '#ff7f0e',   # arancio
            '#9467bd',   # viola
            '#8c564b',   # marrone
            '#17becf',   # azzurro
            '#bcbd22',   # giallo-verde
        ]

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 7),
            gridspec_kw={'height_ratios': [4, 1]},
            sharex=True
        )
        fig.subplots_adjust(hspace=0.05)

        # ── dati ──────────────────────────────────────────────────────────────
        if show_hist:
            errori_y = np.sqrt(self.y)
            errori_y[self.y == 0] = 1
            ax1.errorbar(
                self.x, self.y, yerr=errori_y,
                fmt='o', color='#333333', markersize=3,
                linewidth=0.8, capsize=2, capthick=0.8,
                label='Dati MCA', zorder=3
            )

        # ── curva di best fit ─────────────────────────────────────────────────
        x_fit = np.linspace(self.x[0], self.x[-1], 2000)
        kwargs = {p: self.minuit.values[p] for p in self.minuit.parameters}
        bin_width = self.x[1] - self.x[0]
        y_fit = self.modello_pdf(x_fit, **kwargs) * bin_width

        ax1.plot(
            x_fit, y_fit,
            color='#c0392b', linewidth=2.2,
            linestyle='--', dashes=(6, 2),
            label='Fit Extended Likelihood', zorder=4
        )

        # ── componenti individuali ────────────────────────────────────────────
        if componenti is not None:
            for i, (label_comp, func_comp) in enumerate(componenti):
                colore = _COLORI_COMP[i % len(_COLORI_COMP)]
                y_comp = func_comp(x_fit, **kwargs) * bin_width
                ax1.plot(
                    x_fit, y_comp,
                    color=colore, linewidth=1.8,
                    linestyle='--', dashes=(4, 3),
                    label=label_comp, zorder=2, alpha=0.85
                )

        ax1.set_ylabel(self.ylabel, fontsize=12)
        ax1.set_title(self.title, fontsize=13, pad=8)
        ax1.legend(framealpha=0.92, fontsize=9, loc='upper left')
        ax1.grid(True, linestyle=':', linewidth=0.6, alpha=0.7)
        ax1.tick_params(labelbottom=False)

        # ── box statistiche ───────────────────────────────────────────────────
        chi2_line = (f"$\\chi^2 / n_{{\\mathrm{{dof}}}}$"
                     f" = {self.chi2_val:.1f} / {self.ndof}")
        pval_line  = f"p-value = {self.p_value:.3f}"
        par_lines  = "\n".join(
            f"{p} = {v:.4g} $\\pm$ {e:.3g}"
            for p, (v, e) in self.fit_result.items()
        )
        res_text = f"{chi2_line}\n{pval_line}\n\n{par_lines}"

        box_coords = self._get_info_box_coords(info_box_pos)
        ax1.annotate(
            res_text,
            xy=box_coords['xy'], xycoords='axes fraction',
            va=box_coords['va'], ha=box_coords['ha'],
            fontsize=9,
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='white', alpha=0.93,
                edgecolor='#aaaaaa', linewidth=0.8
            )
        )

        # ── pulls ─────────────────────────────────────────────────────────────
        attesi_safe = np.where(self.expected_counts > 0, self.expected_counts, 1.0)
        pulls = (self.y - self.expected_counts) / np.sqrt(attesi_safe)

        # Banda ±1σ e ±2σ di riferimento
        ax2.axhspan(-1, 1, color='#f0c27f', alpha=0.35, zorder=0)
        ax2.axhspan(-2, 2, color='#d4e8b0', alpha=0.25, zorder=0)
        ax2.axhline(0, color='#c0392b', linestyle='--', linewidth=1.0, zorder=2)

        ax2.plot(
            self.x, pulls,
            'o', color='#2980b9', markersize=3,
            linewidth=0, zorder=3
        )

        ax2.set_xlabel(self.xlabel, fontsize=12)
        ax2.set_ylabel('Pulls', fontsize=11)
        ax2.set_ylim(-4, 4)
        ax2.set_yticks([-3, -2, -1, 0, 1, 2, 3])
        ax2.grid(True, linestyle=':', linewidth=0.6, alpha=0.7)

        plt.tight_layout()
        plt.show()






def Tstudent(val1, val2, sigma1, sigma2, val1_name="Valore 1", val2_name="Valore 2", use_ttest=False, custom_df=None, significance_level=0.05):
    """
    Esegue un test di compatibilità tra due valori con le loro incertezze.
    """
    # Inizializzazione di default
    T_score = np.nan
    p_value = np.nan
    compatibili = False # Assumiamo non compatibili fino a prova contraria
    dist_type = "N/A"
    df_eff = np.nan # Gradi di libertà effettivi usati

    if sigma1 < 0 or sigma2 < 0:
        print(f"Errore nel test tra {val1_name} e {val2_name}: Le incertezze (sigma) non possono essere negative.")
        # T_score, p_value, compatibili rimangono ai loro valori di default (nan, nan, False)
        return T_score, p_value, compatibili
        
    denominatore_T = np.sqrt(sigma1**2 + sigma2**2)

    if denominatore_T < 1e-15: # Denominatore quasi zero
        if np.abs(val1 - val2) < 1e-15: # Anche differenza quasi zero
            T_score = 0.0
            p_value = 1.0
            compatibili = True
            dist_type = "N/A (Valori identici con sigma denominatore zero)"
        else: # Differenza non zero
            T_score = np.inf
            p_value = 0.0
            # compatibili rimane False
            dist_type = "N/A (Valori diversi con sigma denominatore zero)"
    else: # Denominatore_T è valido
        T_score = np.abs(val1 - val2) / denominatore_T

        if use_ttest:
            if custom_df is not None:
                df_eff = custom_df
                if df_eff <= 0:
                    print(f"Errore nel test tra {val1_name} e {val2_name}: custom_df per t-test ({df_eff}) deve essere > 0.")
                    # T_score è calcolato, p_value e compatibili rimangono ai default
                    return T_score, p_value, compatibili 
            else: # Calcola df approssimato
                s1_sq = sigma1**2
                s2_sq = sigma2**2
                if s1_sq < 1e-15 and s2_sq < 1e-15:
                    df_eff = np.inf # Se entrambe le sigma sono ~0, la t si approssima alla normale
                else:
                    df_num = (s1_sq + s2_sq)**2
                    df_den = s1_sq**2 + s2_sq**2 
                    if df_den < 1e-15:
                        print(f"Attenzione nel test tra {val1_name} e {val2_name}: Denominatore per df è zero. Si userà df=infinito.")
                        df_eff = np.inf
                    else:
                        df_eff = df_num / df_den
            
            # Calcolo p_value per t-test
            if np.isinf(df_eff):
                p_value = 2 * (1 - norm.cdf(T_score))
                dist_type = f"t di Student (df=infinito, equivale a Normale)"
            elif df_eff < 1:
                print(f"Attenzione nel test tra {val1_name} e {val2_name}: df calcolato ({df_eff:.2f}) < 1. Il p-value del t-test potrebbe non essere affidabile.")
                p_value = np.nan # Non calcoliamo p_value per df < 1 in questo esempio
                dist_type = f"t di Student (df={df_eff:.2f} - problematico)"
            else:
                p_value = 2 * (1 - t.cdf(T_score, df_eff))
                dist_type = f"t di Student (df={df_eff:.2f})"
        
        else: # Usa la distribuzione Normale standard (use_ttest = False)
            df_eff = np.inf # Gradi di libertà per la normale
            p_value = 2 * (1 - norm.cdf(T_score))
            dist_type = "Normale Standard"

        # Determina compatibilità basata sul p_value calcolato (se non è NaN)
        if not np.isnan(p_value):
            compatibili = p_value > significance_level
        # Se p_value è NaN, compatibili rimane False (dall'inizializzazione)

    # Stampa finale dei risultati
    print(f"\nTest di compatibilità tra {val1_name} ({val1:.3e} ± {sigma1:.2e}) e {val2_name} ({val2:.3e} ± {sigma2:.2e}):")
    if not np.isnan(T_score):
        print(f"  Differenza: {np.abs(val1 - val2):.2e}")
        print(f"  Incertezza sulla differenza (denominatore di T): {denominatore_T:.2e}" if denominatore_T >= 1e-15 else "N/A (denominatore T zero)")
        print(f"  Statistica del test (T o Z): {T_score:.2f}")
    else: # T_score potrebbe essere NaN se si esce prima
         print("  Statistica del test non calcolata a causa di errore precedente.")

    print(f"  Distribuzione usata: {dist_type}")
    # print(f"  Gradi di libertà effettivi (df): {df_eff if not np.isnan(df_eff) else 'N/A'}") # Opzionale
    
    if not np.isnan(p_value):
        print(f"  P-value (due code): {p_value:.4f}")
        if compatibili:
            print(f"  I due valori sono COMPATIBILI (p > {significance_level}).")
        else:
            print(f"  I due valori NON sono compatibili (p <= {significance_level}).")
    else:
        print(f"  P-value non calcolato (probabilmente a causa di df < 1 o altri errori).")
        print(f"  Impossibile determinare la compatibilità.")
        
    return T_score, p_value, compatibili

# --- ESEMPIO DI UTILIZZO ---
'''if __name__ == '__main__':
    print("--- Esempio 1: Valori compatibili (Normale) ---")
    test_compatibilita(10.0, 10.8, 0.5, 0.4, "Val A1", "Val B1")

    print("\n--- Esempio 2: Valori non compatibili (Normale) ---")
    test_compatibilita(10.0, 10.8, 0.1, 0.1, "Val A2", "Val B2")

    print("\n--- Esempio 3: Valori compatibili (t-test, df approssimato) ---")
    test_compatibilita(10.0, 10.8, 0.5, 0.4, "Val A1", "Val B1", use_ttest=True)

    print("\n--- Esempio 4: Valori compatibili (t-test, df custom alto) ---")
    test_compatibilita(10.0, 10.8, 0.5, 0.4, "Val A1", "Val B1", use_ttest=True, custom_df=100)

    print("\n--- Esempio 5: Valori identici, sigma zero ---")
    test_compatibilita(10.0, 10.0, 0.0, 0.0, "Val C1", "Val C2")

    print("\n--- Esempio 6: Valori diversi, sigma zero ---")
    test_compatibilita(10.0, 10.1, 0.0, 0.0, "Val D1", "Val D2")
    
    print("\n--- Esempio 7: Una sigma zero (t-test forzato) ---")
    test_compatibilita(10.0, 10.1, 0.5, 0.0, "Val E1", "Val E2", use_ttest=True) # df sarà inf

    print("\n--- Esempio 8: custom_df non valido ---")
    test_compatibilita(10.0, 10.8, 0.5, 0.4, "Val A1", "Val B1", use_ttest=True, custom_df=0)

    print("\n--- Esempio 9: df calcolato < 1 (molto diverse sigma) ---")
    # df_num = (0.01^2 + 1^2)^2 ~ 1
    # df_den = (0.01^4 + 1^4) ~ 1
    # df ~ 1
    # Proviamo sigma molto diverse per vedere se df_num/df_den diventa < 1
    # (sigma1^2+sigma2^2)^2 / (sigma1^4+sigma2^4)
    # Se sigma1 -> 0, df -> sigma2^4 / sigma2^4 = 1
    # Questa formula per df non scenderà facilmente sotto 1 se almeno una sigma è non nulla.
    # Ma se forzassimo df<1 con custom_df:
    test_compatibilita(10.0, 10.1, 0.5, 0.4, "Val G1", "Val G2", use_ttest=True, custom_df=0.5)'''


""" Funzione unica per fare i fit 
ESEMPIO DI UTILIZZO
    def exp_model(x, A, B):
        return A * np.exp(B * x)

    x_exp = np.linspace(1, 5, 15)
    y_exp_true = exp_model(x_exp, 100, -0.8)
    y_exp_noise = np.random.normal(0, y_exp_true * 0.2, len(x_exp)) # Errore proporzionale
    y_exp_data = y_exp_true + y_exp_noise
    sigma_y_exp = np.abs(y_exp_true * 0.2) # Stima dell'errore

    # Rendi positivi i dati y per scala log
    y_exp_data[y_exp_data <= 0] = 1e-3
    sigma_y_exp[sigma_y_exp <= 0] = 1e-3

    data_exp = {'x': x_exp, 'y': y_exp_data, 'sigma_y': sigma_y_exp}
    init_exp = {'A': 90, 'B': -0.7}

    fit_log = FitBomberone(exp_model, data_exp, init_exp, fit_method='Scipy',
                        title="Fit Esponenziale con Scala Log", ylabel = 'Dio stronzo', xlabel = 'Dio merda')
    fit_log.perform_fit().print_results()
    fit_log.plot_results(log_scale_y=True, log_scale_x=True, info_box_pos='upper right')"""






class FitBomberone:
    """
    Classe unificata per eseguire fit di dati con diversi metodi.

    Permette di scegliere tra:
    - 'LeastSquares': Minimi quadrati usando iminuit.cost.LeastSquares (richiede iminuit).
    - 'Chi2': Minimizzazione del Chi-quadro usando Minuit (richiede iminuit).
    - 'Scipy': Minimi quadrati usando scipy.optimize.curve_fit (richiede scipy).
    - 'ODR': Orthogonal Distance Regression (richiede scipy). Gestisce errori su x e y.

    Offre opzioni per personalizzare il plot dei risultati, inclusa la posizione
    del box informativo e l'uso della scala logaritmica sull'asse y.
    """
    def __init__(self, model_func, data_arrays, initial_params,
                 fit_method='LeastSquares', xlabel="x", ylabel="y", title="Risultati del fit"):
        """

        Args:
            model_func (callable): La funzione modello da fittare.
                Deve accettare come primo argomento l'array delle x e poi i parametri
                del fit come argomenti nominali (keyword arguments), es: def my_model(x, a, b, c): ...
            data_arrays (dict): Dizionario contenente i dati. Deve avere le chiavi:
                'x': array dei valori x.
                'y': array dei valori y.
                'sigma_y': array degli errori su y (deviazioni standard).
                Può opzionalmente contenere:
                'sigma_x': array degli errori su x (deviazioni standard). Necessario solo per il metodo 'ODR'.
                           Se non fornito e si usa 'ODR', verrà assunto come zero.
            initial_params (dict): Dizionario con i valori iniziali per i parametri del fit.
                                   Le chiavi devono corrispondere ai nomi dei parametri in model_func.
            fit_method (str, optional): Il metodo di fit da utilizzare.
                                        Opzioni: 'LeastSquares', 'Chi2', 'Scipy', 'ODR'.
                                        Default: 'LeastSquares'.
            xlabel (str, optional): Etichetta per l'asse x del grafico. Default: "x".
            ylabel (str, optional): Etichetta per l'asse y del grafico. Default: "y".
            title (str, optional): Titolo del grafico. Default: "Risultati del fit".
        """
        # --- Validazione Input Iniziale ---
        valid_methods = ['LeastSquares', 'Chi2', 'Scipy', 'ODR']
        if fit_method not in valid_methods:
            raise ValueError(f"Metodo di fit '{fit_method}' non valido. Scegliere tra: {valid_methods}")

        if not callable(model_func):
             raise TypeError("model_func deve essere una funzione eseguibile.")
        if not isinstance(data_arrays, dict):
             raise TypeError("data_arrays deve essere un dizionario.")
        if not isinstance(initial_params, dict):
             raise TypeError("initial_params deve essere un dizionario.")

        # --- Memorizzazione Attributi ---
        self.model = model_func
        self.fit_method = fit_method
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

        # --- Estrazione Dati ---
        self.x = np.asarray(data_arrays['x'])
        self.y = np.asarray(data_arrays['y'])
        self.sigma_y = np.asarray(data_arrays.get('sigma_y', np.ones_like(self.y)))

        # Gestione sigma_x (rilevante solo per ODR)
        if 'sigma_x' in data_arrays:
             self.sigma_x = np.asarray(data_arrays['sigma_x'])
        elif fit_method == 'ODR':
             print("Attenzione: sigma_x non fornito per il metodo ODR. Verrà assunto nullo.")
             self.sigma_x = np.zeros_like(self.x)
        else:
             self.sigma_x = None # Non necessario per altri metodi

        # --- Estrazione Parametri ---
        sig = inspect.signature(model_func)
        all_param_names = list(sig.parameters.keys())
        if not all_param_names:
             raise ValueError("La funzione modello non sembra accettare argomenti.")
        # Il primo argomento è assunto essere la variabile indipendente (x)
        self.param_names = all_param_names[1:]
        if not self.param_names:
             raise ValueError("La funzione modello deve accettare almeno un parametro oltre alla variabile indipendente.")

        self.initial_params = initial_params

        # --- Inizializzazione Risultati ---
        self.m: Any = None            # Oggetto Minuit
        self.fit_result: Any = None   # Dizionario con (valore, errore) dei parametri
        self.chi2_val: Any = None     # Chi quadro
        self.dof: Any = None          # Gradi di libertà
        self.p_value: Any = None      # p-value
        self.is_fit_valid: bool = False # Flag di validità del fit
        self.scipy_popt: Any = None   # Parametri ottimali per curve_fit
        self.scipy_pcov: Any = None   # Matrice di covarianza per curve_fit
        self.odr_output: Any = None   # Output completo di scipy.odr

        # --- Validazione Input Dettagliata ---
        self._validate_inputs()

    def _validate_inputs(self):
        """Controlla la consistenza degli array di input e dei parametri."""
        if not (len(self.x) == len(self.y) == len(self.sigma_y)):
            raise ValueError("Gli array x, y e sigma_y devono avere la stessa lunghezza.")
        if self.sigma_x is not None and len(self.x) != len(self.sigma_x):
             raise ValueError("Gli array x e sigma_x devono avere la stessa lunghezza.")

        # Verifica che sigma_y non contenga zeri o valori non positivi se usati come errori
        if np.any(self.sigma_y <= 0):
            print("Attenzione: sigma_y contiene valori nulli o negativi. Questo può causare problemi nel fit.")
            # Potresti voler sollevare un errore o sostituire questi valori a seconda del contesto
            # raise ValueError("sigma_y non può contenere valori nulli o negativi.")

        # Verifica parametri iniziali
        missing_params = set(self.param_names) - set(self.initial_params.keys())
        extra_params = set(self.initial_params.keys()) - set(self.param_names)
        if missing_params:
            raise ValueError(f"Parametri iniziali mancanti per: {missing_params}")
        if extra_params:
            raise ValueError(f"Parametri iniziali non richiesti dalla funzione modello: {extra_params}")

    # --- Funzioni Ausiliarie per Fit Specifici ---

    def _odr_model_wrapper(self, B, x):
        """Wrapper per la funzione modello richiesta da scipy.odr."""
        params_dict = {name: value for name, value in zip(self.param_names, B)}
        return self.model(x, **params_dict)

    def _chi2_func_wrapper(self, *args):
        """Wrapper per la funzione Chi-quadro richiesta da Minuit."""
        params_dict = {name: val for name, val in zip(self.param_names, args)}
        y_model = self.model(self.x, **params_dict)
        # Evita divisione per zero se sigma_y è zero (anche se _validate_inputs avverte)
        # Si potrebbe usare np.where o aggiungere un piccolo epsilon, ma qui usiamo mask
        valid_sigma = self.sigma_y > 0
        residuals = np.zeros_like(self.y)
        residuals[valid_sigma] = (self.y[valid_sigma] - y_model[valid_sigma]) / self.sigma_y[valid_sigma]
        return np.sum(residuals**2)

    # --- Metodi di Fit ---

    def _fit_odr(self):
        """Esegue il fit usando Orthogonal Distance Regression (ODR)."""
        if self.sigma_x is None:
             # Questo non dovrebbe accadere grazie al check in __init__, ma per sicurezza
             print("Avviso: ODR richiede sigma_x. Assumendo sigma_x = 0.")
             self.sigma_x = np.zeros_like(self.x)

        beta0 = [self.initial_params[name] for name in self.param_names]
        odr_model = Model(self._odr_model_wrapper)
        data = RealData(self.x, self.y, sx=self.sigma_x, sy=self.sigma_y)
        odr = ODR(data, odr_model, beta0=beta0)
        self.odr_output = odr.run()

        if self.odr_output.info > 0: # Controlla se il fit è terminato con successo
            self.fit_result = {name: (val, err) for name, val, err in zip(self.param_names, self.odr_output.beta, self.odr_output.sd_beta)}
            self.chi2_val = self.odr_output.sum_square # ODR chiama chi2 'sum_square'
            self.dof = len(self.x) - len(self.param_names)
            self.p_value = 1 - chi2.cdf(self.chi2_val, self.dof) if self.dof > 0 else 0
            self.is_fit_valid = True
        else:
             print(f"Errore ODR: Il fit non è converguto (codice info: {self.odr_output.info}).")
             # Potresti voler popolare self.fit_result con NaN o valori iniziali
             self.fit_result = {name: (self.initial_params[name], np.nan) for name in self.param_names}
             self.is_fit_valid = False


    def _fit_least_squares(self):
        """Esegue il fit usando iminuit.cost.LeastSquares."""
        try:
             # LeastSquares accetta direttamente la funzione modello con keyword args
             least_squares_cost = LeastSquares(self.x, self.y, self.sigma_y, self.model)
             self.m = Minuit(least_squares_cost, **self.initial_params)
             self.m.migrad()  # Esegui minimizzazione
             self.m.hesse()   # Calcola errori accurati (matrice di Hesse)

             self.is_fit_valid = self.m.valid
             if not self.is_fit_valid:
                  print("Attenzione: La covarianza del fit (LeastSquares) non è valida. Gli errori potrebbero essere inaffidabili.")

             self.fit_result = {name: (self.m.values[name], self.m.errors[name]) for name in self.param_names}
             self.chi2_val = self.m.fval # Minuit chiama chi2 'fval'
             self.dof = self.m.ndof
             self.p_value = self.m.fval / self.dof if self.dof > 0 else 0 # Chi2 p-value
             self.p_value = chi2.sf(self.m.fval, self.m.ndof) # Modo corretto con scipy.stats.chi2.sf (survival function)

        except Exception as e:
             print(f"Errore durante il fit LeastSquares con Minuit: {e}")
             self.is_fit_valid = False
             self.fit_result = {name: (self.initial_params[name], np.nan) for name in self.param_names}


    def _fit_chi2(self):
        """Esegue il fit minimizzando il Chi-quadro con Minuit."""
        try:
            # Minuit richiede che la funzione da minimizzare accetti parametri posizionali
            initial_values_list = [self.initial_params[name] for name in self.param_names]
            # Assegna nomi ai parametri per Minuit per migliore output
            self.m = Minuit(self. _chi2_func_wrapper, *initial_values_list, name=self.param_names)
            self.m.errordef = Minuit.LEAST_SQUARES # Imposta errordef a 1 per Chi2/Least Squares
            self.m.migrad()
            self.m.hesse()

            self.is_fit_valid = self.m.valid
            if not self.is_fit_valid:
                 print("Attenzione: La covarianza del fit (Chi2) non è valida. Gli errori potrebbero essere inaffidabili.")

            # Recupera i risultati usando i nomi dei parametri
            self.fit_result = {name: (self.m.values[name], self.m.errors[name]) for name in self.param_names}
            self.chi2_val = self.m.fval
            self.dof = len(self.x) - len(self.param_names) # m.ndof potrebbe non essere corretto qui
            # self.p_value = self.m.fcn(self.m.values) / self.dof if self.dof > 0 else 0
            self.p_value = chi2.sf(self.m.fval, self.dof) if self.dof > 0 else 0

        except Exception as e:
            print(f"Errore durante il fit Chi2 con Minuit: {e}")
            self.is_fit_valid = False
            self.fit_result = {name: (self.initial_params[name], np.nan) for name in self.param_names}

    def _fit_scipy(self):
        """Esegue il fit usando scipy.optimize.curve_fit."""
        try:
            # curve_fit preferisce parametri posizionali, ma può gestire keyword se la firma corrisponde
            # Per sicurezza e coerenza, potremmo creare un wrapper, ma proviamo direttamente
            # La funzione modello DEVE avere la forma func(x, p1, p2, ...) per curve_fit

            # Assicurati che l'ordine dei parametri iniziali corrisponda a self.param_names
            initial_params_list = [self.initial_params[name] for name in self.param_names]

            self.scipy_popt, self.scipy_pcov = curve_fit(
                self.model,
                self.x,
                self.y,
                p0=initial_params_list,
                sigma=self.sigma_y,
                absolute_sigma=True # Tratta sigma come deviazioni standard assolute
            )

            errors = np.sqrt(np.diag(self.scipy_pcov))
            self.fit_result = {name: (val, err) for name, val, err in zip(self.param_names, self.scipy_popt, errors)}

            # Calcola Chi2 manualmente
            residuals = self.y - self.model(self.x, *self.scipy_popt)
            # Evita divisione per zero se sigma_y è zero
            valid_sigma = self.sigma_y > 0
            chisq_terms = np.zeros_like(self.y)
            chisq_terms[valid_sigma] = (residuals[valid_sigma] / self.sigma_y[valid_sigma])**2
            self.chi2_val = np.sum(chisq_terms)

            self.dof = len(self.x) - len(self.scipy_popt)
            self.p_value = chi2.sf(self.chi2_val, self.dof) if self.dof > 0 else 0
            self.is_fit_valid = True # curve_fit non ha un flag di validità diretto come Minuit, ma se non lancia eccezioni...

        except RuntimeError as e:
             print(f"Errore durante il fit con Scipy (curve_fit): {e}. Il fit potrebbe non essere converguto.")
             self.is_fit_valid = False
             self.scipy_popt = [self.initial_params[name] for name in self.param_names]
             self.scipy_pcov = np.full((len(self.param_names), len(self.param_names)), np.nan)
             self.fit_result = {name: (self.initial_params[name], np.nan) for name in self.param_names}
             self.chi2_val = np.nan
             self.dof = len(self.x) - len(self.param_names)
             self.p_value = np.nan
        except Exception as e:
             print(f"Errore imprevisto durante il fit con Scipy: {e}")
             self.is_fit_valid = False
             self.fit_result = {name: (self.initial_params[name], np.nan) for name in self.param_names}
             # Inizializza gli altri attributi a NaN o valori predefiniti
             self.scipy_popt, self.scipy_pcov, self.chi2_val, self.dof, self.p_value = None, None, None, None, None


    # --- Metodo Principale per Eseguire il Fit ---

    def perform_fit(self):
        """Esegue il fit utilizzando il metodo specificato in __init__."""
        print(f"--- Esecuzione Fit con Metodo: {self.fit_method} ---")
        if self.fit_method == 'ODR':
            self._fit_odr()
        elif self.fit_method == 'LeastSquares':
            self._fit_least_squares()
        elif self.fit_method == 'Chi2':
            self._fit_chi2()
        elif self.fit_method == 'Scipy':
            self._fit_scipy()
        else:
            # Questo non dovrebbe accadere grazie al check in __init__
            raise ValueError(f"Metodo di fit '{self.fit_method}' non riconosciuto.")

        if self.fit_result is None:
             print("Errore: Il fit non ha prodotto risultati.")
             self.is_fit_valid = False
        elif not self.is_fit_valid:
             print("Avviso: Il fit potrebbe non essere valido o non essere converguto.")
        else:
            print("Fit completato.")

        # Restituisce self per permettere il chaining, es. fit.perform_fit().print_results()
        return self

    # --- Metodi per Visualizzare i Risultati ---

    def print_results(self):
        """Stampa i risultati del fit (parametri, chi2, p-value)."""
        if self.fit_result is None:
            print("Errore: Nessun risultato del fit disponibile. Eseguire prima perform_fit().")
            return

        print(f"\n--- Risultati del Fit ({self.fit_method}) ---")
        print(f"Fit Valido: {'Sì' if self.is_fit_valid else 'No'}")

        print("\nParametri Ottimizzati:")
        for name in self.param_names:
            val, err = self.fit_result.get(name, (np.nan, np.nan)) # Gestisce il caso di fit fallito
            print(f"  {name} = {val:.4g} ± {err:.2g}") # Formato più leggibile

        if self.chi2_val is not None and self.dof is not None and self.dof > 0:
            chi2_rid = self.chi2_val / self.dof
            print(f"\nStatistiche del Fit:")
            print(f"  Chi-quadro (χ²): {self.chi2_val:.4f}")
            print(f"  Gradi di libertà (DoF): {self.dof}")
            print(f"  Chi-quadro Ridotto (χ²/DoF): {chi2_rid:.4f}")
            if self.p_value is not None:
                print(f"  p-value: {self.p_value:.4f}")
        elif self.dof == 0:
             print("\nStatistiche del Fit:")
             print(f"  Chi-quadro (χ²): {self.chi2_val:.4f}")
             print(f"  Gradi di libertà (DoF): {self.dof}")
             print("  Attenzione: Con 0 gradi di libertà, Chi2 ridotto e p-value non sono definiti.")
        else:
            print("\nStatistiche del Fit non disponibili (Chi2/DoF potrebbero non essere stati calcolati).")

        # Stampa informazioni aggiuntive da Minuit se disponibili
        if self.m and (self.fit_method == 'LeastSquares' or self.fit_method == 'Chi2'):
             # print(f"\nMinuit Fit Status: {self.m.fmin}") # fmin contiene info dettagliate
             if hasattr(self.m, 'covariance') and self.m.covariance is not None:
                  print("\nMatrice di Covarianza (Minuit):")
                  # Stampa la matrice in modo leggibile
                  # for row in self.m.covariance:
                  #      print("  [" + ", ".join(f"{x: .2e}" for x in row) + "]")
                  pass # Potrebbe essere troppo verboso, lo lascio commentato
             else:
                  print("\nMatrice di Covarianza (Minuit): non disponibile o non valida.")
        # Stampa matrice covarianza da Scipy
        elif self.scipy_pcov is not None and self.fit_method == 'Scipy':
             print("\nMatrice di Covarianza (Scipy):")
             # for row in self.scipy_pcov:
             #      print("  [" + ", ".join(f"{x: .2e}" for x in row) + "]")
             pass # Anche qui, potenzialmente verboso


    def _get_info_box_coords(self, position='upper left', pad=0.05):
        """Restituisce coordinate e allineamento per plt.annotate."""
        positions = {
            'upper left':   {'xy': (pad, 1 - pad), 'ha': 'left', 'va': 'top'},
            'upper right':  {'xy': (1 - pad, 1 - pad), 'ha': 'right', 'va': 'top'},
            'lower left':   {'xy': (pad, pad), 'ha': 'left', 'va': 'bottom'},
            'lower right':  {'xy': (1 - pad, pad), 'ha': 'right', 'va': 'bottom'},
            'upper center': {'xy': (0.5, 1 - pad), 'ha': 'center', 'va': 'top'},
            'lower center': {'xy': (0.5, pad), 'ha': 'center', 'va': 'bottom'},
            'center left':  {'xy': (pad, 0.5), 'ha': 'left', 'va': 'center'},
            'center right': {'xy': (1 - pad, 0.5), 'ha': 'right', 'va': 'center'},
            'center':       {'xy': (0.5, 0.5), 'ha': 'center', 'va': 'center'},
        }
        # Se viene passata una tupla (x, y), usala direttamente
        if isinstance(position, (tuple, list)) and len(position) == 2:
             return {'xy': tuple(position), 'ha': 'center', 'va': 'center'} # Default alignment for custom coords

        return positions.get(position.lower().replace("_", " "), positions['upper left']) # Default a upper left

    def plot_results(self, title_fontsize=14, label_fontsize=12,
                     info_box_pos='upper right', log_scale_y=False, log_scale_x=False):
        """
        Genera un grafico dei dati fittati con la curva di fit e un box informativo.

        Args:
            title_fontsize (int, optional): Dimensione del font per il titolo. Default: 14.
            label_fontsize (int, optional): Dimensione del font per le etichette degli assi. Default: 12.
            info_box_pos (str or tuple, optional): Posizione del box informativo.
                Può essere una stringa come 'upper left', 'center right', etc.,
                o una tupla (x, y) in coordinate relative agli assi (0-1).
                Default: 'upper right'.
            log_scale_y (bool, optional): Se True, imposta la scala logaritmica per l'asse y.
                                         Default: False.
        """
        if self.fit_result is None:
            print("Errore: Nessun risultato del fit disponibile per il plot. Eseguire prima perform_fit().")
            return
        if not self.is_fit_valid:
             print("Attenzione: Si sta plottando un fit non valido o non converguto.")

        plt.figure(figsize=(10, 7)) # Leggermente più alta per dare spazio
        ax = plt.gca() # Get current axes

        # --- Plot Dati ---
        plot_kwargs = {'fmt': 'o', 'label': 'Dati', 'markersize': 6, 'capsize': 4, 'elinewidth': 1.5}
        # Includi errori x se disponibili e significativi (o se ODR)
        if self.sigma_x is not None and np.any(self.sigma_x > 1e-9):
             ax.errorbar(self.x, self.y, xerr=self.sigma_x, yerr=self.sigma_y, **plot_kwargs)
             print("Plotting con errori su X e Y.")
        else:
             ax.errorbar(self.x, self.y, yerr=self.sigma_y, **plot_kwargs)

        # --- Plot Curva di Fit ---
        # Genera più punti per una curva liscia
        if len(self.x) > 1:
             x_min, x_max = np.min(self.x), np.max(self.x)
             # Estendi leggermente il range per non tagliare la curva ai bordi
             range_ext = (x_max - x_min) * 0.02
             x_fit = np.linspace(x_min - range_ext, x_max + range_ext, 500)
        else:
             # Gestisce caso con un solo punto dati
             x_fit = np.array([self.x[0] - 1, self.x[0], self.x[0] + 1]) # Un piccolo range intorno

        # Prendi i valori fittati (ignora gli errori qui)
        fitted_params_dict = {name: val_err[0] for name, val_err in self.fit_result.items()}
        try:
             y_fit = self.model(x_fit, **fitted_params_dict)
             ax.plot(x_fit, y_fit, '-r', label=f'Fit ({self.fit_method})', linewidth=2)
        except Exception as e:
             print(f"Errore nel calcolare la curva di fit per il plot: {e}")
             print("La curva di fit potrebbe non essere visualizzata.")


        # --- Impostazioni Grafico ---
        ax.set_xlabel(self.xlabel, fontsize=label_fontsize)
        ax.set_ylabel(self.ylabel, fontsize=label_fontsize)
        ax.set_title(self.title, fontsize=title_fontsize, pad=15) # Aumenta pad per spazio

        if log_scale_y:
            ax.set_yscale('log')
            # Aggiusta limiti y se necessario in scala log
            # Potrebbe essere necessario gestire y <= 0 nei dati
            min_y = np.min(self.y[self.y > 0]) if np.any(self.y > 0) else 1e-9
            # ax.set_ylim(bottom=min_y * 0.5) # Esempio di aggiustamento

        if log_scale_x:
            ax.set_xscale('log')
            min_x = np.min(self.x[self.x > 0]) if np.any(self.x > 0) else 1e-9

        ax.grid(True, which='major', linestyle='-', linewidth='0.5', color='gray', alpha=0.7)
        ax.grid(True, which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.4)
        ax.minorticks_on() # Abilita minor ticks

        # --- Box Informativo ---
        box_text_lines = []
        for name in self.param_names:
             val, err = self.fit_result.get(name, (np.nan, np.nan))
             # Usa notazione scientifica se necessario, altrimenti float più leggibile
             if abs(val) > 1e4 or abs(val) < 1e-3:
                  val_str = f"{val:.2e}"
             else:
                  val_str = f"{val:.3f}" # Aumenta precisione per float
             err_str = f"{err:.2g}" # Usa 'g' per precisione automatica sull'errore
             box_text_lines.append(f"${name} = {val_str} \\pm {err_str}$")

        if self.chi2_val is not None and self.dof is not None and self.dof > 0:
             chi2_rid_str = f"{self.chi2_val / self.dof:.3f}"
             box_text_lines.append(f"$\\chi^2/N_{{dof}} = {chi2_rid_str}$")
             if self.p_value is not None:
                  p_val_str = f"{self.p_value:.3f}"
                  box_text_lines.append(f"$p$-value $= {p_val_str}$")
        elif self.chi2_val is not None:
             # Mostra chi2 anche se DoF=0
             chi2_str = f"{self.chi2_val:.3f}"
             box_text_lines.append(f"$\\chi^2 = {chi2_str}$ (DoF=0)")


        info_text = "\n".join(box_text_lines)
        box_props = self._get_info_box_coords(info_box_pos)

        ax.annotate(info_text,
                    xy=box_props['xy'],
                    xycoords='axes fraction',
                    va=box_props['va'],
                    ha=box_props['ha'],
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.85, edgecolor='gray'),
                    fontsize=11, # Leggermente più piccolo per non sovrapporsi troppo
                    linespacing=1.4)

        ax.legend(fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.97]) # Aggiusta layout per dare spazio al titolo
        plt.show()






class FitBomberone2:
    """
    Classe unificata per eseguire fit di dati con diversi metodi (incluso ODR).
    Supporta calcolo delle bande di confidenza e plot avanzati.
    """
    def __init__(self, model_func, data_arrays, initial_params,
                 fit_method='LeastSquares', xlabel="x", ylabel="y", title="Risultati del fit"):
        # --- Validazione Input Iniziale ---
        valid_methods = ['LeastSquares', 'Chi2', 'Scipy', 'ODR'] 
        if fit_method not in valid_methods:
            raise ValueError(f"Metodo di fit '{fit_method}' non valido. Scegliere tra: {valid_methods}")

        if not callable(model_func):
             raise TypeError("model_func deve essere una funzione eseguibile.")
        if not isinstance(data_arrays, dict):
             raise TypeError("data_arrays deve essere un dizionario.")
        if not isinstance(initial_params, dict):
             raise TypeError("initial_params deve essere un dizionario.")

        # --- Memorizzazione Attributi ---
        self.model = model_func
        self.fit_method = fit_method
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

        # --- Estrazione Dati ---
        self.x = np.asarray(data_arrays['x'])
        self.y = np.asarray(data_arrays['y'])
        self.sigma_y = np.asarray(data_arrays.get('sigma_y', np.ones_like(self.y)))
        
        # Gestione sigma_x (per ODR)
        if 'sigma_x' in data_arrays:
             self.sigma_x = np.asarray(data_arrays['sigma_x'])
        elif fit_method == 'ODR':
             print("Attenzione: sigma_x non fornito per il metodo ODR. Verrà assunto nullo.")
             self.sigma_x = np.zeros_like(self.x)
        else:
             self.sigma_x = None

        # --- Estrazione Parametri ---
        sig = inspect.signature(model_func)
        all_param_names = list(sig.parameters.keys())
        if not all_param_names:
             raise ValueError("La funzione modello non sembra accettare argomenti.")
        self.param_names = all_param_names[1:] # Assume il primo sia x
        if not self.param_names:
             raise ValueError("La funzione modello deve accettare almeno un parametro oltre a x.")

        self.initial_params = initial_params

        # --- Inizializzazione Risultati ---
        self.fit_result: Any = None
        self.chi2_val: Any = None
        self.dof: Any = None
        self.p_value: Any = None
        self.is_fit_valid: bool = False
        self.odr_output: Any = None
        self.m: Any = None
        self.scipy_popt: Any = None
        self.scipy_pcov: Any = None

        self._validate_inputs()

    def _validate_inputs(self):
        if not (len(self.x) == len(self.y) == len(self.sigma_y)):
            raise ValueError("Gli array x, y e sigma_y devono avere la stessa lunghezza.")
        if self.sigma_x is not None and len(self.x) != len(self.sigma_x):
             raise ValueError("Gli array x e sigma_x devono avere la stessa lunghezza.")
        if np.any(self.sigma_y <= 0):
            print("Attenzione: sigma_y contiene valori <= 0.")
        missing_params = set(self.param_names) - set(self.initial_params.keys())
        extra_params = set(self.initial_params.keys()) - set(self.param_names)
        if missing_params: raise ValueError(f"Parametri iniziali mancanti: {missing_params}")
        if extra_params: raise ValueError(f"Parametri iniziali extra: {extra_params}")

    def _chi2_func_wrapper(self, *args):
        params_dict = {name: val for name, val in zip(self.param_names, args)}
        y_model = self.model(self.x, **params_dict)
        valid_sigma = self.sigma_y > 0
        residuals = np.zeros_like(self.y)
        residuals[valid_sigma] = (self.y[valid_sigma] - y_model[valid_sigma]) / self.sigma_y[valid_sigma]
        return np.sum(residuals**2)

    def _odr_model_wrapper(self, B, x):
        """Wrapper per la funzione modello richiesta da scipy.odr."""
        params_dict = {name: val for name, val in zip(self.param_names, B)}
        return self.model(x, **params_dict)

    def _fit_odr(self):
        """Esegue il fit usando Orthogonal Distance Regression (ODR)."""
        try:
            beta0 = [self.initial_params[name] for name in self.param_names]
            odr_model = Model(self._odr_model_wrapper)
            data = RealData(self.x, self.y, sx=self.sigma_x, sy=self.sigma_y)
            odr = ODR(data, odr_model, beta0=beta0)
            self.odr_output = odr.run()

            if self.odr_output.info > 0: # Controlla se il fit è terminato
                self.fit_result = {name: (val, err) for name, val, err in zip(self.param_names, self.odr_output.beta, self.odr_output.sd_beta)}
                
                # IMPORTANTISSIMO per far funzionare le bande di confidenza
                self.pcov = self.odr_output.cov_beta 
                
                self.chi2_val = self.odr_output.sum_square
                self.dof = len(self.x) - len(self.param_names)
                self.p_value = chi2.sf(self.chi2_val, self.dof) if self.dof > 0 else 0
                self.is_fit_valid = True
            else:
                 print(f"Errore ODR: Fit non converguto (info: {self.odr_output.info}).")
                 self._handle_fit_failure()
                 
        except Exception as e:
            print(f"Errore durante il fit ODR: {e}")
            self._handle_fit_failure()

    def _fit_least_squares(self):
        try:
             least_squares_cost = LeastSquares(self.x, self.y, self.sigma_y, self.model)
             self.m = Minuit(least_squares_cost, **self.initial_params)
             self.m.migrad()
             self.m.hesse()
             self.is_fit_valid = self.m.valid
             if not self.is_fit_valid: print("Attenzione: Covarianza (LeastSquares) non valida.")
             self.fit_result = {name: (self.m.values[name], self.m.errors[name]) for name in self.param_names}
             self.pcov = self.m.covariance
             self.chi2_val = self.m.fval
             self.dof = self.m.ndof
             self.p_value = chi2.sf(self.m.fval, self.m.ndof)
        except Exception as e:
             print(f"Errore Fit LeastSquares: {e}")
             self._handle_fit_failure()

    def _fit_chi2(self):
        try:
            initial_values_list = [self.initial_params[name] for name in self.param_names]
            self.m = Minuit(self._chi2_func_wrapper, *initial_values_list, name=self.param_names)
            self.m.errordef = Minuit.LEAST_SQUARES
            self.m.migrad()
            self.m.hesse()
            self.is_fit_valid = self.m.valid
            if not self.is_fit_valid: print("Attenzione: Covarianza (Chi2) non valida.")
            self.fit_result = {name: (self.m.values[name], self.m.errors[name]) for name in self.param_names}
            self.pcov = self.m.covariance
            self.chi2_val = self.m.fval
            self.dof = len(self.x) - len(self.param_names)
            self.p_value = chi2.sf(self.m.fval, self.dof) if self.dof > 0 else 0
        except Exception as e:
            print(f"Errore Fit Chi2: {e}")
            self._handle_fit_failure()

    def _fit_scipy(self):
        try:
            initial_params_list = [self.initial_params[name] for name in self.param_names]
            self.scipy_popt, self.scipy_pcov = curve_fit(
                self.model, self.x, self.y, p0=initial_params_list,
                sigma=self.sigma_y, absolute_sigma=True
            )
            errors = np.sqrt(np.diag(self.scipy_pcov))
            self.fit_result = {name: (val, err) for name, val, err in zip(self.param_names, self.scipy_popt, errors)}
            residuals = self.y - self.model(self.x, *self.scipy_popt)
            valid_sigma = self.sigma_y > 0
            chisq_terms = np.zeros_like(self.y)
            chisq_terms[valid_sigma] = (residuals[valid_sigma] / self.sigma_y[valid_sigma])**2
            self.chi2_val = np.sum(chisq_terms)
            self.dof = len(self.x) - len(self.scipy_popt)
            self.p_value = chi2.sf(self.chi2_val, self.dof) if self.dof > 0 else 0
            self.is_fit_valid = True
        except RuntimeError as e:
             print(f"Errore Fit Scipy: {e}. No convergenza.")
             self._handle_fit_failure(is_scipy=True)
        except Exception as e:
             print(f"Errore imprevisto Fit Scipy: {e}")
             self._handle_fit_failure(is_scipy=True)

    def _handle_fit_failure(self, is_scipy=False):
        self.is_fit_valid = False
        self.fit_result = {name: (self.initial_params[name], np.nan) for name in self.param_names}
        if is_scipy:
            self.scipy_popt = [self.initial_params[name] for name in self.param_names]
            self.scipy_pcov = np.full((len(self.param_names), len(self.param_names)), np.nan)
        self.chi2_val, self.dof, self.p_value = np.nan, len(self.x) - len(self.param_names), np.nan


    def perform_fit(self):
        print(f"--- Esecuzione Fit con Metodo: {self.fit_method} ---")
        if self.fit_method == 'LeastSquares': self._fit_least_squares()
        elif self.fit_method == 'Chi2': self._fit_chi2()
        elif self.fit_method == 'Scipy': self._fit_scipy()
        elif self.fit_method == 'ODR': self._fit_odr() # ODR COMPLETAMENTE INTEGRATO!
        else: raise ValueError(f"Metodo '{self.fit_method}' non riconosciuto per il fit interno.")
        
        if self.fit_result is None: print("Errore: Fit non ha prodotto risultati.")
        elif not self.is_fit_valid: print("Avviso: Fit potrebbe non essere valido.")
        else: print("Fit completato.")
        return self

    def print_results(self):
        if self.fit_result is None:
            print("Nessun risultato del fit. Eseguire perform_fit().")
            return
        print(f"\n--- Risultati del Fit ({self.fit_method}) ---")
        print(f"Fit Valido: {'Sì' if self.is_fit_valid else 'No'}")
        print("\nParametri Ottimizzati:")
        for name in self.param_names:
            val, err = self.fit_result.get(name, (np.nan, np.nan))
            print(f"  {name} = {val:.4g} ± {err:.2g}")
        if not np.isnan(self.chi2_val) and self.dof is not None and self.dof > 0:
            chi2_rid = self.chi2_val / self.dof
            print(f"\nStatistiche del Fit:")
            print(f"  Chi-quadro (χ²): {self.chi2_val:.4f}")
            print(f"  Gradi di libertà (DoF): {self.dof}")
            print(f"  Chi-quadro Ridotto (χ²/DoF): {chi2_rid:.4f}")
            if not np.isnan(self.p_value): print(f"  p-value: {self.p_value:.4f}")
        elif not np.isnan(self.chi2_val):
             print(f"\n  Chi-quadro (χ²): {self.chi2_val:.4f} (DoF={self.dof})")
        return self

    def calculate_confidence_band(self, x_points, num_sigma=1):
        if not self.is_fit_valid or self.fit_result is None:
            print("Fit non valido o non eseguito. Impossibile calcolare la banda di confidenza.")
            return None, None
        if self.pcov is None:
            print("Matrice di covarianza non disponibile. Impossibile calcolare la banda di confidenza.")
            return None, None
        if np.all(np.isnan(self.pcov)):
            print("Matrice di covarianza contiene NaN. Impossibile calcolare la banda di confidenza.")
            return None, None

        popt_values = np.array([self.fit_result[name][0] for name in self.param_names])
        y_model_on_x_points = self.model(x_points, *popt_values)
        dy_confidence_band = np.zeros_like(x_points, dtype=float)

        for i, x_val in enumerate(x_points):
            if list(self.param_names) == ['A', 'B'] and self.model.__name__ == 'cauchy': 
                grad = np.array([1.0, 1.0 / (x_val**2) if x_val != 0 else float('inf')])
            else:
                eps = 1e-8 
                grad = np.zeros(len(popt_values))
                for j in range(len(popt_values)):
                    p_plus = popt_values.copy()
                    p_minus = popt_values.copy()
                    p_plus[j] += eps
                    p_minus[j] -= eps
                    grad[j] = (self.model(x_val, *p_plus) - self.model(x_val, *p_minus)) / (2 * eps)
            
            try:
                var_y = grad.T @ self.pcov @ grad
                if var_y < 0: 
                    dy_confidence_band[i] = 0.0
                else:
                    dy_confidence_band[i] = num_sigma * np.sqrt(var_y)
            except Exception as e:
                print(f"Errore nel calcolo della varianza della banda a x={x_val}: {e}")
                dy_confidence_band[i] = np.nan 

        return y_model_on_x_points, dy_confidence_band

    def _get_info_box_coords(self, position='upper left', pad=0.05):
        positions = {
            'upper left':   {'xy': (pad, 1 - pad), 'ha': 'left', 'va': 'top'},
            'upper right':  {'xy': (1 - pad, 1 - pad), 'ha': 'right', 'va': 'top'},
            'lower left':   {'xy': (pad, pad), 'ha': 'left', 'va': 'bottom'},
            'lower right':  {'xy': (1 - pad, pad), 'ha': 'right', 'va': 'bottom'},
        }
        return positions.get(position.lower().replace("_", " "), positions['upper right'])

    def plot_results(self, title_fontsize=14, label_fontsize=12,
                     info_box_pos='upper right', log_scale_y=False, log_scale_x=False,
                     plot_confidence_band=False, confidence_sigma_level=1):
        if self.fit_result is None:
            print("Nessun risultato. Eseguire perform_fit().")
            return
        
        plt.style.use('seaborn-v0_8-whitegrid') 
        plt.figure(figsize=(10, 7))
        ax = plt.gca()

        # --- Plot Dati ---
        plot_kwargs = {'fmt': 'o', 'label': 'Dati', 'markersize': 5, 
                       'capsize': 3, 'elinewidth': 1, 'markeredgecolor': 'k', 'zorder': 10}
        if self.sigma_x is not None and np.any(self.sigma_x > 1e-9):
             ax.errorbar(self.x, self.y, xerr=self.sigma_x, yerr=self.sigma_y, **plot_kwargs)
        else:
             ax.errorbar(self.x, self.y, yerr=self.sigma_y, **plot_kwargs)

        # --- Plot Curva di Fit e Banda di Confidenza ---
        if len(self.x) > 1:
             x_min_data, x_max_data = np.min(self.x), np.max(self.x)
             range_ext_factor = 0.05
             
             if log_scale_x:
                 if x_min_data <= 0: x_log_min = np.log10(np.min(self.x[self.x > 0]) * (1-range_ext_factor*2)) if np.any(self.x > 0) else -1
                 else: x_log_min = np.log10(x_min_data * (1-range_ext_factor))
                 x_log_max = np.log10(x_max_data * (1+range_ext_factor))
                 x_fit_plot = np.logspace(x_log_min, x_log_max, 400)
             else:
                 data_range = x_max_data - x_min_data
                 if data_range == 0: data_range = 1 
                 x_fit_plot = np.linspace(x_min_data - data_range*range_ext_factor, 
                                      x_max_data + data_range*range_ext_factor, 400)
        elif len(self.x) == 1:
             x_fit_plot = np.array([self.x[0] * 0.9, self.x[0], self.x[0] * 1.1]) if self.x[0]!=0 else np.array([-0.1, 0, 0.1])
        else:
            x_fit_plot = np.linspace(0,1,100) 

        fitted_p_values = np.array([self.fit_result[name][0] for name in self.param_names])
        
        try:
             y_fit_curve = self.model(x_fit_plot, *fitted_p_values)
             ax.plot(x_fit_plot, y_fit_curve, color='crimson', label=f'Fit ({self.fit_method})', linewidth=1.5, zorder= 5)

             if plot_confidence_band:
                y_model_band, dy_band = self.calculate_confidence_band(x_fit_plot, num_sigma=confidence_sigma_level)
                if y_model_band is not None and dy_band is not None:
                    valid_band_indices = np.isfinite(y_model_band) & np.isfinite(dy_band)
                    ax.fill_between(x_fit_plot[valid_band_indices],
                                     (y_model_band - dy_band)[valid_band_indices],
                                     (y_model_band + dy_band)[valid_band_indices],
                                     color='salmon', alpha=0.35,
                                     label=f'Banda conf. ({confidence_sigma_level}σ)')
        except Exception as e:
             print(f"Errore calcolo curva/banda di fit: {e}")

        ax.set_xlabel(self.xlabel, fontsize=label_fontsize)
        ax.set_ylabel(self.ylabel, fontsize=label_fontsize)
        ax.set_title(self.title, fontsize=title_fontsize, pad=15)

        if log_scale_y: ax.set_yscale('log')
        if log_scale_x: ax.set_xscale('log')

        ax.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')
        ax.grid(True, which='minor', linestyle=':', linewidth='0.3', color='lightgrey')
        ax.minorticks_on()

        box_text_lines = []
        for name in self.param_names:
             val, err = self.fit_result.get(name, (np.nan, np.nan))
             val_str = f"{val:.3e}" if (abs(val) > 1e4 or (abs(val) < 1e-3 and val !=0)) else f"{val:.4g}"
             err_str = f"{err:.2e}" if (abs(err) > 1e3 or (abs(err) < 1e-4 and err !=0)) else f"{err:.2g}"
             box_text_lines.append(f"${name} = {val_str} \\pm {err_str}$")

        if not np.isnan(self.chi2_val) and self.dof is not None and self.dof > 0:
             box_text_lines.append(f"$\\chi^2/N_{{dof}} = {self.chi2_val / self.dof:.2f}$ ($N_{{dof}}={self.dof}$)")
             if not np.isnan(self.p_value): box_text_lines.append(f"$p$-value $= {self.p_value:.3f}$")
        elif not np.isnan(self.chi2_val):
             box_text_lines.append(f"$\\chi^2 = {self.chi2_val:.2f}$ (DoF={self.dof})")
        if not self.is_fit_valid: box_text_lines.append("Fit non valido!")
        
        info_text = "\n".join(box_text_lines)
        box_coords = self._get_info_box_coords(info_box_pos)
        ax.annotate(info_text, xy=box_coords['xy'], xycoords='axes fraction',
                    va=box_coords['va'], ha=box_coords['ha'], fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.9, ec='grey'))

        ax.legend(fontsize=9, loc='best')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()




def noise_generator_uniform(x_true, y_true, sigma_x, sigma_y):
    y = sigma_y * np.sqrt(3)
    noise_y = np.random.uniform(-y, y, size=len(y_true))
    y_noise = y + noise_y

    x = sigma_x * np.sqrt(3)
    noise_x = np.random.uniform(-x, x, size=len(x_true))
    x_noise = x + noise_x

    return x_noise, y_noise

#Se invece i miei dati non seguono una distribuzione precisa posso usare questo tipo di funzione
def bootstrap(y_true, residui):
    indices = np.random.randint(0, len(residui), size=len(y_true))
    noise = residui[indices]
    return y_true + noise



def verifica_montecarlo(modello, x, sigma_x, sigma_y, N, true_params, fit_method, noise_generator=None, initial_params= None, initial_params_generator= None, silent=True, **kwargs):
    if noise_generator is None:
        def default_noise(x_atteso, y_atteso, sig_x, sig_y):
            # Spalma la y
            y_rand = y_atteso + np.random.normal(0, sig_y, size=len(y_atteso))
            # Spalma la x (Necessario solo se passi una sigma_x non nulla, es. per ODR)
            if sig_x is not None and np.any(sig_x > 0):
                x_rand = x_atteso + np.random.normal(0, sig_x, size=len(x_atteso))
            else:
                x_rand = x_atteso
            return x_rand, y_rand
        noise_generator = default_noise
    

    param_names = list(true_params.keys())
    params_values = {name: [] for name in param_names}
    params_errors = {name: [] for name in param_names}
    chi2_list = []
    ndof = len(x) - len(true_params)
    
    y_expected = modello(x, **true_params)  

    if silent:
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    try:
        for i in range(N):
            x_random, y_random = noise_generator(x, y_expected, sigma_x, sigma_y)

            data_toy = {'x': x_random, 'y': y_random, 'sigma_y': sigma_y, 'sigma_x': sigma_x}
            if initial_params_generator is not None:
                init = initial_params_generator(i)
            elif initial_params is not None:
                init = initial_params
            else:
                init = true_params

            fit_toy = lib.FitBomberone2(model_func=modello, data_arrays=data_toy, initial_params=init, fit_method=fit_method,   
                                        xlabel='SIUM', ylabel='Uomo Falena', title=f'Toy #{i+1}')
            fit_toy.perform_fit()

            if fit_toy.is_fit_valid:
                for name in param_names:
                    val, err = fit_toy.fit_result[name]
                    params_values[name].append(val)
                    params_errors[name].append(err)
                chi2_list.append(fit_toy.chi2_val)
            else:
                continue
    finally:
        # Ripristina l'output della console anche se il ciclo si interrompe per un errore
        if silent:
            sys.stdout.close()
            sys.stdout = old_stdout
    
    chi2_array = np.array(chi2_list) if chi2_list else np.array([])

    return {
        'params_values': params_values,
        'params_errors': params_errors,
        'chi2': chi2_array,
        'true_params': true_params,
        'ndof': ndof,
        'n_successi': len(chi2_list)
    }

def verifica_montecarlo_binned(modello_cdf, modello_pdf, canali, true_params, N, initial_params=None, initial_params_generator=None, silent=True):
    """
    Toy Monte Carlo specifico per istogrammi (statistica di Poisson).
    Genera i conteggi estratti per ogni bin partendo dai conteggi attesi veri (CDF) e refitta con FitLikelihoodBomberone.
    Restituisce un dizionario direttamente compatibile con `plot_montecarlo_results`.
    """
    param_names = list(true_params.keys())
    params_values = {name: [] for name in param_names}
    params_errors = {name: [] for name in param_names}
    chi2_list = []
    
    # NDOF tipico: n_bin - n_params_liberi
    ndof = len(canali) - 1 - len(true_params)

    # 1. Calcolo conteggi Attesi per bin (Verità Montecarlo) tramite CDF
    # Assumiamo i dati continui per CDF: bin_edges = np.append(x - step/2, x[-1] + step/2)
    step = canali[1] - canali[0]
    bin_edges = np.append(canali - step/2, canali[-1] + step/2)
    expected_counts = np.diff(modello_cdf(bin_edges, **true_params))

    if silent:
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
    try:
        for i in range(N):
            # 2. Estrazione Poissoniana per simulare i dati sperimentali 
            # (Ogni bin oscilla poissonianamente intorno al suo valore atteso)
            toy_counts = np.random.poisson(expected_counts)
            
            if initial_params_generator is not None:
                init = initial_params_generator(i)
            elif initial_params is not None:
                init = initial_params
            else:
                init = true_params

            # 3. Fit del Toy Spectrum appena generato
            fit_toy = FitLikelihoodBomberone(
                canali=canali, 
                conteggi=toy_counts, 
                modello_cdf=modello_cdf, 
                modello_pdf=modello_pdf, 
                initial_params=init,
                xlabel='Canale', 
                ylabel='Conteggi', 
                title=f'Toy MC #{i+1}'
            )
            fit_toy.perform_fit()

            # 4. Estrapolazione Risultati
            if fit_toy.is_fit_valid:
                for name in param_names:
                    val, err = fit_toy.fit_result[name]
                    params_values[name].append(val)
                    params_errors[name].append(err)
                chi2_list.append(fit_toy.chi2_val)
            else:
                continue
    finally:
        if silent:
            sys.stdout.close()
            sys.stdout = old_stdout
            
    chi2_array = np.array(chi2_list) if chi2_list else np.array([])

    return {
        'params_values': params_values,
        'params_errors': params_errors,
        'chi2': chi2_array,
        'true_params': true_params,
        'ndof': ndof,
        'n_successi': len(chi2_list)
    }




def plot_montecarlo_results(mc_results):
    """
    Stampa i grafici diagnostici (Pulls e Chi2) partendo dall'output di verifica_montecarlo
    """
    true_params = mc_results['true_params']
    param_names = list(true_params.keys())
    
    print(f"Toy MC completati con successo: {mc_results['n_successi']}")
    if mc_results['n_successi'] == 0:
        return
        
    for name in param_names:
        # 1. FORZIAMO A NUMPY ARRAY QUI
        vals = np.array(mc_results['params_values'][name])
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        print(f"Parametro {name}: Vero = {true_params[name]} | Media Toy = {mean_val:.4g} ± {std_val:.4g}")

    n_params = len(param_names)
    fig, axes = plt.subplots(1, n_params + 1, figsize=(5 * (n_params + 1), 4))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot dei Pulls per ogni parametro
    for i, name in enumerate(param_names):
        # 2. FORZIAMO A NUMPY ARRAY ANCHE QUI
        vals = np.array(mc_results['params_values'][name])
        errs = np.array(mc_results['params_errors'][name])
        true_val = true_params[name]
        
        # Calcolo dei Pulls (Ora funzionerà perfettamente!)
        pulls = (vals - true_val) / errs
        
        ax = axes[i]
        ax.hist(pulls, bins=30, color='skyblue', edgecolor='black', density=True)
        ax.set_title(rf'Pull Distribution: {name}\n$\mu={np.mean(pulls):.2f}, \sigma={np.std(pulls):.2f}$')
        ax.set_xlabel('Zio pera')
        
        # Disegna la gaussiana attesa N(0,1)
        x_g = np.linspace(-4, 4, 100)
        ax.plot(x_g, sc.norm.pdf(x_g, 0, 1), 'r--', label='Atteso N(0,1)')
        ax.legend()

    # Plot del Chi-Quadro
    ax_chi = axes[-1]
    
    # 3. FORZIAMO A NUMPY ARRAY PER IL CHI2
    chi2_vals = np.array(mc_results['chi2'])
    ndof = mc_results['ndof']
    
    ax_chi.hist(chi2_vals, bins=30, color='lightgreen', edgecolor='black', density=True)
    ax_chi.set_title(rf'$\chi^2$ Distribution (ndof={ndof})\nMedia={np.mean(chi2_vals):.2f}')
    ax_chi.set_xlabel(r'$\chi^2$')
    
    # Disegna la distribuzione chi2 teorica
    x_chi = np.linspace(max(0, np.min(chi2_vals)-5), np.max(chi2_vals)+5, 100)
    ax_chi.plot(x_chi, sc.chi2.pdf(x_chi, ndof), 'r--', label='Teorico')
    ax_chi.legend()

    plt.tight_layout()
    plt.show()



def validate_toy_montecarlo(mc_results, alpha=0.05):
    """
    Esegue test quantitativi sui risultati di toy Monte Carlo su tutti i parametri.
    Accetta direttamente in input il dizionario generato da 'verifica_montecarlo'.
    """
    true_params = mc_results['true_params']
    
    # FORZIAMO IL CHI2 A DIVENTARE UN ARRAY NUMPY
    chi2_array = np.array(mc_results['chi2']) 
    
    ndof = mc_results['ndof']
    n_toys = mc_results['n_successi']
    
    if n_toys == 0:
        return {"error": "Nessun toy MC completato con successo."}

    results = {'parameters': {}, 'goodness_of_fit': {}}

    # 1. ANALISI PARAMETRI E PULL (Per ogni parametro)
    for name, p_true in true_params.items():
        
        # ECCO LA MAGIA: FORZIAMO I PARAMETRI A DIVENTARE ARRAY NUMPY
        p_fit_array = np.array(mc_results['params_values'][name])
        p_err_array = np.array(mc_results['params_errors'][name])
        
        param_res = {}

        # Test Bias (Media parametri)
        mean_p = np.mean(p_fit_array)
        std_p = np.std(p_fit_array, ddof=1)
        std_err_mean = std_p / np.sqrt(n_toys)
        t_stat_mean = (mean_p - p_true) / std_err_mean if std_err_mean > 0 else 0
        p_value_mean = 2 * (1 - stats.t.cdf(np.abs(t_stat_mean), df=n_toys-1))
        
        param_res['mean_test'] = {
            'desc': "Bias Media (Atteso: True Value)",
            'val': mean_p, 'err': std_err_mean, 'p_value': p_value_mean, 'reject_H0': p_value_mean < alpha
        }

        # Analisi Pull (Ora la sottrazione funzionerà perfettamente!)
        pull = (p_fit_array - p_true) / p_err_array
        mean_pull = np.mean(pull)
        std_pull = np.std(pull, ddof=1)
        
        # Media Pull
        t_pull = mean_pull / (std_pull / np.sqrt(n_toys))
        p_value_mean_pull = 2 * (1 - stats.t.cdf(np.abs(t_pull), df=n_toys-1))
        param_res['pull_mean'] = {
            'desc': "Media Pull (Atteso: 0)",
            'val': mean_pull, 'err': std_pull / np.sqrt(n_toys), 'p_value': p_value_mean_pull, 'reject_H0': p_value_mean_pull < alpha
        }

        # Varianza Pull
        var_pull = std_pull**2
        chi2_var = (n_toys - 1) * var_pull
        p_value_var = 2 * min(stats.chi2.cdf(chi2_var, df=n_toys-1), 1 - stats.chi2.cdf(chi2_var, df=n_toys-1))
        param_res['pull_variance'] = {
            'desc': "Varianza Pull (Attesa: 1)",
            'val': var_pull, 'p_value': p_value_var, 'reject_H0': p_value_var < alpha
        }

        # Normalità Pull
        if n_toys <= 5000:
            stat, p_norm = stats.shapiro(pull)
            test_name = "Shapiro-Wilk"
        else:
            stat, p_norm = stats.kstest(pull, 'norm')
            test_name = "Kolmogorov-Smirnov"
            
        param_res['pull_normality'] = {
            'desc': f"Normalità Pull ({test_name})",
            'stat': stat, 'p_value': p_norm, 'reject_H0': p_norm < alpha
        }

        # Copertura (Coverage)
        inside = (p_fit_array - p_err_array <= p_true) & (p_true <= p_fit_array + p_err_array)
        coverage_1sigma = np.mean(inside)
        p_val_cov = 2 * min(stats.binom.cdf(int(coverage_1sigma * n_toys), n_toys, 0.6827),
                            1 - stats.binom.cdf(int(coverage_1sigma * n_toys) - 1, n_toys, 0.6827))
        param_res['coverage_1sigma'] = {
            'desc': "Copertura 1σ (Attesa: ~68.3%)",
            'val': coverage_1sigma * 100, 'p_value': p_val_cov, 'reject_H0': p_val_cov < alpha
        }

        results['parameters'][name] = param_res

    # 2. ANALISI DEL CHI-QUADRO (Globale)
    mean_chi2 = np.mean(chi2_array)
    z_chi2 = (mean_chi2 - ndof) / np.sqrt(2 * ndof / n_toys)
    p_value_chi2_mean = 2 * (1 - stats.norm.cdf(np.abs(z_chi2)))
    
    results['goodness_of_fit']['chi2_mean'] = {
        'desc': f"Media χ² (Attesa: {ndof})",
        'val': mean_chi2, 'p_value': p_value_chi2_mean, 'reject_H0': p_value_chi2_mean < alpha
    }

    ks_chi2_stat, ks_chi2_p = stats.kstest(chi2_array, 'chi2', args=(ndof,))
    results['goodness_of_fit']['chi2_dist'] = {
        'desc': "Distribuzione χ² (KS test)",
        'stat': ks_chi2_stat, 'p_value': ks_chi2_p, 'reject_H0': ks_chi2_p < alpha
    }

    return results




def print_validation_report(val_results):
    """Stampa un report pulito e formattato dell'analisi di validazione."""
    if "error" in val_results:
        print(val_results["error"])
        return

    print("="*60)
    print("   REPORT VALIDAZIONE TOY MONTE CARLO   ")
    print("="*60)
    
    for param, tests in val_results['parameters'].items():
        print(f"\n[{param}] ANALISI PARAMETRO:")
        print("-" * 50)
        for test_key, data in tests.items():
            status = "❌ FALLITO" if data['reject_H0'] else "✅ PASSATO"
            
            # Formattazione intelligente del valore
            val_str = ""
            if 'val' in data:
                if 'err' in data:
                    val_str = f"Valore: {data['val']:.4g} ± {data['err']:.4g}  |"
                elif test_key == 'coverage_1sigma':
                    val_str = f"Valore: {data['val']:.1f}%  |"
                else:
                    val_str = f"Valore: {data['val']:.4g}  |"
            
            print(f"{status} | {data['desc']:<35} | {val_str} p-value: {data['p_value']:.3e}")

    print("\n[BONTÀ DEL FIT] ANALISI CHI-QUADRO GLOBALE:")
    print("-" * 50)
    for test_key, data in val_results['goodness_of_fit'].items():
        status = "❌ FALLITO" if data['reject_H0'] else "✅ PASSATO"
        val_str = f"Valore: {data['val']:.4g}  |" if 'val' in data else ""
        print(f"{status} | {data['desc']:<35} | {val_str} p-value: {data['p_value']:.3e}")
    print("="*60)









"""Funzione per fare il test di Student, in xname basta mettere il nome che voglio printi"""

def Tstudent_secondo(x, y, sigma_x, sigma_y, x_name="x", y_name="y"):
    T = np.abs((x - y) / np.sqrt(sigma_x**2 + sigma_y**2))
    df = (sigma_x**2 + sigma_y**2)**2 / ((sigma_x**4) + (sigma_y**4))
    p_value = 2 * (1 - t.cdf(T, df))
    
    print(f"Il p-value del test di compatibilità tra il valore di {x_name} e {y_name} vale: {p_value:.4f}")
    if p_value > 0.05:
        print("I due valori sono compatibili")
    else:
        print("I due valori NON sono compatibili")




"""Funzione per fare lo Z-test"""
def Ztest(x, y, sigma_x, x_name='x', y_name='y'):
    Z = np.abs((x - y) / sigma_x)

    p_value = 2 * (1 - sc.norm.cdf(Z))
    
    print(f"Il p-value del test di compatibilità tra il valore di {x_name} e {y_name} vale: {p_value:.4f}")
    if p_value > 0.05:
        print("I due valori sono compatibili")
    else:
        print("I due valori NON sono compatibili")


"""Funzione per fare lo scatter"""

def scatter_plot_with_error(x, y, sigma_y, xlabel, ylabel, title, sigma_x=None, axhline_value=None):
    """
    Crea uno scatter plot dei dati con barre d'errore e linea connettente tra i punti.
    
    Parametri:
      x: array-like, dati dell'asse x
      y: array-like, dati dell'asse y
      sigma_y: array-like, errori associati a y
      sigma_x: array-like, errori associati a x (opzionale)
      axhline_value: float, valore y dove disegnare una linea orizzontale (default: None, nessuna linea)
      xlabel: string, etichetta per l'asse x
      ylabel: string, etichetta per l'asse y
      title: string, titolo del grafico
    """

    plt.figure(figsize=(10, 5), dpi=100)
    plt.style.use('seaborn-v0_8-notebook')

    plt.errorbar(
        x,
        y,
        xerr=sigma_x,
        yerr=sigma_y,
        fmt='-',
        color='purple',             # colore della linea connettente
        ecolor='orange',               # colore delle barre di errore
        elinewidth=1.5,               # spessore delle linee degli errori
        capsize=4,                  
        alpha=0.8,
        zorder=1
    )

    sc = plt.scatter(
        x,
        y,
        c=np.abs(y),
        cmap='viridis',
        s=45,
        alpha=0.8,
        edgecolors='k',
        linewidths=0.5,
        zorder=3
    )

    if axhline_value is not None:
        plt.axhline(axhline_value, color='gray', linestyle='--', linewidth=0.8, zorder=1)

    plt.grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.5)

    plt.title(title, fontsize=14, pad=20)
    plt.xlabel(xlabel, fontsize=12, labelpad=10)
    plt.ylabel(ylabel, fontsize=12, labelpad=10)

    ax = plt.gca()
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Barra dei colori
    cbar = plt.colorbar(sc)
    cbar.set_label('Ampiezza (V)', rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=9)
    
    plt.tight_layout()
    plt.show()


def scatter_plot_log_with_error(x, y, sigma_y, xlabel, ylabel, title, sigma_x=None, axhline_value=None):
    """
    Crea uno scatter plot dei dati con barre d'errore e linea connettente tra i punti,
    con scala bilogaritmica (log-log).
    
    Parametri:
      x: array-like, dati dell'asse x
      y: array-like, dati dell'asse y
      sigma_y: array-like, errori associati a y
      sigma_x: array-like, errori associati a x (opzionale)
      axhline_value: float, valore y dove disegnare una linea orizzontale (default: None, nessuna linea)
      xlabel: string, etichetta per l'asse x
      ylabel: string, etichetta per l'asse y
      title: string, titolo del grafico
    """
    plt.figure(figsize=(10, 5), dpi=100)
    plt.style.use('seaborn-v0_8-notebook')
    
    # Imposta entrambi gli assi in scala logaritmica
    plt.xscale('log')
    plt.yscale('log')
    
    plt.errorbar(
        x,
        y,
        xerr=sigma_x,
        yerr=sigma_y,
        fmt='-',
        color='purple',             # colore della linea connettente
        ecolor='orange',            # colore delle barre di errore
        elinewidth=1.5,             # spessore delle linee degli errori
        capsize=4,                  
        alpha=0.8,
        zorder=1
    )
    sc = plt.scatter(
        x,
        y,
        c=np.abs(y),
        cmap='viridis',
        s=45,
        alpha=0.8,
        edgecolors='k',
        linewidths=0.5,
        zorder=3
    )
    if axhline_value is not None:
        plt.axhline(axhline_value, color='gray', linestyle='--', linewidth=0.8, zorder=1)
    
    # Configurazione della griglia specifica per i grafici logaritmici
    plt.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.5)
    plt.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.3)
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel(xlabel, fontsize=12, labelpad=10)
    plt.ylabel(ylabel, fontsize=12, labelpad=10)
    
    # Configurazione dei tick per scale logaritmiche
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    
    # Barra dei colori
    cbar = plt.colorbar(sc)
    cbar.set_label('Ampiezza (V)', rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=9)
    
    plt.tight_layout()
    plt.show()



"""
Funzione per fare i fit con ODR
ESEMPIO:
def V_Ohm(I, R, q):
  return R*I + q

# Prepara il dizionario dei dati e dei parametri iniziali
data_arrays = {'x': x, 'y': y, 'sigma_y': sy * np.ones_like(y), 'sigma_x': sx * np.ones_like(x)}
initial_params = {'R': 10.0, 'q': 0.0}

# Istanzia la classe per il fit ODR
fit_odr = FitODR(V_Ohm, data_arrays, initial_params,
                               xlabel="x", ylabel="y", title="Fit lineare con ODR")

# Esegue il fit
fit_odr.perform_fit()

# Stampa i risultati
fit_odr.print_results()

# Visualizza il grafico dei dati e del fit
fit_odr.plot_results()
"""


class FitODR:
    def __init__(self, model_func, data_arrays, initial_params, xlabel="x", ylabel="y", title="Risultati del fit"):
        """
        model_func: funzione Python che prende x e i parametri come keyword (es. def model(x, a, b): …)
        data_arrays: dizionario con 'x', 'y', 'sigma_y' e opzionalmente 'sigma_x'
        initial_params: dizionario con parametri e valori iniziali
        """
        self.model = model_func
        self.x = data_arrays['x']
        self.y = data_arrays['y']
        self.sigma_y = data_arrays.get('sigma_y', np.ones_like(self.y))
        self.sigma_x = data_arrays.get('sigma_x', np.zeros_like(self.x))  # se non fornito, si assume zero
        
        # Estrae i nomi dei parametri dalla firma della funzione (esclude x)
        sig = inspect.signature(model_func)
        all_param_names = list(sig.parameters.keys())
        self.param_names = all_param_names[1:]
        
        self.initial_params = initial_params
        self.fit_result = None
        
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

        self._validate_inputs()
        
    def _validate_inputs(self):
        if len(self.x) != len(self.y) or len(self.y) != len(self.sigma_y):
            raise ValueError("Gli array x, y e sigma_y devono avere la stessa lunghezza")
        if len(self.x) != len(self.sigma_x):
            raise ValueError("Gli array x e sigma_x devono avere la stessa lunghezza")
        if not all(p in self.param_names for p in self.initial_params.keys()):
            raise ValueError("I nomi dei parametri iniziali non corrispondono a quelli della funzione modello")
    
    def _odr_model(self, B, x):
        """
        Funzione wrapper per ODR.
        B: array dei parametri, nell'ordine definito da self.param_names
        x: array (o array 2D) delle variabili indipendenti
        """
        # Mappa l'array B in un dizionario con i nomi dei parametri
        params_dict = {name: value for name, value in zip(self.param_names, B)}
        return self.model(x, **params_dict)
    
    def perform_fit(self):
        beta0 = [self.initial_params[name] for name in self.param_names]
        
        odr_model = Model(self._odr_model)
        
        data = RealData(self.x, self.y, sx=self.sigma_x, sy=self.sigma_y)
        
        odr = ODR(data, odr_model, beta0=beta0)
        output = odr.run()
        
        self.fit_result = {name: (val, err) for name, val, err in zip(self.param_names, output.beta, output.sd_beta)}
        self.odr_output = output 
        
        return output
    
    def print_results(self):
        if self.fit_result is None:
            raise RuntimeError("Devi eseguire prima il fit con perform_fit()")
            
        print("Risultati del fit (ODR):")
        for name in self.param_names:
            val, err = self.fit_result[name]
            print(f"{name} = {val:.3e} ± {err:.3e}")
        
        chi2_val = self.odr_output.sum_square
        dof = len(self.x) - len(self.param_names)
        print(f"\nChi-quadro ridotto: {chi2_val/dof:.3f}")
        print(f"Gradi di libertà: {dof}")
        print(f"p-value: {1 - chi2.cdf(chi2_val, dof):.3f}")
        
    def plot_results(self, title_fontsize=14, label_fontsize=12):
        if self.fit_result is None:
            raise RuntimeError("Devi eseguire prima il fit con perform_fit()")
        
        plt.figure(figsize=(10, 6))

        plt.errorbar(self.x, self.y, xerr=self.sigma_x, yerr=self.sigma_y,
                     fmt='o', label='Dati', markersize=7, capsize=4)
        

        x_fit = np.linspace(np.min(self.x), np.max(self.x), 500)

        params_dict = {name: self.fit_result[name][0] for name in self.param_names}
        y_fit = self.model(x_fit, **params_dict)
        plt.plot(x_fit, y_fit, '-r', label='Fit (ODR)', linewidth=2.5)
        
        plt.xlabel(self.xlabel, fontsize=label_fontsize)
        plt.ylabel(self.ylabel, fontsize=label_fontsize)
        plt.title(self.title, fontsize=title_fontsize, pad=20)
        
        # Box con informazioni sui risultati
        text_lines = [f"${name} = {val:.2e} \\pm {err:.2e}$" 
                      for name, (val, err) in self.fit_result.items()]
        dof = len(self.x) - len(self.param_names)
        chi2_red = self.odr_output.sum_square / dof
        text_lines.append(f"$\\chi^2/NdoF = {chi2_red:.3f}$")
        text = "\n".join(text_lines)
        plt.annotate(text, 
                     xy=(0.05, 0.95), 
                     xycoords='axes fraction',
                     va='top', 
                     ha='left', 
                     bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.7', edgecolor='gray'),
                     fontsize=13, 
                     linespacing=1.5)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


class FitODR_2:
    def __init__(self, model_func, data_arrays, initial_params, xlabel="x", ylabel="y", title="Risultati del fit"):
        """
        model_func: funzione Python che prende x e i parametri come keyword (es. def model(x, a, b): …)
        data_arrays: dizionario con 'x', 'y', 'sigma_y' e opzionalmente 'sigma_x'
        initial_params: dizionario con parametri e valori iniziali
        """
        self.model = model_func
        self.x = data_arrays['x']
        self.y = data_arrays['y']
        self.sigma_y = data_arrays.get('sigma_y', np.ones_like(self.y))
        self.sigma_x = data_arrays.get('sigma_x', np.zeros_like(self.x))  # se non fornito, si assume zero
        
        # Estrae i nomi dei parametri dalla firma della funzione (esclude x)
        sig = inspect.signature(model_func)
        all_param_names = list(sig.parameters.keys())
        self.param_names = all_param_names[1:]
        
        self.initial_params = initial_params
        self.fit_result = None
        
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

        self._validate_inputs()
        
    def _validate_inputs(self):
        if len(self.x) != len(self.y) or len(self.y) != len(self.sigma_y):
            raise ValueError("Gli array x, y e sigma_y devono avere la stessa lunghezza")
        if len(self.x) != len(self.sigma_x):
            raise ValueError("Gli array x e sigma_x devono avere la stessa lunghezza")
        if not all(p in self.param_names for p in self.initial_params.keys()):
            raise ValueError("I nomi dei parametri iniziali non corrispondono a quelli della funzione modello")
    
    def _odr_model(self, B, x):
        """
        Funzione wrapper per ODR.
        B: array dei parametri, nell'ordine definito da self.param_names
        x: array (o array 2D) delle variabili indipendenti
        """
        # Mappa l'array B in un dizionario con i nomi dei parametri
        params_dict = {name: value for name, value in zip(self.param_names, B)}
        return self.model(x, **params_dict)
    
    def perform_fit(self):
        beta0 = [self.initial_params[name] for name in self.param_names]
        
        odr_model = Model(self._odr_model)
        
        data = RealData(self.x, self.y, sx=self.sigma_x, sy=self.sigma_y)
        
        odr = ODR(data, odr_model, beta0=beta0)
        output = odr.run()
        
        self.fit_result = {name: (val, err) for name, val, err in zip(self.param_names, output.beta, output.sd_beta)}
        self.odr_output = output 
        
        return output
    
    def print_results(self):
        if self.fit_result is None:
            raise RuntimeError("Devi eseguire prima il fit con perform_fit()")
            
        print("Risultati del fit (ODR):")
        for name in self.param_names:
            val, err = self.fit_result[name]
            print(f"{name} = {val:.3e} ± {err:.3e}")
        
        chi2_val = self.odr_output.sum_square
        dof = len(self.x) - len(self.param_names)
        print(f"\nChi-quadro ridotto: {chi2_val/dof:.3f}")
        print(f"Gradi di libertà: {dof}")
        print(f"p-value: {1 - chi2.cdf(chi2_val, dof):.3f}")
        
    def plot_results(self, title_fontsize=14, label_fontsize=12):
        if self.fit_result is None:
            raise RuntimeError("Devi eseguire prima il fit con perform_fit()")
        
        plt.figure(figsize=(10, 6))

        plt.errorbar(self.x, self.y, xerr=self.sigma_x, yerr=self.sigma_y,
                     fmt='o', label='Dati', markersize=7, capsize=4)
        

        x_fit = np.linspace(np.min(self.x), np.max(self.x), 500)

        params_dict = {name: self.fit_result[name][0] for name in self.param_names}
        y_fit = self.model(x_fit, **params_dict)
        plt.plot(x_fit, y_fit, '-r', label='Fit (ODR)', linewidth=2.5)
        
        plt.xlabel(self.xlabel, fontsize=label_fontsize)
        plt.ylabel(self.ylabel, fontsize=label_fontsize)
        plt.title(self.title, fontsize=title_fontsize, pad=20)
        
        # Box con informazioni sui risultati
        text_lines = [f"${name} = {val:.2e} \\pm {err:.2e}$" 
                      for name, (val, err) in self.fit_result.items()]
        dof = len(self.x) - len(self.param_names)
        chi2_red = self.odr_output.sum_square / dof
        text_lines.append(f"$\\chi^2/NdoF = {chi2_red:.3f}$")
        text = "\n".join(text_lines)
        plt.annotate(text, 
                     xy=(0.55, 0.95), 
                     xycoords='axes fraction',
                     va='top', 
                     ha='left', 
                     bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.7', edgecolor='gray'),
                     fontsize=13, 
                     linespacing=1.5)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


"""
Funzione per fittare e plottare utilizzando i minimi quadrati
ESEMPIO:
x_data = I_bobina
y_data = array_funz
sigma_y = sigma_ang

# Configurazione del fit
data_dict = {
    'x': x_data,
    'y': y_data,
    'sigma_y': sigma_y
}

initial_params = {
    'Be': 1e-8,
    'b': 0.0
}

# Esegui il fit
fit = FitMinimiQuadrati(
    bobina,
    data_dict,
    initial_params,
    xlabel="Corrente (A)",
    ylabel="sos (V)",
    title="Livio Laido"
)
fit.perform_fit()
fit.print_results()
fit.plot_results(
    title_fontsize=16,
    label_fontsize=12
)
"""

class FitMinimiQuadrati:
    def __init__(self, model_func, data_arrays, initial_params, xlabel="x", ylabel="y", title="Risultati del fit"):
        """
        model_func: funzione Python che prende x e i parametri
        data_arrays: dizionario con 'x', 'y', 'sigma_y'
        initial_params: dizionario con parametri e valori iniziali
        """
        self.model = model_func
        self.x = data_arrays['x']
        self.y = data_arrays['y']
        self.sigma = data_arrays.get('sigma_y', np.ones_like(self.y))
        
        # Estrae i nomi dei parametri dalla firma della funzione
        sig = inspect.signature(model_func)
        self.param_names = list(sig.parameters.keys())[1:]  # Esclude il primo parametro (x)
        
        self.initial_params = initial_params
        self.fit_result = None
                
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

        self._validate_inputs()
        
    def _validate_inputs(self):
        if len(self.x) != len(self.y) or len(self.y) != len(self.sigma):
            raise ValueError("Tutti gli array devono avere la stessa lunghezza")
            
        if not all(p in self.param_names for p in self.initial_params.keys()):
            raise ValueError("Nomi parametri non corrispondenti alla funzione")
    
    def perform_fit(self):
        least_squares = LeastSquares(self.x, self.y, self.sigma, self.model) 
        self.m = Minuit(least_squares, **self.initial_params)
        self.m.migrad()
        
        # Memorizza i risultati
        self.fit_result = {name: (self.m.values[name], self.m.errors[name]) 
                          for name in self.param_names}
        
        return self.m
    
    def print_results(self):
        print(self.m.valid)
        print("\nRisultati del fit:")
        for name in self.param_names:
            val, err = self.fit_result[name]
            print(f"{name} = {val:.3e} ± {err:.3e}")
        
        chi2_val = self.m.fval
        dof = len(self.x) - len(self.param_names)
        print(f"\nChi-quadro ridotto: {chi2_val/dof:.3f}")
        print(f"gradi di libertà: {dof:.3f}")
        print(f"p-value: {1 - chi2.cdf(chi2_val, dof):.3f}")
        
    def plot_results(self, title_fontsize=14, label_fontsize=12):
        plt.figure(figsize=(10, 6))
    
        plt.errorbar(self.x, self.y, yerr=self.sigma, fmt='o', label='Dati', markersize=7, capsize=4)
        params_dict = {name: value for name, value in zip(self.param_names, self.m.values)}  
        x_fit = np.linspace(self.x.min(), self.x.max(), 500)
        y_fit = self.model(x_fit, **params_dict)
    
        plt.plot(x_fit, y_fit, '-r', label='Fit', linewidth=2.5)
        plt.xlabel("x", fontsize=12)
        plt.ylabel("y", fontsize=12)
        plt.xlabel(self.xlabel, fontsize=label_fontsize)
        plt.ylabel(self.ylabel, fontsize=label_fontsize)
        plt.title(self.title, fontsize=title_fontsize, pad=20)
    
        # Box informazioni
        text = "\n".join([f"${n} = {v:.2e} \\pm {e:.2e}$" 
                    for n, (v, e) in self.fit_result.items()])
        text += f"\n$\\chi^2/NdoF = {self.m.fval/(len(self.x)-len(self.param_names)):.3f}$"
        plt.annotate(text, 
                xy=(0.05, 0.95), 
                xycoords='axes fraction',
                va='top', 
                ha='left', 
                bbox=dict(
                    facecolor='white',
                    alpha=0.9,
                    boxstyle='round,pad=0.7',  
                    edgecolor='gray'
                ),
                fontsize=13,  
                linespacing=1.5)  
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()  
        plt.show()


"""
Funzione per fare il fit con il chi2
"""

class FitChi2:
    def __init__(self, model_func, data_arrays, initial_params, xlabel="x", ylabel="y", title="Risultati del fit"):
        """
        model_func: funzione Python che prende x e i parametri
        data_arrays: dizionario con 'x', 'y', 'sigma_y'
        initial_params: dizionario con parametri e valori iniziali
        """
        self.model = model_func
        self.x = data_arrays['x']
        self.y = data_arrays['y']
        self.sigma = data_arrays.get('sigma_y', np.ones_like(self.y))
        
        # Estrae i nomi dei parametri dalla firma della funzione
        sig = inspect.signature(model_func)
        self.param_names = list(sig.parameters.keys())[1:]  # Esclude il primo parametro (x)
        
        self.initial_params = initial_params
        self.fit_result = None
                
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

        #print("Parametri rilevati nella funzione:", self.param_names)
        #print("Parametri forniti:", initial_params.keys()) 

        self._validate_inputs()
        
    def _validate_inputs(self):
        if len(self.x) != len(self.y) or len(self.y) != len(self.sigma):
            raise ValueError("Tutti gli array devono avere la stessa lunghezza")
            
        if not all(p in self.param_names for p in self.initial_params.keys()):
            raise ValueError("Nomi parametri non corrispondenti alla funzione")
        
        missing = set(self.param_names) - set(self.initial_params.keys())
        if missing:
          raise ValueError(f"Parametri mancanti: {missing}")
    
    def chi2_function(self, *args):
        params = {name: val for name, val in zip(self.param_names, args)}
        y_model = self.model(self.x, **params)
        return np.sum(((self.y - y_model) / self.sigma)**2)

    def perform_fit(self):       
        self.m = Minuit(self.chi2_function, *self.initial_params.values())
        self.m.errordef = 1.0
        self.m.migrad()
        
        # Memorizza i risultati
        self.fit_result = {name: (self.m.values[i], self.m.errors[i]) 
                          for i, name in enumerate(self.param_names)}
        
        return self.m
    
    def print_results(self):
        print(self.m.valid)
        print("\nRisultati del fit:")
        for name in self.param_names:
            val, err = self.fit_result[name]
            print(f"{name} = {val:.3e} ± {err:.3e}")
        
        chi2_val = self.m.fval
        dof = len(self.x) - len(self.param_names)
        print(f"\nChi-quadro ridotto: {chi2_val/dof:.3f}")
        print(f"gradi di libertà: {dof:.3f}")
        print(f"p-value: {1 - chi2.cdf(chi2_val, dof):.3f}")
        
    def plot_results(self, title_fontsize=14, label_fontsize=12):
        plt.figure(figsize=(10, 6))
    
        plt.errorbar(self.x, self.y, yerr=self.sigma, fmt='o', label='Dati', markersize=7, capsize=4)
        params_dict = {name: value for name, value in zip(self.param_names, self.m.values)}  
        x_fit = np.linspace(self.x.min(), self.x.max(), 1000)
        y_fit = self.model(x_fit, **params_dict)
    
        plt.plot(x_fit, y_fit, '-r', label='Fit', linewidth=2.5)
        plt.xlabel("x", fontsize=12)
        plt.ylabel("y", fontsize=12)
        plt.xlabel(self.xlabel, fontsize=label_fontsize)
        plt.ylabel(self.ylabel, fontsize=label_fontsize)
        plt.title(self.title, fontsize=title_fontsize, pad=20)
    
        # Box informazioni
        text = "\n".join([f"${n} = {v:.2e} \\pm {e:.2e}$" 
                    for n, (v, e) in self.fit_result.items()])
        text += f"\n$\\chi^2/NdoF = {self.m.fval/(len(self.x)-len(self.param_names)):.3f}$"
        plt.annotate(text, 
                     xy=(0.05, 0.95), 
                     xycoords='axes fraction',
                     va='top', 
                     ha='left', 
                     bbox=dict(
                              facecolor='white',
                              alpha=0.9,
                              boxstyle='round,pad=0.7',  
                              edgecolor='gray'
                              ),
                     fontsize=13,  
                     linespacing=1.5)  
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()  
        plt.show()



class FitChi2_0:
    def __init__(self, model_func, data_arrays, initial_params, xlabel="x", ylabel="y", title="Risultati del fit"):
        """
        model_func: funzione Python che prende x e i parametri
        data_arrays: dizionario con 'x', 'y', 'sigma_y'
        initial_params: dizionario con parametri e valori iniziali
        """
        self.model = model_func
        self.x = data_arrays['x']
        self.y = data_arrays['y']
        self.sigma = data_arrays.get('sigma_y', np.ones_like(self.y))
        
        # Estrae i nomi dei parametri dalla firma della funzione
        sig = inspect.signature(model_func)
        self.param_names = list(sig.parameters.keys())[1:]  # Esclude il primo parametro (x)
        
        self.initial_params = initial_params
        self.fit_result = None
                
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

        #print("Parametri rilevati nella funzione:", self.param_names)
        #print("Parametri forniti:", initial_params.keys()) 

        self._validate_inputs()
        
    def _validate_inputs(self):
        if len(self.x) != len(self.y) or len(self.y) != len(self.sigma):
            raise ValueError("Tutti gli array devono avere la stessa lunghezza")
            
        if not all(p in self.param_names for p in self.initial_params.keys()):
            raise ValueError("Nomi parametri non corrispondenti alla funzione")
        
        missing = set(self.param_names) - set(self.initial_params.keys())
        if missing:
          raise ValueError(f"Parametri mancanti: {missing}")
    
    def chi2_function(self, *args):
        params = {name: val for name, val in zip(self.param_names, args)}
        y_model = self.model(self.x, **params)
        return np.sum(((self.y - y_model) / self.sigma)**2)

    def perform_fit(self):       
        self.m = Minuit(self.chi2_function, *self.initial_params.values())
        self.m.errordef = 1.0
        self.m.migrad()
        
        # Memorizza i risultati
        self.fit_result = {name: (self.m.values[i], self.m.errors[i]) 
                          for i, name in enumerate(self.param_names)}
        
        return self.m
    
    def print_results(self):
        print(self.m.valid)
        print("\nRisultati del fit:")
        for name in self.param_names:
            val, err = self.fit_result[name]
            print(f"{name} = {val:.3e} ± {err:.3e}")
        
        chi2_val = self.m.fval
        dof = len(self.x) - len(self.param_names)
        print(f"\nChi-quadro ridotto: {chi2_val/dof:.3f}")
        print(f"gradi di libertà: {dof:.3f}")
        print(f"p-value: {1 - chi2.cdf(chi2_val, dof):.3f}")
        
    def plot_results(self, title_fontsize=14, label_fontsize=12):
        plt.figure(figsize=(10, 6))
    
        plt.errorbar(self.x, self.y, yerr=self.sigma, fmt='o', label='Dati', markersize=7, capsize=4)
        params_dict = {name: value for name, value in zip(self.param_names, self.m.values)}  
        x_fit = np.linspace(self.x.min(), self.x.max(), 1000)
        y_fit = self.model(x_fit, **params_dict)
    
        plt.plot(x_fit, y_fit, '-r', label='Fit', linewidth=2.5)
        plt.xlabel("x", fontsize=12)
        plt.ylabel("y", fontsize=12)
        plt.xlabel(self.xlabel, fontsize=label_fontsize)
        plt.ylabel(self.ylabel, fontsize=label_fontsize)
        plt.title(self.title, fontsize=title_fontsize, pad=20)
    
        # Box informazioni
        text = "\n".join([f"${n} = {v:.2e} \\pm {e:.2e}$" 
                    for n, (v, e) in self.fit_result.items()])
        text += f"\n$\\chi^2/NdoF = {self.m.fval/(len(self.x)-len(self.param_names)):.3f}$"
        plt.annotate(text, 
                     xy=(0.55, 0.95), 
                     xycoords='axes fraction',
                     va='top', 
                     ha='left', 
                     bbox=dict(
                              facecolor='white',
                              alpha=0.9,
                              boxstyle='round,pad=0.7',  
                              edgecolor='gray'
                              ),
                     fontsize=13,  
                     linespacing=1.5)  
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()  
        plt.show()



class FitScipy:
    def __init__(self, model_func, data_arrays, initial_params, xlabel="x", ylabel="y", title="Risultati del fit"):
        """
        model_func: funzione Python che prende x e i parametri come argomenti posizionali
        data_arrays: dizionario con 'x', 'y', 'sigma_y'
        initial_params: dizionario con parametri e valori iniziali
        """
        self.model = model_func
        self.x = data_arrays['x']
        self.y = data_arrays['y']
        self.sigma = data_arrays.get('sigma_y', np.ones_like(self.y))
        
        # Estrae i nomi dei parametri dalla firma della funzione
        sig = inspect.signature(model_func)
        self.param_names = list(sig.parameters.keys())[1:]  # Esclude il primo parametro (x)
        
        # Mappatura parametri -> ordine per curve_fit
        self.initial_params_list = [initial_params[name] for name in self.param_names]
        
        self.fit_result = None
        self.cov_matrix = None
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

    def perform_fit(self):
        """Esegue il fit usando curve_fit"""
        popt, pcov = curve_fit(
            self.model,
            self.x,
            self.y,
            p0=self.initial_params_list,
            sigma=self.sigma,
            absolute_sigma=True
        )
        
        self.fit_result = {name: (val, np.sqrt(pcov[i,i])) 
                          for i, (name, val) in enumerate(zip(self.param_names, popt))}
        self.cov_matrix = pcov

        residuals = self.y - self.model(self.x, *popt)
        self.chi2_val = np.sum((residuals / self.sigma)**2)
        self.dof = len(self.x) - len(popt)
        
    def print_results(self):
        """Stampa i risultati del fit in formato leggibile"""
        print("\nRisultati del fit:")
        for name in self.param_names:
            val, err = self.fit_result[name]
            print(f"{name} = {val:.3e} ± {err:.3e}")
        
        print(f"\nChi-quadro ridotto: {self.chi2_val/self.dof:.3f}")
        print(f"Gradi di libertà: {self.dof}")
        print(f"p-value: {1 - chi2.cdf(self.chi2_val, self.dof):.3f}")

    def plot_results(self, title_fontsize=14, label_fontsize=12):
        """Genera il plot dei risultati"""
        plt.figure(figsize=(10, 6))
        plt.errorbar(self.x, self.y, yerr=self.sigma, fmt='o', label='Dati', markersize=7, capsize=4)

        x_fit = np.linspace(self.x.min(), self.x.max(), 1000)
        params = [self.fit_result[name][0] for name in self.param_names]
        y_fit = self.model(x_fit, *params)
        
        plt.plot(x_fit, y_fit, '-r', label='Fit', linewidth=2.5)
        plt.xlabel(self.xlabel, fontsize=label_fontsize)
        plt.ylabel(self.ylabel, fontsize=label_fontsize)
        plt.title(self.title, fontsize=title_fontsize, pad=20)
        
        # Box informazioni
        text = "\n".join([f"${n} = {v:.2e} \\pm {e:.2e}$" 
                    for n, (v, e) in self.fit_result.items()])
        text += f"\n$\\chi^2/N_{{doF}} = {self.chi2_val/self.dof:.3f}$"
        
        plt.annotate(text, 
                     xy=(0.05, 0.95), 
                     xycoords='axes fraction',
                     va='top', 
                     ha='left', 
                     bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.7'),
                     fontsize=13)
        
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()




class FitScipy2_0:
    def __init__(self, model_func, data_arrays, initial_params, xlabel="x", ylabel="y", title="Risultati del fit"):
        """
        model_func: funzione Python che prende x e i parametri come argomenti posizionali
        data_arrays: dizionario con 'x', 'y', 'sigma_y'
        initial_params: dizionario con parametri e valori iniziali
        """
        self.model = model_func
        self.x = data_arrays['x']
        self.y = data_arrays['y']
        self.sigma = data_arrays.get('sigma_y', np.ones_like(self.y))
        
        # Estrae i nomi dei parametri dalla firma della funzione
        sig = inspect.signature(model_func)
        self.param_names = list(sig.parameters.keys())[1:]  # Esclude il primo parametro (x)
        
        self.initial_params_list = [initial_params[name] for name in self.param_names]
        
        self.fit_result = None
        self.cov_matrix = None
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

    def perform_fit(self):
        """Esegue il fit usando curve_fit"""
        popt, pcov = curve_fit(
            self.model,
            self.x,
            self.y,
            p0=self.initial_params_list,
            sigma=self.sigma,
            absolute_sigma=True
        )
        
        self.fit_result = {name: (val, np.sqrt(pcov[i,i])) 
                          for i, (name, val) in enumerate(zip(self.param_names, popt))}
        self.cov_matrix = pcov
        
        # Calcola chi2 ridotto
        residuals = self.y - self.model(self.x, *popt)
        self.chi2_val = np.sum((residuals / self.sigma)**2)
        self.dof = len(self.x) - len(popt)
        
    def print_results(self):
        """Stampa i risultati del fit in formato leggibile"""
        print("\nRisultati del fit:")
        for name in self.param_names:
            val, err = self.fit_result[name]
            print(f"{name} = {val:.3e} ± {err:.3e}")
        
        print(f"\nChi-quadro ridotto: {self.chi2_val/self.dof:.3f}")
        print(f"Gradi di libertà: {self.dof}")
        print(f"p-value: {1 - chi2.cdf(self.chi2_val, self.dof):.3f}")

    def plot_results(self, title_fontsize=14, label_fontsize=12):
        """Genera il plot dei risultati"""
        plt.figure(figsize=(10, 6))
        plt.errorbar(self.x, self.y, yerr=self.sigma, fmt='o', label='Dati', markersize=7, capsize=4)
        
        # Genera curva di fit
        x_fit = np.linspace(self.x.min(), self.x.max(), 1000)
        params = [self.fit_result[name][0] for name in self.param_names]
        y_fit = self.model(x_fit, *params)
        
        plt.plot(x_fit, y_fit, '-r', label='Fit', linewidth=2.5)
        plt.xlabel(self.xlabel, fontsize=label_fontsize)
        plt.ylabel(self.ylabel, fontsize=label_fontsize)
        plt.title(self.title, fontsize=title_fontsize, pad=20)
        
        # Box informazioni
        text = "\n".join([f"${n} = {v:.2e} \\pm {e:.2e}$" 
                    for n, (v, e) in self.fit_result.items()])
        text += f"\n$\\chi^2/N_{{doF}} = {self.chi2_val/self.dof:.3f}$"
        
        plt.annotate(text, 
                     xy=(0.55, 0.95), 
                     xycoords='axes fraction',
                     va='top', 
                     ha='left', 
                     bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.7'),
                     fontsize=13)
        
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()







"""
Formule per trovare zeri, massimi e minimi e integrali per via numerica"""

def bisezione_iterativa(f, a, b, tolleranza=1e-6, max_iter=100):
    """
    Bisezione usa il metodo di bisezione per trovare lo zero di funzione sfruttando 
    il teorema degli zeri
    Argomenti = la funzione, gli estremi 'a' e 'b', la tolleranza che posso impostare

    Return = valore medio dell'interallo finale
    """
    if f(a) * f(b) >= 0:
        raise ValueError("La funzione deve avere segni opposti ai due estremi dell'intervallo [a, b].")
    iterazioni = 0
    while (b - a) / 2 > tolleranza and iterazioni < max_iter:
        c = (a + b) / 2  
        if f(c) == 0:    #abbiamo trovato lo zero esatto
            return c
        elif f(a) * f(c) < 0:
            b = c       #lo zero è nell'intervallo [a, c]
        else:
            a = c       #lo zero è nell'intervallo [c, b]
        iterazioni += 1
    return (a + b) / 2 
    

def bisezione_ricorsiva(f, a, b, tolleranza=1e-6):
    """
    Metodo ricorsivo per trovare lo zero di una funzione usando il metodo di bisezione.
    Lancia un'eccezione se l'intervallo fornito non contiene uno zero.
    """
    if f(a) * f(b) > 0:
        raise ValueError("L'intervallo fornito non contiene uno zero (segni uguali ai due estremi).")
    c = (a + b) / 2  # Punto medio
    if abs(f(c)) < tolleranza or abs(b - a) / 2 < tolleranza:
        return c
    if f(a) * f(c) < 0:
        if f(a) * f(b) > 0:
            raise ValueError("Intervallo non valido durante la ricorsione.")
        return bisezione_ricorsiva(f, a, c, tolleranza)
    else:
        if f(a) * f(b) > 0:
            raise ValueError("Intervallo non valido durante la ricorsione.")
        return bisezione_ricorsiva(f, c, b, tolleranza)


def minimi_sezioneAurea(f, a, b, tolleranza=1e-6):
  phi = (1 + np.sqrt(5)) / 2
  x1 = a + (b-a)/(phi**2)
  x2 = a + (b-a)/(phi)
  if (b-a) < tolleranza:
    min = (a + b) / 2
    minf = f(min)
    return [min, minf]
  elif f(x1) < f(x2):
    return minimi_sezioneAurea(f, a, x2, tolleranza)
  else:
    return minimi_sezioneAurea(f, x1, b, tolleranza)


def massimi_sezioneAurea(f, a, b, tolleranza=1e-6):
    phi = (1 + np.sqrt(5)) / 2
    x1 = a + (b-a)/(phi**2)
    x2 = a + (b-a)/(phi)
    if (b-a) < tolleranza:
        max = (a + b) / 2
        maxf = f(max)
        return [max, maxf]
    elif f(x1) > f(x2):
        return massimi_sezioneAurea(f, a, x2, tolleranza)
    else:
        return massimi_sezioneAurea(f, x1, b, tolleranza)


def hit_or_miss_integrate(f, a, b, m, M, N):
    """
    Argomenti: funzione f, [a, b] base rettangolo, [m, M] altezza rettangolo,
    N numero di x e y generati
    """
    x = np.random.uniform(a, b, N)
    y = np.random.uniform(m, M, N)
    hits = 0
    for i in range(N):
        if y[i] <= f(x[i]):
            hits += 1
    successi = np.sum(hits)
    A_rect = (b - a) * (M-m)
    p = successi / N
    integral = p * A_rect
    std_integral = np.sqrt(((A_rect**2) / N) * p * (1 - p))
    return integral, std_integral 


def crude_montecarlo_integrate(f, a, b, N):
    """
    Montecarlo Method stima numericamente l'integrale sfruttando E[f(x)]
    Argomenti: la funzione, gli estremi e il numero di elementi N
    """
    x = np.linspace(a, b, N)
    f_values = f(x)  
    mean = np.mean(f_values)
    std_dev = np.std(f_values)
    integral = (b - a) * mean
    err = (b - a) * std_dev / np.sqrt(N)
    return integral, err


def MC_bidimensionale(f, xmin, xmax, ymin, ymax, M, N):
  """
  La funzione calcola un integrale di una funzione in due variabili
  """
  x = np.random.uniform(xmin, xmax, N)
  y = np.random.uniform(ymin, ymax, N)
  z = np.random.uniform(0, M, N)
  hits = z <= f(x, y)
  successi = np.sum(hits)
  A_rect = (xmax - xmin) * (ymax - ymin) * M
  p = successi / N
  integral = p * A_rect
  std_integral = np.sqrt(((A_rect**2) / N) * p * (1 - p))
  return integral, std_integral


"""
Funzioni per generare numeri pseudo casuali
"""

def generatore_casual_pdf(pdf, n, xmin, xmax, N):
	"""
  	Questa funzione prende in ingresso: una pdf qualunque, 
	massimo e minimo del dominio delle x e quanto voglio grande questo dominio N
	e infine quanti numeri n piccolo voglio generare"""
	
	x = np.linspace(xmin, xmax, N)
	pdf_values = pdf(x)
	pdf_values /= np.trapz(pdf_values) #normalizzo la pdf con l'integrale

	cdf = np.cumsum(pdf_values)
	cdf /= cdf[-1]  

	random_numbers = np.random.uniform(0, 1, size=n)
	samples = np.interp(random_numbers, cdf, x)  #interp mi trasforma i dati con la cdf
	return samples, x, pdf_values


def TCL(media, sigma, Neventi):                
    """
    Funzione che genera eventi gaussiani con il TCL conosciuti media e sigma
    """
    risultati = [] 
    for i in range(Neventi):
        eventi = 0
        delta = np.sqrt(3*Neventi)*sigma
        xmin = media - delta
        xmax = media + delta
        for i in range (Neventi):
            eventi += np.random.uniform(xmin, xmax)
        eventi /= Neventi
        risultati.append(eventi) 
    return np.array(risultati) 


def inversa_exponential(t0, N):
	"""
	Funzione che mi genera N numeri distribuiti esponenzialmente con lambda=1/t0
	"""
	u = np.random.uniform(0, 1, N)
	f = -t0 * np.log(1 - u)
	return f


def generatore_poisson(t, N):                   
	#il contatore mi restituisce un numero che è sostanzialmente per quante volte moltiplico p per un numero casuale
	risultati = []
	for i in range(N):
		contatore = 0
		p = 1
		while p > np.exp(-t):
			contatore += 1
			p *= np.random.uniform(0, 1)     	#p mi accumula il prodotto di numeri casuali fino a quando questo non è minore del tempo tra eventi successivi
                                          			#di una distribuzione esponenziale, quindi sostanzialmente il numero di eventi tra due eventi successivi
		risultati.append(contatore - 1)
	return np.array(risultati)


def generatore_gaussiani(Ntoy, Neventi):      
    """
    Argomenti:
    - Ntoy (int): Numero di simulazioni da generare.
    - Neventi (int): Numero di eventi per ogni simulazione.

    Ritorna:
    - medie (list): Lista delle medie calcolate per ogni simulazione.
    - sigma (list): Lista delle stime dell'errore sulla media per ogni simulazione.
    """
    medie = []
    for _ in range(Ntoy):
        eventi = np.random.uniform(0, 1, Neventi)  # Genera eventi uniformi
        toy_stats = np.mean(eventi)               # Calcola la media
        medie.append(toy_stats)                   # Aggiungi alla lista delle medie

    sigma_valore = np.std(np.random.uniform(0, 1, Neventi)) / np.sqrt(Neventi)
    sigma = [sigma_valore] * Ntoy  # Stima costante per ogni toy

    return medie, sigma


def hit_or_miss_generator(f, a, b, m, M, N):
  """
  Argomenti: 
  f = funzione, [a, b] = estremi della base del rettangolo, 
  [m, M] = estremi altezza del rettangolo (spesso m = 0), N = numeri da generare
  """
  sample = []
  hits = 0
  x_values = np.random.uniform(a, b, N)
  y_values = np.random.uniform(m, M, N)
  for i in range(N):
    if y_values[i] <= f(x_values[i]):
      sample.append(x_values[i])
      hits += 1
  Area_rettangolo = (hits/N)*(b-a)*(M-m)
  return np.array(sample), Area_rettangolo


def try_and_catch_generator(f, a, b, M, N):
    """
    Argomenti: 
    f: la pdf, funzione da campionare
    a, b: insieme di generazione (limiti di campionamento)
    M: valore tale che la pdf è minore di M (fattore di normalizzazione)
    N: numero di eventi da generare

    Return: sample di eventi generati tramite il metodo Try-and-Catch
    """
    sample = []
    attempts = 0  
    while len(sample) < N:  
        x = np.random.uniform(a, b)  
        u = np.random.uniform(0, 1)  
        if u <= f(x)/M:  
            sample.append(x)  
        attempts += 1
        if attempts >= 100000:  # Massimo numero di tentativi
            print("Maximum attempts reached!")
            break
    return np.array(sample)


def try_and_catch_generator_v2(f, a, b, M, N):
  """
  Argomenti: pdf f, (a, b) insieme di generazione, M=valore tale per cui
  f è minore di M, N=numero eventi da generare

  Return: sample di eventi
  """
  sample = []
  x = np.random.uniform(a, b, N)
  u = np.random.uniform(0, 1, N)
  for i in range(N):
    if u[i] <= f(x[i])/M:
      sample.append(x[i])
  return sample


def TCL_pdf(f, a, b, Neventi):
  """
  Argomenti: una pdf f, (a,b) intervallo di generazione
  """
  sample = []
  n = 10
  for i in range(Neventi):
    media = np.mean(mr.generatore_casual_pdf(f, n, a, b, n))
    sample.append(media)
  return np.array(sample)


def generate_gauss_Box_Muller(Neventi, mu, sigma):
  """
  La funzione genera Neventi gaussiani con media mu e dev sigma
  """
  sample = []
  g1_tot = []
  g2_tot = []
  transform = lambda x : x * sigma + mu 
  #Moltiplicare per sigma mi trasforma la larghezza da 1 a sigma e aggiungo mu per traslare
  for i in range(N):
    x1 = np.random.uniform(0, 1)
    x2 = np.random.uniform(0, 1)
    p = np.sqrt(-2*np.log(x1))
    g1 = p*np.cos(2*np.pi*x2)
    g2 = p*np.sin(2*np.pi*x2)
    g1_tot.append(transform(g1))
    g2_tot.append(transform(g2))
  sample = np.concatenate([g1_tot, g2_tot])
  return sample


def generate_gauss_bm(Neventi):
  """
  La funzione genera Neventi gaussiani
  """
  sample = []
  g1_tot = []
  g2_tot = []
  for i in range(N):
    x1 = np.random.uniform(0, 1)
    x2 = np.random.uniform(0, 1)
    p = np.sqrt(-2*np.log(x1))
    g1 = p*np.cos(2*np.pi*x2)
    g2 = p*np.sin(2*np.pi*x2)
    g1_tot.append(g1)
    g2_tot.append(g2)
  sample = np.concatenate([g1_tot, g2_tot])
  return sample

def uniform(media, sigma, N):
    
    l = sigma*(np.sqrt(12))         
    a = media - l/2           
    b = media + l/2
    x = np.random.uniform(a, b, N)
    return x

def sturges (n):		#STURGES PER IL NUMERO DI BIN
	return int(np.ceil(1+3.322*np.log(n)))


def two_data_binning(data1, data2):
	"""Questa funzione, dati due set di dati diversi, mi trova
	il binning ottimale per plottare l'istogramma"""

	xMin = floor (min (min (data1), min (data2)))
	xMax = ceil (max (max (data1), max (data2)))
	N_bins = sturges (min (len (data1), len (data2)))
	bin_edges = np.linspace (xMin, xMax, N_bins)
	return bin_edges


def one_data_binning(data):
  	"""
	Preso un array di dati, mi produce i binedges e il bin_content;
  	Se voglio i primi N elementi, devo fare data_new = data[:N]
  	"""
  	N = len(data)
  	Nbins = sturges(N)
  	binnaggio, binedges = np.histogram(data, bins = Nbins, range = (min(data), max(data)))
  	return binnaggio, binedges

def loglikelihood(pdf, theta, sample):
    """
    Calcola il logaritmo della likelihood dato un parametro theta (es. media) e un set di dati (sample).
    Argomenti:
      - pdf: funzione di densità di probabilità.
      - theta: parametro singolo (es. media).
      - sample: array di dati osservati.
    Ritorna:
      - log-likelihood (float).
    """
    logL = 0
    for x in sample:
      if (pdf(x, theta)) > 0:
        logL = logL + np.log(pdf(x, theta))
      else:
        raise ValueError(f"PDF value is zero or negative at x={x}, theta={theta}.")
    return logL


def loglikelihood(pdf, theta, sample):    #Logaritmo della likelihood
	risultato = 0
	for x in sample:
		if (pdf(x, theta) > 0):
			risultato += np.log(pdf(x, theta))
	return (risultato)


def maximumlikelihood(loglikelihood, pdf, sample, a, b, tolleranza=1e-6):
	phi = (1 + np.sqrt(5)) / 2
	x1 = a + (b - a) / phi**2
	x2 = a + (b - a) / phi

	while (b - a) > tolleranza:
		L1 = loglikelihood(pdf, x1, sample)
		L2 = loglikelihood(pdf, x2, sample)
		if L1 > L2:
			b = x2
			x2 = x1
			x1 = a + (b - a) / phi**2
		else:
			a = x1
			x1 = x2
			x2 = a + (b - a) / phi
	x_max = (a + b) / 2
	return x_max

# Calcola integrale [Scipy]
def integral_scipy(f, a, b) : 
  integral = quad(f, a,b)
  return integral[0], integral[1]

def media_pesata(x, sigma) :
  m = np.sum(x/sigma**2)/np.sum(1/sigma**2)
  sigma_m = 1/np.sqrt(np.sum(1/sigma**2))
  return m, sigma_m

# PDF's & CDF's
def Gaussian(x, mu = 0, sigma = 1) :
	return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((x - mu)**2) / (2 * sigma**2))

# Gaussiana standardizzata
def Gaussian_standard(z):
  return (1/np.sqrt(2*np.pi))*np.exp((-z**2)/2) 

def Gaussian_cdf_ext(bin_edges, s, mu, sigma) :
  return s*norm.cdf(bin_edges, mu, sigma)

def Gaussian_cdf(bin_edges, mu, sigma) :
  return norm.cdf(bin_edges, mu, sigma)


# HYPOTHESIS TESTING 
def p_value(chi_square, x, ndof) :
  s = 1-chi2.cdf(chi_square, len(x)-ndof)
  r = s*100
  return r

# z_test double sided
def z_test1(x1,x2,s1,s2) : 
  z = np.absolute(x1-x2)/np.sqrt(s1**2+s2**2)  #t di confronto
  R = quad(Gaussian_standard,-t,t) #calcolo del rapporto con l'integrale
  p_value = (1 - R[0])
  return p_value

# z test di ipotesi con un valore calcolato
def z_test2(x1,X,s) :  
  z = np.absolute(x1-X)/s  #t di confronto
  R = quad(Gaussian_standard,-t,t) #calcolo del rapporto con l'integrale
  p_value = (1 - R[0])
  return p_value

# t test con 1 vincolo
def t_test1(x1, X, err_media) :  
	t = np.absolute(x1-X)/err_media
	R = t.cdf(-t, df=len(x1)-1)
	p_value = R*2
	return p_value
	
