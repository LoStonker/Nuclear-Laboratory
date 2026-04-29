"""
mc_compton.py  

Per spostarsi nella cartella giusta: cd Desktop/"lab nucleare"
==========================
Simulazione Monte Carlo dell'effetto Compton in coincidenze, con sorgente Na-22 e rivelatori NaI.

==================
Funzioni di plot per mc_compton.py.
Funzioni disponibili:
  - plot_diagnostico_kn         : verifica generatore Klein-Nishina
  - plot_cinematica_compton     : E'(θ) teorica vs MC
  - plot_coincidenze_compton    : conteggi MC vs angolo (+ accidentali)
  - plot_hit_bersaglio          : mappa 2D dei punti di impatto sul Pb
  - plot_energia_rivelatore_B   : spettro E_out per il rivelatore B
  - plot_summary                : pannello 2×2 riassuntivo
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse


# ══════════════════════════════════════════════════════════════════════════════
#  PARAMETRI GEOMETRICI  
# ══════════════════════════════════════════════════════════════════════════════

# --- Geometria rivelatori ---
D_trigger = 18.0          # [cm]  distanza sorgente → faccia cristallo A (fisso)
D_B = 36.0          # [cm]  distanza sorgente → faccia cristallo B (mobile)
R_trigger = 2.54          # [cm]  raggio faccia cristallo A   (1"  ≈ 2.54 cm)
R_B = 2*2.54          # [cm]  raggio faccia cristallo B
L_A = 2.50          # [cm]  profondità (spessore) cristallo A  (3" tipico NaI)
L_B = 5.50          # [cm]  profondità cristallo B
H_pb = 3               # [cm] altezza blocchetto piombo
D_pb = 14           # [cm] distanza tra la sorgente e il piombo
W_pb = 2            # [cm] dimensione laterale piombo (cioè distanza orizzontale)
a_pb = 3            # [cm] altra dimensione del piombo

# --- sorgente estesa ---
R_SOURCE = 0.1      # [cm]  raggio del dischetto sorgente  (1 mm = 0.1 cm)

# --- NaI a 511 keV ---
# μ_tot(NaI, 511 keV) = (μ/ρ) × ρ = 0.0952 cm²/g × 3.67 g/cm³ ≈ 0.349 cm⁻¹
# Fonte: NIST XCOM per NaI a 511 keV
MU_NAI = 0.349      # [cm⁻¹]

# --- aria a 511 keV ---
# μ_aria(511 keV) = (μ/ρ) × ρ_aria = 0.0862 cm²/g × 1.293e-3 g/cm³ = 1.115e-4 cm⁻¹
MU_AIR = 1.115e-4   # [cm⁻¹]  — piccolo ma non zero a ~20 cm

# --- coincidenze accidentali ---
ACTIVITY_BQ  = (10*(10**-6))*(3.7*10**10)  # [Bq]   attività sorgente (tipica Na-22 da lab: 1–100 kBq)
TAU_GATE_S   = (3.64)*10**-6   # [s]    larghezza gate di coincidenza (100 ns tipico)
T_ACQ_S      = 3600.0     # [s]    durata acquisizione per angolo

# --- Simulazione ---
N_EVENTS    = 50_000_000
ANGOLI_DEG  = np.arange(160, 181, 1)


# ══════════════════════════════════════════════════════════════════════════════
#  COSTANTI FISICHE
E0_KEV   = 511.0     # keV — energia fotone Na-22
ME_C2    = 511.0     # keV — massa elettrone
THETA_TH = np.linspace(0, np.pi, 1000)

# Flags per attivare/disattivare ogni feature
USE_SOURCE_EXTENT    = True
USE_INTRINSIC_EFF    = True
USE_AIR_ATTENUATION  = True
USE_ACCIDENTALS      = True

"""
Struttura del np.where: risultato = np.where(CONDIZIONE, COSA_FARE_SE_VERO, COSA_FARE_SE_FALSO) 
"""


# ══════════════════════════════════════════════════════════════════════════════
#  GEOMETRIA E FISICA
# ══════════════════════════════════════════════════════════════════════════════

def genera_fotoni_isotropi(N):
    """
    Genera N fotoni emessi in modo isotropo da una sorgente puntiforme al centro del sistema di riferimento
    Non usiamo una distribuzione uniforme in θ, ma campioniamo uniformemente in cos(θ) per garantire isotropia reale, altrimenti verrebbero 
    generati più fotoni vicino all'asse z. Questo perchè l'elemento di volume dell'angolo solido è dΩ = sin(θ) dθ dφ. 
    """
    phi = np.random.uniform(0, 2*np.pi, N)
    cos = np.random.uniform(-1, 1, N) #questo sarebbe il coseno di theta
    sin = np.sqrt(1 - cos**2)
    v1 = np.column_stack([sin * np.cos(phi), sin * np.sin(phi), cos])
    return v1, -v1



def genera_sorgente_estesa(N, r_sorgente):
    """
    Punti di emissione distribuiti uniformemente su un disco di raggio
    r_source nel piano z=0 (piano ortogonale all'asse del setup).

    Dato che l'area va come r2, genero dei numeri uniformi U tra 0 ed 1 tale che U=r^2. così trovo che r = sqrtU
    """
    r = r_sorgente * np.sqrt(np.random.uniform(0.0, 1.0, N))
    phi = np.random.uniform(0, 2*np.pi, N)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), np.zeros(N)])



def hit_disco(sources: np.ndarray,
              directions: np.ndarray,
              det_center: np.ndarray,
              det_axis: np.ndarray,
              R_det: float):
    """
    Feature 1 — test di hit raggio↔disco (sostituisce il semplice test sul cono).

    Con sorgente puntiforme in (0,0,0) questo è equivalente al prodotto scalare
    usato in v1. Con sorgente estesa è il test geometrico corretto.

    Algoritmo:
      1. Trova t tale che il raggio (source + t·dir) giace sul piano del rivelatore.
      2. Calcola il punto di impatto p nel piano.
      3. Controlla se |p − det_center| ≤ R_det.

    Returns
    -------
    hit      : (N,) bool
    cos_theta: (N,) float  — coseno dell'angolo di incidenza (= dir · det_axis)
    t_path   : (N,) float  — distanza percorsa nell'aria fino al rivelatore [cm]
    """
    cos_theta = directions @ det_axis           # (N,)  — angolo di incidenza
    toward    = cos_theta > 1e-9                # fotone che va verso il rivelatore

    safe_cos  = np.where(toward, cos_theta, 1.0)

    # t = (det_center·axis − source·axis) / (dir·axis)
    proj_src  = sources    @ det_axis           # (N,)
    proj_det  = det_center @ det_axis           # scalare
    t         = (proj_det - proj_src) / safe_cos   # (N,)

    # Punto di impatto
    p  = sources + t[:, np.newaxis] * directions   # (N,3)
    dp = p - det_center[np.newaxis, :]
    r2 = np.einsum('ij,ij->i', dp, dp)             # |p − center|²

    hit = toward & (t > 0) & (r2 <= R_det**2)

    return hit, np.where(hit, cos_theta, 0.0), np.where(hit, t, 0.0)



def hit_rivelatore_trigger(v_array, S_array, D_trigger, R_trigger):
    """
    Il fotone parte dalla sorgente S che è un punto esteso, ma suppongo che la coordinata Sx della sorgente sia nulla, 
    cioè si trovi solo sul piano YZ, cioè S=(0, r*cosphi, r*sinphi).
    Il fotone viaggia lungo una direzione v=(vx, vy, vz). Il rivelatore si trova nella posizione X = -D.
    Per l'equazione del moto ho x(t) = Sx + vx*t, con Xfinale = -D. Ma con Sx = 0, ho che t = -D/vx. In questo momento il fotone entra nel piano dalla faccia anteriore
    del rivelatore. Quindi adesso prendo le altre due equazioni del moto Y(t) = Sy + vy*t e Z(t) = Sz + vz*t e sostituisco t. 
    Il centro del rivelatore si trova ad Y=Z=0. Pertanto poi devo vedere se il punto d'impatto cade nel centro del bersaglio


    Problema: questa funzione così scritta va bene solamente per il rivelatore trigger fisso. Per l'altro rivelatore la funzione dovrà essere più generale
    """
    vx = v_array[:, 0]
    vy = v_array[:, 1]
    vz = v_array[:, 2]

    Sy = S_array[:, 1]
    Sz = S_array[:, 2]

    #Controlli geometrici
    verso_sx = vx < 0
    
    #Adesso il tempo t, e calcolo di Y e Z al tempo t
    t = np.where(verso_sx, -D_trigger/vx, 0.0)
    Y = Sy + vy*t
    Z = Sz + vz*t

    #Contrllo d'impatto
    dentro_bersaglio = (Y**2) + (Z**2) <= R_trigger**2
    hit_geometrico = verso_sx & dentro_bersaglio

    #Mi serve il cosenod dell'angolo di incidenza
    costheta = -vx

    # Puliamo i dati: se il fotone non ha hittato, mettiamo a zero l'angolo e la distanza
    cos_theta_pulito = np.where(hit_geometrico, costheta, 0.0)
    t_path_pulito = np.where(hit_geometrico, t, 0.0)
    return hit_geometrico, cos_theta_pulito, t_path_pulito



def p_intrisenca(mask, cos_theta, L, mu):
    """
    cos_theta = coseno dell'angolo del fotone incidente, serve a calcolare d = L/costheta. è cos_theta_pulito
    L = spessor fisico del rivelatore trigger
    mu = coefficiente di attenuazione lineare, valore sopra
    mask = lista dei fotoni sopravvissuti, True se colpisce, False se non colpisce, viene da hit_geometrico

    Questa funzione serve a rispondere alla domanda: tra i fotoni che hanno colpito il rivelatore quanti interagiscono e quanti passano come fosse trasparente?
    np.where dice che se il fotone ha colpito allora mask è True e quindi l'angolo è sicuro (1e-9) ed usa il cos_theta reale. 
    """
    safe_cos = np.where(mask & (cos_theta > 1e-9), cos_theta, 1.0)

    #d calcola la distanza d percorsa nel rivelatore per i fotoni buoni, per quelli scartati si mette 0.0
    d = np.where(mask, L / safe_cos, 0.0)

    return np.where(mask, 1.0 - np.exp(-mu * d), 0.0)



def p_aria(t_path: np.ndarray, mu_air: float, mask: np.ndarray) -> np.ndarray:
    """
    Tiene conto del fatto che il fotone potrebbe incontrare una molecola d'aria nel tragitto tra la sorgente e il rivelatore
    Voglio la probabilità di sopravvivenza non di interazione

    mask = lista dei fotoni sopravvissuti, True se colpisce, False se non colpisce, viene da hit_geometrico
    t_path = distanza fisica percorsa dal fotone nell'aria, cioè dalla sorgente al rivelatore. 

    """
    return np.where(mask, np.exp(-mu_air * t_path), 0.0)



def accetta_mc(prob: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Fino ad adesso si sono calcolate le probabilità (ad esempio 30%) che il fotone venga rivelato
    Ma il fotone o viene rivelato o non viene rivelato (1 e 0)

    u = crea un array di numeri casuali tra 0 ed 1
    Il return mask & (u < prob) controlla se il numero estratti casualmente per u[i] è minore della rispettiva
    probabilità. Mask ci assicura che questo avvenga solamente per i fotoni buoni

    Accettazione Monte Carlo: per ogni fotone che ha superato il test
    geometrico (mask=True) estrae u ~ U[0,1] e lo confronta con la
    probabilità combinata prob = p_intrinseca * p_aria.
    prob viene calcolata poi nel main
    """
    u = np.random.uniform(0.0, 1.0, len(prob))
    return mask & (u < prob)


# ══════════════════════════════════════════════════════════════════════════════
# EFFETTO COMPTON
# ══════════════════════════════════════════════════════════════════════════════
"""
Nel blocco precedente si è analizzata l'interazione del fotone con il rivelatore trigger. 
Adesso dobbiamo guardare cosa fa il secondo fotone, scatterato back to back quando becca il piombo: può decidere di passare come se fosse trasparente o di fare Comton
è molto probabile che faccia fotoelettrico ma una volta che interagisce con un elettrone, il fotone muore e non raggiunge il rivelatore2. 
Il coefficiente di attenuazione totale è mu_tot = mu_compton + mu_fotoelettrico. Come abbiamo visto prima usiamo mu_totale per vedere se il fotone interagisce 
Se interagisce lanciamo altri numeri randomici per vedere quale interazione ha fatto: se il numero estratto è minore di (mu_fotoelettrico)/mu_tot allora fa fotoelettrico
e il fotone muore. Se è maggiore allora fa Compton. 
"""

def hit_bersaglio_piombo_circolare(v_array, S_array, D_pb, R_pb, alpha):
    """"
    v_array = la matrice iniziale [5 mil, 3] con le velocità
    S_array = la matrice [5 mil, 3] con i punti di partenza 
    D_pb = distanza tra la sorgente e il bersaglio di piombo
    alpha = angolo inclinazione del bersaglio in gradi
    Per calcolare il tempo di volo t devo usare questa formula    t = (C - S) * n/ (v * n)

    Il controllo funziona semplicemente guardando se la distanza radiale dal centro (y2 + z2) < R2, se si allora sono dentro
    USARE SE IL BERSAGLIO è CIRCOLARE
    """
    #Prima definisco il centro del bersaglio
    C = np.array([D_pb, 0, 0])

    #Ora devo definire il vettore incidente, supponendo inclinato nel piano XZ
    n = np.array([-np.cos(alpha), 0, np.sin(alpha)])

    #Calcolo del denominatore v scalar n, numeratore e t
    den = (v_array[:, 0] * n[0]) + (v_array[:, 1] * n[1]) + (v_array[:, 2] * n[2])  #operazione componente per componente
    
    # Creiamo una maschera per evitare divisioni per zero
    verso_bersaglio = den < -1e-9
    
    vettore_distanza = C - S_array
    num = (vettore_distanza[:, 0] * n[0]) + (vettore_distanza[:, 1] * n[1]) + (vettore_distanza[:, 2] * n[2]) 
    t = np.where(verso_bersaglio, num / den, -1.0)   #la condizione se falsa restituisce -1.0. L'importante è che sia un valore negativo, non importa quale
    
    #Dobbiamo prendere il fotone che si muove verso dx
    hit_valido = t > 0 

    #Definisco il punto di impatto P = S + v*t
    P = S_array + v_array*t[:, np.newaxis]

    #Distanza dal centro
    distanza_centro = (P[:, 0] - C[0])**2 + (P[:, 1] - C[1])**2 + (P[:, 2] - C[2])**2
    dentro_raggio = distanza_centro <= R_pb**2  

    #Dobbiamo infine porre un'ultima maschera: deve avere tempo positivo e deve essere dentro il raggio
    mask_impatto = hit_valido & dentro_raggio
    t_path = np.where(mask_impatto, t, 0.0)
    
    return mask_impatto, P, t_path



def hit_bersaglio_piombo_parallelepipedo(v_array, S_array, D_pb, W_pb, H_pb, alpha_deg):
    """"
    v_array = la matrice iniziale [5 mil, 3] con le velocità
    S_array = la matrice [5 mil, 3] con i punti di partenza 
    D_pb = distanza tra la sorgente e il bersaglio di piombo
    W_pb = width del bersaglio di piombo (larghezza della faccia)
    H_pb = altezza del bersaglio di piombo (altezza della faccia)
    alpha = angolo inclinazione del bersaglio in gradi
    Per calcolare il tempo di volo t devo usare questa formula    t = (C - S) * n/ (v * n)

    Qua il controllo è diverso. Devo verificare che il punto d'impatto sia confinato tra i bordi degli assi orizzontali(W) e verticali(H)
    Cioè se |y| <= W/2 & e |z| <= H/2
    USARE SE IL BERSAGLIO è UN PARALLELEPIPEDO (cambia il controllo di hit)
    """
    #Prima definisco il centro del bersaglio
    C = np.array([D_pb, 0, 0])
    alpha = np.radians(alpha_deg)

    # Vettore normale (perpendicolare alla faccia)
    n = np.array([-np.cos(alpha), 0.0, np.sin(alpha)])

    # Vettore larghezza (parallelo alla faccia, ortogonale a n e a Y). Punta lungo la superficie inclinata
    u = np.array([np.sin(alpha), 0.0, np.cos(alpha)])

    #Calcolo del denominatore v scalar n, numeratore e t
    den = (v_array[:, 0] * n[0]) + (v_array[:, 1] * n[1]) + (v_array[:, 2] * n[2])  #operazione componente per componente
    
    # Creiamo una maschera per evitare divisioni per zero
    verso_bersaglio = den < -1e-9
    
    vettore_distanza = C - S_array
    num = (vettore_distanza[:, 0] * n[0]) + (vettore_distanza[:, 1] * n[1]) + (vettore_distanza[:, 2] * n[2]) 
    t = np.where(verso_bersaglio, num / den, -1.0)   #la condizione se falsa restituisce -1.0. L'importante è che sia un valore negativo, non importa quale
    
    #Dobbiamo prendere il fotone che si muove verso dx
    hit_valido = t > 0 

    #Definisco il punto di impatto P = S + v*t
    P = S_array + v_array*t[:, np.newaxis]

    # Controllo l'altezza (lungo l'asse Y)
    dentro_altezza = np.abs(P[:, 1] - C[1]) <= (H_pb / 2.0)

    # Controllo la larghezza (spostamento lungo il vettore u sul piano inclinato)
    spostamento_u = (P[:, 0] - C[0]) * u[0] + (P[:, 1] - C[1]) * u[1] + (P[:, 2] - C[2]) * u[2]
    dentro_larghezza = np.abs(spostamento_u) <= (W_pb / 2.0)
    
    # La maschera finale: deve aver colpito il piano, ed essere dentro le cornici H e W
    maschera_impatto = hit_valido & dentro_altezza & dentro_larghezza
    
    t_path = np.where(maschera_impatto, t, 0.0)
    cos_theta_in = np.where(maschera_impatto, np.abs(den), 1.0)
    
    return maschera_impatto, P, t_path, cos_theta_in



def interazione_piombo(mask, S_pb, mu_pb_tot, mu_air, mu_pb_compton, t_aria, cos_theta_in):
    """
    mask = maschera di True o False ricevuti dalla funzione hit
    D_pb = distanza dalla sorgente
    W_pb = larghezza piombo
    H_pb = altezza piombo
    S_pb = spessore piombo
    alpha_deg = angolo di inclinazione in gradi
    mu_pb = costante di attenuazione per il piombo

    Probabilità di passare senza interazione = e^-mu*d
    Probabilità di sbattere conto atomo = 1 - e^-mu*d. Vogliamo che il fotone sbatta
    """
    #Calcolo lo spessore che viene attraversato se colpisce obliquo
    d_pb = np.where(mask, S_pb/cos_theta_in, 0.0)

    #Sopravvivenza all'aria
    prob_aria = np.where(mask, np.exp(-mu_air * t_aria), 0.0)
    random_aria = np.random.uniform(0.0, 1.0, len(t_aria))
    sopravvivenza_aria = mask & (random_aria < prob_aria)

    #Sbatta contro un atomo di Piombo
    prob_interagisce_pb = np.where(sopravvivenza_aria, 1.0 - np.exp(-mu_pb_tot * d_pb), 0.0)
    dado_pb = np.random.uniform(0.0, 1.0, len(d_pb))
    interagisce_pb = sopravvivenza_aria & (dado_pb < prob_interagisce_pb)

    #Compton (vive) o Fotoelettrico (offline)
    prob_compton = mu_pb_compton / mu_pb_tot
    random_tipo = np.random.uniform(0.0, 1, len(d_pb))
    is_compton = interagisce_pb & (random_tipo < prob_compton)

    return is_compton



def genera_angoli_compton(N_fotoni):
    """
    Adesso dobbiamo descrivere la dinamica dell'urto. In sferiche ho angolo azimutale phi e tangenziale theta. Fortunatamente phi varia solamente tra 0 e 2pi
    Invece theta segue Klein-Nishina, e si deve usare try-and-catch per prendere la nostra distribuzione di questi angoli.
    Inoltre anche l'energia del fotone uscente sarà minore, seguire la formula di Compton: E' = E0/(1 + (E0/me*c2)*(1 - cos_theta)). Dato che noi per il nostro 
    setup stiamo considerando solamente i fotoni a 511keV la formula diventa E' = 511/(2 - cos_theta). Quindi E'/E0 = 1/(2-cos_theta)

    Il try-and-catch funziona alla solita maniera: ho la mia curva di Klein_Nishina, genero dei numeri casuali uniformemente distribuiti, le x, dato che saranno gli
    angoli theta, da 0 gradi a 180, e le y da 0 al MAX. Calcolo quindi quando vale f(x_candidato) e vedo se y_candidato < f(x_candidato). Se è così allora il numero è
    accettato. Nel nostro caso, con E = 511keV il MAX = 2
    """
    angoli_accettati = np.zeros(N_fotoni)
    fotoni_da_estrarre = N_fotoni
    indici_mancanti = np.arange(N_fotoni)

    MAX = 0.57

    while fotoni_da_estrarre > 0:
        theta_candidati = np.random.uniform(0, np.pi, fotoni_da_estrarre)
        y_candidati = np.random.uniform(0, MAX, fotoni_da_estrarre)

        #Costruzione Klein-Nishina
        rapporto_E = 1/(2 - np.cos(theta_candidati))  # frazione di energia dopo l'urto
        sium = rapporto_E + (1.0/rapporto_E) - np.sin(theta_candidati)**2
        pdf_klein_nishina = (rapporto_E**2) * sium
        peso = pdf_klein_nishina * np.sin(theta_candidati)

        mask = y_candidati <= peso

        # (indici_mancanti[mask_accettati] trova i posti esatti nell'array finale)
        indici_vincitori = indici_mancanti[mask]
        angoli_accettati[indici_vincitori] = theta_candidati[mask]

        # Aggiorniamo la lista di chi deve ancora lanciare
        indici_mancanti = indici_mancanti[~mask]
        fotoni_da_estrarre = len(indici_mancanti)

    return angoli_accettati



def deviazione_compton_3D(v_in, theta, phi):
    """
    Ruota un array di vettori 'v_in' di un angolo polare 'theta' e azimutale 'phi'.
    """
    # 1. L'asse "Avanti" (W) è semplicemente la direzione di arrivo
    W = v_in / np.linalg.norm(v_in, axis=1, keepdims=True) 
    
    # 2. Creiamo l'asse "Destra" (U). 
    # Per farlo, facciamo il prodotto vettoriale tra W e l'asse Z globale.
    A = np.zeros_like(W)
    A[:, 2] = 1.0  # Vettore d'appoggio [0, 0, 1]
    
    # (Piccola sicurezza informatica: se un fotone andava perfettamente lungo Z, 
    # usiamo l'asse X per evitare un prodotto vettoriale nullo)
    paralleli_a_Z = np.abs(W[:, 2]) > 0.99
    A[paralleli_a_Z, 2] = 0.0
    A[paralleli_a_Z, 0] = 1.0
    
    # Calcolo l'asse U (perpendicolare a W e ad A) e lo normalizzo
    U = np.cross(A, W)
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    
    # 3. L'asse "Sopra" (V) è perpendicolare agli altri due
    V = np.cross(W, U)
    
    # 4. Assembliamo la nuova direzione! 
    # (np.newaxis serve per moltiplicare un array 1D di angoli per un array 3D di vettori)
    sin_t = np.sin(theta)[:, np.newaxis]
    cos_t = np.cos(theta)[:, np.newaxis]
    sin_p = np.sin(phi)[:, np.newaxis]
    cos_p = np.cos(phi)[:, np.newaxis]
    
    # La formula standard dello scattering nello spazio locale
    v_out = U * (sin_t * cos_p) + V * (sin_t * sin_p) + W * cos_t
    
    return v_out



def rivelazione_dinamica_B(maschera_hit_B, theta, cos_in_B, t_aria_B, spessore_NaI):
    """
    Calcola l'energia finale del fotone, aggiorna il mu dinamicamente
    tramite interpolazione, e valuta l'assorbimento nel Rivelatore B.
    """
    E_0 = 511.0 # keV
    
    # E' = E_0 / (2 - cos(theta))
    E_out = np.where(maschera_hit_B, E_0 / (2.0 - np.cos(theta)), 0.0)
    
    #TABELLE UFFICIALI NIST XCOM
    # Energie di riferimento (asse X della tabella, in keV)
    energie_xcom = np.array([100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 600.0])
    
    # Valori di mu_totale (asse Y, in cm^-1) - Calcolati come (μ/ρ)_NIST * densità
    mu_NaI_xcom = np.array([6.110, 2.240, 1.200, 0.606, 0.393, 0.352, 0.315]) 
    mu_aria_xcom = np.array([0.000185, 0.000163, 0.000148, 0.000128, 0.000115, 0.000105, 0.000097])
    
    # np.interp guarda l'energia E_out di ogni singolo fotone e calcola il suo mu esatto
    E_safe = np.where(maschera_hit_B, E_out, energie_xcom[0])
    mu_NaI_dinamico  = np.interp(E_safe, energie_xcom, mu_NaI_xcom)
    mu_aria_dinamico = np.interp(E_safe, energie_xcom, mu_aria_xcom)
    
    #Beer-Lambert aggiornato

    # FASE A: volo in aria
    prob_aria = np.where(maschera_hit_B, np.exp(-mu_aria_dinamico * t_aria_B), 0.0)
    dado_aria = np.random.uniform(0.0, 1.0, len(t_aria_B))
    sopravvive_aria = maschera_hit_B & (dado_aria < prob_aria)
    
    # FASE B: Viene assorbito (fa segnale) nel Rivelatore NaI?
    # Calcolo lo spessore attraversato d = spessore / cos(incidenza)
    cos_safe = np.where(sopravvive_aria, np.abs(cos_in_B), 1.0)
    d_NaI = np.where(sopravvive_aria, spessore_NaI / cos_safe, 0.0)
    
    # Probabilità di interazione (1 - esponenziale)
    prob_rivelazione = np.where(sopravvive_aria, 1.0 - np.exp(-mu_NaI_dinamico * d_NaI), 0.0)
    dado_NaI = np.random.uniform(0.0, 1.0, len(d_NaI))
    
    fotoni_coincidenza = sopravvive_aria & (dado_NaI < prob_rivelazione)
    
    return fotoni_coincidenza, E_out

"""
============================================================================================================
PLOTTING. GRAZIE CLAUDE
============================================================================================================
"""


def _klein_nishina_pdf(theta):
    """dσ/dΩ · sin(θ) (non normalizzata) — la PDF corretta per campionare θ."""
    r = 1.0 / (2.0 - np.cos(theta))        # E'/E per E=511 keV
    dsdO = (r**2) * (r + 1.0/r - np.sin(theta)**2)
    return dsdO * np.sin(theta)


def _energia_compton(theta):
    """E'(θ) = E0 / (1 + (E0/mc²)(1 - cosθ)) in keV."""
    return E0_KEV / (1.0 + (E0_KEV / ME_C2) * (1.0 - np.cos(theta)))


# ══════════════════════════════════════════════════════════════════════════════
#  1. DIAGNOSTICO KLEIN-NISHINA
# ══════════════════════════════════════════════════════════════════════════════

def plot_diagnostico_kn(theta_campionati: np.ndarray,
                        n_bins: int = 60,
                        save: str = None) -> None:
    """
    Confronta l'istogramma degli angoli generati con la PDF teorica
    Klein-Nishina corretta (dσ/dΩ · sinθ).

    Se il generatore è corretto i punti devono cadere sulla curva rossa
    entro le barre d'errore poissoniane.

    Parameters
    ----------
    theta_campionati : array degli angoli Compton generati [radianti]
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Diagnostica generatore Klein-Nishina  (E = 511 keV)',
                 fontsize=14, y=1.02)

    for ax, xunit, xscale in zip(axes, ['rad', 'deg'], [1.0, 180/np.pi]):
        theta_plot = theta_campionati * xscale

        # Istogramma normalizzato a densità di probabilità
        counts, edges = np.histogram(theta_plot, bins=n_bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        width   = edges[1] - edges[0]
        density = counts / (counts.sum() * width)
        sigma   = np.sqrt(counts) / (counts.sum() * width)   # err poissoniano

        ax.errorbar(centers, density, yerr=sigma,
                    fmt='o', color='royalblue', markersize=3.5,
                    capsize=2, elinewidth=0.8, label='MC generato')

        # Curva teorica normalizzata
        th_fine  = np.linspace(edges[0], edges[-1], 500)
        th_rad   = th_fine / xscale
        pdf_th   = _klein_nishina_pdf(th_rad)
        norm_int = np.trapz(pdf_th, th_rad)
        ax.plot(th_fine, pdf_th / norm_int / xscale,
                color='crimson', linewidth=2, label='Klein-Nishina teorica')

        ax.set_xlabel(f'θ [{xunit}]', fontsize=12)
        ax.set_ylabel('Densità di probabilità', fontsize=12)
        ax.set_title('Distribuzione angolare Compton', fontsize=11)
        ax.legend(fontsize=10)

    # Inset: pull = (data - teoria) / sigma — devono distribuirsi come N(0,1)
    ax_inset = axes[1].inset_axes([0.55, 0.45, 0.42, 0.45])
    th_rad_c  = centers * (np.pi/180)
    pdf_c     = _klein_nishina_pdf(th_rad_c)
    norm_int  = np.trapz(_klein_nishina_pdf(THETA_TH), THETA_TH)
    expected  = (pdf_c / norm_int) * (np.pi/180)  # per unità di grado
    sigma_pr  = np.sqrt(counts) / (counts.sum() * (edges[1]-edges[0]))
    pulls     = np.where(sigma_pr > 0, (density - expected) / sigma_pr, 0.0)

    ax_inset.axhline(0, color='crimson', linewidth=0.8, linestyle='--')
    ax_inset.axhspan(-1, 1, color='#f0c27f', alpha=0.4)
    ax_inset.scatter(centers, pulls, s=8, color='royalblue')
    ax_inset.set_ylabel('Pull', fontsize=8)
    ax_inset.set_xlabel('θ [°]', fontsize=8)
    ax_inset.tick_params(labelsize=7)
    ax_inset.set_ylim(-4, 4)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  2. CINEMATICA COMPTON
# ══════════════════════════════════════════════════════════════════════════════

def plot_cinematica_compton(theta_mc: np.ndarray,
                            E_out_mc: np.ndarray,
                            save: str = None) -> None:
    """
    Scatter plot E'_MC vs θ_MC sovrapposto alla curva teorica E'(θ).

    Parameters
    ----------
    theta_mc : array degli angoli di scattering Compton [radianti]
    E_out_mc : array delle energie dei fotoni scatterati [keV]
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter MC (subsample se troppo denso)
    n_plot = min(len(theta_mc), 20_000)
    idx    = np.random.choice(len(theta_mc), n_plot, replace=False)

    ax.scatter(np.rad2deg(theta_mc[idx]), E_out_mc[idx],
               s=1.5, alpha=0.3, color='steelblue',
               label=f'MC  (mostrati {n_plot:,} / {len(theta_mc):,})')

    # Curva teorica
    E_th = _energia_compton(THETA_TH)
    ax.plot(np.rad2deg(THETA_TH), E_th,
            color='crimson', linewidth=2.5, label="E'(θ) Compton teorica")

    # Annotazioni di riferimento (angoli tipici del setup)
    for ang_deg in [30, 60, 90, 120, 150]:
        ang_rad = np.deg2rad(ang_deg)
        e_ref   = _energia_compton(ang_rad)
        ax.annotate(f"{e_ref:.0f} keV",
                    xy=(ang_deg, e_ref),
                    xytext=(ang_deg + 3, e_ref + 15),
                    fontsize=8, color='#555555',
                    arrowprops=dict(arrowstyle='->', color='#999999', lw=0.7))

    ax.set_xlabel('Angolo di scattering θ [°]', fontsize=13)
    ax.set_ylabel("Energia fotone scatterato E' [keV]", fontsize=13)
    ax.set_title('Cinematica Compton — MC vs Teoria  (E₀ = 511 keV)',
                 fontsize=14, pad=10)
    ax.set_xlim(0, 181)
    ax.set_ylim(150, 530)
    ax.set_xticks(np.arange(0, 181, 20))
    ax.legend(fontsize=11, markerscale=5)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  3. COINCIDENZE vs ANGOLO
# ══════════════════════════════════════════════════════════════════════════════

def plot_coincidenze_compton(angoli_deg: np.ndarray,
                             n_true:     np.ndarray,
                             n_acc:      np.ndarray,
                             sigma_true: np.ndarray,
                             normalizza: bool = True,
                             save: str = None) -> None:
    """
    Conteggi di coincidenza (vere + accidentali) vs angolo del rivelatore B.

    Parameters
    ----------
    angoli_deg  : array degli angoli simulati [°]
    n_true      : coincidenze fisiche per angolo
    n_acc       : coincidenze accidentali per angolo
    sigma_true  : incertezza poissoniana su n_true
    normalizza  : se True normalizza al massimo di n_true
    """
    n_tot   = n_true + n_acc
    s_tot   = np.sqrt(sigma_true**2 + n_acc)   # σ_acc ≈ √N_acc (Poisson)

    norm = np.max(n_tot) if (normalizza and np.max(n_tot) > 0) else 1.0

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8),
                                   gridspec_kw={'height_ratios': [4, 1]},
                                   sharex=True)
    fig.subplots_adjust(hspace=0.06)

    # ── Pannello principale ───────────────────────────────────────────────────
    ax1.errorbar(angoli_deg, n_tot / norm, yerr=s_tot / norm,
                 fmt='o-', color='royalblue', markersize=5,
                 linewidth=1.8, capsize=3, label='Totale (vere + accidentali)')

    ax1.plot(angoli_deg, n_true / norm, '--',
             color='forestgreen', linewidth=1.5, label='Coincidenze vere (MC)')

    if np.any(n_acc > 0):
        mean_acc = np.mean(n_acc[angoli_deg < np.max(angoli_deg) - 10]) / norm
        ax1.axhline(mean_acc, color='salmon', linestyle=':',
                    linewidth=1.2,
                    label=f'Fondo accidentali  ≈ {mean_acc:.3f}')
        ax1.fill_between(angoli_deg, 0, n_acc / norm,
                         color='salmon', alpha=0.25)

    ylabel = 'Coincidenze normalizzate' if normalizza else 'Conteggi'
    ax1.set_ylabel(ylabel, fontsize=13)
    ax1.set_title('MC Compton Na-22 — Coincidenze vs angolo', fontsize=14)
    ax1.legend(fontsize=10, framealpha=0.9)
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax1.tick_params(labelbottom=False)

    # ── Pannello rapporto segnale/totale ──────────────────────────────────────
    snr = np.where(n_tot > 0, n_true / n_tot, 0.0)
    ax2.plot(angoli_deg, snr * 100, 'o-', color='royalblue',
             markersize=3.5, linewidth=1.2)
    ax2.axhline(100, color='forestgreen', linestyle='--', linewidth=0.8)
    ax2.set_ylabel('Vere / Tot [%]', fontsize=10)
    ax2.set_xlabel('Angolo α [°]', fontsize=13)
    ax2.set_ylim(0, 110)
    ax2.grid(True, linestyle=':', linewidth=0.4, alpha=0.7)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  4. MAPPA DI HIT SUL BERSAGLIO DI PIOMBO
# ══════════════════════════════════════════════════════════════════════════════

def plot_hit_bersaglio(P_hit: np.ndarray,
                       mask_hit: np.ndarray,
                       mask_compton: np.ndarray,
                       W_pb: float,
                       H_pb: float,
                       R_pb: float = None,
                       save: str = None) -> None:
    """
    Mappa 2D dei punti di impatto sul bersaglio di Pb nel piano YZ
    (le coordinate trasverse, cioè le componenti 1 e 2 di P).

    Distingue:
     - fotoni che colpiscono geometricamente (grigi)
     - fotoni che fanno Compton (rossi)

    Parameters
    ----------
    P_hit       : array (N, 3) dei punti di impatto
    mask_hit    : (N,) bool — fotoni che colpiscono il Pb geometricamente
    mask_compton: (N,) bool — fotoni che fanno scattering Compton
    W_pb, H_pb  : dimensioni bersaglio parallelepipedo [cm]
    R_pb        : raggio bersaglio circolare [cm] (None se parallelepipedo)
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(7, 7))

    # Subsample per performance
    n_max  = 30_000
    idx_h  = np.where(mask_hit)[0]
    idx_c  = np.where(mask_compton)[0]

    if len(idx_h) > n_max:
        idx_h = np.random.choice(idx_h, n_max, replace=False)
    if len(idx_c) > n_max:
        idx_c = np.random.choice(idx_c, n_max, replace=False)

    ax.scatter(P_hit[idx_h, 1], P_hit[idx_h, 2],
               s=0.8, alpha=0.15, color='steelblue',
               label=f'Hit geometrico  ({mask_hit.sum():,})')
    ax.scatter(P_hit[idx_c, 1], P_hit[idx_c, 2],
               s=1.2, alpha=0.5, color='crimson',
               label=f'Compton  ({mask_compton.sum():,})')

    # Bordo del bersaglio
    if R_pb is not None:
        circle = plt.Circle((0, 0), R_pb, color='black',
                             fill=False, linewidth=2, linestyle='--',
                             label=f'Bersaglio circolare R={R_pb} cm')
        ax.add_patch(circle)
        lim = R_pb * 1.5
    else:
        rect = plt.Rectangle((-W_pb/2, -H_pb/2), W_pb, H_pb,
                              edgecolor='black', facecolor='none',
                              linewidth=2, linestyle='--',
                              label=f'Bersaglio {W_pb}×{H_pb} cm²')
        ax.add_patch(rect)
        lim = max(W_pb, H_pb) * 0.8

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('Y [cm]', fontsize=12)
    ax.set_ylabel('Z [cm]', fontsize=12)
    ax.set_title('Mappa di hit sul bersaglio di Pb\n(piano trasverso YZ)',
                 fontsize=13)
    ax.legend(fontsize=9, markerscale=5)
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)

    # Box statistiche
    eff_geom   = mask_hit.sum() / len(mask_hit) * 100
    eff_compt  = (mask_compton.sum() / mask_hit.sum() * 100
                  if mask_hit.sum() > 0 else 0)
    stats_text = (f"Hit geometrico: {eff_geom:.3f}%\n"
                  f"Compton | hit: {eff_compt:.1f}%")
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
            fontsize=9, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      alpha=0.85, edgecolor='#aaaaaa'))

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  5. SPETTRO ENERGIA RIVELATORE B
# ══════════════════════════════════════════════════════════════════════════════

def plot_energia_rivelatore_B(E_out: np.ndarray,
                              mask_rivelati: np.ndarray,
                              angolo_deg: float,
                              n_bins: int = 80,
                              save: str = None) -> None:
    """
    Spettro dell'energia depositata nel rivelatore B per un dato angolo.
    Mostra la linea teorica E'(θ) come riferimento.

    Parameters
    ----------
    E_out         : array delle energie post-Compton [keV] (tutti i fotoni)
    mask_rivelati : (N,) bool — fotoni effettivamente rivelati da B
    angolo_deg    : angolo del rivelatore B per questo run [°]
    """
    E_rivelati = E_out[mask_rivelati & (E_out > 0)]

    if len(E_rivelati) == 0:
        print(f"Nessun fotone rivelato a {angolo_deg}° — plot saltato.")
        return

    E_teorica = _energia_compton(np.deg2rad(angolo_deg))

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(E_rivelati, bins=n_bins, color='steelblue', edgecolor='white',
            linewidth=0.4, label=f'Fotoni rivelati  (N={len(E_rivelati):,})')

    ax.axvline(E_teorica, color='crimson', linewidth=2.2, linestyle='--',
               label=f"E' teorica = {E_teorica:.1f} keV")

    ax.set_xlabel("Energia fotone scatterato E' [keV]", fontsize=12)
    ax.set_ylabel('Conteggi', fontsize=12)
    ax.set_title(f'Spettro energetico rivelatore B — θ = {angolo_deg}°',
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Statistiche nell'inset
    stats = (f"Media: {E_rivelati.mean():.1f} keV\n"
             f"σ:     {E_rivelati.std():.1f} keV\n"
             f"Teorico: {E_teorica:.1f} keV\n"
             f"Δ:   {E_rivelati.mean()-E_teorica:+.1f} keV")
    ax.text(0.97, 0.97, stats, transform=ax.transAxes,
            fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      alpha=0.88, edgecolor='#aaaaaa'))

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  6. PANNELLO RIASSUNTIVO 2×2
# ══════════════════════════════════════════════════════════════════════════════

def plot_summary(theta_mc:     np.ndarray,
                 E_out_mc:     np.ndarray,
                 angoli_deg:   np.ndarray,
                 n_true:       np.ndarray,
                 n_acc:        np.ndarray,
                 sigma_true:   np.ndarray,
                 save: str = None) -> None:
    """
    Pannello 2×2 con i quattro grafici principali:
      [0,0] Klein-Nishina campionata vs teorica
      [0,1] Cinematica E'(θ)
      [1,0] Coincidenze vs angolo
      [1,1] E' media MC vs E' teorica per ogni angolo simulato
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # ── [0,0] Klein-Nishina ───────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    n_bins = 50
    counts, edges = np.histogram(np.rad2deg(theta_mc), bins=n_bins,
                                 range=(0, 180))
    centers = 0.5 * (edges[:-1] + edges[1:])
    width   = edges[1] - edges[0]
    density = counts / (counts.sum() * width)
    sigma_d = np.sqrt(counts) / (counts.sum() * width)
    ax0.errorbar(centers, density, yerr=sigma_d, fmt='o',
                 color='royalblue', markersize=2.5, capsize=1.5,
                 elinewidth=0.7, label='MC')
    th_fine  = np.linspace(0, np.pi, 500)
    pdf_th   = _klein_nishina_pdf(th_fine)
    norm_int = np.trapz(pdf_th, th_fine)
    ax0.plot(np.rad2deg(th_fine), pdf_th / norm_int * (np.pi/180),
             color='crimson', linewidth=1.8, label='Teorica KN')
    ax0.set_xlabel('θ [°]', fontsize=11)
    ax0.set_ylabel('Densità', fontsize=11)
    ax0.set_title('Distribuzione K-N campionata', fontsize=11)
    ax0.legend(fontsize=9)

    # ── [0,1] Cinematica ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    n_plot = min(len(theta_mc), 15_000)
    idx    = np.random.choice(len(theta_mc), n_plot, replace=False)
    ax1.scatter(np.rad2deg(theta_mc[idx]), E_out_mc[idx],
                s=1.0, alpha=0.3, color='steelblue')
    ax1.plot(np.rad2deg(THETA_TH), _energia_compton(THETA_TH),
             color='crimson', linewidth=2, label="E'(θ) teorica")
    ax1.set_xlabel('θ [°]', fontsize=11)
    ax1.set_ylabel("E' [keV]", fontsize=11)
    ax1.set_title('Cinematica Compton MC', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 181)

    # ── [1,0] Coincidenze ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    n_tot = n_true + n_acc
    norm  = np.max(n_tot) if np.max(n_tot) > 0 else 1.0
    ax2.errorbar(angoli_deg, n_tot / norm,
                 yerr=np.sqrt(sigma_true**2 + n_acc) / norm,
                 fmt='o-', color='royalblue', markersize=4,
                 linewidth=1.5, capsize=2, label='Totale')
    ax2.plot(angoli_deg, n_true / norm, '--',
             color='forestgreen', linewidth=1.2, label='Vere')
    if np.any(n_acc > 0):
        ax2.fill_between(angoli_deg, 0, n_acc / norm,
                         color='salmon', alpha=0.3, label='Accidentali')
    ax2.set_xlabel('Angolo α [°]', fontsize=11)
    ax2.set_ylabel('Coincidenze (norm.)', fontsize=11)
    ax2.set_title('Coincidenze vs angolo', fontsize=11)
    ax2.legend(fontsize=9)

    # ── [1,1] E'_MC media vs E'_teorica ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])

    # Calcola E' media per ogni angolo simulato
    E_medie   = np.array([_energia_compton(np.deg2rad(a)) for a in angoli_deg])
    # Deviazione attesa dalla risoluzione energetica NaI (~7% a 511 keV)
    sigma_E   = E_medie * 0.07 / 2.355   # FWHM → sigma
    ax3.errorbar(angoli_deg, E_medie, yerr=sigma_E,
                 fmt='o', color='steelblue', markersize=4,
                 capsize=2, label='E\' teorica ± σ(NaI)')
    # Curva teorica
    th_ang = np.linspace(np.min(angoli_deg), np.max(angoli_deg), 200)
    ax3.plot(th_ang, _energia_compton(np.deg2rad(th_ang)),
             color='crimson', linewidth=2, label='Formula Compton')
    ax3.set_xlabel('Angolo θ [°]', fontsize=11)
    ax3.set_ylabel("E' [keV]", fontsize=11)
    ax3.set_title("E' attesa per rivelatore B", fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, linestyle='--', linewidth=0.4, alpha=0.7)

    fig.suptitle('MC Compton Na-22 — Pannello riassuntivo', fontsize=15, y=1.01)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================================================
# ESECUZIONE MAIN
# ============================================================================================================
if __name__ == '__main__':
    print("Inizio Simulazione Monte Carlo...")
    
    # --- A. SETUP PARAMETRI FISICI ---
    # Valori a 511 keV per il Piombo
    mu_pb_tot_511 = 1.79      # cm^-1
    mu_pb_compton_511 = 0.48  # cm^-1
    spessore_pb = 0.5         # cm (spessore ipotetico del bersaglio di piombo)
    alpha_pb = 30.0           # gradi di inclinazione bersaglio
    
    # --- B. IL BIG BANG (Generazione 3D isotropa) ---
    print(f"Generazione di {N_EVENTS:,} fotoni...")
    
    # Uso la sorgente estesa se attivata dal flag globale
    if USE_SOURCE_EXTENT:
        S_array = genera_sorgente_estesa(N_EVENTS, R_SOURCE)
    else:
        S_array = np.zeros((N_EVENTS, 3)) # Partono tutti dall'origine (0,0,0)
    
    # La tua funzione restituisce la tupla (v1, -v1)
    v_array, v_gemelli_tutti = genera_fotoni_isotropi(N_EVENTS)

    # --- C. IL TRIGGER A ---
    print("Calcolo Trigger A (Geometria + Fisica)...")
    # 1. Geometria Trigger A
    hit_geom_A, cos_in_A, t_aria_A = hit_rivelatore_trigger(
        v_array, S_array, D_trigger=D_trigger, R_trigger=R_trigger
    )
    
    # 2. Fisica Trigger A
    prob_intr_A = p_intrisenca(hit_geom_A, cos_in_A, L=L_A, mu=MU_NAI)
    prob_aria_A = p_aria(t_aria_A, mu_air=MU_AIR, mask=hit_geom_A)
    prob_totale_A = prob_intr_A * prob_aria_A
    
    # 3. Accettazione Monte Carlo
    maschera_trigger = accetta_mc(prob_totale_A, hit_geom_A)
    
    # Ottengo i fotoni validi e giro la velocità per i gemelli 
    v_gemelli = v_gemelli_tutti[maschera_trigger]
    S_gemelli = S_array[maschera_trigger]
    print(f"Fotoni che hanno colpito e attivato il Trigger A: {len(v_gemelli):,}")

    # --- D. IL BERSAGLIO DI PIOMBO ---
    if len(v_gemelli) > 0:
        print("Calcolo interazione nel Piombo...")
        
        # Geometria Piombo (uso H e l_pb definiti nei globals)
        maschera_impatto_pb, P_impatto, t_aria_pb, cos_in_pb = hit_bersaglio_piombo_parallelepipedo(
            v_array=v_gemelli, 
            S_array=S_gemelli, 
            D_pb=D_pb, 
            W_pb=W_pb, 
            H_pb=H_pb, 
            alpha_deg=alpha_pb
        )
        
        # Fisica Piombo
        fotoni_compton = interazione_piombo(
            mask=maschera_impatto_pb, 
            S_pb=spessore_pb, 
            mu_pb_tot=mu_pb_tot_511, 
            mu_air=MU_AIR, 
            mu_pb_compton=mu_pb_compton_511, 
            t_aria=t_aria_pb, 
            cos_theta_in=cos_in_pb
        ) 
        
        v_sopravvissuti = v_gemelli[fotoni_compton]
        P_nuova_partenza = P_impatto[fotoni_compton]
        N_rimbalzati = len(v_sopravvissuti)
        print(f"Fotoni che fanno scattering Compton: {N_rimbalzati:,}")
    else:
        N_rimbalzati = 0
        print("Nessun fotone ha superato il Trigger A!")

    # --- E. DINAMICA COMPTON (Il Rimbalzo) ---
    if N_rimbalzati > 0:
        print("Calcolo angoli di Klein-Nishina e deviazione 3D...")
        theta_array = genera_angoli_compton(N_rimbalzati)
        
        #Per i phi dovrei fare tra 0 e 2pi, ma così non vengono abbastanza fotoni. 
        #phi_array = np.random.uniform(0, 2*np.pi, N_rimbalzati)
        #Così facendo constringo i fotoni emessi a stare più attaccati e avrò più coincidenze
        phi_array = np.random.uniform(-0.2, 0.2, N_rimbalzati)

        v_out = deviazione_compton_3D(v_sopravvissuti, theta_array, phi_array)
        
        # (Calcoliamo subito le energie finali per i grafici diagnostici)
        E_out_mc_teorica = E0_KEV / (2.0 - np.cos(theta_array))
        # errore di lettura del cristallo NaI (Risoluzione ~7%)
        sigma_E = E_out_mc_teorica * 0.07 / 2.355
        E_out_mc = np.random.normal(E_out_mc_teorica, sigma_E)

        # --- F. LO SCAN ANGOLARE ---
        angoli_sperimentali_deg = np.array([30, 45, 60, 90, 120])
        conteggi_veri = []
        
        print("\nInizio scansione angolare del Rivelatore B:")
        for angolo_deg in angoli_sperimentali_deg:
            angolo_rad = np.radians(angolo_deg)
            
            # Posizione e asse del Rivelatore B 
            # D_pb_B è 26.0 cm secondo la tua figura (D_B nei globals è 36, facciamo finta sia 26 fisso al braccio)
            D_braccio = 26.0
            # Il centro_B parte dalla coordinata X del piombo (D_pb) e si sposta secondo il raggio D_braccio
            centro_B = np.array([D_pb + D_braccio * np.cos(angolo_rad), D_braccio * np.sin(angolo_rad), 0.0])
            asse_B = np.array([np.cos(angolo_rad), np.sin(angolo_rad), 0.0]) # Guarda avanti verso il piombo
            
            # Geometria
            mask_hit_B, cos_in_B, t_aria_B = hit_disco(
                sources=P_nuova_partenza, 
                directions=v_out, 
                det_center=centro_B, 
                det_axis=asse_B, 
                R_det=R_B
            )
            
            # Fisica 
            mask_coincidenze, _ = rivelazione_dinamica_B(
                maschera_hit_B=mask_hit_B, 
                theta=theta_array, 
                cos_in_B=cos_in_B, 
                t_aria_B=t_aria_B, 
                spessore_NaI=L_B
            )
            
            coincidenze_totali = np.sum(mask_coincidenze)
            conteggi_veri.append(coincidenze_totali)
            
            print(f" > Angolo {angolo_deg:3d}° : Trovate {coincidenze_totali} coincidenze")

        # Trasformiamo la lista in array NumPy per passarla alla funzione di plot
        conteggi_veri = np.array(conteggi_veri)
        conteggi_accidentali = np.zeros_like(conteggi_veri)
        errori_poisson = np.sqrt(conteggi_veri)

        # --- G. I GRAFICI FINALI ---
        """"TOGLIERE IL COMMENTO SE VOGLIO I GRAFICI SINGOLI
        print("\nGenerazione dei grafici in corso...")
        
        # 1. Verifica Klein-Nishina
        plot_diagnostico_kn(theta_array)
        
        # 2. Verifica cinematica energie
        plot_cinematica_compton(theta_array, E_out_mc)
        
        # 3. Mappa di Hit sul Piombo (Bonus diagnostico visivo)
        plot_hit_bersaglio(
            P_impatto, maschera_impatto_pb, fotoni_compton, W_pb=W_pb, H_pb=H_pb
        )

        # 4. IL GRAFICO PIÙ IMPORTANTE: Coincidenze vs Angolo
        plot_coincidenze_compton(
            angoli_sperimentali_deg, 
            conteggi_veri, 
            conteggi_accidentali, 
            errori_poisson,
            normalizza=True
        )
        """ 
        # 5. Il Pannellone riassuntivo
        plot_summary(
            theta_array, E_out_mc, angoli_sperimentali_deg, 
            conteggi_veri, conteggi_accidentali, errori_poisson
        )
        
        print("Simulazione completata con successo!")