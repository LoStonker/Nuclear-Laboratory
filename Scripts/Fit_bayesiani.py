





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
import pymc as pm
import arviz as az
from math import floor, ceil



import inspect
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from scipy.stats import chi2  # Necessario per il p-value

class FitBayesiano:
    def __init__(self, model_func, data_arrays, initial_params, custom_priors=None, xlabel="x", ylabel="y", title="Risultati del fit Bayesiano"):
        self.model = model_func
        self.x = np.asarray(data_arrays['x'])
        self.y = np.asarray(data_arrays['y'])
        self.sigma = np.asarray(data_arrays.get('sigma_y', np.ones_like(self.y)))
        
        sig = inspect.signature(model_func)
        self.param_names = list(sig.parameters.keys())[1:]
        
        self.initial_params = initial_params
        self.custom_priors = custom_priors if custom_priors is not None else {}
        
        self.fit_result: Any = None
        self.trace: Any = None
        self.chi2_val: Any = None
        self.dof: Any = None
        self.p_value: Any = None
        
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

    def perform_fit(self, draws=2000, tune=1000):
        with pm.Model() as self.pm_model:
            priors = {}
            for name in self.param_names:
                init_val = self.initial_params[name]
                prior_sigma = self.custom_priors.get(name, abs(init_val) * 10 if init_val != 0 else 1.0)
                
                # Se il valore iniziale è > 0, forziamo il sampler a non andare nei negativi
                if init_val > 0:
                    priors[name] = pm.TruncatedNormal(name, mu=init_val, sigma=prior_sigma, lower=0.0)
                else:
                    priors[name] = pm.Normal(name, mu=init_val, sigma=prior_sigma)
            
            # Calcolo del modello
            mu = self.model(self.x, **priors)
            
            # Likelihood
            pm.Normal('obs', mu=mu, sigma=self.sigma, observed=self.y)
            
            print("Avvio campionamento MCMC...")
            try:
                self.trace = pm.sample(draws=draws, tune=tune, chains=2, cores=1, 
                                       nuts_sampler="nutpie", return_inferencedata=True, progressbar=True)
            except Exception as e:
                print(f"Uso il sampler base (nutpie non disp: {e})")
                self.trace = pm.sample(draws=draws, tune=tune, chains=2, cores=1, 
                                       return_inferencedata=True, progressbar=True)

        # 1. Salvataggio risultati medi
        self.fit_result = {}
        for name in self.param_names:
            val = self.trace.posterior[name].mean().item()
            err = self.trace.posterior[name].std().item()
            self.fit_result[name] = (val, err)
            
        # 2. Calcolo pseudo-Chi2 e p-value usando le medie posteriori
        try:
            params_dict = {name: self.fit_result[name][0] for name in self.param_names}
            y_fit_mean = self.model(self.x, **params_dict)
            
            residuals = self.y - y_fit_mean
            valid_sigma = self.sigma > 0
            chisq_terms = np.zeros_like(self.y)
            chisq_terms[valid_sigma] = (residuals[valid_sigma] / self.sigma[valid_sigma])**2
            
            self.chi2_val = np.sum(chisq_terms)
            self.dof = len(self.x) - len(self.param_names)
            self.p_value = chi2.sf(self.chi2_val, self.dof) if self.dof > 0 else 0.0
            
        except Exception as e:
            print(f"Impossibile calcolare il Chi2/p-value: {e}")
            self.chi2_val, self.dof, self.p_value = np.nan, len(self.x) - len(self.param_names), np.nan

    def print_results(self):
        if self.fit_result is None:
            return print("Esegui prima perform_fit()!")
            
        print("\n--- Risultati del Fit Bayesiano ---")
        for name in self.param_names:
            val, err = self.fit_result[name]
            print(f"{name} = {val:.4e} ± {err:.4e}")
            
        if self.chi2_val is not None and not np.isnan(self.chi2_val):
            chi2_rid = self.chi2_val / self.dof if self.dof > 0 else np.nan
            print(f"\nStatistiche del Fit (calcolate sulle medie posteriori):")
            print(f"  Chi-quadro (χ²): {self.chi2_val:.4f}")
            print(f"  Gradi di libertà (DoF): {self.dof}")
            print(f"  Chi-quadro Ridotto (χ²/DoF): {chi2_rid:.4f}")
            print(f"  p-value: {self.p_value:.4f}")

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

    def plot_results(self, title_fontsize=14, label_fontsize=12,
                     info_box_pos='upper right', log_scale_y=False, log_scale_x=False,
                     param_labels=None): # <-- NUOVO ARGOMENTO
                     
        if self.fit_result is None:
            return print("Nessun risultato. Eseguire perform_fit().")
            
        if param_labels is None:
            param_labels = {}
        
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(10, 7))
        ax = plt.gca()

        # Dati Sperimentali
        plot_kwargs = {'fmt': 'o', 'label': 'Dati', 'markersize': 5, 
                       'capsize': 3, 'elinewidth': 1, 'markeredgecolor': 'k', 'zorder': 10}
        ax.errorbar(self.x, self.y, yerr=self.sigma, **plot_kwargs)

        # Curva di Fit
        if len(self.x) > 1:
             x_min_data, x_max_data = np.min(self.x), np.max(self.x)
             range_ext_factor = 0.05
             if log_scale_x:
                 if x_min_data <= 0: 
                     x_log_min = np.log10(np.min(self.x[self.x > 0]) * (1-range_ext_factor*2)) if np.any(self.x > 0) else -1
                 else: 
                     x_log_min = np.log10(x_min_data * (1-range_ext_factor))
                 x_log_max = np.log10(x_max_data * (1+range_ext_factor))
                 x_fit_plot = np.logspace(x_log_min, x_log_max, 400)
             else:
                 data_range = x_max_data - x_min_data
                 if data_range == 0: data_range = 1 
                 x_fit_plot = np.linspace(x_min_data - data_range*range_ext_factor, 
                                          x_max_data + data_range*range_ext_factor, 400)
        else:
             x_fit_plot = np.linspace(0, 1, 100)

        params_dict = {name: self.fit_result[name][0] for name in self.param_names}
        try:
             y_fit_curve = self.model(x_fit_plot, **params_dict)
             ax.plot(x_fit_plot, y_fit_curve, color='crimson', label='Fit Bayesiano', linewidth=2, zorder=5)
        except Exception as e:
             print(f"Errore calcolo curva di fit: {e}")

        # Assi
        ax.set_xlabel(self.xlabel, fontsize=label_fontsize)
        ax.set_ylabel(self.ylabel, fontsize=label_fontsize)
        ax.set_title(self.title, fontsize=title_fontsize, pad=15)
        if log_scale_y: ax.set_yscale('log')
        if log_scale_x: ax.set_xscale('log')
        ax.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')
        ax.grid(True, which='minor', linestyle=':', linewidth='0.3', color='lightgrey')
        ax.minorticks_on()

        # --- Box Informativo ---
        box_text_lines = ["$\\bf{Fit\\ Bayesiano\\ (MCMC)}$"]
        for name in self.param_names:
             val, err = self.fit_result.get(name, (np.nan, np.nan))
             val_str = f"{val:.3e}" if (abs(val) > 1e4 or (abs(val) < 1e-3 and val !=0)) else f"{val:.4g}"
             err_str = f"{err:.2e}" if (abs(err) > 1e3 or (abs(err) < 1e-4 and err !=0)) else f"{err:.2g}"
             
             # TRUCCO MAGICO: Se hai passato un nome custom, usa quello, altrimenti usa il nome della variabile
             display_name = param_labels.get(name, name)
             box_text_lines.append(f"${display_name} = {val_str} \\pm {err_str}$")
             
        # Aggiunta Chi2 e p-value nel box
        if self.chi2_val is not None and not np.isnan(self.chi2_val):
            if self.dof > 0:
                box_text_lines.append(f"$\\chi^2/N_{{dof}} = {self.chi2_val / self.dof:.2f}$ ($N_{{dof}}={self.dof}$)")
                if self.p_value is not None and not np.isnan(self.p_value):
                    box_text_lines.append(f"$p$-value $= {self.p_value:.3f}$")
            else:
                box_text_lines.append(f"$\\chi^2 = {self.chi2_val:.2f}$ (DoF={self.dof})")

        info_text = "\n".join(box_text_lines)
        box_coords = self._get_info_box_coords(info_box_pos)
        ax.annotate(info_text, xy=box_coords['xy'], xycoords='axes fraction',
                    va=box_coords['va'], ha=box_coords['ha'], fontsize=11,
                    bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.9, ec='grey'))

        ax.legend(fontsize=10, loc='best')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        
    def plot_diagnostics(self):
        if self.trace is not None:
            az.plot_posterior(self.trace, var_names=self.param_names)
            plt.show()



class FitBayesiano2:
    def __init__(self, model_func, data_arrays, initial_params, custom_priors=None, xlabel="x", ylabel="y", title="Risultati del fit Bayesiano"):
        self.model = model_func
        self.x = np.asarray(data_arrays['x'])
        self.y = np.asarray(data_arrays['y'])
        self.sigma = np.asarray(data_arrays.get('sigma_y', np.ones_like(self.y)))
        
        # Novità: Estrazione dell'errore su X (se esiste)
        self.sigma_x = np.asarray(data_arrays.get('sigma_x', np.zeros_like(self.x)))
        self.has_x_err = np.any(self.sigma_x > 0)
        
        sig = inspect.signature(model_func)
        self.param_names = list(sig.parameters.keys())[1:]
        
        self.initial_params = initial_params
        self.custom_priors = custom_priors if custom_priors is not None else {}
        
        self.fit_result = None
        self.trace = None
        self.chi2_val = None
        self.dof = None
        self.p_value = None
        
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

    def perform_fit(self, draws=2000, tune=1000):
        with pm.Model() as self.pm_model:
            # 1. Creazione Priors
            priors = {}
            for name in self.param_names:
                init_val = self.initial_params[name]
                prior_sigma = self.custom_priors.get(name, abs(init_val) * 10 if init_val != 0 else 1.0)
                
                if init_val > 0:
                    priors[name] = pm.TruncatedNormal(name, mu=init_val, sigma=prior_sigma, lower=0.0)
                else:
                    priors[name] = pm.Normal(name, mu=init_val, sigma=prior_sigma)
            
            # 2. LOGICA ODR (Error-in-Variables)
            if self.has_x_err:
                print("Rilevati errori su X: Attivazione modello Error-in-Variables (Bayesian ODR).")
                # PyMC tratterà ogni x reale come una variabile incognita distribuita attorno a x_misurata
                x_vero = pm.Normal('x_vero', mu=self.x, sigma=self.sigma_x, shape=len(self.x))
            else:
                x_vero = self.x
            
            # 3. Calcolo del modello e Likelihood
            mu = self.model(x_vero, **priors)
            pm.Normal('obs', mu=mu, sigma=self.sigma, observed=self.y)
            
            # 4. Campionamento
            print("Avvio campionamento MCMC...")
            try:
                self.trace = pm.sample(draws=draws, tune=tune, chains=2, cores=1, 
                                       nuts_sampler="nutpie", return_inferencedata=True, 
                                       progressbar=True, random_seed=42)
            except Exception as e:
                print(f"Uso il sampler base (nutpie non disp: {e})")
                self.trace = pm.sample(draws=draws, tune=tune, chains=2, cores=1, 
                                       return_inferencedata=True, progressbar=True, 
                                       random_seed=42)

        # 5. Salvataggio risultati (Ignoriamo x_vero per non affollare l'output)
        self.fit_result = {}
        for name in self.param_names:
            val = self.trace.posterior[name].mean().item()
            err = self.trace.posterior[name].std().item()
            self.fit_result[name] = (val, err)
            
        # 6. Pseudo-Chi2 (Calcolato come residuo ortogonale o verticale sulle medie)
        try:
            params_dict = {name: self.fit_result[name][0] for name in self.param_names}
            y_fit_mean = self.model(self.x, **params_dict)
            
            if self.has_x_err:
                # Approssimazione del Chi2 ortogonale (metodo delle differenze finite per la varianza totale)
                eps = 1e-8
                var_effettiva = np.zeros_like(self.y)
                for i in range(len(self.x)):
                    # Derivata numerica locale
                    dy_dx = (self.model(self.x[i] + eps, **params_dict) - self.model(self.x[i] - eps, **params_dict)) / (2 * eps)
                    var_effettiva[i] = self.sigma[i]**2 + (dy_dx * self.sigma_x[i])**2
                chisq_terms = ((self.y - y_fit_mean)**2) / var_effettiva
            else:
                valid_sigma = self.sigma > 0
                chisq_terms = np.zeros_like(self.y)
                chisq_terms[valid_sigma] = ((self.y[valid_sigma] - y_fit_mean[valid_sigma]) / self.sigma[valid_sigma])**2
            
            self.chi2_val = np.sum(chisq_terms)
            self.dof = len(self.x) - len(self.param_names)
            self.p_value = chi2.sf(self.chi2_val, self.dof) if self.dof > 0 else 0.0
            
        except Exception as e:
            print(f"Impossibile calcolare il Chi2/p-value: {e}")
            self.chi2_val, self.dof, self.p_value = np.nan, len(self.x) - len(self.param_names), np.nan

    def print_results(self):
        if self.fit_result is None:
            return print("Esegui prima perform_fit()!")
            
        print("\n--- Risultati del Fit Bayesiano ---")
        for name in self.param_names:
            val, err = self.fit_result[name]
            print(f"{name} = {val:.4e} ± {err:.4e}")
            
        if self.chi2_val is not None and not np.isnan(self.chi2_val):
            chi2_rid = self.chi2_val / self.dof if self.dof > 0 else np.nan
            print(f"\nStatistiche del Fit (sulle medie posteriori):")
            print(f"  Chi-quadro (χ²): {self.chi2_val:.4f}")
            print(f"  Gradi di libertà (DoF): {self.dof}")
            print(f"  Chi-quadro Ridotto (χ²/DoF): {chi2_rid:.4f}")
            print(f"  p-value: {self.p_value:.4f}")

    def calculate_confidence_band(self, x_points, num_sigma=1):
        if self.trace is None:
            print("Fit non valido o non eseguito. Impossibile calcolare la banda.")
            return None, None
            
        samples = {name: self.trace.posterior[name].values.flatten() for name in self.param_names}
        n_tot_samples = len(samples[self.param_names[0]])
        
        max_samples = min(1000, n_tot_samples)
        idx = np.random.choice(n_tot_samples, max_samples, replace=False)
        
        y_evals = np.zeros((max_samples, len(x_points)))
        for i in range(max_samples):
            p_dict = {name: samples[name][idx[i]] for name in self.param_names}
            y_evals[i, :] = self.model(x_points, **p_dict)
            
        y_model_on_x_points = np.mean(y_evals, axis=0)
        
        if num_sigma == 1:
            lower, upper = np.percentile(y_evals, [15.865, 84.135], axis=0)
        elif num_sigma == 2:
            lower, upper = np.percentile(y_evals, [2.275, 97.725], axis=0)
        elif num_sigma == 3:
            lower, upper = np.percentile(y_evals, [0.135, 99.865], axis=0)
        else:
            lower, upper = np.percentile(y_evals, [15.865, 84.135], axis=0)
            
        dy_confidence_band = (upper - lower) / 2.0
        
        return y_model_on_x_points, dy_confidence_band

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

    def plot_results(self, title_fontsize=14, label_fontsize=12,
                     info_box_pos='upper right', log_scale_y=False, log_scale_x=False,
                     param_labels=None, plot_confidence_band=False, confidence_sigma_level=1):
        if self.fit_result is None:
            return print("Nessun risultato. Eseguire perform_fit().")
            
        if param_labels is None:
            param_labels = {}
        
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(10, 7))
        ax = plt.gca()

        # --- Dati Sperimentali ---
        plot_kwargs = {'fmt': 'o', 'label': 'Dati', 'markersize': 5, 
                       'capsize': 3, 'elinewidth': 1, 'markeredgecolor': 'k', 'zorder': 10}
        if self.has_x_err:
            ax.errorbar(self.x, self.y, xerr=self.sigma_x, yerr=self.sigma, **plot_kwargs)
        else:
            ax.errorbar(self.x, self.y, yerr=self.sigma, **plot_kwargs)

        # --- Creazione asse X esteso ---
        if len(self.x) > 1:
             x_min_data, x_max_data = np.min(self.x), np.max(self.x)
             range_ext_factor = 0.05
             if log_scale_x:
                 if x_min_data <= 0: 
                     x_log_min = np.log10(np.min(self.x[self.x > 0]) * (1-range_ext_factor*2)) if np.any(self.x > 0) else -1
                 else: 
                     x_log_min = np.log10(x_min_data * (1-range_ext_factor))
                 x_log_max = np.log10(x_max_data * (1+range_ext_factor))
                 x_fit_plot = np.logspace(x_log_min, x_log_max, 400)
             else:
                 data_range = x_max_data - x_min_data
                 if data_range == 0: data_range = 1 
                 x_fit_plot = np.linspace(x_min_data - data_range*range_ext_factor, 
                                          x_max_data + data_range*range_ext_factor, 400)
        else:
             x_fit_plot = np.linspace(0, 1, 100)

        # --- Calcolo Curva e Bande ---
        params_dict = {name: self.fit_result[name][0] for name in self.param_names}
        try:
             y_fit_curve = self.model(x_fit_plot, **params_dict)
             ax.plot(x_fit_plot, y_fit_curve, color='crimson', label='Fit Bayesiano', linewidth=2, zorder=5)
             
             if plot_confidence_band:
                 y_model_band, dy_band = self.calculate_confidence_band(x_fit_plot, num_sigma=confidence_sigma_level)
                 if y_model_band is not None and dy_band is not None:
                     valid_band_indices = np.isfinite(y_model_band) & np.isfinite(dy_band)
                     ax.fill_between(x_fit_plot[valid_band_indices],
                                     (y_model_band - dy_band)[valid_band_indices],
                                     (y_model_band + dy_band)[valid_band_indices],
                                     color='salmon', alpha=0.35, zorder=3,
                                     label=f'Banda Cred. ({confidence_sigma_level}σ)')
        except Exception as e:
             print(f"Errore calcolo curva/banda di fit: {e}")

        # --- Assi e Griglia ---
        ax.set_xlabel(self.xlabel, fontsize=label_fontsize)
        ax.set_ylabel(self.ylabel, fontsize=label_fontsize)
        ax.set_title(self.title, fontsize=title_fontsize, pad=15)
        if log_scale_y: ax.set_yscale('log')
        if log_scale_x: ax.set_xscale('log')
        ax.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')
        ax.grid(True, which='minor', linestyle=':', linewidth='0.3', color='lightgrey')
        ax.minorticks_on()

        # --- Box Informativo ---
        box_text_lines = ["$\\bf{Fit\\ Bayesiano\\ (MCMC)}$"]
        if self.has_x_err:
             box_text_lines[0] = "$\\bf{Fit\\ Bayesiano\\ (ODR)}$"
             
        for name in self.param_names:
             val, err = self.fit_result.get(name, (np.nan, np.nan))
             val_str = f"{val:.3e}" if (abs(val) > 1e4 or (abs(val) < 1e-3 and val !=0)) else f"{val:.4g}"
             err_str = f"{err:.2e}" if (abs(err) > 1e3 or (abs(err) < 1e-4 and err !=0)) else f"{err:.2g}"
             display_name = param_labels.get(name, name)
             box_text_lines.append(f"${display_name} = {val_str} \\pm {err_str}$")
             
        if self.chi2_val is not None and not np.isnan(self.chi2_val):
            if self.dof > 0:
                box_text_lines.append(f"$\\chi^2/N_{{dof}} = {self.chi2_val / self.dof:.2f}$ ($N_{{dof}}={self.dof}$)")
                if self.p_value is not None and not np.isnan(self.p_value):
                    box_text_lines.append(f"$p$-value $= {self.p_value:.3f}$")
            else:
                box_text_lines.append(f"$\\chi^2 = {self.chi2_val:.2f}$ (DoF={self.dof})")

        info_text = "\n".join(box_text_lines)
        box_coords = self._get_info_box_coords(info_box_pos)
        ax.annotate(info_text, xy=box_coords['xy'], xycoords='axes fraction',
                    va=box_coords['va'], ha=box_coords['ha'], fontsize=11,
                    bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.9, ec='grey'),
                    zorder=10)

        ax.legend(fontsize=10, loc='best')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()