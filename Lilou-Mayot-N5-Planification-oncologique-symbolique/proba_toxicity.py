"""
Couche probabiliste PyMC pour OncoPlan-Symbolique (Sujet N5).

Modelise la reponse individuelle d'un patient a la chimiotherapie via un
profil de toxicite latent (Resistant / Normal / Sensible), inferee par
inference bayesienne (MCMC/NUTS) a partir de l'historique des doses
administrees et des taux de globules blancs observes.

Choix de modelisation par rapport a une implementation Pyro/SVI classique :
- La variable latente discrete `profil_toxicite` (3 categories) est
  MARGINALISEE via un melange de gaussiennes (pm.Mixture), plutot
  qu'echantillonnee directement. PyMC gere mal le melange de variables
  latentes discretes et continues dans un meme modele sequentiel ; la
  marginalisation est le pattern standard et recommande pour ce cas (elle
  evite aussi les problemes de "missing observations" que rencontre Pyro
  avec ses guides variationnels mal alignes).
- L'inference utilise NUTS (MCMC complet), pas une approximation
  variationnelle (SVI) : on obtient ainsi un echantillon de la distribution
  a posteriori complete, ce qui permet des diagnostics de convergence
  (R-hat, ESS) et des intervalles de credibilite, plutot qu'un simple point
  estimate.

Dynamique avec delai d'effet (PK/PD) :
    Le squelette CoursIA applique l'effet toxique de la dose au MEME pas de
    temps que l'administration. Cliniquement, le nadir des globules blancs
    survient typiquement 7-14 jours apres l'administration (cf. litterature
    sur la cinetique de la neutropenie chimio-induite), ce qui correspond
    dans la grille temporelle du dataset (J1/J8/J15/J21, pas de ~7 jours) a
    un delai d'environ UN pas de temps. On modelise donc l'effet toxique
    d'une dose comme reparti sur le pas de temps courant ET le pas suivant
    (avec un poids plus fort sur le pas suivant), plutot qu'instantane :

        effet_immediat(t)  = alpha * dose(t)
        effet_retarde(t)   = (1-alpha) * dose(t-1)   [contribution du pas precedent]
        toxicite_cumulee(t) = decroissance * toxicite_cumulee(t-1)
                              + gain * sensibilite * (effet_immediat(t) + effet_retarde(t))
        taux_GB(t) ~ Normal(8000 - 1000 * toxicite_cumulee(t), sigma_obs)

    Avec alpha=0.3 (30% d'effet immediat, 70% retarde d'un pas), le nadir
    apparait au pas suivant l'administration, conformement a la litterature
    clinique -- et conformement au pattern observe dans le dataset
    (GB minimal a J15, soit 1-2 pas apres l'administration de J1/J8).

Normalisation des doses (correction empirique post-validation) :
    Les doses brutes (mg) ne sont PAS comparables entre protocoles : 150mg
    de Carboplatine et 75mg de Cisplatine n'ont pas la meme signification
    toxique. Utiliser GAIN_TOXICITE * dose_brute en l'etat conflait
    "magnitude de dose propre au protocole" et "sensibilite propre au
    patient", ce qui produisait de faux positifs (patients sous protocoles
    a dose nominale elevee, comme Carbo-Taxol a 150mg, signales a risque
    alors qu'ils tolerent tres bien le traitement en realite -- cf.
    validation.py, cas P003/P005). La correction : on normalise chaque
    dose par la dose de reference du protocole (dose_prevue_mg a J1/cycle1,
    consideree comme "1.0 dose standard"), de sorte que GAIN_TOXICITE
    s'applique a une magnitude relative (fraction de dose standard) et non
    a un nombre de mg brut. Toute la variation inter-patient reste alors
    portee par le profil de sensibilite latent (Resistant/Normal/Sensible),
    qui est precisement ce que le modele est censé inferer.
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az


PROFILS = ["Resistant", "Normal", "Sensible"]
SENSIBILITE_MAP = np.array([0.5, 1.0, 2.0])  # Resistant, Normal, Sensible
PRIOR_PROFIL = np.array([0.1, 0.6, 0.3])     # meme prior que CoursIA
# Concentration du prior Dirichlet sur les poids de melange. Une
# concentration totale elevee (ex: x10, soit 10 "pseudo-observations" a
# priori) ancre fortement le posterior pres du prior meme avec des donnees
# informatives -- ce qui, empiriquement sur ce dataset, empechait le
# posterior de bien distinguer "Resistant" de "Normal" pour des patients
# dont la trajectoire reelle correspond clairement au profil resistant
# (cf. validation.py, cas P003/P005). Une concentration plus faible (x3)
# laisse les 6-8 observations reelles peser davantage que le prior.
PRIOR_CONCENTRATION = 3.0
TAUX_BASE_GB = 8000.0

# --- Conversion ANC <-> WBC total -----------------------------------------
# La vraie mesure clinique de la neutropenie severe est l'ANC (Absolute
# Neutrophil Count), pas le WBC total : neutropenie legere ANC<1500/uL,
# moderee ANC<1000/uL, severe ANC<500/uL (NCI-CTCAE / IDSA -- Management of
# Neutropenia in Cancer Patients, PMC4059501). Le dataset ne fournit que le
# WBC total (pas de differentiel neutrophiles/lymphocytes), donc le modele
# doit convertir.
#
# Chez un patient SAIN, les neutrophiles representent typiquement 55-70% du
# WBC total (News-Medical, "What is ANC and How is it Measured"). On
# retient 60% comme ratio par defaut (RATIO_ANC_WBC_DEFAUT ci-dessous).
#
# LIMITE IMPORTANTE, identifiee explicitement : ce ratio n'est PAS stable
# sous chimiotherapie myelosuppressive. Le Blood Project (thebloodproject.com)
# donne l'exemple d'un patient sous forte dose de chimiotherapie avec un WBC
# total tres bas (0.1x10^9/L) ou les neutrophiles representent PRES DE 100%
# du differentiel -- l'inverse du ratio "sain". La conversion lineaire
# ci-dessous est donc une approximation qui peut SOUS-estimer le risque
# reel en phase de nadir profond (le vrai ANC peut chuter relativement plus
# que le WBC total ne le suggere, ou au contraire moins si la proportion de
# neutrophiles augmente -- la direction de l'erreur depend du patient). Sans
# differentiel reel dans les donnees, aucune conversion ne peut etre exacte ;
# le ratio est expose comme parametre pour permettre une analyse de
# sensibilite plutot que de masquer l'approximation derriere une constante
# opaque.
RATIO_ANC_WBC_DEFAUT = 0.60
SEUIL_ANC_NEUTROPENIE_SEVERE = 500.0   # cellules/uL, NCI-CTCAE/IDSA
SEUIL_ANC_NEUTROPENIE_MODEREE = 1000.0
SEUIL_ANC_NEUTROPENIE_LEGERE = 1500.0


def seuil_wbc_equivalent(seuil_anc=SEUIL_ANC_NEUTROPENIE_SEVERE,
                          ratio_anc_wbc=RATIO_ANC_WBC_DEFAUT):
    """Convertit un seuil clinique exprime en ANC vers son equivalent en
    WBC total, via seuil_wbc = seuil_anc / ratio_anc_wbc.

    Args:
        seuil_anc: seuil clinique ANC (cellules/uL). Par defaut, le seuil
            de neutropenie SEVERE (500/uL, NCI-CTCAE/IDSA).
        ratio_anc_wbc: proportion de neutrophiles dans le WBC total.
            Par defaut 0.60 (valeur "patient sain", cf. note ci-dessus sur
            l'instabilite de ce ratio sous chimiotherapie).

    Returns:
        Seuil equivalent en WBC total (cellules/uL).
    """
    if not (0.0 < ratio_anc_wbc <= 1.0):
        raise ValueError(f"ratio_anc_wbc doit etre dans (0, 1], recu {ratio_anc_wbc}.")
    return seuil_anc / ratio_anc_wbc


# Seuil de "danger" utilise par le reste du module, exprime en WBC total
# (cf. seuil_wbc_equivalent ci-dessus). Avec le ratio par defaut (0.60) et
# le seuil de neutropenie severe (500/uL ANC) : 500 / 0.60 ~= 833/uL.
# On retient toutefois 1500/uL comme seuil de "danger" operationnel -- plus
# conservateur que la conversion stricte -- car il correspond a la zone de
# neutropenie LEGERE en ANC (1500/uL) AVEC un ratio ANC/WBC degrade a ~1.0
# (cas extreme documente par le Blood Project, cf. note ci-dessus), c'est-
# a-dire le pire cas plausible plutot que le cas moyen. Ce choix delibere
# privilegie la securite du patient (moins de faux negatifs) au prix de
# plus de faux positifs -- coherent avec l'esprit prudent du reste du
# pipeline (cf. Section VI du notebook, decomposition bayesienne du risque).
SEUIL_CRITIQUE_GB = 1500.0
DECROISSANCE_TOXICITE = 0.8  # recuperation entre administrations
# GAIN_TOXICITE recalibre pour des doses NORMALISEES (fraction de dose
# standard du protocole, ~1.0 typiquement, cf. normaliser_doses ci-dessous)
# plutot que des mg bruts. Calibre empiriquement pour reproduire l'amplitude
# de chute observee sur les donnees reelles (ex: P001/FOLFOX, profil
# "Normal" : GB ~7300 -> ~5000 apres 2 administrations a dose standard).
GAIN_TOXICITE = 2.2
# Proportion de l'effet toxique d'une dose ressentie AU MEME pas de temps
# (le reste, 1-ALPHA_EFFET_IMMEDIAT, est ressenti au pas suivant). Modelise
# le delai entre administration et nadir hematologique (cf. docstring).
ALPHA_EFFET_IMMEDIAT = 0.3


def normaliser_doses(doses, dose_reference):
    """Normalise un array de doses brutes (mg) par la dose de reference du
    protocole (dose_prevue_mg a J1/cycle 1), de sorte qu'une dose standard
    pleine corresponde a 1.0. Permet de comparer le risque toxique entre
    protocoles a dose nominale tres differente (cf. docstring du module).

    Args:
        doses: array de doses brutes (mg), 0 pour les jours sans administration.
        dose_reference: dose de reference du protocole (mg), > 0.

    Returns:
        array de doses normalisees (sans unite, ~1.0 = dose standard).
    """
    doses = np.asarray(doses, dtype=float)
    if dose_reference is None or dose_reference <= 0:
        raise ValueError("dose_reference doit etre un nombre strictement positif.")
    return doses / dose_reference


def calculer_trajectoire_toxicite(doses, sensibilite):
    """Calcule la trajectoire deterministe de toxicite cumulee et le taux
    de GB moyen attendu, pour UNE valeur scalaire de sensibilite, avec un
    delai d'effet partiel d'un pas de temps (cf. docstring du module).

    Args:
        doses: array (T,) des doses administrees (mg) a chaque temps t.
        sensibilite: scalaire (facteur de sensibilite du profil).

    Returns:
        mu_gb: array (T,) des taux de GB moyens attendus.
    """
    T = len(doses)
    toxicite = pt.zeros(())
    mu_list = []
    for t in range(T):
        effet_immediat = ALPHA_EFFET_IMMEDIAT * doses[t]
        effet_retarde = (1 - ALPHA_EFFET_IMMEDIAT) * (doses[t - 1] if t > 0 else 0.0)
        apport_dose = effet_immediat + effet_retarde
        toxicite = DECROISSANCE_TOXICITE * toxicite + GAIN_TOXICITE * apport_dose * sensibilite
        mu = TAUX_BASE_GB - 1000.0 * toxicite
        # Plancher abaisse (50% du seuil de danger) plutot qu'au seuil lui-
        # meme : sinon mu ne peut jamais descendre sous le seuil de danger,
        # ce qui rendrait toute detection de risque impossible par
        # construction (seul le bruit d'observation pourrait alors
        # declencher une alerte, jamais la tendance centrale du modele).
        mu = pt.maximum(mu, SEUIL_CRITIQUE_GB * 0.3)
        mu_list.append(mu)
    return pt.stack(mu_list)


def construire_modele(doses, observations, dose_reference, sigma_obs=500.0):
    """Construit le modele PyMC de melange sur les 3 profils de toxicite.

    Le profil latent discret est marginalise : la vraisemblance des
    observations est un melange (pm.Mixture) des 3 vraisemblances
    conditionnelles (une par profil), pondere par le prior sur le profil.
    PyMC met alors a jour ce prior en un posterior sur les memes 3 poids
    via l'inference NUTS standard (pas de variable discrete latente a
    echantillonner directement).

    Args:
        doses: array (T,) doses administrees (mg), historique reel observe.
        observations: array (T,) taux de GB observes (meme longueur que
            doses -- tronque a l'historique reel, jamais aux doses futures
            sans observation correspondante).
        dose_reference: dose standard du protocole (mg, dose_prevue_mg a
            J1/cycle1), utilisee pour normaliser les doses (cf. docstring
            du module : evite de conflater magnitude de dose et sensibilite
            patient).
        sigma_obs: ecart-type de bruit de mesure sur le taux de GB.

    Returns:
        modele PyMC (context manager).
    """
    doses = normaliser_doses(doses, dose_reference)
    observations = np.asarray(observations, dtype=float)
    if len(doses) != len(observations):
        raise ValueError(
            f"doses et observations doivent avoir la meme longueur "
            f"(recu {len(doses)} doses et {len(observations)} observations). "
            f"Tronquer les doses futures sans observation correspondante."
        )
    if len(doses) == 0:
        raise ValueError("doses et observations ne peuvent pas etre vides.")

    with pm.Model() as modele:
        # Prior categoriel sur le profil (poids du melange)
        poids_profil = pm.Dirichlet(
            "poids_profil", a=PRIOR_PROFIL * PRIOR_CONCENTRATION
        )

        # Pour chaque profil, calculer la trajectoire deterministe de mu_gb
        composantes = []
        for k, sens in enumerate(SENSIBILITE_MAP):
            mu_k = calculer_trajectoire_toxicite(doses, sens)
            composantes.append(mu_k)

        # mu_stack : shape (3, T) -> (T, 3) pour le Mixture
        mu_stack = pt.stack(composantes, axis=0).T  # (T, 3)

        pm.Mixture(
            "taux_gb_obs",
            w=poids_profil,
            comp_dists=pm.Normal.dist(mu=mu_stack, sigma=sigma_obs),
            observed=observations,
        )

    return modele


def inferer_profil(doses, observations, dose_reference, n_samples=1000,
                    n_tune=1000, chains=2):
    """Lance l'inference MCMC/NUTS et retourne le posterior sur le profil
    de toxicite ainsi que les diagnostics de convergence.

    Args:
        dose_reference: dose standard du protocole (mg), pour normaliser
            les doses brutes avant inference (cf. construire_modele).

    Returns:
        dict avec "idata" (InferenceData ArviZ), "probs_posterior"
        (array de 3 probabilites), "rhat_max", "profil_le_plus_probable".
    """
    modele = construire_modele(doses, observations, dose_reference)
    with modele:
        idata = pm.sample(
            draws=n_samples, tune=n_tune, chains=chains, cores=1,
            progressbar=False, random_seed=42,
            target_accept=0.9,
        )

    probs_posterior = idata.posterior["poids_profil"].mean(dim=("chain", "draw")).values
    rhat = az.rhat(idata)
    rhat_max = float(rhat["poids_profil"].max())
    profil_idx = int(np.argmax(probs_posterior))

    return {
        "idata": idata,
        "probs_posterior": probs_posterior,
        "profil_probable": PROFILS[profil_idx],
        "rhat_max": rhat_max,
    }


def calculer_trajectoire_toxicite_numpy(doses, sensibilite):
    """Version numpy (non-symbolique) de la meme dynamique que
    calculer_trajectoire_toxicite, utilisee pour les simulations Monte
    Carlo de risque futur (simuler_risque_futur). Garder les deux
    implementations synchronisees est important : c'est exactement le
    type d'incoherence (deux copies de la meme formule qui divergent) qui
    avait initialement masque le sous-risque de P008 dans la validation.
    """
    toxicite = 0.0
    mu_list = []
    for t, d in enumerate(doses):
        effet_immediat = ALPHA_EFFET_IMMEDIAT * d
        effet_retarde = (1 - ALPHA_EFFET_IMMEDIAT) * (doses[t - 1] if t > 0 else 0.0)
        apport_dose = effet_immediat + effet_retarde
        toxicite = DECROISSANCE_TOXICITE * toxicite + GAIN_TOXICITE * apport_dose * sensibilite
        mu = max(TAUX_BASE_GB - 1000.0 * toxicite, SEUIL_CRITIQUE_GB * 0.3)
        mu_list.append(mu)
    return mu_list


def simuler_risque_futur(probs_posterior, doses_historique, dose_future,
                          dose_reference, n_simulations=5000,
                          seuil_danger=SEUIL_CRITIQUE_GB, fenetre_suivi=3,
                          seed=42):
    """Simule l'effet d'une dose future sur le taux de GB, en tirant le
    profil dans le posterior inferre, et estime P(GB < seuil_danger) sur
    le NADIR (minimum) atteint dans la fenetre de suivi post-administration,
    et non uniquement sur la valeur au dernier pas de temps simule.

    Justification clinique : le risque de neutropenie severe / febrile
    survient typiquement au nadir (creux), qui apparait quelques jours
    apres l'administration (J15 dans le cycle standard a J1/J8/J21), pas
    necessairement au moment exact ou l'on evalue le modele. Ne regarder
    que le dernier point simule sous-estime systematiquement le risque
    pour les protocoles a administration peu frequente (ex: un seul jour
    d'administration par cycle), ou la "recuperation" (decroissance de la
    toxicite cumulee) a deja partiellement eu lieu avant le point evalue.

    C'est l'approche "Jumeau Numerique" : on rejoue la dynamique deterministe
    (avec son delai d'effet, cf. calculer_trajectoire_toxicite_numpy) + bruit
    d'observation pour chaque tirage de profil, sur plusieurs pas de temps
    suivant l'administration, afin de capturer le pire moment plutot qu'un
    instantane arbitraire.

    Args:
        dose_reference: dose standard du protocole (mg), pour normaliser
            les doses brutes (historique + future), cf. normaliser_doses.
        fenetre_suivi: nombre de pas de temps supplementaires simules
            APRES la dose future, sans nouvelle administration (= periode
            de suivi/repos), pour capturer le nadir reel.
    """
    rng = np.random.default_rng(seed)
    # On simule : historique + dose future + (fenetre_suivi-1) pas de repos
    # (dose = 0), pour observer toute la descente jusqu'au nadir puis le
    # debut de la recuperation. Normalisation appliquee sur l'ensemble de
    # la sequence (les zeros de repos restent des zeros apres division).
    doses_brutes = np.concatenate([
        np.asarray(doses_historique, dtype=float),
        [dose_future],
        np.zeros(fenetre_suivi - 1),
    ])
    doses_completes = normaliser_doses(doses_brutes, dose_reference)

    risques = []
    for _ in range(n_simulations):
        profil_idx = rng.choice(3, p=probs_posterior)
        sensibilite = SENSIBILITE_MAP[profil_idx]

        mu_list = calculer_trajectoire_toxicite_numpy(doses_completes, sensibilite)
        trajectoire_gb = [rng.normal(mu, 500.0) for mu in mu_list]

        # On ne regarde que les pas de temps APRES l'administration de la
        # dose future (l'historique passe n'est pas le risque qu'on evalue)
        nadir_futur = min(trajectoire_gb[len(doses_historique):])
        risques.append(nadir_futur < seuil_danger)

    p_danger = float(np.mean(risques))
    return p_danger


def analyse_sensibilite_anc_wbc(probs_posterior, doses_historique, dose_future,
                                 dose_reference, ratios_testes=None,
                                 seuil_anc=SEUIL_ANC_NEUTROPENIE_SEVERE,
                                 n_simulations=5000, fenetre_suivi=3, seed=42):
    """Analyse de sensibilite : fait varier l'hypothese de ratio ANC/WBC
    (cf. seuil_wbc_equivalent et la limite documentee sur son instabilite
    sous chimiotherapie) et mesure l'impact sur le risque estime P(danger).

    Cette fonction repond directement a la limite "architecture
    sequentielle, incertitude non propagee entre couches" : plutot que de
    se fier a un seuil de danger fixe (SEUIL_CRITIQUE_GB), elle quantifie
    explicitement combien la decision de dose serait differente sous des
    hypotheses cliniques alternatives sur le ratio ANC/WBC -- une forme de
    propagation d'incertitude (ici, sur un parametre du modele plutot que
    sur le profil patient lui-meme, qui est deja propage via le posterior
    complet dans simuler_risque_futur).

    Args:
        ratios_testes: liste de ratios ANC/WBC a tester. Par defaut, balaie
            de 0.5 (ratio sain, borne basse de la litterature) a 1.0 (cas
            extreme documente sous myelosuppression severe, ou les
            neutrophiles peuvent representer la quasi-totalite du WBC
            restant).
        seuil_anc: seuil clinique ANC de reference (cellules/uL). Par
            defaut, le seuil de neutropenie SEVERE (500/uL).

    Returns:
        DataFrame-like list of dicts avec les colonnes "ratio_anc_wbc",
        "seuil_wbc_equivalent", "p_danger", pour visualisation ou tableau.
    """
    if ratios_testes is None:
        ratios_testes = [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]

    resultats = []
    for ratio in ratios_testes:
        seuil_wbc = seuil_wbc_equivalent(seuil_anc=seuil_anc, ratio_anc_wbc=ratio)
        p_danger = simuler_risque_futur(
            probs_posterior, doses_historique, dose_future, dose_reference,
            n_simulations=n_simulations, seuil_danger=seuil_wbc,
            fenetre_suivi=fenetre_suivi, seed=seed,
        )
        resultats.append({
            "ratio_anc_wbc": ratio,
            "seuil_wbc_equivalent": round(seuil_wbc, 1),
            "p_danger": round(p_danger, 4),
        })
    return resultats


def recommander_dose(probs_posterior, doses_historique, dose_prevue,
                      dose_reference, seuil_risque_acceptable=0.05,
                      pas_reduction=0.25, max_reductions=3):
    """Applique la regle de decision du sujet : si P(GB < seuil) > 5% avec
    la dose prevue, propose une reduction de dose par paliers jusqu'a
    revenir sous le seuil de risque acceptable.

    Args:
        dose_reference: dose standard du protocole (mg), propagee a
            simuler_risque_futur pour la normalisation.
    """
    dose_test = dose_prevue
    for reduction_idx in range(max_reductions + 1):
        p_danger = simuler_risque_futur(
            probs_posterior, doses_historique, dose_test, dose_reference
        )
        if p_danger <= seuil_risque_acceptable:
            return {
                "dose_recommandee": dose_test,
                "reduction_pct": reduction_idx * pas_reduction * 100,
                "p_danger": p_danger,
                "decision": "Dose prevue maintenue." if reduction_idx == 0
                            else f"Dose reduite de {reduction_idx * pas_reduction * 100:.0f}%.",
            }
        dose_test = dose_prevue * (1 - (reduction_idx + 1) * pas_reduction)

    return {
        "dose_recommandee": 0.0,
        "reduction_pct": 100.0,
        "p_danger": p_danger,
        "decision": "Report du traitement recommande (risque residuel trop eleve "
                    "meme a dose minimale).",
    }


if __name__ == "__main__":
    # --- Test sur un patient reel du dataset (P001, FOLFOX) ---
    df = pd.read_csv("/home/claude/work/onco_project/data/patients_oncology.csv")
    p1 = df[df.patient_id == "P001"].sort_values(["cycle_numero", "jour_cycle"])

    doses_historique = p1["dose_reelle_mg"].values[:6].astype(float)  # jusqu'a J15 cycle 2
    observations = p1["taux_globules_blancs"].values[:6].astype(float)
    dose_reference = float(
        p1[(p1.cycle_numero == 1) & (p1.jour_cycle == 1)]["dose_prevue_mg"].iloc[0]
    )

    print("Doses (historique tronque) :", doses_historique)
    print("Observations GB :", observations)
    print("Dose de reference du protocole (J1/C1) :", dose_reference)
    print()

    print("=== Inference MCMC/NUTS sur le profil de toxicite (P001) ===")
    resultat = inferer_profil(doses_historique, observations, dose_reference)
    print(f"Probabilites a posteriori : "
          f"Resistant={resultat['probs_posterior'][0]:.3f}, "
          f"Normal={resultat['probs_posterior'][1]:.3f}, "
          f"Sensible={resultat['probs_posterior'][2]:.3f}")
    print(f"Profil le plus probable : {resultat['profil_probable']}")
    print(f"R-hat max (diagnostic convergence, doit etre proche de 1.0) : "
          f"{resultat['rhat_max']:.4f}")

    print("\n=== Simulation jumeau numerique : dose prevue J21 (85 mg) ===")
    p_danger = simuler_risque_futur(
        resultat["probs_posterior"], doses_historique,
        dose_future=85.0, dose_reference=dose_reference
    )
    print(f"P(neutropenie severe a J21 si dose maintenue) = {p_danger:.3f}")

    print("\n=== Recommandation de dose (regle : risque < 5%) ===")
    reco = recommander_dose(resultat["probs_posterior"], doses_historique,
                             dose_prevue=85.0, dose_reference=dose_reference)
    print(reco)
