"""
Ontologie OWL pour OncoPlan-Symbolique (Sujet N5).

Modelise les protocoles de chimiotherapie, les agents (medicaments), les
pathologies, et leurs relations (contre-indication, synergie) en utilisant
owlready2 (OWL 2 DL) avec raisonnement HermiT.

Bases cliniques :
- Cisplatine  : nephrotoxique majeur (CrCl doit etre >= 60 mL/min), ototoxique,
  neurotoxique a haute dose cumulee.
- Oxaliplatine: neurotoxicite cumulative dose-dependante (seuil ~500-600 mg/m2),
  toxicite renale/hematologique moderee.
- Carboplatine: profil renal plus favorable que le cisplatine, mais
  myelosuppresseur (thrombocytopenie).
- Docetaxel / Paclitaxel / Nab-Paclitaxel (taxanes): neurotoxicite
  peripherique (CIPN).
- Doxorubicine (anthracycline, utilisee dans AC-T, R-CHOP): cardiotoxique,
  dose cumulee maximale a vie ~450-550 mg/m2.
- Cyclophosphamide (agent alkylant, utilise dans AC-T, R-CHOP) :
  cardiotoxicite, cystite hemorragique.
- Vincristine (vinca-alcaloide, R-CHOP) : tres neurotoxique. Vinorelbine :
  profil neuro plus favorable, toxicite limitante hematologique.
- 5-FU: cardiotoxicite possible (rare), peu nephrotoxique.
- Gemcitabine: toxicite hematologique, peu nephro/neurotoxique aux doses
  standards.
- Irinotecan (inhibiteur de topoisomerase I, FOLFIRI) : toxicite
  hematologique et digestive.
- Rituximab (anticorps monoclonal, R-CHOP) : pas de toxicite renale/neuro
  significative dans ce modele.
- Prednisone (corticosteroide, R-CHOP) : pas de toxicite d'organe modelisee
  ici.
- Capecitabine (antimetabolite, prodrogue orale du 5-FU) : meme profil
  cardiotoxique que le 5-FU (spectre similaire de fluoropyrimidines :
  infarctus, angine, arythmies, insuffisance cardiaque -- Xeloda IB) ;
  neurotoxicite rare associee a un deficit en DPD.
- Methotrexate (antimetabolite, anti-folate) : nephrotoxique a haute dose
  (>500 mg/m2 : precipitation de cristaux dans les tubules distaux,
  necrose tubulaire directe -- Widemann & Adamson 2006, Nephrol Dial
  Transplant 2017). Seuil de clairance comparable au cisplatine.
- Trastuzumab (anticorps monoclonal anti-HER2) : CONTRAIREMENT au
  Rituximab, cardiotoxique documente (dysfonction VG, IC -- Bovelli et al.,
  Ann Oncol). Le risque de cardiotoxicite est lui-meme majore par une
  fonction renale diminuee (GFR < 78 mL/min/1.73m2 -- Tarantini et al.,
  PubMed 22714882), illustrant une interaction renal/cardiaque non
  modelisee explicitement ici (cf. limites).
- Bevacizumab (anticorps monoclonal anti-VEGF) : profil de toxicite dominé
  par l'hypertension et les evenements thromboemboliques, hors des 4 axes
  de toxicite modelises dans cette ontologie (renal/cardiaque/neuro/
  hematologique) -- limite assumee, cf. section Limites du notebook.
- Busulfan (agent alkylant) : cardiotoxicite rare mais documentee
  (tamponnade, fibrose endomyocardique -- Bovelli et al., Ann Oncol).

Sources : FDA drug labels (docetaxel, cisplatin, capecitabine/Xeloda),
revues sur la nephrotoxicite du cisplatine et du methotrexate (Widemann &
Adamson 2006 ; Launay-Vacher et al., Nephrol Dial Transplant 2017),
revues sur la neuropathie peripherique induite par chimiotherapie (CIPN),
litterature cardio-onco sur les anthracyclines, agents alkylants et
anticorps monoclonaux anti-HER2 (Bovelli et al., Ann Oncol ; Tarantini et
al. 2012, role de la fonction renale dans la cardiotoxicite du
trastuzumab).
agents alkylants.
"""

from owlready2 import (
    get_ontology, Thing, ObjectProperty, DataProperty,
    FunctionalProperty, sync_reasoner, World,
    AllDisjoint, destroy_entity, Or, Not
)

import itertools
_onto_counter = itertools.count()


def build_ontology():
    """Construit et retourne l'ontologie OWL OncoPlan.

    Utilise un World() owlready2 dedie (et non le default_world global) et
    un IRI unique a chaque appel. owlready2 fait
    correspondre une IRI a un objet ontologie UNIQUE dans un meme world --
    appeler get_ontology() deux fois avec la meme IRI retourne le MEME objet
    (et le meme etat RDF sous-jacent), au lieu d'une ontologie fraiche. Sans
    cette isolation, plusieurs appels a build_ontology() dans un meme
    processus (ex: plusieurs cellules d'un notebook) partageraient et
    pollueraient le meme graphe, ce qui peut notamment faire echouer le
    raisonneur HermiT sur des incoherences residuelles d'un appel precedent.
    """
    world = World()
    onto = world.get_ontology(f"http://onco-plan.epita.fr/ontologie_{next(_onto_counter)}.owl")

    with onto:

        # ----------------------------------------------------------------
        # Classes principales
        # ----------------------------------------------------------------
        class Agent(Thing):
            """Un agent de chimiotherapie (medicament)."""

        class Protocole(Thing):
            """Un protocole therapeutique (combinaison d'agents)."""

        class Pathologie(Thing):
            """Une pathologie ou comorbidite du patient."""

        class Patient(Thing):
            """Un patient (instance utilisee pour la verification)."""

        # Sous-classes d'Agent par famille pharmacologique
        class AgentPlatine(Agent):
            """Derives du platine : cisplatine, carboplatine, oxaliplatine."""

        class Taxane(Agent):
            """Stabilisateurs de microtubules : paclitaxel, docetaxel, nab-paclitaxel."""

        class VincaAlcaloide(Agent):
            """Inhibiteurs de polymerisation des microtubules : vincristine, vinorelbine."""

        class Anthracycline(Agent):
            """Intercalants de l'ADN : doxorubicine."""

        class Antimetabolite(Agent):
            """Analogues de nucleotides : 5-FU, gemcitabine."""

        class AnticorpsMonoclonal(Agent):
            """Immunotherapies ciblees : rituximab."""

        class TopoisomeraseInhibiteur(Agent):
            """Inhibiteurs de topoisomerase I : irinotecan."""

        class AgentAlkylant(Agent):
            """Agents alkylants de l'ADN : cyclophosphamide."""

        class Corticosteroide(Agent):
            """Anti-inflammatoires steroidiens utilises en oncologie : prednisone."""

        # Les familles pharmacologiques sont mutuellement exclusives :
        # un agent ne peut appartenir qu'a un seul mecanisme d'action
        # parmi ceux modelises ici.
        AllDisjoint([
            AgentPlatine, Taxane, VincaAlcaloide, Anthracycline,
            Antimetabolite, AnticorpsMonoclonal, TopoisomeraseInhibiteur,
            AgentAlkylant, Corticosteroide
        ])

        # Sous-classes de Pathologie (egalement disjointes : un objet
        # "Pathologie" instancie represente une seule comorbidite)
        class InsuffisanceRenale(Pathologie):
            pass

        class InsuffisanceCardiaque(Pathologie):
            pass

        class NeuropathiePreexistante(Pathologie):
            pass

        class Immunodepression(Pathologie):
            pass

        AllDisjoint([
            InsuffisanceRenale, InsuffisanceCardiaque,
            NeuropathiePreexistante, Immunodepression
        ])

        # ----------------------------------------------------------------
        # Proprietes de donnees (booleens / valeurs)
        # ----------------------------------------------------------------
        class toxicite_renale(Agent >> bool, FunctionalProperty):
            """Vrai si l'agent presente une nephrotoxicite cliniquement significative."""

        class toxicite_cardiaque(Agent >> bool, FunctionalProperty):
            """Vrai si l'agent presente une cardiotoxicite cliniquement significative."""

        class toxicite_neurologique(Agent >> bool, FunctionalProperty):
            """Vrai si l'agent presente une neurotoxicite peripherique cliniquement significative."""

        class toxicite_hematologique(Agent >> bool, FunctionalProperty):
            """Vrai si l'agent presente une myelosuppression (neutropenie,
            thrombocytopenie) cliniquement significative."""

        class dose_cumulee_max_mg_m2(Agent >> float, FunctionalProperty):
            """Seuil de dose cumulee a vie (mg/m2), au-dela duquel la toxicite
            devient inacceptable. Valeur absente = pas de seuil documente
            dans ce modele."""

        class seuil_crcl_min(Agent >> float, FunctionalProperty):
            """Clairance de la creatinine minimale (mL/min) requise pour
            administrer cet agent sans adaptation de dose."""

        # ----------------------------------------------------------------
        # Proprietes objet (relations)
        # ----------------------------------------------------------------
        class contre_indication(Agent >> Pathologie):
            """Un agent est contre-indique pour une pathologie donnee."""

        class incompatible_avec(Agent >> Agent):
            """Incompatibilite pharmacologique directe entre deux agents.
            Relation symetrique : peuplee dans les deux sens lors de la
            population de l'ontologie (cf. populate_ontology)."""

        class synergie(Agent >> Agent):
            """Effet synergique connu entre deux agents (information clinique,
            n'implique pas une contre-indication)."""

        class contient_agent(Protocole >> Agent):
            """Un protocole contient tel agent."""

        class possede_pathologie(Patient >> Pathologie):
            """Un patient presente telle pathologie/comorbidite."""

        class suit_protocole(Patient >> Protocole):
            """Un patient suit tel protocole."""

        # ----------------------------------------------------------------
        # Classe inferee (DL reasoning) : un agent est a "haut risque
        # toxique" s'il cumule au moins deux des trois toxicites d'organe
        # majeures. Cette classe n'est JAMAIS assignee manuellement : elle
        # est definie par une expression de classe (equivalent_to) et
        # c'est HermiT qui en deduit l'appartenance lors du raisonnement.
        # ----------------------------------------------------------------
        class AgentHautRisqueRenalCardiaque(Agent):
            """Agent inferable : nephrotoxique ET cardiotoxique."""
            equivalent_to = [
                Agent
                & toxicite_renale.value(True)
                & toxicite_cardiaque.value(True)
            ]

        class AgentHautRisqueRenalNeuro(Agent):
            """Agent inferable : nephrotoxique ET neurotoxique."""
            equivalent_to = [
                Agent
                & toxicite_renale.value(True)
                & toxicite_neurologique.value(True)
            ]

    return onto


def populate_ontology(onto):
    """Peuple l'ontologie avec des agents, protocoles et pathologies reels,
    bases sur la litterature pharmacologique (cf. docstring du module)."""

    AgentPlatine = onto.AgentPlatine
    Taxane = onto.Taxane
    VincaAlcaloide = onto.VincaAlcaloide
    Anthracycline = onto.Anthracycline
    Antimetabolite = onto.Antimetabolite
    AnticorpsMonoclonal = onto.AnticorpsMonoclonal
    TopoisomeraseInhibiteur = onto.TopoisomeraseInhibiteur
    AgentAlkylant = onto.AgentAlkylant
    Corticosteroide = onto.Corticosteroide
    InsuffisanceRenale = onto.InsuffisanceRenale
    InsuffisanceCardiaque = onto.InsuffisanceCardiaque
    NeuropathiePreexistante = onto.NeuropathiePreexistante
    Immunodepression = onto.Immunodepression
    Protocole = onto.Protocole

    # --- Pathologies ---
    insuff_renale = InsuffisanceRenale("insuffisance_renale_1")
    insuff_cardiaque = InsuffisanceCardiaque("insuffisance_cardiaque_1")
    neuropathie = NeuropathiePreexistante("neuropathie_preexistante_1")
    immunodepression = Immunodepression("immunodepression_1")

    # --- Agents : (classe, toxicite_renale, toxicite_cardiaque,
    #               toxicite_neuro, toxicite_hemato, dose_max_cumulee,
    #               seuil_crcl_min) ---
    # Bases cliniques : voir docstring du module pour les references.
    agents_spec = {
        "Cisplatine":      (AgentPlatine,   True,  False, True,  True,  None,  60.0),
        "Carboplatine":    (AgentPlatine,   False, False, False, True,  None,  None),
        "Oxaliplatine":    (AgentPlatine,   False, False, True,  True,  600.0, None),
        "Docetaxel":       (Taxane,         False, False, True,  True,  None,  None),
        "Paclitaxel":      (Taxane,         False, False, True,  True,  1000.0, None),
        "Nab_Paclitaxel":  (Taxane,         False, False, True,  True,  1000.0, None),
        "Vincristine":     (VincaAlcaloide, False, False, True,  False, None,  None),
        "Vinorelbine":     (VincaAlcaloide, False, False, False, True,  None,  None),
        "Doxorubicine":    (Anthracycline,  False, True,  False, True,  500.0, None),
        "5-FU":            (Antimetabolite, False, True,  False, False, None,  None),
        "Gemcitabine":     (Antimetabolite, False, False, False, True,  None,  None),
        "Capecitabine":    (Antimetabolite, False, True,  False, False, None,  None),
        "Methotrexate":    (Antimetabolite, True,  False, False, True,  None,  60.0),
        "Rituximab":       (AnticorpsMonoclonal, False, False, False, False, None, None),
        "Trastuzumab":     (AnticorpsMonoclonal, False, True,  False, False, None, None),
        "Bevacizumab":     (AnticorpsMonoclonal, False, False, False, False, None, None),
        "Irinotecan":      (TopoisomeraseInhibiteur, False, False, False, True, None, None),
        "Cyclophosphamide": (AgentAlkylant, False, True,  False, True,  None,  None),
        "Busulfan":        (AgentAlkylant, False, True,  False, True,  None,  None),
        "Prednisone":      (Corticosteroide, False, False, False, False, None, None),
    }

    agents = {}
    for name, (cls, renale, cardiaque, neuro, hemato, dose_max, crcl_min) in agents_spec.items():
        a = cls(name)
        a.toxicite_renale = renale
        a.toxicite_cardiaque = cardiaque
        a.toxicite_neurologique = neuro
        a.toxicite_hematologique = hemato
        if dose_max is not None:
            a.dose_cumulee_max_mg_m2 = dose_max
        if crcl_min is not None:
            a.seuil_crcl_min = crcl_min
        agents[name] = a

    # --- Contre-indications (agent -> pathologie) ---
    # Sources : labels FDA, revues CIPN, litterature cardio-onco (cf. module docstring)
    agents["Cisplatine"].contre_indication.append(insuff_renale)
    agents["Methotrexate"].contre_indication.append(insuff_renale)
    agents["Doxorubicine"].contre_indication.append(insuff_cardiaque)
    agents["Cyclophosphamide"].contre_indication.append(insuff_cardiaque)
    agents["Busulfan"].contre_indication.append(insuff_cardiaque)
    agents["Trastuzumab"].contre_indication.append(insuff_cardiaque)
    agents["Capecitabine"].contre_indication.append(insuff_cardiaque)
    agents["Oxaliplatine"].contre_indication.append(neuropathie)
    agents["Paclitaxel"].contre_indication.append(neuropathie)
    agents["Nab_Paclitaxel"].contre_indication.append(neuropathie)
    agents["Docetaxel"].contre_indication.append(neuropathie)
    agents["Vincristine"].contre_indication.append(neuropathie)
    # Myelosuppresseurs forts contre-indiques en cas d'immunodepression
    # preexistante (risque de neutropenie febrile potentiellement fatale)
    for name in ["Cisplatine", "Carboplatine", "Oxaliplatine", "Docetaxel",
                 "Paclitaxel", "Nab_Paclitaxel", "Vinorelbine", "Doxorubicine",
                 "Gemcitabine", "Irinotecan", "Cyclophosphamide",
                 "Methotrexate", "Busulfan"]:
        agents[name].contre_indication.append(immunodepression)

    # --- Incompatibilites (symetriques) ---
    def add_incompatible(a, b):
        agents[a].incompatible_avec.append(agents[b])
        agents[b].incompatible_avec.append(agents[a])

    # Cisplatine + Docetaxel : majoration documentee de la neuro/nephrotoxicite
    # en l'absence d'hydratation adequate (illustration pedagogique d'une
    # interaction directe entre deux agents, distincte d'une contre-indication
    # liee a une pathologie du patient).
    add_incompatible("Cisplatine", "Docetaxel")
    # Deux vinca-alcaloides neurotoxiques cumules : majoration du risque de
    # neuropathie severe (vincristine est le plus neurotoxique de la classe).
    add_incompatible("Vincristine", "Vinorelbine")

    # --- Synergies (informatives, correspondent aux associations reelles
    # des protocoles standards) ---
    def add_synergie(a, b):
        agents[a].synergie.append(agents[b])
        agents[b].synergie.append(agents[a])

    add_synergie("Carboplatine", "Paclitaxel")        # Carbo-Taxol
    add_synergie("Cisplatine", "Vinorelbine")         # Cisplatin-Vinorelbine
    add_synergie("Gemcitabine", "Nab_Paclitaxel")     # Gemcitabine-nab-Paclitaxel
    add_synergie("Doxorubicine", "Cyclophosphamide")  # AC-T, R-CHOP
    add_synergie("5-FU", "Irinotecan")                # FOLFIRI

    # --- Protocoles (correspondant aux 8 protocoles du dataset patients,
    # compositions completes conformes aux protocoles cliniques standards) ---
    protocoles_spec = {
        "FOLFOX":                      ["Oxaliplatine", "5-FU"],
        "FOLFIRI":                     ["Irinotecan", "5-FU"],
        "AC-T":                        ["Doxorubicine", "Cyclophosphamide", "Paclitaxel"],
        "Carbo-Taxol":                 ["Carboplatine", "Paclitaxel"],
        "Cisplatin-Vinorelbine":       ["Cisplatine", "Vinorelbine"],
        "Gemcitabine-nab-Paclitaxel":  ["Gemcitabine", "Nab_Paclitaxel"],
        "R-CHOP":                      ["Rituximab", "Cyclophosphamide", "Doxorubicine",
                                         "Vincristine", "Prednisone"],
        "Docetaxel":                   ["Docetaxel"],
    }

    protocoles = {}
    for name, agent_names in protocoles_spec.items():
        p = Protocole(name.replace("-", "_").replace(" ", "_"))
        for an in agent_names:
            p.contient_agent.append(agents[an])
        protocoles[name] = p

    return {
        "agents": agents,
        "pathologies": {
            "InsuffisanceRenale": insuff_renale,
            "InsuffisanceCardiaque": insuff_cardiaque,
            "NeuropathiePreexistante": neuropathie,
            "Immunodepression": immunodepression,
        },
        "protocoles": protocoles,
    }


def run_reasoner(onto):
    """Lance le raisonneur HermiT pour verifier la coherence de l'ontologie
    et inferer les classifications implicites.

    Note d'implementation : on passe `[onto]` (une LISTE contenant
    l'ontologie) plutot que `onto` directement. C'est necessaire avec un
    World() dedie (cf. build_ontology) : en interne, owlready2 serialise
    l'ontologie via `world.save(...)` lorsque l'argument n'est pas une
    liste, mais `world.save()` sans cible explicite peut serialiser une
    ontologie "anonyme" vide plutot que celle qu'on a construite, lorsque
    le world contient plusieurs ontologies. Passer une liste force
    owlready2 a appeler `onto.save(...)` directement sur l'ontologie
    voulue, qui serialise correctement son contenu reel."""
    with onto:
        sync_reasoner([onto], infer_property_values=True)


def verifier_prescription(onto, refs, protocole_nom, patient_pathologies):
    """Verifie une prescription via une requete SPARQL sur le graphe RDF
    sous-jacent a l'ontologie OWL (apres raisonnement HermiT).

    Args:
        onto: l'ontologie owlready2 (deja raisonnee).
        refs: dict retourne par populate_ontology.
        protocole_nom: nom du protocole (cle de refs["protocoles"]).
        patient_pathologies: liste de noms de pathologies presentes chez le
            patient (cles de refs["pathologies"]).

    Returns:
        Liste d'alertes (str). Liste vide = prescription validee.

    Raises:
        ValueError: si protocole_nom ou une pathologie de
            patient_pathologies n'existe pas dans `refs`. Le message
            d'erreur liste les cles valides pour faciliter le debogage --
            sans cette validation explicite, l'erreur par defaut serait un
            KeyError peu informatif leve au milieu de la fonction.
    """
    if protocole_nom not in refs["protocoles"]:
        raise ValueError(
            f"Protocole inconnu : '{protocole_nom}'. "
            f"Protocoles disponibles : {sorted(refs['protocoles'].keys())}"
        )
    pathologies_inconnues = set(patient_pathologies) - set(refs["pathologies"].keys())
    if pathologies_inconnues:
        raise ValueError(
            f"Pathologie(s) inconnue(s) : {sorted(pathologies_inconnues)}. "
            f"Pathologies disponibles : {sorted(refs['pathologies'].keys())}"
        )

    alertes = []
    protocole = refs["protocoles"][protocole_nom]
    pathologies_patient = [refs["pathologies"][p] for p in patient_pathologies]

    agents_du_protocole = list(protocole.contient_agent)

    # Le prefixe SPARQL doit correspondre a l'IRI reelle de CETTE ontologie
    # (generee dynamiquement par build_ontology, cf. note sur l'isolation
    # des World()) ; on ne peut pas le coder en dur.
    base_iri = onto.base_iri
    world = onto.world

    # --- Requete SPARQL : agents du protocole contre-indiques pour une
    # pathologie du patient ---
    for pathologie in pathologies_patient:
        query = f"""
        PREFIX onto: <{base_iri}>
        SELECT ?agent WHERE {{
            ?agent onto:contre_indication <{pathologie.iri}> .
        }}
        """
        results = list(world.sparql(query))
        agents_contre_indiques = {row[0] for row in results}
        for agent in agents_du_protocole:
            if agent in agents_contre_indiques:
                alertes.append(
                    f"ALERTE : {agent.name} (protocole {protocole_nom}) est "
                    f"contre-indique pour {pathologie.name}."
                )

    # --- Requete SPARQL : paires d'agents incompatibles dans le protocole ---
    query_incompat = f"""
    PREFIX onto: <{base_iri}>
    SELECT ?a1 ?a2 WHERE {{
        ?a1 onto:incompatible_avec ?a2 .
    }}
    """
    results = list(world.sparql(query_incompat))
    paires_incompatibles = {(row[0], row[1]) for row in results}

    for i, a1 in enumerate(agents_du_protocole):
        for a2 in agents_du_protocole[i + 1:]:
            if (a1, a2) in paires_incompatibles or (a2, a1) in paires_incompatibles:
                alertes.append(
                    f"ALERTE : incompatibilite entre {a1.name} et {a2.name} "
                    f"dans le protocole {protocole_nom}."
                )

    return alertes


def agents_haut_risque(onto):
    """Retourne les agents classifies par HermiT comme 'haut risque' (classes
    inferees AgentHautRisqueRenalCardiaque / AgentHautRisqueRenalNeuro).

    Ces classes ne sont JAMAIS peuplees manuellement : leur appartenance est
    deduite par le raisonneur a partir des proprietes de toxicite, ce qui
    illustre l'apport reel d'un raisonneur de description logique (HermiT)
    par rapport a un simple graphe de triplets (RDFLib sans raisonneur).
    """
    renal_cardiaque = list(onto.AgentHautRisqueRenalCardiaque.instances())
    renal_neuro = list(onto.AgentHautRisqueRenalNeuro.instances())
    return {
        "renal_cardiaque": renal_cardiaque,
        "renal_neuro": renal_neuro,
    }


if __name__ == "__main__":
    onto = build_ontology()
    refs = populate_ontology(onto)
    print(f"Ontologie construite : {len(list(onto.individuals()))} individus, "
          f"{len(list(onto.classes()))} classes.")
    run_reasoner(onto)
    print("Raisonnement HermiT termine sans incoherence.\n")

    # --- Classes inferees ---
    risques = agents_haut_risque(onto)
    print("--- Agents classifies 'haut risque' par inference HermiT ---")
    print("Renal + Cardiaque :", [a.name for a in risques["renal_cardiaque"]])
    print("Renal + Neuro     :", [a.name for a in risques["renal_neuro"]])

    # --- Tests de verification ---
    print("\n--- Test 1 : FOLFOX, patient sans comorbidite ---")
    print(verifier_prescription(onto, refs, "FOLFOX", []) or "OK, aucune alerte.")

    print("\n--- Test 2 : Cisplatin-Vinorelbine, patient insuffisant renal ---")
    print(verifier_prescription(onto, refs, "Cisplatin-Vinorelbine",
                                 ["InsuffisanceRenale"]))

    print("\n--- Test 3 : AC-T, patient insuffisant cardiaque ---")
    print(verifier_prescription(onto, refs, "AC-T", ["InsuffisanceCardiaque"]))

    print("\n--- Test 4 : Docetaxel, patient avec neuropathie preexistante ---")
    print(verifier_prescription(onto, refs, "Docetaxel",
                                 ["NeuropathiePreexistante"]))

    print("\n--- Test 5 : R-CHOP, patient immunodeprime ---")
    print(verifier_prescription(onto, refs, "R-CHOP", ["Immunodepression"]))

    print("\n--- Test 6 : FOLFIRI, patient sans comorbidite ---")
    print(verifier_prescription(onto, refs, "FOLFIRI", []) or "OK, aucune alerte.")
