import networkx as nx
from ortools.sat.python import cp_model


def _build_model(AF, required_in=frozenset()):
    model = cp_model.CpModel()

    args = list(AF.nodes())
    x = {a: model.NewBoolVar(f"x_{a}") for a in args}

    # Arguments imposés
    for a in required_in:
        model.Add(x[a] == 1)

    # Sans conflit
    for a, b in AF.edges():
        model.Add(x[a] + x[b] <= 1)

    # Défense
    for a in args:
        for attacker in AF.predecessors(a):

            defenders = list(AF.predecessors(attacker))

            if not defenders:
                model.Add(x[a] == 0)
            else:
                model.Add(sum(x[d] for d in defenders) >= x[a])

    return model, x, args


def _solve(AF, required_in=frozenset()):
    model, x, args = _build_model(AF, required_in)

    solver = cp_model.CpSolver()

    status = solver.Solve(model)

    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return None

    return {a for a in args if solver.Value(x[a])}


def preferred_extension(AF: nx.DiGraph, must_contain=None):
    """
    Retourne une extension preferred.

    Paramètres
    ----------
    AF : nx.DiGraph
        Graphe d'argumentation.

    must_contain : hashable | None
        Si spécifié, l'extension preferred retournée doit contenir
        cet argument. Si aucune n'existe, retourne set().

    Retour
    -------
    set
    """

    required = set()

    if must_contain is not None:

        if must_contain not in AF:
            return set()

        required.add(must_contain)

    current = _solve(AF, frozenset(required))

    if current is None:
        return set()

    while True:

        model, x, args = _build_model(AF, current)

        outside = [a for a in args if a not in current]

        if not outside:
            return current

        # sur-ensemble strict
        model.Add(sum(x[a] for a in outside) >= 1)

        solver = cp_model.CpSolver()

        status = solver.Solve(model)

        if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            return current

        current = {a for a in args if solver.Value(x[a])}