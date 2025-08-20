from typing import FrozenSet, Iterable, List
import warnings

from dowhy import CausalModel as DowhyCausalModel
import networkx as nx
from pgmpy.inference.CausalInference import CausalInference
from pgmpy.models import DiscreteBayesianNetwork


class Identifier:
    def __init__(self, causal_model):
        self.causal_model = causal_model
        self.X = causal_model.interventions[0].label
        self.Y = causal_model.target.label
        self.latent_labels = {u.label for u in (causal_model.unobservables or [])}
        self.G = causal_model.graph.DAG
        dbn = DiscreteBayesianNetwork()
        dbn.add_edges_from(self.G.edges())
        self.infer = CausalInference(dbn)
        self.data_obs = causal_model.data.drop(
            columns=[u for u in self.latent_labels if u in causal_model.data.columns]
        )

    def find_backdoor(self):
        """
        Find a backdoor adjustment set for the causal model.
        """
        backdoors = self.infer.get_all_backdoor_adjustment_sets(X=self.X, Y=self.Y)
        obs_bds = self._filter_observed_sets(backdoors)
        if obs_bds:
            Z = min(obs_bds, key=len)
            return Z

        return None

    def find_frontdoor(self):
        """
        Find a frontdoor adjustment set for the causal model.
        """
        frontdoors = self.infer.get_all_frontdoor_adjustment_sets(X=self.X, Y=self.Y)
        obs_fds = self._filter_observed_sets(frontdoors)
        if obs_fds:
            Z = min(obs_fds, key=len)
            return Z

        return None

    def find_instrumental_variable(self):
        raise NotImplementedError

    def check_unobservable_confounding(self) -> bool:
        """
        Check for an unobservable confounder in the causal model.
        """
        for U in self.causal_model.unobservables or []:
            if nx.has_path(self.G, U.label, self.X) and nx.has_path(
                self.G, U.label, self.Y
            ):
                return True

        return False

    def graphical_identification(self):
        """
        Check if X and Y are d-separated after removing X's incoming edges.
        """
        G_do = self.G.copy()
        G_do.remove_edges_from(list(G_do.in_edges(self.X)))
        if nx.is_d_separator(G_do, {self.X}, {self.Y}, set()):
            return True

        return False

    def id_algorithm_identification(self):
        """
        Identify the causal effect using the ID algorithm.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dowhy_model = DowhyCausalModel(
                data=self.data_obs, graph=self.G, treatment=self.X, outcome=self.Y
            )
            try:
                estimand = dowhy_model.identify_effect(
                    method_name="id-algorithm", proceed_when_unidentifiable=False
                )
                return estimand
            except Exception:
                return None

    def _filter_observed_sets(
        self, sets: Iterable[FrozenSet[str]]
    ) -> List[FrozenSet[str]]:
        """
        Returns only those adjustment-sets Z that do not contain any latent variable.
        """
        return [Z for Z in sets if Z.isdisjoint(self.latent_labels)]
