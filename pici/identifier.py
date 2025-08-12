from pgmpy.inference.CausalInference import CausalInference
from pgmpy.models import DiscreteBayesianNetwork
import networkx as nx
from typing import Iterable, Set, FrozenSet, List
import warnings
from dowhy import CausalModel as DowhyCausalModel


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

        backdoors = self.infer.get_all_backdoor_adjustment_sets(X=self.X, Y=self.Y)
        obs_bds = self._filter_observed_sets(backdoors)
        if obs_bds:
            Z = min(obs_bds, key=len)
            return Z

        return None

    def find_frontdoor(self):

        frontdoors = self.infer.get_all_frontdoor_adjustment_sets(X=self.X, Y=self.Y)
        obs_fds = self._filter_observed_sets(frontdoors)
        if obs_fds:
            Z = min(obs_fds, key=len)
            return Z

        return None

    def find_instrumental_variable(self):

        observed = set(self.G.nodes()) - self.latent_labels
        for z in observed - {self.X, self.Y}:
            if not self.G.has_edge(z, self.X):
                continue
            bds_zy = self.infer.get_all_backdoor_adjustment_sets(X=z, Y=self.Y)
            if not self._all_sets_observed(bds_zy):
                continue
            paths = list(self._all_directed_paths(self.G, z, self.Y))
            if paths and all(self.X in path[1:-1] for path in paths):
                return z

        return None

    def check_unobservable_confounding(self) -> bool:

        for U in self.causal_model.unobservables or []:
            if nx.has_path(self.G, U.label, self.X) and nx.has_path(
                self.G, U.label, self.Y
            ):
                return True

        return False

    def graphical_identification(self):

        G_do = self.G.copy()
        G_do.remove_edges_from(list(G_do.in_edges(self.X)))
        if nx.is_d_separator(G_do, {self.X}, {self.Y}, set()):
            return True

        return False

    def id_algorithm_identification(self):

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

    @staticmethod
    def _all_directed_paths(G, src, dst):
        for path in nx.all_simple_paths(G, src, dst):
            if all(G.has_edge(u, v) for u, v in zip(path, path[1:])):
                yield path

    def _all_sets_observed(self, sets: Iterable[FrozenSet[str]]) -> bool:
        """
        Returns True iff every adjustment-set in sets is disjoint from the latents.
        """
        return all(Z.isdisjoint(self.latent_labels) for Z in sets)
