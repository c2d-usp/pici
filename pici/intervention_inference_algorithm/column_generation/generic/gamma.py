from gurobipy import GRB, Var
from pici.graph.node import Node
from pici.intervention_inference_algorithm.column_generation.generic.bits import Bit, BitProduct
from pici.utils.probabilities_helper import find_conditional_probability


class Gamma:
    def __init__(self) -> None:
        pass

    def get_coef_from_objective_function(self, w_header: list, w_realization: list):
        coefw = 1
        for w_conditional_probability in self.Pw:
            w_target, w_conditioned = w_conditional_probability

            w_target.value = w_realization[w_header.index(w_target.label)]
            for node in w_conditioned:
                node.value = w_realization[w_header.index(node.label)]

            coefw *= find_conditional_probability(dataFrame=self.df, target_realization=[w_target], condition_realization=w_conditioned)
            

        if len(self.objective_function_vars_not_in_W) <= 0:
            return coefw

        coefq = 0
        q_header = self.realization_objective_function_vars_not_in_W[0]
        q_realizations = self.realization_objective_function_vars_not_in_W[1:]

        for q_realization in q_realizations:
            coef_parcial = 1
            for q_conditional_probability in self.Pq:
                q_target, q_conditioned = q_conditional_probability
                
                if q_target in self.reversed_ordered_W:
                    q_target.value = w_realization[w_header.index(q_target.label)]
                else:
                    q_target.value = q_realization[q_header.index(q_target.label)]
                
                for node in q_conditioned:
                    if node in self.reversed_ordered_W:
                        node.value = w_realization[w_header.index(node.label)]
                    else:
                        node.value = q_realization[q_header.index(node.label)]
                coef_parcial *= find_conditional_probability(dataFrame=self.df, target_realization=[q_target],condition_realization=q_conditioned)
            coefq += coef_parcial
        return coefq * coefw


    def linearize(self, W_realizations: list[list], considered_c_component_in_topological_order: list[Node]) -> dict:
        '''
        Gera o Yu (Gamma U): gamma_u_map_bit_product_to_linearized_variable
        Mapeia o produtório de bits e seu coef a uma linearização
        '''
        # TODO: Edge cases: intervention and target, apenas desprezat na realization
        gamma_u_map_bit_product_to_linearized_variable: dict[BitProduct, Var] = {}
        header = W_realizations[0]
        cartesian_products = W_realizations[1:]

        for realization in cartesian_products:
            # quais são as condições para essa função?
            coef = self.get_coef_from_objective_function(header, realization)
            bit_product = BitProduct()
            bit_product.set_coef(coef)

            # TODO: Devo desprezar a intervention e a target aqui?
            for node in considered_c_component_in_topological_order:
                parents_label = [parent.label for parent in node.parents]
                parents_realization = [realization[header.index(parent_label)] for parent_label in parents_label]
                realization_key: str = self.get_realization_key(parents_label, parents_realization)
                bit_gurobi_var = self.cluster_bits[node.label][realization_key]

                node_idx = header.index(node.label)
                sign = 1
                if realization[node_idx] == 0:
                    sign = -1
                new_bit = Bit(bit_gurobi_var, sign)

                bit_product.add_bit(new_bit)
            
            # TODO: Add variable name
            gamma_u_map_bit_product_to_linearized_variable[bit_product] = self.model.addVar(obj=coef, vtype=GRB.BINARY)
        
        '''
        TODO: Pode ser que o gurobi sabe linearizar o produtório.
        Basicamente teriamos uma lista de produtórios ao inveés de um dicionário mapeando uma nova variável.
        Para cada produtório:
            addConstr(0 <= produtorio <= 1)
        '''

        return gamma_u_map_bit_product_to_linearized_variable

    def _get_node_bit_variable_given_parents_realization(self, node: Node, w_realization: list[int], w_header: list[str]) -> Var:
        parents_label = [parent.label for parent in node.parents]
        parents_realization = [w_realization[w_header.index(parent_label)] for parent_label in parents_label]
        realization_key: str = self.get_realization_key(parents_label, parents_realization)
        return self.cluster_bits[node.label][realization_key]


    def add_linearized_bit_products_constraints(self, gamma_u_map_bit_product_to_linearized_variable: dict[BitProduct, Var]) -> None:
        for bit_product, variable in gamma_u_map_bit_product_to_linearized_variable.items():
            # TODO: Add constraint name
            self.model.addConstr(variable >= 0)
            # TODO: Add constraints name
            self.model.addConstr(variable <= 1)

            sum_bits = 0
            for bit in bit_product.bit_list:
                one_or_zero = 0
                if bit.sign == -1:
                    one_or_zero = 1
            
                # TODO: Add constraint name
                self.model.addConstr(variable <= one_or_zero + bit.sign*bit.gurobi_var)
                sum_bits += one_or_zero + bit.sign*bit.gurobi_var

            n = len(bit_product.bit_list)
            self.model.addConstr(variable >= 1 - n + sum_bits)
