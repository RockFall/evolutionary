import numpy as np


class CFG:
    def __init__(self, grammar, X_size=2, max_tree_depth=None, min_tree_depth=None, shortest_path=None):
        self.grammar = grammar
        self.max_tree_depth = max_tree_depth
        self.min_tree_depth = min_tree_depth
        self.shortest_path = shortest_path

        self.non_terminals = list(self.grammar.keys())
        self.terminals = grammar['<op>'] + grammar['<var>']
        self.start_rule = '<start>'
        self.start_rule_tuple = ('<start>', 'NT')
        self.n_options_by_non_terminal = {'<start>': 1, '<expr>': 3, '<op>': X_size+5, '<var>': 2}

        self.index_of_non_terminal = {}
        self.gen_pcfg()

    def gen_pcfg(self):
        # Generating the PCFG
        max_num_of_options = max(self.n_options_by_non_terminal.values())
        arr = np.zeros(shape=(len(self.grammar.keys()), max_num_of_options))

        for i, nt in enumerate(self.grammar):
            number_probs = len(self.grammar[nt])
            prob = 1.0 / number_probs
            arr[i,:number_probs] = prob
            if nt not in self.index_of_non_terminal:
                self.index_of_non_terminal[nt] = i

        self.pcfg = arr
        self.pcfg_mask = self.pcfg != 0

    def mapping(self, genotype, to_map):
        # Mapping the genotype to the phenotype
        output = []
        max_depth = self._mapping_aux(genotype, to_map, self.start_rule_tuple, 0, output)
        output = "".join(output)
        return output, max_depth

    def _mapping_aux(self, genotype, to_map, symbol, curr_depth, output):
        depths = [curr_depth]
        if symbol[1] == "T":
            output.append(symbol[0])
        else:
            curr_symbol_index = self.index_of_non_terminal[symbol[0]]
            choices = self.grammar[symbol[0]]
            codon = np.random.uniform()

            if to_map[curr_symbol_index] >= len(genotype[curr_symbol_index]):
                # Experiencia
                if curr_depth > self.max_tree_depth:
                    prob_non_recursive = 0.0
                    for rule in self.shortest_path[symbol][1:]:
                        index = self.grammar[symbol[0]].index(rule)
                        prob_non_recursive += self.pcfg[self.index_of_non_terminal[symbol[0]],index]
                    prob_aux = 0.0
                    for rule in self.shortest_path[symbol][1:]:
                        index = self.grammar[symbol[0]].index(rule)
                        new_prob = self.pcfg[self.index_of_non_terminal[symbol[0]],index] / prob_non_recursive
                        prob_aux += new_prob
                        if codon <= round(prob_aux,3):
                            expansion_possibility = index
                            break
                else:
                    prob_aux = 0.0
                    for index, option in enumerate(self.grammar[symbol[0]]):
                        prob_aux += self.pcfg[self.index_of_non_terminal[symbol[0]],index]
                        if codon <= round(prob_aux,3):
                            expansion_possibility = index
                            break
                genotype[curr_symbol_index].append([expansion_possibility,codon])
            else:
                # re-mapping with new probabilities                
                codon = genotype[curr_symbol_index][to_map[curr_symbol_index]][1]
                if curr_depth > self.max_tree_depth:
                    prob_non_recursive = 0.0
                    for rule in self.shortest_path[(symbol[0],'NT')][1:]:
                        index = self.grammar[symbol[0]].index(rule)
                        prob_non_recursive += self.pcfg[self.index_of_non_terminal[symbol[0]],index]
                    prob_aux = 0.0
                    for rule in self.shortest_path[(symbol[0],'NT')][1:]:
                        index = self.grammar[symbol[0]].index(rule)
                        new_prob = self.pcfg[self.index_of_non_terminal[symbol[0]],index] / prob_non_recursive
                        prob_aux += new_prob
                        if codon <= round(prob_aux,3):
                            expansion_possibility = index
                            break
                else:
                    prob_aux = 0.0
                    for index, option in enumerate(self.grammar[symbol[0]]):
                        prob_aux += self.pcfg[self.index_of_non_terminal[symbol[0]],index]
                        if codon <= round(prob_aux,3):
                            expansion_possibility = index
                            break
            # update mapping rules com a updated expansion possibility
            genotype[curr_symbol_index][to_map[curr_symbol_index]] = [expansion_possibility, codon]
            current_production = expansion_possibility
            to_map[curr_symbol_index] += 1
            next_to_expand = choices[current_production]
            for next_sym in next_to_expand:
                depths.append(
                    self._mapping_aux(genotype, to_map, next_sym, curr_depth + 1, output))
        return max(depths)
  
