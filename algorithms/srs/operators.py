import numpy as np
import copy

def tournament(population, tsize=2):
    pool = np.random.choice(population, tsize)
    pool= sorted(pool, key = lambda i: i['fitness'])
    return copy.deepcopy(pool[0])

def roulette(population):
    fitnesses = np.array([p['fitness'] for p in population])
    total_fitness = fitnesses.sum()
    probs = [f/total_fitness for f in fitnesses]
    return np.random.choice(population, p=probs)

def crossover(p1, p2, grammar):
    xover_p_value = 0.5
    gen_size = len(p1['genotype'])
    mask = [np.random.uniform() for i in range(gen_size)]
    genotype = []
    for index, prob in enumerate(mask):
        if prob < xover_p_value:
            genotype.append(p1['genotype'][index][:])
        else:
            genotype.append(p2['genotype'][index][:])
    mapping_values = [0] * gen_size
    _, tree_depth = grammar.mapping(genotype, mapping_values)
    return {'genotype': genotype, 'fitness': None, 'mapping_values': mapping_values, 'tree_depth': tree_depth}

def mutate(p, pmutation, grammar):
    p = copy.deepcopy(p)
    p['fitness'] = None
    size_of_genes = grammar.n_options_by_non_terminal
    mutable_genes = [index for index, nt in enumerate(grammar.non_terminals) if size_of_genes[nt] != 1 and len(p['genotype'][index]) > 0]
    for at_gene in mutable_genes:
        nt = list(grammar.non_terminals)[at_gene]
        temp = p['mapping_values']
        mapped = temp[at_gene]
        for position_to_mutate in range(0, mapped):
            if np.random.uniform() < pmutation:
                current_value = p['genotype'][at_gene][position_to_mutate]
                # gaussian mutation
                codon = np.random.normal(current_value[1], 0.5)
                codon = min(codon,1.0)
                codon = max(codon,0.0)
                if p['tree_depth'] >= grammar.max_tree_depth:
                    prob_non_recursive = 0.0
                    for rule in grammar.shortest_path[(nt,'NT')][1:]:
                        index = grammar.grammar[nt].index(rule)
                        prob_non_recursive += grammar.pcfg[grammar.index_of_non_terminal[nt],index]
                    prob_aux = 0.0
                    for rule in grammar.shortest_path[(nt,'NT')][1:]:
                        index = grammar.grammar[nt].index(rule)
                        new_prob = grammar.pcfg[grammar.index_of_non_terminal[nt],index] / prob_non_recursive
                        prob_aux += new_prob
                        if codon <= round(prob_aux,3):
                            expansion_possibility = index
                            break
                else:
                    prob_aux = 0.0
                    for index, option in enumerate(grammar.grammar[nt]):
                        prob_aux += grammar.pcfg[grammar.index_of_non_terminal[nt],index]
                        if codon <= round(prob_aux,3):
                            expansion_possibility = index
                            break
                  
                p['genotype'][at_gene][position_to_mutate] = [expansion_possibility, codon]
    return p