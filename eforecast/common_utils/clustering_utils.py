import random
import numpy as np
import skfuzzy as fuzz
from collections.abc import Sequence
from itertools import repeat


def cx_fun(ind1, ind2, alpha):
    if random.random() > 0.5:
        size = min(len(ind1), len(ind2))
        cxpoint1 = random.randint(1, size)
        cxpoint2 = random.randint(1, size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else:  # Swap the two cx points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
            = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
    else:
        for i, (x1, x2) in enumerate(zip(ind1, ind2)):
            gamma = (1. + 2. * alpha) * random.random() - alpha
            ind1[i] = (1. - gamma) * x1 + gamma * x2
            ind2[i] = gamma * x1 + (1. - gamma) * x2

    return ind1, ind2


def mut_fun(individual, mu, sigma, eta, low, up, indpb):
    if random.random() > 0.65:

        size = len(individual)
        if not isinstance(mu, Sequence):
            mu = repeat(mu, size)
        elif len(mu) < size:
            raise IndexError("mu must be at least the size of individual: %d < %d" % (len(mu), size))
        if not isinstance(sigma, Sequence):
            sigma = repeat(sigma, size)
        elif len(sigma) < size:
            raise IndexError("sigma must be at least the size of individual: %d < %d" % (len(sigma), size))

        for i, m, s in zip(range(size), mu, sigma):
            if random.random() < indpb:
                individual[i] += random.gauss(m, s)
    else:
        size = len(individual)
        if not isinstance(low, Sequence):
            low = repeat(low, size)
        elif len(low) < size:
            raise IndexError("low must be at least the size of individual: %d < %d" % (len(low), size))
        if not isinstance(up, Sequence):
            up = repeat(up, size)
        elif len(up) < size:
            raise IndexError("up must be at least the size of individual: %d < %d" % (len(up), size))

        for i, xl, xu in zip(range(size), low, up):
            if random.random() <= indpb:
                x = individual[i]
                delta_1 = (x - xl) / (xu - xl)
                delta_2 = (xu - x) / (xu - xl)
                rand = random.random()
                mut_pow = 1.0 / (eta + 1.)

                if rand < 0.5:
                    xy = 1.0 - delta_1
                    if xy < 0:
                        xy = 1e-6
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1)
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta_2
                    if xy < 0:
                        xy = 1e-6
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
                    delta_q = 1.0 - val ** mut_pow

                x = x + delta_q * (xu - xl) / 2
                x = min(max(x, xl), xu)
                individual[i] = x
    return individual,


def checkBounds(mn, mx):
    def decorator(func):
        def wrappper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in range(len(child)):
                    if child[i] > mx[i]:
                        child[i] = mx[i]
                    elif child[i] < mn[i]:
                        child[i] = mn[i]
            return offspring

        return wrappper

    return decorator


def create_rules(final_rules, model_mfs):
    rules = []
    for mf in sorted(model_mfs.keys()):
        if len(rules) == 0:
            for f in model_mfs[mf]:
                rules.append([f])
        else:
            new_rules = []
            for rule in rules:
                for f in model_mfs[mf]:
                    new_rules.append(rule + [f])
            rules = new_rules

    n_old_rules = len(final_rules)
    for i in range(len(rules)):
        final_rules['rule_' + str(n_old_rules + i)] = rules[i]

    return final_rules


def create_mfs(model_mfs, var_name, num_mf, old_num_mf, abbreviations, gbell=False):
    mfs = []
    type_mf = 'gauss'
    var_range = [-0.005, 1.005]
    for abbreviation in abbreviations:
        if abbreviation.lower() in var_name.lower():
            type_mf = 'gauss'
            break
        else:
            type_mf = 'trap'
    if type_mf == 'gauss':
        if num_mf == 1:
            mean = np.array([(var_range[0] - var_range[1]) / 2])
        else:
            mean = np.linspace(var_range[0], var_range[1], num=num_mf)
        mean[mean < 0] = 0
        std = 1.25 * var_range[1] / num_mf
        for i in range(num_mf):
            mfs.append({'name': 'mf_' + var_name + str(old_num_mf + i),
                        'var_name': var_name,
                        'prange': std,
                        'type': 'gauss',
                        'param': [mean[i], 0.25],
                        'universe': np.arange(var_range[0] - std - .01, var_range[1] + std + .01, .001),
                        'func': fuzz.gaussmf(np.arange(var_range[0] - std - .01, var_range[1] + std + .01, .001),
                                             mean[i], std)})
    else:
        if num_mf == 1:
            mean = np.array([np.mean(var_range)])
        else:
            mean = np.linspace(var_range[0], var_range[1], num=num_mf)
        std = 1.25 * var_range[1] / num_mf
        std1 = 1.125 * var_range[1] / (num_mf)
        for i in range(num_mf):
            if gbell:
                param = [0.25, 2, mean[i]]
                mfs.append({'name': 'mf_' + var_name + str(old_num_mf + i),
                            'var_name': var_name,
                            'prange': std,
                            'type': 'gbell',
                            'param': param,
                            'universe': np.arange(var_range[0] - .01 - std, var_range[1] + std + .01, .001),
                            'func': fuzz.gbellmf(np.arange(var_range[0] - .01 - std, var_range[1] + std + .01, .001, ),
                                                 param[0], param[1], param[2])})
            else:
                param = [mean[i] - std, mean[i] - std1, mean[i] + std1, mean[i] + std]
                mfs.append({'name': 'mf_' + var_name + str(old_num_mf + i),
                            'var_name': var_name,
                            'prange': std,
                            'type': 'trap',
                            'param': param,
                            'universe': np.arange(var_range[0] - .01 - std, var_range[1] + std + .01, .001),
                            'func': fuzz.trapmf(np.arange(var_range[0] - .01 - std, var_range[1] + std + .01, .001, ),
                                                param)})
    model_mfs[var_name] = mfs
    return model_mfs


def check_if_all_nans(activations, thres_act):
    indices = np.where(np.all(activations.values <= thres_act, axis=1))[0]
    if indices.shape[0] > 0:
        for ind in indices:
            act = activations.loc[activations.index[ind]]
            clust = act.idxmax()
            activations.loc[activations.index[ind], clust] = thres_act

    return activations
