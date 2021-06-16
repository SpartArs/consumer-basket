import math
import itertools
from mlxtend.frequent_patterns import fpcommon as fpc


def fpgrowth(df, min_support=0.5, use_colnames=False, max_len=None, verbose=0):
    fpc.valid_input_check(df)

    if min_support <= 0.:
        raise ValueError('`min_support` must be a positive '
                         'number within the interval `(0, 1]`. '
                         'Got %s.' % min_support)

    colname_map = None
    if use_colnames:
        colname_map = {idx: item for idx, item in enumerate(df.columns)}

    tree, _ = fpc.setup_fptree(df, min_support)
    minsup = math.ceil(min_support * len(df.index))
    generator = fpg_step(tree, minsup, colname_map, max_len, verbose)

    return fpc.generate_itemsets(generator, len(df.index), colname_map)


def fpg_step(tree, minsup, colnames, max_len, verbose):
    count = 0
    items = tree.nodes.keys()
    if tree.is_path():
        size_remain = len(items) + 1
        if max_len:
            size_remain = max_len - len(tree.cond_items) + 1
        for i in range(1, size_remain):
            for itemset in itertools.combinations(items, i):
                count += 1
                support = min([tree.nodes[i][0].count for i in itemset])
                yield support, tree.cond_items + list(itemset)
    elif not max_len or max_len > len(tree.cond_items):
        for item in items:
            count += 1
            support = sum([node.count for node in tree.nodes[item]])
            yield support, tree.cond_items + [item]

    if verbose:
        tree.print_status(count, colnames)

    if not tree.is_path() and (not max_len or max_len > len(tree.cond_items)):
        for item in items:
            cond_tree = tree.conditional_tree(item, minsup)
            for sup, iset in fpg_step(cond_tree, minsup,
                                      colnames, max_len, verbose):
                yield sup, iset