from itertools import combinations
import numpy as np
import pandas as pd


def association_rules(df, metric="confidence",
                      min_threshold=0.8, support_only=False):
    if not all(col in df.columns for col in ["support", "itemsets"]):
        raise ValueError("Dataframe needs to contain the\
                         columns 'support' and 'itemsets'")

    def conviction_helper(sAC, sA, sC):
        confidence = sAC/sA
        conviction = np.empty(confidence.shape, dtype=float)
        if not len(conviction.shape):
            conviction = conviction[np.newaxis]
            confidence = confidence[np.newaxis]
            sAC = sAC[np.newaxis]
            sA = sA[np.newaxis]
            sC = sC[np.newaxis]
        conviction[:] = np.inf
        conviction[confidence < 1.] = ((1. - sC[confidence < 1.]) /
                                       (1. - confidence[confidence < 1.]))

        return conviction

    metric_dict = {
        "antecedent support": lambda _, sA, __: sA,
        "consequent support": lambda _, __, sC: sC,
        "support": lambda sAC, _, __: sAC,
        "confidence": lambda sAC, sA, _: sAC/sA,
        "lift": lambda sAC, sA, sC: metric_dict["confidence"](sAC, sA, sC)/sC,
        }

    columns_ordered = ["antecedent support", "consequent support",
                       "support",
                       "confidence", "lift"]

    if support_only:
        metric = 'support'
    else:
        if metric not in metric_dict.keys():
            raise ValueError("Metric must be 'confidence' or 'lift', got '{}'"
                             .format(metric))

    keys = df['itemsets'].values
    values = df['support'].values
    frozenset_vect = np.vectorize(lambda x: frozenset(x))
    frequent_items_dict = dict(zip(frozenset_vect(keys), values))

    rule_antecedents = []
    rule_consequents = []
    rule_supports = []

    for k in frequent_items_dict.keys():
        sAC = frequent_items_dict[k]
        for idx in range(len(k)-1, 0, -1):
            for c in combinations(k, r=idx):
                antecedent = frozenset(c)
                consequent = k.difference(antecedent)

                if support_only:
                    sA = None
                    sC = None

                else:
                    try:
                        sA = frequent_items_dict[antecedent]
                        sC = frequent_items_dict[consequent]
                    except KeyError as e:
                        s = (str(e) + 'You are likely getting this error'
                                      ' because the DataFrame is missing '
                                      ' antecedent and/or consequent '
                                      ' information.'
                                      ' You can try using the '
                                      ' `support_only=True` option')
                        raise KeyError(s)

                score = metric_dict[metric](sAC, sA, sC)
                if score >= min_threshold:
                    rule_antecedents.append(antecedent)
                    rule_consequents.append(consequent)
                    rule_supports.append([sAC, sA, sC])

    if not rule_supports:
        return pd.DataFrame(
            columns=["antecedents", "consequents"] + columns_ordered)

    else:
        rule_supports = np.array(rule_supports).T.astype(float)
        df_res = pd.DataFrame(
            data=list(zip(rule_antecedents, rule_consequents)),
            columns=["antecedents", "consequents"])

        if support_only:
            sAC = rule_supports[0]
            for m in columns_ordered:
                df_res[m] = np.nan
            df_res['support'] = sAC

        else:
            sAC = rule_supports[0]
            sA = rule_supports[1]
            sC = rule_supports[2]
            for m in columns_ordered:
                df_res[m] = metric_dict[m](sAC, sA, sC)

        return df_res
