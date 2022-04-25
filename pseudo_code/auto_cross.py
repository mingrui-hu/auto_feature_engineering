import collections

import numpy as np

operator = np.cross

class FeatureNode:
    def __init__(self):
        self.feature_set = []
        self.parent = None
        self.children = None
        self.score = None

    @property
    def feature_set(self):
        return self.feature_set

    @feature_set.setter
    def feature_set(self, feature):
        if isinstance(feature, list):
            self.feature_set.extend(feature)
        else:
            self.feature_set.append(feature)


# ----------------------------------------------------------------------------
# preprocessing
# ----------------------------------------------------------------------------
def multi_gran_discretization(features):
    """
    multi-granularity discretization, include a rule-based mechanism to
    select hyperparam p [number of level of granularity]
    :param features:
    :return:
    """
    # in the paper, p is determined automatically
    p = 5

    granularities = [10 ** (i + 1) for i in range(p)]

    # discretization
    multi_gran_feature_set = [features.bin(g) for g in granularities]

    # evaluating using field-wise LR without b_sum (i.e. normal LR)
    # and keep only the best half
    multi_gran_feature_set_scores = {x: evaluate_feature_set(x) for x in
                                     multi_gran_feature_set}

    multi_gran_feature_set = multi_gran_feature_set.sort(
        key=lambda x: multi_gran_feature_set_scores[x])[:p // 2]

    return multi_gran_feature_set


# ----------------------------------------------------------------------------
# feature set generation
# ----------------------------------------------------------------------------
# expand beam-size child -- dfs construction
# note the beam search strategy is applied for each branch, hence
# still within a dfs framework; but may be adapted to a bfs framework
# for higher efficiency?
def generate_children_candidates(node, beam_size=3):
    # one-hot-encoded feature columns
    orig_cross_feats = node.parent.feature_set

    # generate children:
    # pair-wise interactions:
    # 1. crossing btw its own elements(cross features)
    # 2. crossing of a cross features & an original feature

    # list of FeatureNode object
    children_candidates = []

    # evaluates childeren candidates -- no memo caching applied yet

    # ------ the following naive beam search strategy(evaluating all
    # candidates & rank) could be replaced by the proposed
    # `successive-halving-gradient-descent
    # `
    child_candidate_scores = {x: evaluate_feature_set(x) for x in
                              children_candidates}

    # ranking scores & keep top-k depend on beam_size

    child_candidates = children_candidates.sort(
        key=lambda x: child_candidate_scores[x])[:beam_size]

    return child_candidates


# ----------------------------------------------------------------------------
# feature set evaluation
# ----------------------------------------------------------------------------
# global evaluation score memo
feature_set_score_cache = {

}


def field_wise_lr(current_solutions, new_feature_set, metric="auc"):
    """
    logistic regression for categorical crossing features (one-hot encoded)?
    with gradient descent.

    in paper: a hash-trick is mentionedf for storing feature values

    :param current_solutions:
    :param new_feature_set:
    :return:
    """
    # no grad, freeze weight for current_solution
    w_c = torch.zeros()
    return model


def successive_minibatch_gradient_descent():
    optimizer = None
    return optimizer


def evaluate_feature_set(feature_set):
    # split feature_set to current_solutions & new_feature_set
    current_solutions, new_feature_set = feature_set.split()

    # define optimzer
    optimizer = successive_minibatch_gradient_descent()

    # field-wise lr
    model = field_wise_lr(current_solutions, new_feature_set, metric="auc")

    score = model.fit(optimizer=optimizer)
    return score


# ----------------------------------------------------------------------------
# check termination condition
# ----------------------------------------------------------------------------
def check_termination_condition(children_scores):
    """
    1) runtime condition -- a global timer
    2) performance condition -- cache all parent node feature set scores
    3) maximum feature number
    :return:
    """
    if runtime_consumed > condition:
        return True

    if min(children_scores) > parent_node_performance:
        return True

    if maximum_feature_number > threshold:
        return True

    return False


# ----------------------------------------------------------------------------
# full algorithm
# ----------------------------------------------------------------------------
def feature_set_generation_dfs(node: FeatureNode, beam_size: int = 3):
    """
    # version 1: beam search applied for each branch, hence final returned res
    # length: k*beam_size; k=tree depth
    """
    # Todo: check if meet terminating condition: True, return
    if check_termination_condition():
        return []

    child_candidates = generate_children_candidates(node, beam_size=beam_size)
    res = []
    for child in child_candidates:
        res.extend(feature_set_generation_dfs(child))

    return res


def feature_set_generation_beam_search():
    # version2: global beam search, layer-wise transverse
    queue = collections.deque()
    root = FeatureNode()
    queue.append(root)

    node = queue.pop()
    while node:
        nodes: generate_children_candidates()
