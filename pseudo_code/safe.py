import xgboost as xgb


def op_generate_new_features(op, op_candidates):
    # apply the operator on the narrowed path-based feature set

    return op(op_candidates)


def filter_candidates_information_gain_ratio(candidates, base_filter):
    """
    in paper: algo2
    :param candidates:
    :param base_filter:
    :return:
    """
    filtered_candidates = []
    return filtered_candidates


def feature_generation(base_features, registered_ops):
    """
    feature generation within one iteration
    in paper algo1, line3+
    :param base_features:
    :param registered_ops:
    :return:
    """
    # 1. first filtering on base features

    # essentially, the xgb performs as a feature evaluation methods on the
    # feature candidate set generated in the previous iteration
    base_tree = xgb.XGBClassifier()
    base_tree.fit(base_features)

    # 2. generate feature combinations based on tree paths (dfs)
    path_feature_sets = path_transverse(base_tree)

    # 3. Algo2: filter feature combinations base on information gain ratio
    op_path_candidates = filter_candidates_information_gain_ratio(
        path_feature_sets, base_tree)


    # 4. new candidates = new features by applying operators in operator pool
    # on the filtered_features + original base features
    new_candidates = []
    op_candidates = op_path_candidates + base_features
    for op in registered_ops:
        op_path_candidates = op_generate_new_features(op, op_candidates)
        new_candidates.extend(op_path_candidates)

    return new_candidates


def feature_selection(candidates):
    # 1. iv
    # 2. pearson correlation
    # 2. xgb feature importance score on average gain across the splits
    filterd_features = []
    return filterd_features


def check_termination_condition():
    # runtime
    # computational resources
    # maximum iteration
    return False


def main():
    base_features = []
    # function hooks
    registered_ops = {}
    while check_termination_condition():
        candidates = feature_generation(base_features, registered_ops)
        base_features = feature_selection(candidates)

    return base_features
