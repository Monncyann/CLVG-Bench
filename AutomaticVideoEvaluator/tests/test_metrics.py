from ave.metrics import aggregate, calculate


def test_metrics():
    matrix = aggregate([{"TP": 2, "TN": 3, "FP": 1, "FN": 1}])
    scores = calculate(matrix)
    assert matrix == {"TP": 2.0, "TN": 3.0, "FP": 1.0, "FN": 1.0}
    assert round(scores["f1"], 6) == 0.666667
    assert round(scores["recall_minus_fpr"], 6) == 0.416667

