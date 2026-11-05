from valkit.bench import VAR_DEFECTS, _var_status, wilson


def test_wilson_known_values():
    lo, hi = wilson(0, 10)
    assert lo == 0 or lo < 1e-12 and 0.25 < hi < 0.32
    assert wilson(40, 40)[0] > 0.9


def test_var_defect_is_flagged_on_a_fresh_seed():
    assert _var_status(VAR_DEFECTS["VD-01 understated x0.70"], 999) >= 1
