"""Monotone constraints for the ranking stages.

A pre-ranker must never score a candidate lower because it is MORE similar:
LightGBM without constraints learned, e.g., "pass-A rank >= 4 => not a match"
and pushed identical-name, identical-address true pairs to p ~ 1e-9 (found in
recall_lab / prerank_misses.py). Constraints: increasing in every pass score and
similarity, decreasing in every pass rank; flags stay free."""

INCREASING_PREFIX = ("score_", "sig_name", "sig_tri", "sig_skel", "sig_addr", "sig_nums",
                     "sig_house_eq", "sig_city_eq", "sig_state_eq", "sig_prefix_eq", "sig_len_ratio")
INCREASING = {"n_passes", "p0", "name_core_tset", "name_core_jw", "addr_tset", "house_match",
              "name_join_partial"}
DECREASING = {"house_conflict"}


def constraints(cols):
    out = []
    for c in cols:
        if c.startswith("rank_") or c in DECREASING:
            out.append(-1)
        elif c.startswith(INCREASING_PREFIX) or c in INCREASING:
            out.append(1)
        else:
            out.append(0)
    return out


def params_for(base, cols):
    return dict(base, monotone_constraints=constraints(cols), monotone_constraints_method="advanced")
