import random
import statistics as st
import zero_relationship_lwe as z


def summarize(cfg):
    b_auc = []
    c_auc = []
    b_ent = []
    c_ent = []
    b_corr = []
    c_corr = []
    b_gap = []
    c_gap = []

    for seed in range(20260403, 20260403 + 16):
        b = z.BaselineGaussianSecretSampler(sigma=2.0, secret_bound=8, rng=random.Random(seed + 1))
        c = z.ZeroRelationshipSecretSampler(secret_bound=8, rng=random.Random(seed + 2), **cfg)

        bm = z.evaluate_secret_sampler("b", b, 256, 120, 1.0, random.Random(seed + 101))
        cm = z.evaluate_secret_sampler("c", c, 256, 120, 1.0, random.Random(seed + 102))

        b_auc.append(bm["zero_nonzero_auc"])
        c_auc.append(cm["zero_nonzero_auc"])
        b_ent.append(bm["entropy"])
        c_ent.append(cm["entropy"])
        b_corr.append(bm["avg_abs_dim_corr"])
        c_corr.append(cm["avg_abs_dim_corr"])
        b_gap.append(bm["zero_nonzero_score_gap"])
        c_gap.append(cm["zero_nonzero_score_gap"])

    return {
        "b_auc": st.fmean(b_auc),
        "c_auc": st.fmean(c_auc),
        "b_ent": st.fmean(b_ent),
        "c_ent": st.fmean(c_ent),
        "b_corr": st.fmean(b_corr),
        "c_corr": st.fmean(c_corr),
        "b_gap": st.fmean(b_gap),
        "c_gap": st.fmean(c_gap),
    }


cfgs = [
    ("default", {}),
    (
        "blur_heavy",
        {
            "output_gain": 0.45,
            "warp_strength": 0.12,
            "jitter_sigma": 0.62,
            "zero_to_nonzero_prob": 0.28,
            "unit_to_zero_prob": 0.40,
            "boundary_mix_prob": 0.66,
            "magnitude_scale": 0.58,
            "high_mag_shrink_prob": 0.78,
            "intra_vector_refresh_prob": 0.65,
        },
    ),
    (
        "balanced",
        {
            "output_gain": 0.50,
            "warp_strength": 0.15,
            "jitter_sigma": 0.50,
            "zero_to_nonzero_prob": 0.22,
            "unit_to_zero_prob": 0.32,
            "boundary_mix_prob": 0.56,
            "magnitude_scale": 0.68,
            "high_mag_shrink_prob": 0.68,
            "intra_vector_refresh_prob": 0.55,
        },
    ),
]

for name, cfg in cfgs:
    r = summarize(cfg)
    print(name)
    print(
        "  auc  baseline={:.4f} cand={:.4f} delta={:+.4f}".format(
            r["b_auc"], r["c_auc"], r["c_auc"] - r["b_auc"]
        )
    )
    print(
        "  gap  baseline={:.4f} cand={:.4f} delta={:+.4f}".format(
            r["b_gap"], r["c_gap"], r["c_gap"] - r["b_gap"]
        )
    )
    print(
        "  corr baseline={:.4f} cand={:.4f} delta={:+.4f}".format(
            r["b_corr"], r["c_corr"], r["c_corr"] - r["b_corr"]
        )
    )
    print(
        "  ent  baseline={:.4f} cand={:.4f} delta={:+.4f}".format(
            r["b_ent"], r["c_ent"], r["c_ent"] - r["b_ent"]
        )
    )
