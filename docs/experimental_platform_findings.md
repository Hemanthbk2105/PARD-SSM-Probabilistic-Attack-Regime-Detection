# Experimental Platform Findings

## Scope

This note records the synthetic experiments run by the platform CI workflow. These results are diagnostic research evidence only. They do not establish real-world intrusion-detection performance on CICIDS2017, UNSW-NB15, or live network traffic.

The held-out Markov benchmark uses five random seeds (2040–2044), a chronological 70% train / 30% test split, and train-only preprocessing for its benchmark baselines. PARD-SSM uses approximate GPB1 filtering and approximate generalized-EM updates.

## Held-out Markov benchmark

For the default PARD-SSM configuration with causal posterior smoothing window 11:

- Mean ARI: 0.2038 (sample standard deviation 0.0870)
- Mean boundary F1: 0.3447 (sample standard deviation 0.0603)

The Gaussian HMM baseline achieved:

- Mean ARI: 0.3776 (sample standard deviation 0.1082)
- Mean boundary F1: 0.4617 (sample standard deviation 0.0521)

On this synthetic benchmark, the Gaussian HMM is stronger on both reported measures. This should be reported clearly rather than presenting PARD-SSM as the best method.

## PCA initialization ablation

The PCA ablation fits StandardScaler, 2D PCA, and KMeans on training data only, then estimates regime-specific initialization parameters from training clusters. It evaluates five seeds.

| Method | Mean ARI | Mean boundary F1 |
| --- | ---: | ---: |
| Existing PARD-SSM, causal window 11 | 0.2038 | 0.3447 |
| PCA-initialized PARD-SSM, raw | 0.0058 | 0.1519 |
| PCA-initialized PARD-SSM, causal window 11 | 0.0117 | 0.2884 |
| Gaussian HMM | 0.3776 | 0.4617 |

PCA initialization did not improve the existing PARD-SSM configuration. Keep it as an ablation/negative result; do not make it the default initialization.

## Prior sensitivity

### Sticky prior

Across tested strengths 0, 0.5, 1, 2, 4, and 8, the default strength 0 produced the highest mean boundary F1 for causal window 11 (0.3447). Strength 8 produced a slightly higher mean ARI (0.2053 versus 0.2038) but lower boundary F1 (0.3355). This is a metric trade-off, not a universal improvement.

### Transition-matrix prior

Across tested strengths 0, 10, 25, 50, 100, 250, and 500:

- Strength 0: mean ARI 0.2038; boundary F1 0.3447.
- Strength 250: highest mean ARI, 0.2246; boundary F1 0.3313.
- Strength 500: mean ARI 0.2240; boundary F1 0.3407.

No tested nonzero strength improves both ARI and boundary F1 over the zero-strength default. Keep the default transition-prior strength at 0 unless a future experiment justifies a different trade-off for a clearly defined objective.

## Interpretation and limitations

- These are synthetic Markov-regime experiments, not real-world network intrusion evaluations.
- Five seeds are an initial sensitivity check, not a definitive statistical study.
- ARI and boundary F1 measure different aspects of regime segmentation; a change in one can come with a trade-off in the other.
- The current PARD-SSM implementation uses approximate GPB1 filtering and approximate generalized EM, not exact switching Kalman EM.
- Results from the existing dataset pipeline must be interpreted separately; its random sampling/splitting does not provide a chronological holdout.

## Research conclusion

The experiments currently support retaining the existing zero-prior PARD-SSM defaults, rejecting PCA initialization as a default replacement, and reporting the Gaussian HMM as the stronger baseline on this particular held-out synthetic benchmark. Further improvements should focus on diagnosing regime inference and boundary false positives, followed by evaluation on properly ordered network traffic with leakage-safe chronological splits.
