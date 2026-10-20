# Standardization — paired bootstrap (F1 label=1)

- Evaluation: 813 measured `predict_input` rows
- Threshold: 0.5 (submission threshold)
- Resamples: 10,000 (paired, molecule-level)
- Source: `reports/standardization_bootstrap.json`

## Per-variant F1

| variant | F1 | 95% CI | Accuracy |
|---|---:|---:|---:|
| raw | 0.8389 | [0.8115, 0.8646] | 0.8290 |
| strip_neutral_taut | 0.8422 | [0.8153, 0.8677] | 0.8327 |

## Paired difference (strip_neutral_taut − raw)

| metric | value |
|---|---:|
| point estimate | +0.0033 |
| bootstrap mean | +0.0034 |
| 95% CI | **[-0.0100, +0.0168]** |
| P(diff ≤ 0) | 0.314 |

## McNemar (per-molecule correctness)

| raw correct / combined wrong | raw wrong / combined correct | p-value |
|---:|---:|---:|
| 16 | 19 | 0.736 |

## Conclusion

**표준화 적용에 의한 F1 개선은 통계적으로 확정되지 않는다.**

- 점추정은 +0.0033으로 combined가 앞서지만, 95% CI가 0을 포함한다.
- 813건에서 확인 가능한 최소 차이는 대략 ±0.013 수준이며, +0.0033은 그보다 작다.
- discordant 분자도 16 vs 19로 거의 대칭이다 (McNemar p=0.736).
- 따라서 F1 단독 랭킹에서는 `strip_neutral_taut`가 1위이지만, **"raw와 동률"** 로 보는 것이 정직하다.

권장: 현재 베이스라인은 **raw 유지**. 표준화를 채택하려면 더 큰 평가셋 또는 seed 2~3개 반복에서
+0.003 이상이 일관되게 재현되는지 확인한 뒤 결정한다.
