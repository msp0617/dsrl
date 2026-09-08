# DSRL 전체 결과 분석 — 2026-09-08

## 결론부터

**121개 온라인 run, 37개 조건의 재집계를 마쳤다. 이번 결과는 “Cal-QL이 해결했다”가 아니라, critic 초기화·엔트로피 조절·데모 replay가 서로 다른 효과와 상충을 가진다는 결과다.**

- **Can:** 성능 위주로는 prefill이 초기 AUC 0.682, 129k 성공률 0.858로 평균값이 가장 높다. 하지만 초반 최저는 0.093으로 깊게 떨어진다. 고정 α=0.3은 AUC 0.656, 후반 0.739로, 평균곡선의 dip 완화와 성능 사이에서 좋은 절충이다.
- **Square:** TD가 초기 AUC 0.492로 가장 높다. 후반은 prefill 0.622, IQL 0.562가 높다. TD/CQL/Cal-QL의 초기 이득이 후반 우위로 이어지지는 않는다.
- **Cal-QL 단독의 일관된 우위는 없다.** Can의 AUC는 IQL보다 낮고, Square 후반 성공률은 IQL보다 세 seed 모두 낮다. CQL보다 확실히 낫다는 결론도 어렵다.
- **Cal-QL+t12i / +prefill은 Cal-QL 단독보다 좋아진다.** 그러나 t12i 단독 / prefill 단독과 비교하면 Cal-QL을 추가할 이득은 입증되지 않았다.
- **Square hq는 tent12의 후반 손실을 줄인다는 증거를 준다.** 같은 seed 1–3에서 후반 +0.093 ± 0.020, 세 seed 모두 개선이다. 그러나 초기 AUC는 낮아지고 dip은 남는다. “초기와 후반을 모두 해결했다”는 결론은 아니다.
- **중요한 문구 수정:** 새 Square 순수 diffusion policy 기준선은 **0.494**다. 기존의 “Can·Square 모두 dip 제거”는 유지할 수 없다. Can도 “평균곡선이 기준선을 넘는다”와 “모든 seed에서 dip이 없다”를 구분해야 한다.

여기서 “가장 높다”는 현재 실험 조건의 표본 평균 순위다. n=3/5의 작은 표본에서 모집단 최적성이나 통계적 유의성을 선언한 것이 아니다.

## 1. 범위와 원본 검증

| 항목 | 확인 결과 |
|---|---:|
| Can 온라인 run / 조건 | 85 / 27 |
| Square 온라인 run / 조건 | 36 / 10 |
| eval CSV / train CSV | 121 / 121 |
| 원본 eval 행 → 중복·재개 정리 후 | 2,401 → 2,385 |
| 원본 train 행 → 재개 정리 후 | 12,764 → 12,614 |
| 초기 AUC 계산 가능 | 121 / 121 |
| 엄격한 후반 목표 시점의 실제 평가 존재 | 102 / 121 |
| 신규 24개 후반 평가 존재 | 24 / 24 |
| 최신 TD/CQL/Cal-QL 사전학습 로그 | 18 / 18: critic 50k + distill 25k |
| 기본 diffusion policy 평가 | Can·Square 각각 500 episodes × 3 seeds |

Drive의 `logs`에서 직접 읽었고, 각 CSV의 읽은 텍스트 길이를 Drive 원본 byte 크기와 대조했다. 이 CSV들은 ASCII이므로 두 길이가 같아야 한다. 로컬 저장 시 CRLF만 LF로 정규화했다. 원본 Drive 파일은 수정하지 않았다. 원본 ID·크기·수정 시각은 입력 폴더의 `source_manifest.json`, 로컬 SHA-256은 `audit.csv`에 있다.

재개 처리는 파일에 기록된 순서를 따른다. step이 뒤로 돌아가면 그 step 이상인 이전 시도의 행을 제거하고 새 기록을 사용한다. eval에서는 기존 `keep=last`와 결과가 완전히 같았다. train에서는 단순 같은-step 중복 제거만으로는 남는 이전 시도 행 140개가 추가로 있었다. 이것까지 제거한 뒤 진단값을 집계했다. 재시도는 별도 seed/run으로 세지 않았다.

오늘 Drive에서 먼저 생성된 `analysis/20260908_125829_b3a5/figures/comparison.csv`와 **37개 조건의 AUC, 평균곡선 최저, 후반 성공률이 부동소수점 오차 이내로 모두 일치**했다. 따라서 이번의 큰 변화는 성공률 계산 결과가 아니라, 기준선·불확실성·비교 대상과 해석의 수정이다.

완료 여부 121/121은 앞선 `[done]`/checkpoint 전수검사 및 두 run의 0-step finalize 기록을 근거로 한 운영 상태다. CSV 존재만으로 checkpoint 정상성이나 clean shutdown까지 재증명한 것은 아니다. 모델 가중치를 새로 다운로드하거나 재평가하지 않았다.

## 2. 숫자 읽는 법

- 모든 성공률/AUC는 0–1 단위다. `±`는 **seed 간 표본 표준편차 / √n**, 즉 SE다. 95% 신뢰구간이 아니다. n=1은 SE가 없으며 0으로 채우지 않았다.
- 초기 AUC는 기존 정의를 유지했다: **Can online 0–75,984 / Square online 0–67,984**. step 0 평가를 포함하고, 평가점 사이 사다리꼴 적분 후 구간 길이로 나눈다. 마지막 구간은 경계 직전 평가를 유지한다.
- online step은 초기 rollout을 뺀 학습 구간이다. Can은 env−24,016, Square는 env−32,016. 처음 eval(env=0)만 online=0이다. 서로 다른 과제의 AUC를 합쳐 하나의 순위로 만들지 않았다.
- 경계에서 선형 보간하는 민감도 분석도 별도로 계산했다. 비교 가능한 조건의 **그룹 AUC 차이는 최대 0.00175**였다. 주요 결론을 바꾸지 않는다. 뒷 평가가 없는 100k run에는 경계 밖 값을 외삽하지 않았다.
- **최저**는 online 5,008 간격 공통 격자의 seed-평균곡선 최저다. seed별 최저를 먼저 뽑아 평균한 값과 다르다. 실제 관측점별 최저도 `per_seed.csv`에 남겼다.
- 후반 성능은 **Can env=129,152 / Square env=127,136**의 실제 평가다(±32 step만 허용). 마지막 3개 평가 평균은 예산이 섞여 있으므로 대표 성능으로 쓰지 않는다.
- Can `alr_half/double` 6개는 150k까지 학습했지만 평가 격자가 달라 마지막 eval=126,896이다. 엄격한 129,152 열은 비어 있다. 나머지 후반 결측 13개는 100k 예산이다. **후반 결측 19개를 미완료 run으로 해석하면 안 된다.** 125k 양쪽 평가가 있는 run에 한해 보간값을 별도 제공했다.
- 초기 eval은 대체로 100 episodes, 후반은 200 episodes다. 평가 episode의 표본 잡음도 있다. 그래프의 SE는 학습 seed 간 변동을 요약할 뿐 그 모든 불확실성을 분리하지 않는다.
- 같은 seed ID를 맞춘 비교도 추가했다. 다만 평가·학습 RNG 흐름이 동일한 common-random-number 실험이라는 보장은 없다. 단순한 seed 구성 민감도 분석이며 인과 효과를 자동으로 보증하지 않는다.

### 반드시 구분할 두 baseline

| 기준 | Can | Square | 의미 |
|---|---:|---:|---|
| 순수 diffusion policy, N(0,I) noise | 0.4053 ± 0.0177 | 0.4940 ± 0.0072 | 학습하지 않은 기준 정책; 3 seed 평균±seed SE |
| 온라인 `baseline`의 후반 성능 | 0.532 ± 0.057 | 0.407 ± 0.060 | random critic에서 출발해 온라인 학습한 결과; n=5 |

기준 정책의 합산 1,500 episodes로 구한 단순 binomial SE는 Can 0.0127, Square 0.0129다. 위의 seed SE와 다른 양이므로 서로 바꿔 쓰지 않는다. 초기 random noise actor의 step-0 성능 역시 순수 N(0,I) diffusion policy 성능과 같지 않다.

## 3. 새 critic 실험: 어떤 사전학습이 효과가 있었나

| 과제 | 조건 | n | 초기 AUC | 평균곡선 최저 | 후반 성공률 |
|---|---|---:|---:|---:|---:|
| Can | baseline | 5 | 0.409 ± 0.028 | 0.238 | 0.532 ± 0.057 |
| Can | IQL | 5 | 0.494 ± 0.016 | 0.338 | 0.561 ± 0.024 |
| Can | TD | 3 | 0.473 ± 0.033 | 0.393 | 0.558 ± 0.026 |
| Can | CQL-style | 3 | 0.477 ± 0.022 | 0.353 | 0.575 ± 0.046 |
| Can | Cal-QL-style | 3 | 0.431 ± 0.018 | 0.340 | 0.510 ± 0.050 |
| Square | baseline | 5 | 0.401 ± 0.007 | 0.234 | 0.407 ± 0.060 |
| Square | IQL | 3 | 0.437 ± 0.005 | 0.370 | 0.562 ± 0.015 |
| Square | TD | 3 | 0.492 ± 0.020 | 0.440 | 0.422 ± 0.064 |
| Square | CQL-style | 3 | 0.465 ± 0.011 | 0.373 | 0.332 ± 0.076 |
| Square | Cal-QL-style | 3 | 0.457 ± 0.028 | 0.397 | 0.415 ± 0.052 |

Can은 TD/CQL도 초기 저하를 완화하지만 IQL을 일관되게 이기지는 않는다. Cal-QL은 같은 seed 1–3 IQL보다 AUC가 **−0.076 ± 0.017**, 세 seed 모두 낮다. 같은 seed baseline과 비교한 Cal-QL AUC 차이는 **−0.0007 ± 0.034**로 거의 없다. 따라서 n=5 baseline 대비 AUC +0.022만 보고 Cal-QL 효과를 강조하지 않는다.

Square의 TD는 평균곡선 최저가 0.234→0.440으로 올라가고 초기 AUC도 가장 높다. 그러나 127k에는 0.422다. Cal-QL은 IQL보다 early AUC가 +0.021이지만 후반은 **−0.147 ± 0.063**, 세 seed 모두 낮다. “초반 개선 → 최종 개선”은 성립하지 않는다.

기존 “critic 초기화는 dip 시점을 못 바꾼다”도 절대적 표현을 피해야 한다. 이번 TD의 초기창 최저는 Can·Square 모두 online 40,064에 있고, Cal-QL은 Can 35,056 / Square 25,040이다. 초기 저하가 얕아지면 전 구간 최저가 다른 작은 요동으로 이동할 수 있다. **최저 시각 하나와 최초 붕괴 시각은 다른 지표**다.

그림: `critic_ladder.png` / `critic_ladder.pdf`.

## 4. Cal-QL 조합: 단독보다 좋다 ≠ Cal-QL을 추가할 이유가 있다

| Can 조건 | n | 초기 AUC | 평균곡선 최저 | 129k 성공률 |
|---|---:|---:|---:|---:|
| Cal-QL 단독 | 3 | 0.431 ± 0.018 | 0.340 | 0.510 ± 0.050 |
| Cal-QL + t12i | 3 | 0.615 ± 0.038 | 0.450 | 0.692 ± 0.025 |
| t12i 단독 | 3 | 0.594 ± 0.052 | 0.460 | 0.697 ± 0.058 |
| Cal-QL + prefill | 3 | 0.655 ± 0.015 | 0.360 | 0.765 ± 0.015 |
| prefill 단독 | 3 | 0.682 ± 0.019 | 0.093 | 0.858 ± 0.049 |
| IQL + prefill | 3 | 0.601 ± 0.015 | 0.200 | 0.847 ± 0.006 |

`t12i`는 목표 엔트로피 12, 초기 α=0.3이다. Cal-QL에 t12i를 붙이면 AUC +0.184, 후반 +0.182이며 두 지표 모두 세 seed에서 개선이다. 하지만 **t12i 단독 대비** 차이는 AUC **+0.021 ± 0.015**, 후반 **−0.005 ± 0.055**다. 작은 초기 개선 가능성은 있지만, 추가 사전학습의 성능 이득이 명확하다고 쓰기는 어렵다.

Cal-QL에 prefill을 붙이면 AUC +0.223, 후반 +0.255이며 세 seed 모두 개선이다. 하지만 **prefill 단독 대비** AUC **−0.027 ± 0.031**, 후반 **−0.093 ± 0.060**이다. 다만 평균곡선 최저는 0.093→0.360으로 크게 올라간다. 따라서 “더 높은 후반 성능”이 아니라 **초반의 급락을 줄이는 대신 후반 평균을 일부 잃는 조합**으로 해석한다.

Cal-QL+t12i는 실제 초기 평가점 전체가 기준선보다 높았던 seed가 2/3이고, t12i 단독은 1/3이다. n=3의 기술적 관측이며 안정성 보장이나 유의한 차이라는 뜻은 아니다.

## 5. 기존 실험까지 포함한 처방 비교

### Can

**Replay 축:** prefill / fixed mix / linear mix의 AUC는 각각 0.682 / 0.667 / 0.641, 후반은 0.858 / 0.835 / 0.747이다. prefill과 fixed mix의 차이만으로 최적 replay 스케줄을 확정할 수는 없다. prefill은 성능은 높지만 초기 최저가 0.093, linear mix는 0.020이다. AUC만 좋다고 안전한 전환은 아니다. `iql_linear_s1`은 n=1 파일럿이므로 일반 결과처럼 취급하지 않는다.

**α sweep:** α=0.01 / 0.03 / 0.1 / 0.3 / 1.0의 AUC는 0.148 / 0.234 / 0.501 / 0.656 / 0.269다. 큰 α일수록 무조건 좋지 않다. 이번 범위에서는 0.3이 좋은 절충이며 n=5 모두 baseline보다 AUC와 후반 성공률이 높다. 다만 순수 diffusion 기준으로 각 seed의 모든 초기 평가점이 높았던 것은 **1/5**다. “모든 seed에서 dip 제거”는 틀리다.

**목표 엔트로피/초기값:** tent12 / tent12i / tent6 AUC는 0.567 / 0.594 / 0.506이다. tent12i 평균곡선 최저는 0.460으로 기준 0.405보다 높지만, seed 간 차이는 남는다. Can tent6은 100k 예산이라 엄격한 129k 결과가 없다.

**critic/actor warmup:** critic-only warmup은 AUC 0.506, 후반 0.608로 기존 IQL과 비슷한 범주다. actor까지 로드한 warmup은 **step 0 평균 0.0167**이다. 학습 후 격자 최저 0.413만 보고 이 조건이 dip을 피했다고 말하면 초기 실패를 숨기게 된다.

**α 학습률/Q scale/hard backup:** alr_half / alr_double AUC는 0.392 / 0.467, reward×0.25 / ×2는 0.319 / 0.437, hardq는 0.298이다. 이 결과는 “Q 값의 크기만 고치면 좋아진다”는 단순 처방을 지지하지 않는다. 그렇다고 모든 종류의 critic-scale 설명을 보편적으로 반증한 것은 아니다. 이번의 고정 개입 범위에서 효과가 제한적이었다고 표현한다.

### Square

초기 AUC가 최우선이면 현재 표본에서는 TD가 가장 좋다. **후반 성능이면 prefill 0.622, IQL 0.562**가 더 높다. 다만 prefill의 초기 최저는 0.030으로 매우 나쁘다. 목표 엔트로피를 고정적으로 높이는 Can 처방을 그대로 옮기면 후반에 손해를 볼 수 있다.

TD/CQL/Cal-QL을 100k에서 멈추지 않고 연장한 것은 분석상 유용했다. 100k 직전 실제 eval인 **env=97,120**에서 127,136까지 평균은 다음처럼 바뀐다.

| Square 조건 | env 97,120 | env 127,136 |
|---|---:|---:|
| TD | 0.470 | 0.422 |
| CQL | 0.443 | 0.332 |
| Cal-QL | 0.513 | 0.415 |
| IQL | 0.433 | 0.562 |
| prefill | 0.567 | 0.622 |

단, TD는 102,128에서 0.327까지 떨어졌다가 127,136에서 회복했다. 모든 조건을 단조로운 “후반 붕괴” 하나로 묘사하지 않는다. 측정 간격이 약 25k라 중간 변화는 관측되지 않는다.

그림: `recipes_and_tradeoffs.png`, `all_conditions_can.png`, `all_conditions_square.png`.

## 6. Square hq: 메커니즘 증거는 있지만, 만능 해결책은 아니다

| 조건 | n | 42,032 성공률 | 초기 AUC | 127,136 성공률 |
|---|---:|---:|---:|---:|
| baseline | 5 | 0.234 ± 0.019 | 0.401 ± 0.007 | 0.407 ± 0.060 |
| tent12 | 5 | 0.404 ± 0.019 | 0.404 ± 0.022 | 0.369 ± 0.024 |
| tent12 + hard backup | 3 | 0.330 ± 0.052 | 0.378 ± 0.014 | 0.490 ± 0.023 |

같은 seed 1–3로 맞추면 tent12의 후반은 0.397이고 hq는 0.490이다. **+0.093 ± 0.020, 개선 3/3**이다. 반면 초기 AUC 차이는 **−0.050 ± 0.040**다. hq의 전체 초기창 평균곡선 최저는 0.287이다.

baseline에 대한 이득은 seed 구성에 민감하다. n=5 baseline과 비교하면 +0.083이지만, 같은 s1–3 baseline의 후반은 0.472라 차이는 **+0.018 ± 0.067**에 불과하다. hq의 가장 설득력 있는 비교 대상은 baseline이 아니라 **동일한 target-entropy 조건인 tent12**다.

온라인 90–95k 구간, seed별 5k bin 평균을 다시 seed 평균한 진단값:

| Square 조건 | α | −logπ 추정치 | Q_W 평균 |
|---|---:|---:|---:|
| baseline | 0.284 | −0.46 | 571 |
| Cal-QL | 0.277 | 0.22 | 577 |
| tent12 | 15.38 | 11.93 | 22,998 |
| tent12 + hard backup | 0.178 | 12.08 | −75.9 |

hq와 tent12는 엔트로피 추정치를 약 12로 유지하면서 α와 Q 규모가 매우 다르게 전개된다. 후반 성공률도 hq 쪽이 좋아진다. 이는 **critic-target entropy bonus가 성능 상충에 관여한다는 해석을 지지**한다. 그러나 target 변경이 state distribution·critic·actor 학습 전체를 바꾸므로, Q 오프셋 하나가 유일한 매개라는 증명은 아니다. soft Q는 entropy bonus를 포함하므로 Q_W의 큰 양수값 자체를 곧바로 환경 수익에 대한 수치적 발산/오차로 등치하지 않는다.

Can baseline과 Cal-QL은 online 10–15k에서 −logπ가 각각 약 −0.68 / 0.08로 떨어지고, t12i 및 Cal-QL+t12i는 약 11.7 / 11.6을 유지한다. 엔트로피 동역학이 초기 저하와 관련된다는 해석은 남는다. 다만 Square에서 엔트로피를 유지해도 기준 정책 아래로 떨어지므로 **“엔트로피만 유지하면 dip이 사라진다”는 충분조건은 아니다.** 이 값은 연속 noise-policy log-density에서 나온 추정치이며 음수가 가능하다. 물리적 행동의 이산 엔트로피가 아니다.

그림: `diagnostics_can.png`, `diagnostics_square.png`.

## 7. 오프라인 TD/CQL/Cal-QL은 정상 학습된 것인가

최신 18개 로그 모두 critic **50,000 step**, Q_W distillation **25,000 step**을 기록했다. 아래는 마지막 critic 로그 10개(45,500–50,000 step)를 각 seed에서 평균한 후 3개 seed를 평균한 **training-batch 진단값**이다. 독립 holdout 평가가 아니다.

| 과제 | 방법 | Q_data | Q_ood | demo G | Q_ood−G | Cal-QL floor 사용률 |
|---|---|---:|---:|---:|---:|---:|
| Can | TD | −151.7 | −152.0 | −103.1 | −48.9 | 해당 없음 |
| Can | CQL | −97.0 | −171.1 | −101.3 | −69.8 | 해당 없음 |
| Can | Cal-QL | −88.7 | −159.4 | −101.3 | −58.1 | 97.4% |
| Square | TD | −255.2 | −254.7 | −156.3 | −98.4 | 해당 없음 |
| Square | CQL | −138.0 | −151.3 | −153.2 | +1.9 | 해당 없음 |
| Square | Cal-QL | −116.3 | −130.1 | −153.2 | +23.1 | 19.6% |

또한 사용자가 붙인 Square 4,096-state 검사 결과에서 모든 9개 checkpoint의 method와 50k metadata가 맞고 0 reject였다. 그 별도 검사에서 평균 Q_ood−G는 TD 약 −103, CQL 약 −1.4, Cal-QL 약 +19.0이다. minibatch/head 결합 방법이 위 표와 달라 수치를 섞지 않는다.

해석:

1. 이번 최종 CQL은 과거 실패 버전의 −수천 Q처럼 무너진 결과가 아니다. 과거 실패 사례를 현재 CQL의 성능 설명으로 재사용하면 안 된다.
2. `verdict: OK`는 코드의 매우 느슨한 규모 검사(Q_data 양수 또는 Q_ood 극단적 음수만 거부)다. **정확한 calibration, OOD 가치의 정답 일치, online 성능 우위를 검증한 것이 아니다.**
3. Cal-QL의 `max(Q_ood, G)`는 **벌점 항에 들어갈 값**을 바꾼다. 실제 네트워크 출력 전체를 G 이상으로 투영하는 제약이 아니다. Can에서 floor 사용률 97.4%인데 Q_ood−G는 −58.1로 남는 것이 모순은 아니다. “교정 완료된 critic을 만들었다”는 강한 표현은 부적절하다.
4. demo G는 demo 행동을 이어 수행한 return-to-go다. prior diffusion 행동을 택한 반사실적 수익의 정답이 아니다. Q_ood−G가 음수라는 사실만으로 실제 prior-policy Q의 과소평가를 확정할 수도 없다.
5. 이 저장소의 CQL/Cal-QL은 **IQL V target + prior diffusion action 후보 + 데이터 action을 포함한 logsumexp**를 쓰는 critic-only 변형이다. 논문 표준 CQL/Cal-QL 전체 알고리즘의 우열 실험으로 일반화하지 않는다. 특히 TD는 bootstrap target도 달라서 TD→CQL 차이가 순수 벌점 효과만은 아니다. CQL↔Cal-QL 비교가 floor 유무를 더 직접적으로 비교한다.

## 8. 기존 포스터/인수인계에서 고쳐야 할 주장

| 기존 표현 | 현재 데이터에 맞는 표현 |
|---|---|
| Can·Square 모두 dip 제거 | Can 일부 조건의 **평균곡선**에서 기준선 미만 저하가 관측되지 않음. Square는 **완화**이지 제거가 아님 |
| Cal-QL 교정이 전환을 해결 | 현재 critic-only Cal-QL의 일관된 추가 우위는 없음 |
| Cal-QL 조합이 좋으므로 calibration이 핵심 | 조합은 단독보다 좋지만, 구성요소 단독 대비 추가 이득은 불명확 |
| hq가 초기 dip과 후반을 모두 해결 | tent12 대비 후반은 개선하지만 초기 AUC는 낮고 dip은 남음 |
| hq는 baseline보다 확실히 좋음 | n=5/3 비교에서 평균은 높으나 공통 seed 1–3 차이는 작고 불확실 |
| critic 초기화는 최저 시각을 바꿀 수 없음 | 완화된 곡선의 전 구간 최저 시각은 이동함; 최초 하락과 구분 |
| “42k”라고 쓴 열에 각 조건 최저값 사용 | Square **실제 env=42,032** 평가만 사용. 예: IQL은 이 점 0.503, 초기창 최저는 0.370@online30,048로 서로 다름 |
| final 마지막 3개 평균으로 모든 조건 순위 | 같은 raw env step의 관측치와 coverage n으로 비교 |
| n=1의 ±0.000 | 불확실성 추정 불가; 파일럿으로 표기 |

### 현재 사용할 수 있는 결론 문안

> DSRL의 offline-to-online 초기 저하는 critic 초기화뿐 아니라 noise-policy의 엔트로피 동역학과 replay 구성에 민감하다. Can에서는 엔트로피 조절이 초기 평균 성능 저하를 크게 줄이고, demo replay는 깊은 초기 저하에도 높은 후반 성능을 낸다. Square에서는 같은 엔트로피 처방의 이득이 제한되며, critic target의 entropy bonus를 제거하면 엔트로피를 유지하면서 후반 성능을 개선할 수 있지만 초기 저하까지 해결되지는 않는다. 현재 critic-only TD/CQL/Cal-QL 비교에서 calibration의 일관된 추가 이득은 관측되지 않았다.

이 문안은 관측 결과의 범위다. 유일한 원인, 이론적 보장, 모든 환경에서의 일반성, 선행 연구 대비 최초성은 주장하지 않는다. 이번 작업은 원본/코드 재분석이며 문헌 최신성 검증은 별도다.

## 9. 지금 더 돌려야 하나

**현재 포스터를 위해 바로 추가 학습을 시작할 근거는 없다.** 이미 121개 결과가 있고, 우선 필요한 것은 위 주장·그림·숫자의 정합성이다.

- hq seed 4·5는 baseline/tent12와 seed 구성을 맞추는 데는 유용하지만, 초기 성능의 약점을 없애 주지 않는다. n만 늘리면 “인과가 닫힌다”는 주장은 성립하지 않는다.
- Cal-QL+hq는 calibration을 유지하는 조건을 따로 연구할 때의 후속이다. 현재 결과를 살리기 위해 필수로 돌려야 하는 실험은 아니다.
- prefill+t12i는 초기 저하와 후반 성능의 절충을 연구할 때 자연스러운 다음 조합이다. 이번 데이터에는 없으며 좋을 것으로 확정하지 않는다.
- LP-DS 비교·추가 task·스케줄은 별도 확장 범위다. 이번 분석에 포함되거나 이미 검증된 것처럼 쓰지 않는다.

다음 작업은 포스터에서 “두 과제 dip 제거” 문구 수정, 공통 시점/seed n 표기, 현재 그림 반영, 마지막으로 캡션과 수치 대조다. 포스터 파일 자체는 이 분석 작업에서 아직 교체하지 않았다.

## 10. 산출물과 재현

- `ALL_RESULTS.md`: 37개 조건 전체 표.
- `groups.csv`, `per_seed.csv`: 그룹/121개 seed별 모든 지표와 coverage.
- `contrasts_seed_matched.csv`: 공통 seed 비교, seed별 차이와 개선/악화 수.
- `audit.csv`: 중복·재개·행 수·최종 관측·파일 해시.
- `eval_clean.csv`: 정리된 평가 관측 2,385행.
- `diagnostic_per_seed.csv`, `diagnostic_groups.csv`: 정리된 train 로그의 5k bin 진단.
- `references.csv`: 두 task 기준 정책과 두 종류의 SE.
- `offline_per_seed.csv`, `offline_groups.csv`: 최신 18개 critic/distillation 완료 검사와 후반 사전학습 진단.
- `previous_analysis_check.csv`: 오늘 먼저 만든 Drive 집계와의 차이(수치 오차 이내 0).
- PNG/PDF 그림 6쌍: critic 비교, 조합/상충, Can/Square 전체 조건, Can/Square 진단.

저장소 루트에서:

```bash
python scripts/test_completed_analysis.py
python scripts/analyze_completed_runs.py --logs logs/audit_20260908 --out results/2026-09-08 --previous-comparison logs/analysis_source_20260908/comparison.csv
python scripts/plot_completed_analysis.py --results results/2026-09-08
```

분석은 numpy/pandas, 그림은 matplotlib만 필요하다. 시뮬레이터·torch·GPU·conda가 필요 없다. 이번 실행에서는 bundled Python으로 집계했고, 그 환경에 matplotlib이 없어 이미 설치된 프로젝트 환경으로 그림만 렌더링했다. 회귀 테스트 7개를 통과했다. 모든 PNG를 열어 잘림·범례·축·정렬을 확인했다.

원본 Drive 위치:

- [온라인 로그](https://drive.google.com/drive/folders/1hBokNrw8iE8YHKMl8Tr21ffGbWtKnfv6)
- [사전학습 로그](https://drive.google.com/drive/folders/1y6GWyNKcUErJ70AyX9uqd9VCpYZYcXmU)
- [비교에 쓴 오늘의 기존 분석 묶음](https://drive.google.com/drive/folders/1pMru06lt47Fy_5rQIiaST_Wkk6S5v_Db)

기존 분석 코드와 과거 보고서는 재현성을 위해 지우거나 일괄 재작성하지 않았다. 현재 해석은 이 보고서와 `ALL_RESULTS.md`를 기준으로 한다.
