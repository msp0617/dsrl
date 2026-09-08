# 인수인계 (채팅 AI용: Gemini / claude.ai) — DSRL offline-to-online dip 연구, 2026-09-06 18:40 KST 기준

> **2026-09-08 최신 상태는 [HANDOFF_2026-09-08.md](HANDOFF_2026-09-08.md)를 우선한다.**
> 온라인 본 실험 121/121, 오프라인 사전학습 32/32 완료. 신규 TD/CQL/Cal-QL
> 24개 결과는 아직 집계 전이며, 지금은 추가 학습보다 분석·포스터 갱신이 우선이다.

이 문서는 저장소를 못 보는 채팅 AI가 **판단·분석·문안 작성·Colab 셀 작성**을 이어가기 위한 것이다. 이 문서만으로 충분하도록 썼다.
코드 수정·푸시·로컬 분석은 Claude Code(로컬 `C:\Users\msp17\dsrl`, 브랜치 `o2o`, 원격 `msp0617/dsrl`)에서 한다. 전체 기록은 저장소 `HANDOFF.md`(0~19절).

**마감**: 포스터 2026-09-09(수) 제출, 작성은 9/7(월)·9/8(화).
**지금(18:40)**: **90 run 전부 완료.** 실험은 끝났고 남은 건 포스터. Colab 크레딧 9.3(충전은 내일). 후속 실험은 Google Cloud 무료 체험판(₩415k)으로 이사 예정(§10).

---

## 1. 연구 한 줄과 설정

논문 DSRL(Wagenmaker et al. 2025, arXiv:2506.15799, CoRL 2025)의 offline-to-online 학습에서 처음 수만 step 동안 성공률이 사전학습 정책보다 **떨어지는 dip**의 정체를 robomimic **Can**(보조 Square)에서 밝히고 처방을 비교한다.

- DSRL-NA: 고정 diffusion policy π_dp, 그 입력 노이즈를 고르는 SAC actor π_W(w∈[−1,1]^28, tanh-Gaussian), critic Q_A(행동 공간)와 Q_W(노이즈 공간, Q_A에서 증류).
- 초기 rollout 24,016 env step(Square 32,016) 뒤 학습. env step 1당 grad step 1.25. 평가 100 에피소드, 5k 격자(일부 run 2.5k), 100k 이후 25k(마지막 평가 Can 129k, Square 127k).
- SAC auto-α: 1.0 시작, **목표 엔트로피 0**(논문 설정), log α가 Adam step마다 lr(3e-4)만큼 감소. 정책 엔트로피가 목표에 닿으면 α는 그 엔트로피를 유지하는 값에 머문다. 초기 정책 엔트로피(28차원)는 약 17~19.
- π_dp 기준선(N(0,I) 노이즈, 500 에피소드×3): **0.405 ± 0.014**. 무작위 actor를 붙인 step-0 값(0.5~0.7)은 이보다 높다.
- 로그: `train_log.csv`(1,200 env step마다 `ent_coef`=α, `logp_mean`=E[log π] (엔트로피 = −logp), `mu_absmean`=|actor 평균|, `qw_mean`, 기울기 진단 등), `eval_log.csv`(`env_steps, success_rate`).

## 2. 결과 (seed 평균. "최저"는 seed 평균곡선의 최소(공통 5k 격자). **AUC는 9/7에 정의를 통일함**: 초기 평가를 online step 0에 두고 online 0~75,984(Can, = env 100k − rollout 24,016) / 0~67,984(Square) 구간을 seed별 사다리꼴 적분 후 평균. 9/6까지의 표는 두 정의(15절: 0~100k step 0 포함 / 19절: 첫 평가 29k~100k step 0 제외)가 섞여 있었고, 아래 AUC 열은 전부 새 값. 순위·결론은 그대로이며 warmup +0.04, iql_prefill +0.04, warmupc +0.03, linear +0.02가 가장 큰 변화. `scripts/plot_results.py` 90975b4, `metrics.csv`.)

### 2.1 Can, 축 A(critic 초기화)·축 B(리플레이의 데모 비율)
| 조건 | n | 최저 | 회복 | AUC online 0~76k | 100k | 129k | 최종(300k) |
|---|---|---|---|---|---|---|---|
| baseline | 5 | 0.24 @39k | 84k | 0.41 | 0.48 | 0.53 | 0.64~0.75 |
| iql (critic 사전학습) | 5 | 0.34 | 73k | 0.49 | 0.54 | 0.56 | 0.65~0.75 |
| warmupc (critic만 로드) | 3 | 0.32 | 42k | 0.51 | — | 0.61 | |
| warmup (actor까지 로드) | 3 | step 0 **0.02** | 61k | 0.48 | — | 0.54 | **0.80** |
| mix_prefill (데모를 리플레이에 한 번 넣음 = 상류 `load_offline_data` 옵션, 공개 config 기본값은 False; 비율 0.91→0.43 자연 감쇠) | 3 | 0.09 @34k | 47k | **0.68 **| **0.84** | **0.86** | 0.89 |
| mix_fixed 0.5 (RLPD식 대칭 샘플링) | 3 | 0.29 | 44k | 0.67 | — | 0.84 | 0.85 |
| mix_linear 0.8→0.1 | 3 | 0.02 | 51k | 0.64 | — | 0.75 | 0.81 |
| iql_prefill (두 축 교차) | 3 | 0.16 | 52k | 0.60 | — | 0.85 | 0.84 |

### 2.2 Can, α·엔트로피 축 (헤드라인)
| 조건 | n | 최저 | AUC online 0~76k | 100k | 129k |
|---|---|---|---|---|---|
| auto-α, 목표 0 (= baseline) | 5 | 0.24 | 0.41 | 0.48 | 0.53 |
| 고정 0.01 | 3 | 0.00 | 0.15 | 0.23 | 0.30 |
| 고정 0.03 | 3 | 0.00 | 0.23 | 0.37 | 0.41 |
| 고정 0.1 (엔트로피 3~8) | 3 | 0.14 | 0.50 | 0.63 | 0.62 |
| **고정 0.3** (엔트로피 10~14) | **5** | **0.47** (기준선 위, dip 없음) | **0.66** | **0.79** | **0.74** |
| 고정 1.0 (엔트로피 ~17) | 3 | 0.03 | 0.27 | 0.26 | 0.15 |
| tent12 (auto-α, 목표 엔트로피 12, 초기 α 1.0) | 3 | 0.33 | 0.57 | 0.70 | 0.69 |
| **tent12i** (auto-α, 목표 12, 초기 α 0.3) | 3 | **0.46** (dip 없음) | 0.59 | 0.68 | 0.70 |
| alr_double / alr_half (α lr 2배 / 절반) | 3/3 | 0.28 / 0.27 | 0.47 / 0.39 | 0.53 / 0.35 | — |
| rs_025 / rs_2 (보상 ×0.25 / ×2) | 3/3 | 0.08 / 0.23 | 0.32 / 0.44 | 0.38 / 0.55 | — |
| hardq (critic 타깃에서 엔트로피 보너스 제거) | 3 | 0.10 | 0.30 | 0.31 | — |
| tent6 (auto-α, 목표 6, 100k) | 3 | 0.22 @29k (34k에 0.45로 회복) | 0.51 | 0.61 | — |

### 2.3 Square (rollout 32k, 150k, 마지막 평가 127k)
| 조건 | n | 37k / 42k / 47k / 52k (env) | AUC online 0~68k | 100k | 127k (seed별) |
|---|---|---|---|---|---|
| square_baseline | 3 | 0.24 / **0.21** / 0.34 / 0.32 | 0.40 | 0.44 | 0.47 (0.35/0.62/0.43) |
| square_iql | 3 | 0.40 / 0.50 / 0.42 / 0.41 | 0.44 | 0.48 | 0.56 |
| square_mix_prefill | 3 | 0.30 / 0.03 / 0.14 / 0.35 | 0.43 | 0.63 | 0.62 |
| square_fixalpha_03 (고정 0.3) | **5** | 0.33 / 0.36 / 0.34 / 0.44 | 0.41 | 0.40 | **0.36** (0.09/0.32/0.60/0.52/0.30) |
| square_tent12 (목표 12) | 3 | 0.34 / **0.41** / 0.46 / 0.51 | 0.43 | 0.34 | 0.40 (0.34/0.46/0.39) |
| square_tent6 (목표 6) | 3 | 0.31 / 0.28 / 0.34 / 0.44 | 0.38 | 0.30 | **0.29** (0.39/0.20/0.29), 87k 이후 3 seed 모두 0.2~0.3 |
(전부 완료. Square의 마지막 평가는 127k.)

## 3. 해석 (포스터 문장의 뼈대)

1. **dip의 정체 = auto-α의 엔트로피 붕괴.** 목표 엔트로피 0인 auto-α는 학습 시작 15k 안에 정책 엔트로피를 17 → 0으로 무너뜨린다(`logp_mean` −17@26k → −6@30k → 0@40k, `mu_absmean` 0.2 → 0.8). 모든 auto-α 조건의 바닥이 그 창(Can 35k, Square 42k)에 있다. Q_W가 아직 부정확한데 actor가 한 점으로 몰리는 순간이다.
2. **critic 쪽 설명은 시점을 못 움직였다.** critic 사전학습(iql·warmupc)은 dip을 얕게 할 뿐 시점·최종은 그대로. 보상 ×0.25~×2(Q 스케일)로 첫 하락이 격자 한 칸 안. hard backup(엔트로피 보너스 제거)은 더 깊음. α 감쇠 2배는 5k 앞당기지만 절반은 안 밀림. 기울기 비 `ratio_ge_gq`가 첫 하락 순간 0.48로 일정했던 건 auto-α 평형의 결과. "Q 스케일이 방아쇠" 가설과 그 위의 게이트(신호 vs 시계) 설계는 **기각**.
3. **엔트로피 바닥이 dip 깊이를 정한다 — 두 과제에서 같은 용량-반응.**
   | 과도기 엔트로피 바닥 | Can 최저 → 기준선 회복 | Square 최저 → 회복 |
   |---|---|---|
   | 0 (auto-α, 목표 0) | 0.24 → 84k (긴 dip) | 0.21 → 62k |
   | ~6 (can_tent6, 고정 0.1 / square_tent6) | 0.22 → **34k** (고정 0.1은 0.14 → 39k) | 0.28 → 52k |
   | ~11 (고정 0.3, tent12i / square_tent12) | **0.47, dip 없음** | **0.41, dip 없음** |
   dip은 엔트로피가 **떨어지는 동안** 생기고 떨어지기를 멈추면 곧 회복된다. 바닥이 6이면 깊이는 비슷하되 짧고, 11 이상이면 없다. 두 과제가 같다.
4. **Can에서는 처방이 성립한다.** 고정 0.3 또는 목표 12 + 초기 0.3이면 dip이 없고 100k 0.68~0.79(baseline 0.48), 데모 없이 데모 넣은 것과 같은 AUC.
5. **Square에서는 dip 제거와 최종 성능이 상충한다.** 목표 12는 dip을 없애지만 127k 0.40, 목표 6은 dip 절반에 127k 0.29, 둘 다 baseline 0.47 아래. 고정 0.3은 5 seed 전부 엔트로피가 0 아래로 무너지고(logp +1.7~+4.8, |mu| 0.83~0.90) 127k 0.37(편차 큼). **이유(진단으로 확인)**: 엔트로피를 붙들면 α가 계속 올라가고(tent6 0.2 → 1.0~1.8, tent12 0.4 → 15~17), 그 α가 critic 타깃의 엔트로피 보너스 α·log π'에 들어가 Q_W가 폭주한다. `qw_mean`이 baseline 300~600일 때 tent6 1,700~2,800, **tent12 23,000~24,000**. Square는 γ=0.999(Can 0.99)라 보너스가 10배 길게 누적된다. critic 타깃이 보상이 아니라 엔트로피 보너스로 채워지니 후반 정책이 무너진다. auto-α(목표 0)는 α 0.2~0.3에 머물러 이 문제가 없는 대신 dip을 겪는다. Can은 Q가 작고 α ≤ 0.5라 문제가 안 드러난다.
6. **결론 문장**: "DSRL의 O2O dip은 SAC auto-α가 목표 엔트로피 0을 향해 정책을 무너뜨리는 15k 창이다. 엔트로피가 10 아래로 떨어지지 않게 하면 dip은 두 과제에서 사라진다. 그러나 목표 엔트로피를 붙드는 데 필요한 α가 soft backup을 통해 critic 타깃까지 부풀리므로(γ=0.999인 Square에서 Q_W 20,000+) 후반에는 손해다. 다음 설계는 둘 중 하나다: 엔트로피 목표를 초기에만 두고 낮추는 스케줄, 또는 actor에는 엔트로피 목표를 두되 critic 타깃에서는 보너스를 빼는 것(`critic_entropy_scale=0`, 이미 구현됨)."
7. **축 A·B**(포스터 3·4번 패널): critic 사전학습은 dip을 얕게 하지만 최종 무영향, actor까지 로드하면 step 0 붕괴 후 최종 최고. 데모 리플레이는 회복 10배·최종 0.9로 가장 큰 지렛대, 명시적 linear 스케줄은 논문의 자연 감쇠보다 못하고, 두 축은 안 쌓인다.
8. **auto-α 과도기**: auto-α는 오차 크기와 무관하게 log α를 고정 속도로 움직여 초기 1.0에서 내려오는 동안 undershoot가 생긴다(tent12의 34k dip 0.33). 초기값 0.3으로 두면 해결(tent12i 0.46).

## 4. 선행 연구 (9/6 검색, 완전하지 않음. "우리가 아는 한"으로 쓸 것)
- DSRL 원논문(2506.15799): dip 언급 없음, 데모 리플레이 유지, 온도 논의 없음.
- **LP-DS** (arXiv 2606.01151, Simsir & Oguz 2026): DSRL의 noise가 prior 저밀도로 흘러가고 mode collapse → 섭동 크기 ‖Δ‖²에 Lagrangian trust region. Can·Square·Lift. SAC α·목표 엔트로피 언급 없음. **같은 현상(노이즈 공간 집중)을 다른 지렛대로 막음. 필수 인용.** 우리는 그 집중을 SAC 온도가 만든다는 것을 보임.
- SAC Flow (2509.25756): robomimic O2O에서 목표 엔트로피 0 그대로, dip 논의 없음 → "이 계열이 공유하는 설정의 비용을 아무도 안 봤다".
- Latent Policy Steering (2603.05296): 증류 critic의 손실 지적. GoRL(2512.02581): latent에서 PPO 엔트로피 보너스로 탐색 유지.
- 일반 O2O: WSRL(2412.07762), PORL(2505.16856), OCR(2412.18855)은 분포 이동 + 보수적 Q. Wang·White·White(2505.00913)는 "탐색이 오프라인 정책을 덮어씀" + 성능 추정으로 탐색 점진 허용(스케줄 future work 인용처). Three Regimes(2510.01460).
- 엔트로피 붕괴 개념: TES-SAC(2112.02852, 목표 엔트로피 스케줄), Meta-SAC(2007.01932), AEPO(2510.08141, LLM RFT). "알려진 현상을 DSRL에서 원인으로 확인"이 정확한 표현.

## 5. 구현돼 있는 것 (Hydra override, 기본값이면 상류와 동일)
| 키 | 뜻 |
|---|---|
| `variant` | baseline / iql / warmup / warmupc |
| `offline_mix.mode` | none / prefill / fixed(p0) / linear(p0→p1 until_env) |
| `train.ent_coef` | −1 auto, 양수면 고정, `auto_0.3`이면 auto-α 초기값 0.3 |
| `train.target_ent` | auto-α 목표 엔트로피(논문 0). tent12 = 12, tent6 = 6 |
| `train.ent_coef_lr` | α 옵티마이저 lr 분리(−1 공유) |
| `train.reward_scale`, `train.critic_entropy_scale` | critic 타깃의 r 배율, 엔트로피 보너스 배율(0 = hard backup) |
| `gate.*` | 게이트(구현돼 있으나 폐기, 기본 off) |
분석: `scripts/plot_results.py`(축별 성공률·진단 그림, `metrics.csv`), `scripts/alpha_timing.py`(α·ratio 통과 시점 vs 첫 하락). 테스트 36개 통과.

## 6. 지금 상태 (9/6 18:40)
- **90 run 전부 완료**(18:10, 세션 종료). CSV 묶음: 로컬 `Downloads\csv_bundle (5).zip`(전부). 그림·`metrics.csv`: `Downloads\dsrl_figs_0906\`(success_critic/mix/sweep/adaptive/square/scale/alpha, diagnostics_*). Colab 크레딧 9.3.
- 결과는 Drive `dsrl_project/logs/<exp_id>/{eval_log.csv,train_log.csv}`, `<exp_id>.out`.

## 7. 남은 순서
1. 포스터 6패널 문안 확정(§8) → 월·화 제작. 그림은 `Downloads\dsrl_figs_0906\` 그대로 쓰거나, 필요하면 Claude Code에 "이 축만 다시 그려 달라"고 요청(`plot_results.py --axes` 한 줄).
2. 실험은 끝. 추가 run은 포스터 뒤 GCP에서(§10).
3. Drive에서 결과를 다시 뽑아야 하면 CPU 런타임 → 셀 0 → `%%bash` zip 셀(`cd /content/drive/MyDrive/dsrl_project && zip -qr csv_bundle.zip logs -i "logs/*/eval_log.csv" "logs/*/train_log.csv" "logs/*.csv"`).

## 8. 포스터 (6 패널, Can 중심)
1. **문제**: DSRL O2O에서 baseline이 0.5 → 0.24(39k) → 84k에야 기준선 0.405 회복. `success_critic.png`.
2. **정체**: α와 `logp_mean` 곡선을 성공률에 겹침. auto-α가 엔트로피를 17 → 0으로 무너뜨리는 15k 창이 dip. "actor가 부정확한 Q_W의 한 점으로 몰리는 순간". critic 사전학습·Q 스케일 ×0.25~×2·hard backup은 시점을 못 움직임(한 줄).
3. **critic 초기화(축 A)**: dip 얕게, 최종 무영향. actor 로드는 step 0 붕괴 후 최종 최고.
4. **데모 리플레이(축 B)**: 회복 47k, 최종 0.9. 명시적 linear는 자연 감쇠보다 못함. 두 축 안 쌓임. (RLPD 대칭 샘플링 = mix_fixed 0.5, Cal-QL·REDQ 인용)
5. **α 스윕 U자 + 고정 0.3(dip 없음, 100k 0.79, 5 seed) + 적응형 tent12i(dip 없음)**. 문구: "처방은 엔트로피 유지. 목표 엔트로피만 맞게 주면 auto-α가 적응형 제어기". `success_sweep.png`, `success_adaptive.png`.
6. **Square: 같은 용량-반응, 그러나 상충과 그 이유**. 42k 바닥 재현(예측 적중). 엔트로피 바닥 0/6/11 → dip 0.21/0.28/0.41(Can과 같음). 하지만 엔트로피를 붙들면 후반을 잃음(127k 0.29/0.40 vs 0.47): α가 1~17까지 올라가 critic 타깃의 엔트로피 보너스를 부풀림(Q_W 2,000~24,000, γ=0.999). 고정 0.3은 엔트로피가 다시 무너짐(5/5 seed). 결론: "actor의 엔트로피 목표와 critic 타깃의 보너스를 분리하거나, 목표를 초기에만 두는 스케줄". LP-DS 인용. `success_square.png`, `diagnostics_square.png`(ent_coef·qw_mean 패널).

## 9. 표현 주의
- α는 **무작위성**("고집" 아님). 높으면 퍼지고 낮으면 한 점.
- "α=0.3이 답"이라 쓰지 않는다. Square에서 틀렸다. "엔트로피 붕괴가 dip, 붕괴를 막으면 dip이 없다(두 과제), 최종 성능은 과제·시기에 따라 다른 엔트로피를 요구한다"까지.
- Q 스케일 가설·게이트는 기각됐다고 명시. 숨기지 않는다. "우리가 아는 한" 없이 "처음"이라고 쓰지 않는다. LP-DS 필수 인용.
- 평가 노이즈 ±0.1. 개별 seed 한 점으로 말하지 않는다. 3~5 seed 평균±SE. Square 127k는 seed 편차가 특히 커서(고정 0.3: 0.09~0.60) "이득 없음"까지만.
- §3.5의 "α 상승 → Q_W 폭주"는 `diagnostics_square.png`(ent_coef, qw_mean 패널)로 확인된 사실. 다만 "그래서 후반 정책이 무너진다"는 인과의 마지막 고리는 `critic_entropy_scale=0` 실험(§10)으로 닫아야 한다.
- alr_half(α를 오래 붙듦)가 나빴던 건 결국 0.05까지 내려가 엔트로피가 0으로 무너지기 때문. 붙드는 시간이 아니라 **어디까지 내려가느냐**가 문제.

## 10. Google Cloud 이사 (포스터 뒤, 후속 실험용)
- 무료 체험판 ₩414,984(12/6 만료). GPU를 쓰려면 "일반 계정 활성화"(크레딧 유지, 소진 뒤 과금) → 예산 알림 → Compute Engine API(켜짐) → 할당량 `GPUs (all regions)`·`NVIDIA L4 GPUs(us-central1)` 1 요청(승인 몇 분~며칠). 무료 체험판 상태에서는 GPU 할당량 항목이 아예 안 보인다.
- 저장소 `gce/`에 준비됨: `setup_gce.sh`(VM에 Colab과 같은 경로를 만들어 명령이 그대로 돌게 함), `sync_drive.sh`(rclone으로 Drive ↔ VM), `launch.sh`(run 목록 띄우고 끝나면 CSV를 Drive에 올리고 VM 정지), `README.md`(VM 생성 명령, 비용). 추천 머신 `g2-standard-32`(32 vCPU, 128GB, L4, run 6개, 약 $1.5~2/h, Spot 약 $0.5). 승인 뒤 약 1시간이면 이사 완료.
- 후속 실험 후보 (우선순위 순): **① `square_tent12` + `train.critic_entropy_scale=0`** 3 seed(actor는 엔트로피 12 유지, critic 타깃에선 보너스 제거) — §3.5가 맞으면 Square에서 dip 없이 후반도 baseline 이상이어야 한다. 같은 조합을 Can에도(`can_tent12i` + β=0). ② 목표 엔트로피 스케줄(초기 12 → 후반 0, 낮추는 시점 = 원래 게이트의 센서 질문). ③ Lift·Transport 추가, seed 확장. `gce/runs_example.txt` 형식으로 한 줄씩 적으면 `launch.sh`가 띄운다.

## 11. 채팅 AI에게 바라는 것
- 포스터 6패널 문안 초안·비판, 영어 제목·초록, 관련 연구 문단, 심사자 예상 질문과 답.
- 결과 해석의 과장 검출(§9 기준).
- Colab 셀이 필요하면 §7의 형식(`%%bash` 또는 `run_bash(r'''...''')`)으로.
