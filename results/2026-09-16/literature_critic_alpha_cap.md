## 문헌 노트: critic-only alpha cap 관련 선행 연구 (2026-09-16)

검증(verifier) 결과: 9편 모두 confirmed, 반박(refuted)된 항목 없음 — "checked and not applicable"에 해당하는 논문은 없으며, 검증되지 않은 finder-only 항목은 별도 표로 분리했다.

### 1. 요약 단락

(a) critic 타깃의 온도를 actor와 분리하는 발상 자체는 이미 두 편에서 확인된다. Asad et al. (2025, arXiv:2509.09838)는 TD 타깃 r + γ[q(s',a') − ζ ln π(a'|s')]의 critic 엔트로피 계수 ζ를 actor의 자동 조정 온도 τ와 명시적으로 분리하고, 결합(ζ = τ)이 DSAC 성능 저하의 주원인이라며 ζ = 0 + 자동 τ actor를 채택했으나, ζ ∈ {0, τ} 두 끝점만 이산 행동 Atari에서 비교했고 중간값이나 cap은 시험하지 않았다. Zhan et al. (TECRL/DSAC-E, 2025, arXiv:2511.11592)는 "온도 가중 엔트로피 항이 Q 타깃을 비정상(non-stationary)으로 만든다"는 우리와 동일한 진단 아래 reward critic에서 α를 완전히 제거하고(hard backup) α-free 엔트로피 critic Q_e를 별도로 두어 actor에만 되먹이는데, 이는 우리 family의 cap → 0 극한에 추가 구성요소를 얹은 것이며 유한 cap, 보상 스케일링, offline-to-online, 확산 latent-noise 정책은 다루지 않는다. (b) Bellman 타깃 안의 엔트로피/log-policy 항을 제한한 가장 가까운 선행은 MDAC (Iwaki, 2025, arXiv:2502.03854)로, actor loss와 자동 α(목표 엔트로피 −dim(A))는 그대로 두고 critic 타깃의 α log π 항만 g = clip(x/10, −1, 1) 또는 tanh(x/10)로 per-sample 유계화하며, "log π 집중 → α 증가 → α log π 폭발 → Q 훼손"이라는 진단은 우리 Square 실패와 같지만, 계수 α가 아니라 곱 α·log π에 비선형 bound를 거는 점과 Munchausen 항이 함께 있다는 점이 다르다; Munchausen RL (Vieillard et al., NeurIPS 2020)은 현재 행동의 log-policy 보너스를 [τ ln π]_{−1}^{0}으로 클리핑하는 정본 선례이나 이산 행동·고정 τ·다음 상태 엔트로피 항 비클리핑이고, SCQ (Wu et al., 2026, arXiv:2609.12749)는 −log π 자체를 유계 sigmoid 점수로 바꾸되 actor/critic에 동일한 공유 α를 쓰며, You (2023, arXiv:2305.11831)는 같은 α로 타깃에서 α·H0를 빼 보너스를 α(H(π') − H0)로 중심화할 뿐 α를 제한하지 않는다. (c) 보상 스케일이 온도의 역수라는 쌍대성은 SAC v1 (Haarnoja et al., ICML 2018)이 원전이며 "튜닝이 필요한 유일한 하이퍼파라미터"라고 명시하지만, 하나의 전역 스케일이 critic 타깃과 actor loss에 동일하게 들어가고 자동 온도 조정과 결합되지 않는다; 우리가 아는 한, 자동 온도 조정 하에서 α의 성장을 억제할 목적으로 보상 스케일링을 critic 엔트로피 보너스 제어 수단으로 사용·검증한 연구는 확인하지 못했다. (d) soft vs hard backup 절제는 RLPD (Ball et al., ICML 2023)가 연속 제어·offline-to-online에서 backup_entropy 플래그(actor는 자동 α 유지)로 수행했고, 어려운 도메인(AntMaze large-diverse, Adroit sparse, V-D4RL humanoid)에서는 엔트로피 백업이 항상 더 나빴다고 보고하나, on/off 이진 스위치이며 α 성장이나 Q 팽창과 연결짓지 않았고 우리가 관찰한 "hard backup은 초기 성능을 잃는다"는 trade-off에 대한 답은 없다; LP-DS (Simsir & Oguz, ICML 2026)는 robomimic Can/Square에서 엔트로피 항 없는 hard backup + latent trust-region을 쓰지만 soft/hard 절제는 하지 않았다. 종합하면, 우리가 아는 한, actor의 자동 α는 유지한 채 critic TD 타깃 안에서만 α_c = min(α, cap)으로 온도 계수를 유한한 상한(0 < cap < ∞)으로 제한하는 중간 형태를 연속 제어 또는 offline-to-online 설정에서 시험한 연구는 없으며, 검증된 선행들은 모두 이 family의 두 끝점(cap = ∞: SAC v2 기본, cap = 0: RLPD/Asad/DSAC-E hard backup) 또는 다른 레버(항 자체의 유계화, 보상 스케일, 목표 엔트로피)에 해당한다.

### 2. 검증된 선행 연구 표

| 논문 | 연도 | 기제 | 우리 cap과의 관계 | 검증 |
|---|---|---|---|---|
| Asad, Babanezhad, Vaswani, "Dissecting Discrete SAC" (arXiv:2509.09838) | 2025 | critic 타깃 계수 ζ를 actor 자동 τ와 분리; ζ = 0 채택 (이산 Atari) | 같은 발상; 우리 cap = min(τ, cap)은 그들이 시험하지 않은 중간값 (ζ ∈ {0, τ}만 절제, 이산 행동, 기제 설명 없음) | confirmed — "disabling the entropy regularization in the critic update ... keeping ... automatic entropy tuning fixed, yields a stable variant of DSAC" |
| Iwaki, "Mirror Descent Actor Critic via Bounded Advantage Learning" (MDAC, arXiv:2502.03854) | 2025 | critic 타깃의 α log π 항만 tanh/clip(x/10)로 유계화; actor loss·자동 α는 그대로 (연속 MuJoCo) | 같은 범주(b), 다른 함수형: 곱 α·log π에 per-sample 비선형 bound vs 우리는 계수 α만 선형 cap; Munchausen 항 포함; "느리게 포화하는 bound가 낫다"는 결과는 온건한 cap을 지지 | confirmed — "log π concentrates to high value, which makes α grow. Then, α log π terms explode and hinder Q" |
| Vieillard, Pietquin, Geist, "Munchausen RL" (NeurIPS 2020, arXiv:2007.14430) | 2020 | 현재 행동 log-policy 보너스를 [τ ln π]_{l0}^{0}, l0 = −1로 클리핑 (M-DQN, 이산) | 다른 레버: 현재 행동 항 클리핑(다음 상태 엔트로피 항은 코드상 비클리핑), 고정 τ, actor/critic 분리 없음; "타깃 내 log-policy 항은 유계화해야 한다"는 동기 인용 | confirmed — "the log-policy term is not bounded, and can cause numerical issues if the policy becomes too close to deterministic" |
| Ball, Smith, Kostrikov, Levine, "RLPD" (ICML 2023, arXiv:2302.02948) | 2023 | backup_entropy 플래그로 critic 타깃의 엔트로피 항만 on/off; actor는 자동 α 유지 (연속, O2O) | 특수 경우: cap = 0 (우리 β = 0 arm)과 동일; 이진 스위치이며 α 성장·Q 팽창과 무관한 경험적 동기, 중간 cap 없음 | confirmed — "we see using entropy backups and smaller networks always results in worse performance" |
| Wu, Zhang, Chu, Hu, "SCQ: Sigmoid-Bounded Entropy" (arXiv:2609.12749) | 2026 | −log π를 유계 sigmoid 점수 H_sig ∈ (0, d_a h_max)로 대체, actor·critic 모두, 공유 자동 α (O2O, 실로봇) | 다른 레버, 같은 계열: 점수(score)를 유계화하고 α는 제한하지 않음; critic-only 배치·hard backup 절제 없음 | confirmed — "By construction, H_sig ∈ (0, d_a h_max) ... y = r + (1−d)γ(min Q̄(s′,a′) + α H_sig(s′,a′))" |
| Haarnoja, Zhou, Abbeel, Levine, "SAC v1" (ICML 2018, arXiv:1801.01290) | 2018 | α = 1 고정, 과제별 보상 스케일 튜닝; 보상 스케일 = 1/온도 | 다른 레버: 보상 스케일 arm의 원전; 전역 스케일이 critic·actor에 동일 적용, 자동 α 없음, cap 미예견 | confirmed — "reward scale to be the only hyperparameter that requires tuning, and its natural interpretation as the inverse of the temperature" |
| Zhan et al., "Mind Your Entropy" (TECRL/DSAC-E, arXiv:2511.11592) | 2025 | reward critic은 α 없는 hard backup, 별도 α-free 엔트로피 critic Q_e; α는 actor loss·온도 loss에만 (연속 MuJoCo) | 같은 동기(α 갱신이 타깃을 비정상화), 다른 레버: cap = 0 극한 + 엔트로피 critic; "w/o TEC" 절제가 '자동 α actor + α 없는 critic'의 가장 가까운 대응; 유한 cap·보상 스케일·O2O 없음 | confirmed — "This reward-centric critic explicitly excludes entropy bonuses, which ensures a clean value target uninfluenced by policy stochasticity" |
| You, "Regularization of SAC with Automatic Temperature Adjustment" (arXiv:2305.11831, 미심사 단독 저자) | 2023 | critic 타깃에서 α·H0를 빼 보너스를 α(H(π') − H0)로 중심화; actor·critic 같은 α | 다른 레버, 같은 대상 수량: 계수 제한이 아닌 baseline 제거; cap과 결합 가능한 대안 arm; Pendulum 예시뿐 | confirmed — "Q = r + E[Q′ − α_{t+1} log π′ − α_{t+1} H0] (24) ... Besides the item related to H0, (21) is identical to (22)" |
| Simsir, Oguz, "LP-DS" (ICML 2026, arXiv:2606.01151) | 2026 | DSRL latent에 residual 섭동 + Lagrangian trust region; 엔트로피 항 없는 hard TD backup (robomimic Can/Square 포함) | 다른 레버: 온도·엔트로피 항 자체가 없음; hard backup + 비엔트로피 정규화가 후반 붕괴를 피한다는 정황 근거; soft/hard 절제 없음 | confirmed — "y = r + γ Q̄^A(s′,a′) (Alg. 1); DSRL predicts higher-magnitude latent queries that correlate with ... performance degradation" |

### 2-1. finder 확인만 (독립 검증 없음 — 본문 주장에 사용하지 말 것)

| 항목 | 범주 | finder 요약 |
|---|---|---|
| SigEnt-SAC (arXiv:2601.15761) | entropy_bonus_clip | SCQ의 자매 논문; 유계 sigmoid 엔트로피를 critic 타깃·actor 모두에, 공유 α |
| SERL / HIL-SERL, Cal-QL (JaxCQL) 코드베이스 | hard_vs_soft_backup | backup_entropy=False 기본값 (코드 관행, 논문 절제 아님); DSRL 코드에는 해당 스위치 미발견 |
| XQL (arXiv:2301.02328) | hard_vs_soft_backup | Gumbel 회귀로 soft value 직접 추정, 타깃에 샘플 −α log π 항 없음 (초록만) |
| Fable67/Soft-Actor-Critic (M-SAC 코드) | entropy_bonus_clip | 연속 SAC에 Munchausen 항 clamp(M_TAU·logπ, −1, 0) 추가; 엔트로피 보너스는 비클리핑 |
| Stable Discrete SAC (arXiv:2209.10081) | other | actor 엔트로피 페널티 + Q-clip(per-update TD 변화량 [−c, c]); 이산, 엔트로피 항 미클리핑 |
| SAC v2 (arXiv:1812.05905) | target_entropy_schedule | 자동 α, 상한 없음, critic·actor 동일 α (초록·기억 기반) |
| TES-SAC (arXiv:2112.02852) | target_entropy_schedule | 도달 불가 목표 엔트로피 시 α 지수적 발산; 목표 엔트로피 어닐링 (이산) |
| Truly-satisfied Inequality Constraint SAC (arXiv:2303.04356) | target_entropy_schedule | 상태 의존 slack Δ(s)로 α 상승 억제; α cap·critic 타깃 변경 없음 |
| Stable-Baselines3 SAC 문서 | reward_scale_alpha | ent_coef = 1/reward scale 언급, log α 최적화; 상한·critic 분리 없음 |
| EAPO (arXiv:2407.18143) | other | 엔트로피 advantage 분리 + PopArt (on-policy PPO); PopArt 세부 미검증 |
| DrQ-v2 (arXiv:2107.09645) | hard_vs_soft_backup | SAC 대신 DDPG hard backup; 자동 엔트로피 조정의 조기 붕괴 지적 |
| EXPO (arXiv:2507.07986) | hard_vs_soft_backup | 기본 hard backup, 소량 데이터 시에만 엔트로피 백업; robomimic 포함, 전용 절제표 없음 |
| LPS (arXiv:2603.05296) | hard_vs_soft_backup | DSRL 재구현에서 엔트로피 항 의도적 생략 (hard backup, OGBench/DROID) |
| SAC Flow (arXiv:2509.25756) | target_entropy_schedule | full soft backup; robomimic O2O에서 목표 엔트로피 0으로 α 제어 |
| ElegantRL AgentSAC / TorchRL SACLoss | other | log α clamp(−16, 2) 전역 cap(critic·actor 공통) / max_alpha 옵션; SB3·CleanRL·jaxrl·rlpd에는 clamp 없음; TD-MPC2는 hard target |

### 3. Related work (poster footnote)

- Decoupling the critic-target entropy coefficient from the actor's auto-tuned temperature has precedent in discrete-action SAC (Asad et al., 2025, arXiv:2509.09838; ζ ∈ {0, τ} only) and in TECRL/DSAC-E (Zhan et al., 2025, arXiv:2511.11592; α removed from the reward critic, separate α-free entropy critic); neither tests an intermediate cap, continuous control, or offline-to-online RL.
- Bounding the entropy/log-policy term inside the Bellman target: Munchausen RL clips the current-action log-policy bonus (Vieillard et al., NeurIPS 2020); MDAC bounds α log π(a'|s') in the critic target with tanh/clip(x/10) while leaving the actor's auto-α untouched (Iwaki, 2025, arXiv:2502.03854) and diagnoses the same α-growth failure loop; SCQ replaces −log π by a bounded sigmoid score in both actor and critic with a shared α (Wu et al., 2026, arXiv:2609.12749).
- Hard backup with a soft actor (our β = 0 arm, i.e. cap = 0) is an environment-specific switch in RLPD (Ball et al., ICML 2023), reported to be strictly better on hard sparse-reward domains; LP-DS uses an entropy-free hard backup on robomimic Can/Square (Simsir & Oguz, ICML 2026) without a soft/hard ablation.
- Reward scale as the inverse temperature is the original SAC v1 formulation (Haarnoja et al., ICML 2018); SAC v2's automatic temperature (arXiv:1812.05905) replaced it with an unbounded dual variable that enters the critic target and actor loss identically.
- Centering the target bonus to α(H(π') − H0) with the same α in actor and critic is proposed, without experiments beyond Pendulum, by You (2023, arXiv:2305.11831).
- To our knowledge, a finite cap α_c = min(α, cap) applied only inside the critic TD target, with the actor's α left auto-tuned, has not been tested in continuous control or offline-to-online RL.

### 4. 사전 등록(pre-registration)에 인용할 통제/선례

1. RLPD (Ball et al., ICML 2023, arXiv:2302.02948) — hard-backup control arm의 선례: backup_entropy=False + actor 자동 α가 정확히 우리 β = 0 (cap = 0) arm이며, 연속 제어·O2O에서 코드까지 공개된 유일한 검증 선례이므로 통제군 정의와 "엔트로피 백업이 항상 더 나빴다"는 비교 기준으로 인용.
2. Asad, Babanezhad, Vaswani (2025, arXiv:2509.09838) — critic 계수 ζ와 actor 온도 τ의 분리 선례: 우리 cap family의 두 끝점(ζ = τ, ζ = 0)이 이 논문의 절제와 대응하므로, 유한 cap이 "새로운 중간 구간"임을 명시하는 근거로 인용 (이산 행동 한정임을 병기).
3. SAC v1 (Haarnoja et al., ICML 2018, arXiv:1801.01290) — 보상 스케일 arm의 원전: 보상 스케일 = 1/온도 쌍대성과 양극단 실패 양상(과대 α → 균일 정책, 과소 α → 조기 결정론)을 reward-scale arm의 예측과 판정 기준에 인용.
4. MDAC (Iwaki, 2025, arXiv:2502.03854) — 타깃 내 α log π 항 유계화 선례이자 진단 일치: "느리게 포화하는(거의 활성화되지 않는) bound가 공격적 bound보다 낫다"(Fig. 4c)는 결과를 cap 값 선택(온건한 cap 우선)의 사전 근거로 인용; Munchausen RL (NeurIPS 2020)은 "타깃 내 log-policy 항은 유계화해야 한다"는 동기 인용으로만 사용하고 통제군으로는 쓰지 않음.