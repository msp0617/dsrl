# 인수인계 (claude.ai 채팅용) — DSRL offline-to-online dip 연구, 2026-09-06 11:00 기준

저장소를 못 보는 채팅 Claude가 **판단·분석·Colab 셀 작성**을 이어가기 위한 문서.
코드 수정·푸시는 Claude Code(로컬 `C:\Users\msp17\dsrl`, 브랜치 `o2o`, 원격 `msp0617/dsrl`)에서. 전체 기록은 저장소 `HANDOFF.md`(0~19절).

**마감**: 포스터 2026-09-09(수) 제출, 작성은 9/7(월)·9/8(화). 오늘 9/6(일) 오후에 마지막 6 run이 끝난다.
**제약**: 크레딧 약 80(10:30) → 6 run 뒤 약 35. VM 하나 시간당 9. 사용자는 Colab 셀을 실행하고 출력을 붙여 준다.

---

## 1. 연구 한 줄과 설정

논문 DSRL(Wagenmaker et al. 2025, arXiv:2506.15799)의 offline-to-online 학습에서 처음 수만 step 동안 성공률이 사전학습 정책보다 **떨어지는 dip**의 정체를 robomimic **Can**(보조 Square)에서 밝히고 처방을 비교한다.

- DSRL-NA: 고정 diffusion policy π_dp, 그 입력 노이즈를 고르는 SAC actor π_W(w∈[−1,1]^28, tanh), critic Q_A와 Q_W(노이즈 공간, Q_A에서 증류).
- 초기 rollout 24,016 env step(Square 32,016) 뒤 학습. env step 1당 grad step 1.25. 평가 100 에피소드, 5k 격자(일부 run 2.5k), 100k 이후 25k.
- SAC auto-α: 1.0 시작, **목표 엔트로피 0**(논문 설정), log α가 Adam step마다 lr(3e-4)만큼 감소. 정책 엔트로피가 목표에 닿으면 α는 그 엔트로피를 유지하는 값에 머문다.
- π_dp 기준선(N(0,I) 노이즈, 500 에피소드×3): **0.405 ± 0.014**. 무작위 actor를 붙인 step-0 값(0.5~0.7)은 이보다 높다.
- 처리량: G4 VM(vCPU 48, RAM 176GB)에 run 9개까지(run당 RAM 17~19GB). 6개면 run당 약 9 env step/s → 150k에 5시간.

## 2. 결과 (seed 평균. "최저"는 seed 평균곡선의 최소, AUC는 24k~100k 정규화, 공통 5k 격자)

### 2.1 Can, 축 A(critic 초기화)·축 B(리플레이의 데모 비율)
| 조건 | n | 최저 | 회복 | AUC | 100k | 129k | 최종(300k) |
|---|---|---|---|---|---|---|---|
| baseline | 5 | 0.24 @39k | 84k | 0.40 | 0.48 | 0.53 | 0.64~0.75 |
| iql (critic 사전학습) | 5 | 0.34 | 73k | 0.49 | 0.54 | 0.56 | 0.65~0.75 |
| warmupc (critic만 로드) | 3 | 0.32 | 42k | 0.48 | — | 0.61 | |
| warmup (actor까지 로드) | 3 | step 0 **0.02** | 61k | 0.44 | — | 0.54 | **0.80** |
| mix_prefill (논문 기본, 데모 비율 0.91→0.43 자연 감쇠) | 3 | 0.09 @34k | 47k | **0.68** | **0.84** | **0.86** | 0.89 |
| mix_fixed 0.5 (RLPD식) | 3 | 0.29 | 44k | 0.66 | — | 0.84 | 0.85 |
| mix_linear 0.8→0.1 | 3 | 0.02 | 51k | 0.62 | — | 0.75 | 0.81 |
| iql_prefill (두 축 교차) | 3 | 0.16 | 52k | 0.56 | — | 0.85 | 0.84 |

### 2.2 Can, α 축 (헤드라인)
| 조건 | n | 최저 | AUC | 100k | 129k |
|---|---|---|---|---|---|
| auto-α (= baseline) | 5 | 0.24 | 0.40 | 0.48 | 0.53 |
| 고정 0.01 | 3 | 0.00 | 0.18 | 0.23 | 0.30 |
| 고정 0.03 | 3 | 0.00 | 0.24 | 0.37 | 0.41 |
| 고정 0.1 | 3 | 0.14 | 0.51 | 0.63 | 0.62 |
| **고정 0.3** | **5** | **0.47** (기준선 위, dip 없음) | **0.66** | **0.79** | **0.74** |
| 고정 1.0 | 3 | 0.03 | 0.25 | 0.26 | 0.15 |
| alr_double (α lr 2배) | 3 | 0.28 | 0.48 | 0.53 | — |
| alr_half (α lr 절반) | 3 | 0.27 | 0.39 | 0.35 | — |
| tent12 (auto-α, 목표 엔트로피 12) | 3 | 0.33 | 0.57 | 0.70 | 0.69 |
| rs_025 / rs_2 (보상 ×0.25 / ×2) | 3/3 | 0.08 / 0.23 | 0.29 / 0.44 | 0.38 / 0.55 | — |
| hardq (critic 타깃에서 엔트로피 보너스 제거) | 3 | 0.10 | 0.27 | 0.31 | — |

### 2.3 Square (rollout 32k, 150k, 마지막 평가 127k)
| 조건 | 최저 | AUC 37~100k | 100k | 127k |
|---|---|---|---|---|
| square_baseline | 0.21 @42k | 0.40 | 0.44 | 0.47 |
| square_iql | 0.37 | 0.44 | 0.48 | 0.56 |
| square_mix_prefill | 0.03 @42k | 0.43 | 0.63 | 0.62 |
| square_fixalpha_03 (고정 0.3) | 0.33 | 0.42 | 0.39 | **0.34** (seed별 0.09 / 0.32 / 0.60) |

## 3. 해석 (포스터 문장의 뼈대)

1. **dip의 정체 = auto-α의 엔트로피 붕괴.** 목표 엔트로피 0인 auto-α는 학습 시작 15k 안에 정책 엔트로피를 17 → 0으로 무너뜨린다(`logp_mean` −17@26k → −6@30k → 0@40k, `mu_absmean` 0.2 → 0.8). 모든 auto-α 조건의 바닥이 그 창(Can 35k, Square 42k)에 있다. Q_W가 아직 부정확한데 actor가 한 점으로 몰리는 순간이다.
2. **엔트로피는 보호막.** α=0.01·0.03 고정은 즉시 붕괴(최저 0.00, 회복 없음). α=0.3 고정은 엔트로피 10~14를 유지해 **dip이 없고** 데모 없이 AUC가 prefill과 같고 100k에서 baseline보다 0.3 높다(5 seed). 1.0은 너무 퍼져 학습이 안 됨(U자).
3. **α 시계·Q 스케일은 방아쇠가 아니다.** α 감쇠를 2배로 하면 dip이 5k 앞당겨지지만 절반으로 해도 안 밀린다(alr_half는 dip이 길고 100k 최악). 보상을 ×0.25~×2로 바꿔도 첫 하락은 격자 한 칸 안이고, hardq(엔트로피 보너스 제거)는 더 깊다. 기울기 비 `ratio_ge_gq`가 첫 하락 순간 0.48로 일정했던 건 auto-α 평형의 결과(40k에 엔트로피가 0에 닿고 α는 Q 크기에 비례: rs_025 0.012, baseline 0.05, rs_2 0.15). 이 가설과 그 위에 세운 게이트(신호 vs 시계)는 **기각·폐기**. 한 줄로 보고.
4. **Q 스케일은 회복 속도를 정한다.** rs_2 39k, baseline 64k, rs_025 62k, hardq 81k. 100k 성능은 평형 α 순서.
5. **축 A**(iql·warmupc)는 dip을 얕게 하지만 최종 무영향. actor까지 로드하면 step 0 붕괴 후 최종 최고.
6. **축 B**(데모 리플레이)는 회복 10배·최종 0.9로 가장 큰 지렛대. 명시적 linear는 자연 감쇠보다 못함. 두 축은 안 쌓임.
7. **고정 0.3은 Square로 전이 안 됨.** Square는 Q 스케일이 Can의 5배(Q_W 300~700 vs 30~100)라 auto-α가 스스로 0.2~0.3에 앉는다. 거기서 0.3 고정은 높은 α가 아니고, 엔트로피는 0 아래로 무너지며(seed 1 `logp_mean` +6.6, 성공률 0.09) 후반이 baseline보다 나쁘다. auto-α는 붕괴 시 α를 올려 엔트로피 0을 지키지만 고정 α는 못 한다. **주장은 "α=0.3"이 아니라 "엔트로피를 무너뜨리지 말라"**, 전이 가능한 처방 후보는 목표 엔트로피.
8. **적응형(목표 엔트로피 12)**: Can에서 tent12는 baseline·iql보다 낫지만 고정 0.3보다 못하다. α가 1.0에서 0.25까지 내려갔다가(30k) 엔트로피 12에 닿은 뒤 0.35~0.45로 올라오는 과도기가 34k dip(0.33)이다. 초기값을 0.3으로 두면(`auto_0.3`) 해결될 가능성 → 오늘 `can_tent12i`.

## 4. 구현돼 있는 것 (Hydra override, 기본값이면 상류와 동일)

| 키 | 뜻 |
|---|---|
| `variant` | baseline / iql / warmup / warmupc |
| `offline_mix.mode` | none / prefill / fixed(p0) / linear(p0→p1 until_env) |
| `train.ent_coef` | −1 auto, 양수면 고정, `auto_0.3`이면 auto-α 초기값 0.3(SB3 문법, 5df31e6부터 fingerprint 허용) |
| `train.target_ent` | auto-α 목표 엔트로피(논문 0). tent12 = 12 |
| `train.ent_coef_lr` | α 옵티마이저 lr 분리(−1 공유) |
| `train.reward_scale`, `train.critic_entropy_scale` | critic 타깃의 r 배율, 엔트로피 보너스 배율(0 = hard backup) |
| `gate.*` | 게이트(구현돼 있으나 폐기, 기본 off) |

`train_log.csv`(1,200 env step마다): `env_steps, ent_coef, logp_mean, mu_absmean, w_absmean, w_frac_sat, log_std_mean, qw_mean, qw_absmean, gq_norm, ge_norm, ratio_ge_gq, gate_open, gate_open_call, offline_p`. 진단 4열(`qw_absmean`~`ratio_ge_gq`)은 9/5 밤 이후 run에만 있다. `eval_log.csv`: `env_steps, success_rate, avg_reward, mc_return, q_start`.
테스트 36개 통과. 최신 커밋 5df31e6.

## 5. 지금 상태 (9/6 11:00)

- 70 run 완료. 어젯밤 것: `can_rs_025/rs_2/hardq_s{1,2,3}`(100k), `can_tent12_s{1,2,3}`, `can_fixalpha_03_s{4,5}`, `square_fixalpha_03_s{1,2,3}`(150k).
- **10:30 시작, 15:30 완료 예정 (G4 1대, 자동 반납 keepalive)**: `square_tent12_s{1,2,3}`(`train.target_ent=12`), `can_tent12i_s{1,2,3}`(`train.ent_coef=auto_0.3 train.target_ent=12`). 150k.
- 결과는 Drive `dsrl_project/logs/<exp_id>/{eval_log.csv,train_log.csv}`, `<exp_id>.out`.

## 6. 오늘 오후 순서

### 6.1 완료 확인 + CSV 묶음 (GPU 불필요: 노트북에서 런타임 유형을 CPU로 → 셀 0(Drive) → 아래)
```bash
%%bash
PROJ=/content/drive/MyDrive/dsrl_project
for E in square_tent12_s1 square_tent12_s2 square_tent12_s3 can_tent12i_s1 can_tent12i_s2 can_tent12i_s3; do
  printf "%-18s done=%s  last_eval=%s\n" $E "$(grep -c '\[done\]' $PROJ/logs/$E.out)" "$(tail -n 1 $PROJ/logs/$E/eval_log.csv | cut -d, -f2,5)"
done
cd $PROJ && rm -f csv_bundle.zip && zip -qr csv_bundle.zip logs -i "logs/*/eval_log.csv" "logs/*/train_log.csv" "logs/*.csv" && ls -la csv_bundle.zip
```
6개 `done=1`(Can 129152, Square 127136) → Drive에서 `csv_bundle.zip` 다운로드.

### 6.2 로컬 분석 (PowerShell, 저장소 폴더)
```powershell
cd C:\Users\msp17\dsrl
git pull origin o2o
Expand-Archive -Force $HOME\Downloads\csv_bundle.zip $HOME\Downloads\csv_bundle
cd scripts
..\.venv\Scripts\python plot_results.py --logs $HOME\Downloads\csv_bundle\logs --out $HOME\Downloads\figs --axes "sweep=baseline,fixalpha,fixalpha_003,fixalpha_01,fixalpha_03,fixalpha_1;adaptive=baseline,iql,mix_prefill,fixalpha_03,tent12,tent12i;square=square_baseline,square_iql,square_mix_prefill,square_fixalpha_03,square_tent12;scale=baseline,rs_025,rs_2,hardq;critic=baseline,warmup,iql,warmupc,fixalpha;mix=baseline,mix_prefill,mix_fixed,mix_linear,iql_prefill"
..\.venv\Scripts\python alpha_timing.py --logs $HOME\Downloads\csv_bundle\logs --out $HOME\Downloads\figs --groups baseline,fixalpha_03,tent12,tent12i
```
`metrics.csv`의 `min_in_window`는 seed별 원시 최소(평균곡선 최소보다 낮게 나옴), `auc_window`는 0~100k. 위 표의 공통 격자 값과 정의가 달라 숫자가 조금 다르다. 그림: `success_<axis>.png`, `diagnostics_<axis>.png`(ent_coef·mu·포화·log_std·qw·ratio 3×3).

### 6.3 판정
- **square_tent12** vs square_baseline(최저 0.21, 127k 0.47) / prefill(0.62): 최저 ≥ 0.33이고 127k ≥ 0.47이면 "목표 엔트로피 규칙은 과제 전이". `diagnostics_square.png`에서 α가 0.5~1로 올라가고 `logp_mean`이 −12 근처를 유지하는지 확인. 안 되면 Square 패널은 "고정도 적응형도 전이 안 됨, Q 스케일 5배 차이가 원인 후보"로 정직하게.
- **can_tent12i** vs fixalpha_03(최저 0.47, 100k 0.79): 최저 ≥ 0.45, 100k ≥ 0.75면 "적응형 완성". 아니면 포스터는 고정 0.3 + "적응형은 과도기 문제" 한 줄.
- 세 run 중 하나가 `done=0`이면 같은 launch 명령 재실행(체크포인트 25k마다, resume 기본).

### 6.4 크레딧이 남으면 (약 35, 선택)
`can_fixalpha_05_s{1,2,3}`(고정 0.5, 100k, 3개 3시간 ≈ 27): 스윕 U자의 오른쪽 어깨를 채운다. 필수 아님.

### 6.5 keepalive (run 띄운 뒤 필수. nohup은 Colab 눈에 활동이 아니라 유휴로 죽는다)
자동 반납 버전:
```python
import subprocess, time
while True:
    out = subprocess.run("ps aux | grep '[t]rain_dsrl.py' | grep -o 'exp_id=[a-z_0-9]*' | sed 's/exp_id=//' | tr '\n' ' '",
                         shell=True, capture_output=True, text=True).stdout.strip()
    ram = subprocess.run("free -g | awk 'NR==2{print $3\"/\"$2}'", shell=True, capture_output=True, text=True).stdout.strip()
    print(time.strftime('%H:%M'), 'ram', ram, '|', out or '(none running)', flush=True)
    if not out:
        break
    time.sleep(600)
print("all runs finished, releasing this VM", flush=True)
time.sleep(60)
from google.colab import runtime
runtime.unassign()
```
붙들어 두는 버전은 `if not out: break` 이하를 빼고 `time.sleep(600)`만. 시각은 UTC(KST −9).

## 7. 포스터 (6 패널, Can 중심)

1. **문제**: DSRL O2O에서 baseline이 0.5 → 0.24(39k) → 84k에야 기준선 0.405 회복. `success_critic.png`.
2. **정체**: α와 `logp_mean` 곡선을 성공률에 겹침. auto-α가 엔트로피를 17 → 0으로 무너뜨리는 15k 창이 dip. "actor가 부정확한 Q_W의 한 점으로 몰리는 순간". Q 스케일 ×0.25~×2로 시점이 안 움직임(기각, 한 줄).
3. **critic 초기화(축 A)**: dip 얕게, 최종 무영향. actor 로드는 step 0 붕괴 후 최종 최고.
4. **데모 리플레이(축 B)**: 회복 47k, 최종 0.9. 명시적 linear는 자연 감쇠보다 못함. 두 축 안 쌓임. (RLPD 대칭 샘플링 = mix_fixed 0.5, Cal-QL·REDQ 인용)
5. **α 스윕 U자 + 고정 0.3(dip 없음, 100k 0.79, 5 seed) + 적응형(목표 엔트로피)**. 문구: "처방은 숫자가 아니라 엔트로피 유지. 0.3은 Can 5점 스윕의 최적이며 Square로는 전이 안 됨".
6. **Square**: 예측대로 42k 바닥. 고정 0.3은 후반 붕괴(Q 스케일 5배 → auto-α가 이미 0.3). 적응형 결과로 마무리(오늘 오후).

## 8. 표현 주의
- α는 **무작위성**("고집" 아님). 높으면 퍼지고 낮으면 한 점.
- "α=0.3이 답"이라 쓰지 않는다. Square에서 틀렸다. "엔트로피 붕괴가 dip, 붕괴를 막으면 dip이 없다"까지.
- Q 스케일 가설·게이트는 기각됐다고 명시. 숨기지 않는다.
- 평가 노이즈 ±0.1. 개별 seed 한 점으로 말하지 않는다. 3~5 seed 평균±SE.
- alr_half는 "똑같이 추락"이 아니라 dip이 길고 100k 최악 → "α를 오래 붙드는 건 손해"의 근거였으나, 고정 0.3이 최고인 것과 모순처럼 보인다. 차이는 alr_half의 α가 결국 0.05까지 내려가 엔트로피가 0으로 무너진다는 점. 붙드는 시간이 아니라 **어디까지 내려가느냐**가 문제.

## 9. 운영 함정
- 노트북은 GitHub에서 열기(`msp0617/dsrl`, `o2o`, `colab/dsrl_colab_run_v3.ipynb`), 런타임 G4. Drive의 예전 사본은 torch 2.4.0(CUDA 오류). Colab에서 GitHub로 저장 금지.
- 새 VM: 셀 0 → 1 → 2 → 3 → 5b(캐시 복원) → 6 → 7 → 7b → 8 → 9(약 10분). Square run은 그 뒤 π_dp 배치 셀(노트북 6 아래, `=== SQUARE READY ===`) 필수.
- Hydra override는 config에 있는 키만. `--config-path`는 스크립트 위치 기준(`scripts/eval_base_policy.py`에는 안 붙임).
- 처리량: `python colab/throughput.py $PROJ/logs/<exp> --target 150000`.
- 영상: `python scripts/render_episode.py +policy=pi_dp|<exp_id> +episodes=2 +out=$PROJ/videos`.

## 10. 채팅에 붙이면 좋은 것
1. 6.1 완료 확인 출력, 6.2의 `metrics.csv` 해당 행과 `alpha_timing` 출력, 그림은 설명으로.
2. `ps aux | grep "[t]rain_dsrl.py" | grep -o "exp_id=[a-z_0-9]*"` 결과와 남은 크레딧.
