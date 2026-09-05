# 인수인계 (claude.ai 채팅용) — DSRL offline-to-online dip 연구, 2026-09-06 새벽 기준

이 문서는 저장소를 못 보는 채팅 Claude가 **판단·분석·Colab 셀 작성**을 이어가기 위한 것이다.
코드 수정·푸시가 필요하면 Claude Code(로컬 `C:\Users\msp17\dsrl`, 브랜치 `o2o`, 원격 `msp0617/dsrl`)에서 한다. 자세한 기록은 저장소의 `HANDOFF.md`(0~18절).

**마감**: 포스터 2026-09-09(수) 제출, 작성은 9/7(월)·9/8(화). 오늘 9/6(일)은 마지막 실험일.
**제약**: 크레딧이 병목(22:00 추정 약 140). VM 하나 시간당 9. 사용자는 Colab 셀을 실행하고 출력을 붙여 준다.

---

## 1. 연구 한 줄과 설정

논문 DSRL(Wagenmaker et al. 2025, arXiv:2506.15799)의 offline-to-online 학습에서 처음 수만 step 동안 성공률이 사전학습 정책보다 **떨어지는 dip**의 정체를 robomimic **Can**(보조로 Square)에서 밝히고, 줄이는 방법을 비교한다.

- DSRL-NA: 고정된 diffusion policy π_dp, 그 입력 노이즈를 고르는 SAC actor π_W(w∈[−1,1]^28, tanh), critic Q_A(행동 공간)와 Q_W(노이즈 공간, Q_A에서 증류).
- 초기 rollout 24,016 env step(π_dp + N(0,I) 노이즈) 뒤 학습 시작. env step 1당 grad step 1.25. 평가는 100 에피소드, 5k 격자(오늘 run은 2.5k), 100k 이후 25k 격자.
- SAC auto-α: 1.0에서 시작, target entropy 0, log α가 Adam step마다 lr(3e-4)만큼 감소 → **학습 시작 후 약 6k env step에 α<0.1**(baseline 32k 근처).
- π_dp 기준선(N(0,I) 노이즈, seed당 500 에피소드): **0.405 ± 0.014**. 무작위 actor를 붙인 step-0 값(0.5~0.7)은 이보다 높다(tanh로 좁혀진 w가 유리).
- 처리량: G4 VM(vCPU 48, RAM 176GB)에 run 9개까지(run당 RAM 17GB). 9개 동시면 run당 약 7 env step/s → 100k에 4~5시간.

## 2. 지금까지 결과 (전부 CSV 기반, 3~5 seed 평균)

**축 A(critic 초기화)·축 B(리플레이의 데모 비율), Can, 150k~300k**
| 조건 | n | step 0 | 최저(0~100k) | 회복 | AUC 0~100k | 최종 |
|---|---|---|---|---|---|---|
| baseline | 5 | 0.50 | 0.14 (평균곡선 0.24 @35k) | 84k | 0.43 | 0.64~0.75 |
| iql (critic 사전학습) | 5 | 0.57 | 0.24 (평균곡선 0.45) | 73k | 0.50 | 0.65~0.75 |
| warmupc (critic만 로드) | 3 | 0.50 | 0.32 | 42k | 0.48 | 0.58 @129k |
| warmup (actor까지 로드) | 3 | **0.02** | 0.29 | 61k | 0.44 | **0.80~0.82** |
| fixalpha (α=0.01 고정) | 3 | 0.55 | **0.003** | 없음 | 0.18 | 0.23~0.30 |
| mix_prefill (논문 기본, 데모가 리플레이에 있음, 비율 0.91→0.43 자연 감쇠) | 3 | 0.52 | 0.08 (35k 한 점) | 47k | **0.67** | **0.89** |
| mix_fixed 0.5 (RLPD식 대칭 샘플링) | 3 | 0.53 | 0.29 | 44k | 0.66 | 0.85 |
| mix_linear 0.8→0.1 | 3 | — | **0.02** @35k | 51k | 0.62 | 0.81 |
| iql_prefill (두 축 교차) | 3 | — | 0.16 | 52k | 0.56 | 0.84 |

**α 개입, Can, 100~150k**
| 조건 | α<0.1 통과 | 첫 하락(기준선 아래) | 바닥 | Q_W 초기 최대 | 100k 성공률 |
|---|---|---|---|---|---|
| alr_double (α lr 6e-4) | 27.6k | **29.0k** (3/3 seed 앞당김) | 31.6k | 55 | 0.53 |
| baseline (3e-4) | 32k | 34.0k | 41k | 110 | ~0.6 |
| alr_half (1.5e-4) | 40~42k | 34.1k (안 밀림) | 45k | 175 | **0.41** (최악) |

**고정 α 스윕, Can, 129k 성공률 (9/5 밤 완료, 상세 분석 전)**: 0.03 → 0.41, 0.1 → 0.63, **0.3 → 0.72**, 1.0 → 0.15. 0.01은 위 표(붕괴).

**Square(보조 과제, rollout 32,016, 150k 예산)**: baseline 바닥 **42k**에서 0.19(예측 적중: 학습 시작 후 1만 step), iql 최저 0.29(평균곡선 0.37, dip 거의 없음), 127k에서 둘 다 0.48~0.49. **square_mix_prefill** 127k: 0.59 / 0.61 / 0.67(평균 0.62, Can과 같은 방향).

## 3. 해석 (포스터 문장의 뼈대)

1. **dip의 정체**: 모든 조건에서 바닥은 학습 시작 후 약 1만 step(Can 35k, Square 42k). α가 1 → 0.1로 떨어지며 actor가 Q_W의 argmax로 **한 번 이동**하는 순간이다(actor 진단 mu 0.3→0.8, 포화 12→22%, 이후 평평). 그때 Q_W가 부정확하면 해가 된다(H1 critic 부정확이 맞되 **α 감쇠가 방아쇠**).
2. **엔트로피는 보호막**: α=0.01 고정이면 29k에 즉시 붕괴(0.02)하고 회복 없음. fixalpha와 warmupc는 actor 통계가 같은데 결과가 정반대 → 포화 자체가 아니라 "포화된 w가 맞는 Q_W의 argmax인가"가 원인.
3. **축 A**: iql·warmupc는 dip을 얕고 짧게 하지만 최종은 무영향. actor까지 로드하면 step 0 붕괴 후 최종 최고.
4. **축 B가 훨씬 큰 지렛대**: 데모가 리플레이에 있으면 35k 급락은 못 막지만 회복이 10배 빠르고 최종 0.9. 명시적 linear 스케줄은 논문 기본(자연 감쇠)보다 못하다. 두 축은 쌓이지 않는다("데모가 리플레이에 있으면 critic 사전학습은 추가 이득이 없다").
5. **α 감쇠율의 비대칭**: 빠르게 하면 dip이 5k 앞당겨지지만(3/3), 느리게 해도 첫 하락은 안 밀리고 dip만 길어지며 100k 성능 최악. α 단독은 스위치가 아니다. α를 붙들면 actor가 넓게 퍼지고 → 넓은 분포로 학습된 Q_W가 커지고(alr_half 175 > baseline 110 > double 55) → 커진 Q_W가 α를 이긴다. **α를 늦추는 개입은 스스로를 상쇄한다.**
6. **현재 가설**: 스위치는 actor에 걸리는 두 기울기의 균형 `ratio_ge_gq = ‖∂(α log π)/∂u‖ / ‖∂(−Q_W)/∂u‖`(u는 tanh 이전 샘플). 값의 비가 아니라 기울기의 비를 쓰는 이유: 초기 `qw_mean` +55~+175는 보상이 전부 ≤0인데도 양수이므로 critic 타깃의 엔트로피 보너스가 쌓인 **오프셋**이고, 오프셋은 w에 거의 무관해 argmax를 옮기지 않는다.

## 4. 구현돼 있는 것 (Hydra override로 켬, 기본값이면 상류와 동일)

| 키 | 뜻 |
|---|---|
| `train.reward_scale` c | critic 타깃의 r만 c배. 로깅된 성공률·보상은 그대로, `qw_mean`·`q_start`만 c배 |
| `train.critic_entropy_scale` β | 타깃의 α·log π' 보너스 배율. 0이면 hard backup(오프셋 인플레이션 제거) |
| `train.ent_coef_lr` | α 옵티마이저 lr 분리(−1이면 공유 3e-4). alr_double=6e-4, alr_half=1.5e-4 |
| `train.ent_coef` | 양수면 고정 α |
| `variant` | baseline / iql / warmup / warmupc |
| `offline_mix.mode` | none / prefill / fixed(p0) / linear(p0→p1 until_env) |
| `gate.enabled`, `gate.signal`(ratio/clock), `gate.actuator`(hard_backup/alpha_hold), `gate.tau`, `gate.K`, `gate.clock_calls`, `gate.alpha_hi` | 게이트. ratio 신호: `ratio_ge_gq`<τ가 K번 연속(한 번 = train() 호출 ≈ 16 env step)이면 열림. clock: `clock_calls`번 뒤 무조건 열림(대조군). 닫힌 동안 hard_backup은 β=0, alpha_hold는 α=alpha_hi 고정. 한 번 열리면 안 닫힘. 상태는 resume에서 이어짐 |

`train_log.csv` 열(1,200 env step마다 한 행): `env_steps, success_rate, ent_coef, offline_p, w_absmean, w_std, w_frac_sat, mu_absmean, log_std_mean, logp_mean, qw_mean, qw_absmean, gq_norm, ge_norm, ratio_ge_gq, gate_open, gate_open_call`.
**`qw_absmean`·`gq_norm`·`ge_norm`·`ratio_ge_gq`는 9/5 밤 9 run(rs_025, rs_2, hardq)부터만 있다.** 그 전 run에는 없다.
`eval_log.csv`: `env_steps, success_rate, avg_reward, mc_return, q_start`(Q_W가 시작 상태에서 예측한 값, c배 주의).

스모크는 전부 통과(scale, gate, resume). 테스트 35개 통과. 코드 최신 커밋 93f750d.

## 5. 지금 상태 (9/6 새벽)

- **VM 2에서 9 run 진행 중** (22:00 KST 시작, 코드 84dc2af, 100k, 평가 2.5k 격자, 새벽 2~3시 완료 예상, 끝나면 keepalive가 VM을 스스로 반납):
  `can_rs_025_s{1,2,3}`(reward_scale 0.25), `can_rs_2_s{1,2,3}`(2.0), `can_hardq_s{1,2,3}`(critic_entropy_scale 0). rs_05는 크레딧 때문에 뺐다.
  step-0 성공률은 조건과 무관하게 seed별로 같고(s1 0.71, s2 0.55, s3 0.37) 로깅된 보상은 −240 그대로 → reward_scale이 타깃에만 들어감을 확인.
- VM 1은 21:00 반납됨. 오후 15 run(`square_mix_prefill_s{1,2,3}`, `can_fixalpha_{01,03,003,1}_s{1,2,3}`) 전부 `[done]`, CSV는 아직 로컬로 안 받음.
- 결과는 전부 Drive `dsrl_project/logs/<exp_id>/{eval_log.csv,train_log.csv}`와 `<exp_id>.out`에 있다. VM이 바뀌어도 그대로.

**예측(가설이 맞다면)**: rs_025는 첫 하락이 baseline(34k)보다 **뒤**, rs_2는 **앞**. hardq는 오프셋이 없어 dip이 얕거나 늦음.

## 6. 오늘(9/6) 순서

### 6.1 새 VM 띄우기 (약 10분, 2~3 크레딧)
Colab에서 노트북을 **GitHub에서 열기**(파일 → 노트 열기 → GitHub → `msp0617/dsrl`, 브랜치 `o2o`, `colab/dsrl_colab_run_v3.ipynb`). 런타임 **G4**. 셀 순서 **0 → 1 → 2 → 3 → 5b(환경 캐시 복원) → 6 → 7 → 7b(run_bash 헬퍼, 필수) → 8 → 9**. 검증 셀에 `torch 2.7.1+cu128`, `matmul ok True`.
Drive의 예전 노트북 사본은 쓰지 말 것(torch 2.4.0 → CUDA 오류). Colab에서 GitHub로 저장하지 말 것(푸시 충돌).

### 6.2 완료 확인 + CSV 묶음
```python
run_bash(r'''
PROJ=/content/drive/MyDrive/dsrl_project
for E in can_rs_025_s1 can_rs_025_s2 can_rs_025_s3 can_rs_2_s1 can_rs_2_s2 can_rs_2_s3 can_hardq_s1 can_hardq_s2 can_hardq_s3; do
  printf "%-16s done=%s  last_eval=%s\n" $E "$(grep -c '\[done\]' $PROJ/logs/$E.out)" "$(tail -n 1 $PROJ/logs/$E/eval_log.csv | cut -d, -f2,5)"
done
cd $PROJ && rm -f csv_bundle.zip && zip -qr csv_bundle.zip logs -i "logs/*/eval_log.csv" "logs/*/train_log.csv" "logs/*.csv" && ls -la csv_bundle.zip
''')
```
9개 `done=1`, last_eval 100000이면 합격. Drive에서 `csv_bundle.zip` 내려받기.
`done=0`인 run이 있으면 `tail -n 5 $PROJ/logs/<exp>.out`을 보고, 체크포인트에서 이어가려면 같은 launch 명령을 다시 실행(`resume` 기본 True, 25k마다 체크포인트).

### 6.3 로컬 분석 (Windows PowerShell, 저장소 폴더)
```powershell
cd C:\Users\msp17\dsrl
git pull origin o2o
Expand-Archive -Force $HOME\Downloads\csv_bundle.zip $HOME\Downloads\csv_bundle
cd scripts
..\.venv\Scripts\python alpha_timing.py --logs $HOME\Downloads\csv_bundle\logs --out $HOME\Downloads\figs --groups alr_double,baseline,alr_half,rs_025,rs_2,hardq
..\.venv\Scripts\python plot_results.py --logs $HOME\Downloads\csv_bundle\logs --out $HOME\Downloads\figs --axes "scale=baseline,rs_025,rs_2,hardq;sweep=baseline,fixalpha,fixalpha_003,fixalpha_01,fixalpha_03,fixalpha_1;square=square_baseline,square_iql,square_mix_prefill"
```
출력 전체를 채팅에 붙인다. `alpha_timing.py`가 내는 것:
- run별: `alpha_below_{0.3,0.1,0.03}`(α가 처음 그 아래로 간 env step), `ratio_below_{3,1,0.3}`(ratio_ge_gq 3행 이동중앙값이 처음 그 아래로 간 env step), `first_below_ref`(3점 평활 성공률이 처음 0.405 아래), `bottom_at`, `back_above_ref`, `ent_coef_at_first_below`, `ratio_ge_gq_at_first_below`, `qw_absmean_at_first_below`.
- 그룹별 평균·표준편차 표.
- 마지막 블록 "at the first evaluation below pi_dp": 각 양의 중앙값·범위·**spread**(run들에 걸친 log10 표준편차. 0.1이면 1.26배, 0.5면 3배 흔들림). **spread가 작은 양이 첫 하락 순간에 일정한 양 = 스위치.**
- 그림 `alpha_timing.png`, `ratio_timing.png`(ratio<1 통과 시점 vs 첫 하락, 그룹별 첫 하락 시 ratio·α), `success_scale.png`, `diagnostics_scale.png`, `metrics.csv`.

### 6.4 판정 규칙 (Q 스케일 실험)
1. **dip이 움직였나**: `first_below_ref` 평균이 rs_025 > baseline(34k) > rs_2이면 Q 스케일이 dip 시점을 움직인다. 안 움직이면 Q_W 크기는 결과이지 원인이 아니고, 두 번째 시계는 critic 학습 진행(데이터 양)이다. 그것도 결과로 보고한다.
2. **무엇이 스위치인가**: `ratio_ge_gq_at_first_below`의 spread가 `ent_coef_at_first_below`의 spread보다 작으면 비가 스위치 → **τ = ratio의 중앙값**(rs_025·rs_2·hardq 9점; baseline 스케일의 ratio는 없으므로 rs_025와 rs_2 사이로 본다). 반대면 α 시계가 더 좋은 신호 → 게이트 신호를 재검토(clock으로만 가거나 τ 대신 α 기준).
3. **hardq**: dip이 baseline보다 얕거나 늦으면 "초기 hard backup" 자체가 처방이고 게이트 작동기는 `hard_backup`으로 확정. hardq가 오히려 나쁘면 작동기는 `alpha_hold`(alpha_hi=0.3, 스윕 최적)로.
4. `plot_results` 표의 rs 조건 `q_start`는 c배이므로 비교 시 c로 나눈다.

### 6.5 게이트 signal 5 run (τ 확정 직후, 5개라 run당 10 env step/s 이상, 약 3~4시간, 약 35 크레딧)
```bash
%%bash
source /usr/local/etc/profile.d/conda.sh && conda activate dsrl
source /content/env.sh
cd /content/dsrl
git pull origin o2o | tail -n 1
PROJ=/content/drive/MyDrive/dsrl_project
CFG="--config-path=cfg/robomimic --config-name=dsrl_can.yaml"
COMMON="log_dir=$PROJ/logs variant=baseline train.total_env_steps=100000 eval_schedule.every_env_early=2500"
launch () { EXP=$1; shift; nohup python train_dsrl.py $CFG exp_id=$EXP "$@" $COMMON > $PROJ/logs/$EXP.out 2>&1 & echo "started $EXP (pid $!)"; }
TAU=1.0   # 6.4의 2번으로 정한 값
for S in 1 2 3 4 5; do
  launch can_gate_sig_s$S seed=$S gate.enabled=true gate.signal=ratio gate.actuator=hard_backup gate.tau=$TAU gate.K=50
done
```
작동기를 바꾸면 `gate.actuator=alpha_hold gate.alpha_hi=0.3`. 5분 뒤 확인:
```python
run_bash(r'''
PROJ=/content/drive/MyDrive/dsrl_project
grep -H "\[budget\]\|\[eval\] env_steps=0\|Error" $PROJ/logs/can_gate_sig_s?.out
echo "running: $(ps aux | grep -c '[t]rain_dsrl.py')"; free -g | awk 'NR==2{print "ram", $3"/"$2}'
''')
```
그다음 keepalive(6.8). 도는 중간 확인(게이트가 열렸는지):
```python
run_bash(r'''
PROJ=/content/drive/MyDrive/dsrl_project
for S in 1 2 3 4 5; do
  F=$PROJ/logs/can_gate_sig_s$S/train_log.csv
  [ -f $F ] && echo "s$S $(tail -n 1 $F | cut -d, -f2) gate_open,open_call = $(tail -n 1 $F | awk -F, '{print $(NF-1)","$NF}') ratio = $(tail -n 1 $F | awk -F, '{print $(NF-2)}')"
  tail -n 1 $PROJ/logs/can_gate_sig_s$S/eval_log.csv | cut -d, -f2,5
done
''')
```
(`train_log.csv`의 마지막 두 열이 `gate_open, gate_open_call`, 그 앞이 `ratio_ge_gq`.) 열린 env step ≈ 24,016 + 16 × `gate_open_call`.

### 6.6 게이트 clock 대조군 (signal 5개 완료 뒤)
N* = signal 5 seed의 `gate_open_call` 평균(정수). 크레딧이 모자라면 seed 1~3만.
```bash
# 6.5 셀과 같고 마지막 루프만 교체
NSTAR=1000   # signal run들의 gate_open_call 평균
for S in 1 2 3 4 5; do
  launch can_gate_clk_s$S seed=$S gate.enabled=true gate.signal=clock gate.clock_calls=$NSTAR gate.actuator=hard_backup
done
```

### 6.7 게이트 판정 (기여의 생사)
| 결과 | 뜻 |
|---|---|
| sig > clk (dip 얕고/회복 빠름) | "언제 풀지를 critic이 알려준다" — 능동 조절 성립 |
| sig ≈ clk > baseline | 지연이 도움되지 신호는 무관. 방법은 살지만 '능동'은 죽음. 그대로 보고 |
| sig ≈ baseline | 게이트 무효. 신호·τ·작동기 재검토 |
| sig ≈ hardq | 게이트가 상시 hard backup과 같음 → "초기 hard backup"이 처방 |
| sig ≈ iql | 데모 없이 iql 수준 → 강한 결과 |

### 6.8 keepalive (run 띄운 뒤 반드시. nohup 프로세스는 Colab 눈에 활동이 아니라 유휴로 죽는다)
반납 없는 버전(낮에 붙어 있을 때):
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
```
자동 반납 버전(자리를 뜰 때): 위 코드 끝에 아래 4줄 추가.
```python
print("all runs finished, releasing this VM", flush=True)
time.sleep(60)
from google.colab import runtime
runtime.unassign()
```
다른 셀을 돌리려면 keepalive 정지 → 셀 → keepalive 재실행. 시각은 UTC(KST −9).

### 6.9 크레딧 계획
9/6 아침 약 100 예상(밤 9 run이 약 45 씀). signal 5개 약 35, clock 5개 약 35 → 딱 맞고 여유 없음. 남은 수가 60 아래면 clock은 3 seed. 그림·표 재생성은 로컬이라 무료.

## 7. 포스터 (6 패널 초안, Can 중심, Square는 마지막 패널)

1. **문제**: DSRL O2O에서 baseline 성공률이 step 0 0.5에서 35k에 0.14~0.24로 떨어져 84k에야 π_dp 기준선(0.405)을 회복. 그림 `success_critic.png`.
2. **정체**: α 곡선을 겹치면 모든 조건의 바닥이 α<0.1 직후(학습 시작 후 1만 step). actor 진단은 그때 한 번 이동하고 평평. "actor가 부정확한 Q_W를 믿기 시작하는 순간".
3. **critic 초기화(축 A)**: iql·warmupc는 dip을 얕게(0.45/0.34), 회복 42~73k, 최종 무영향. actor 로드는 step 0 붕괴 후 최종 최고.
4. **데모 리플레이(축 B)**: prefill·fixed는 회복 47k, 최종 0.9. 명시적 linear는 자연 감쇠보다 못함. 두 축은 안 쌓임. (RLPD의 대칭 샘플링 = mix_fixed 0.5, Cal-QL·REDQ 인용)
5. **α 개입**: 고정 0.01 붕괴(엔트로피 = 보호막), 고정 스윕 최적 0.3, 감쇠율 비대칭(빠르면 5k 앞당김, 느려도 안 밀림·성능 최악) → α 단독은 스위치가 아니고 Q 스케일과의 균형(가설). 오늘 Q 스케일·게이트 결과가 나오면 여기 추가.
6. **Square 일반성**: 예측대로 42k 바닥(0.19), iql은 dip 거의 없음, prefill 0.62@127k. "메커니즘은 과제 무관".

## 8. 표현 주의
- α는 "고집"이 아니라 **무작위성**. 높으면 actor가 퍼지고 낮으면 한 점으로 몰린다.
- Q_W 110→175는 1.6배. "폭증"·"괴물" 금지.
- "비가 스위치"는 오늘 결과 전까지 **가설**. "관측된 비대칭을 Q 스케일이 설명한다"까지만.
- alr_half는 "똑같이 추락"이 아니다. 첫 하락은 같고 바닥은 뒤(45k), **dip이 더 길고 100k 성능 최악** → "α를 오래 붙드는 건 손해"의 근거.
- 평가 노이즈 ±0.1. 개별 seed 한 점으로 말하지 않는다. 3~5 seed 평균±SE.

## 9. 운영 함정
- VM 2는 노트북 **사본**(Drive에 사본 저장)으로 연다. 노트북 하나에 런타임 하나. 두 VM이 같은 exp_id를 돌리면 체크포인트가 섞인다.
- `--config-path`는 스크립트 위치 기준 → `scripts/eval_base_policy.py`에는 붙이지 않는다.
- Hydra override는 config에 있는 키만 된다(없으면 "Could not override"). Square config는 Can과 같은 키로 재작성돼 있다.
- 처리량 확인: `python colab/throughput.py $PROJ/logs/<exp> --target 100000`(마지막 resume 이후 행만 잰다).
- 영상: `python scripts/render_episode.py +policy=pi_dp|<exp_id> +episodes=2 +out=$PROJ/videos`.

## 10. 채팅에 붙이면 좋은 것
1. 6.2 완료 확인 출력, 6.3 분석 출력 전체(표 + spread 블록), 그림은 설명으로.
2. `ps aux | grep "[t]rain_dsrl.py" | grep -o "exp_id=[a-z_0-9]*"` 결과와 남은 크레딧.
3. 게이트 run은 6.5 중간 확인 출력(`gate_open`, `gate_open_call`, ratio).
