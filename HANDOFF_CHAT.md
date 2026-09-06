# 인수인계 (claude.ai 채팅용) — DSRL offline-to-online dip 연구, 2026-09-06 14:00 기준

저장소를 못 보는 채팅 Claude가 **판단·분석·Colab 셀 작성**을 이어가기 위한 문서.
코드 수정·푸시는 Claude Code(로컬 `C:\Users\msp17\dsrl`, 브랜치 `o2o`, 원격 `msp0617/dsrl`)에서. 전체 기록은 저장소 `HANDOFF.md`(0~19절, 오늘 결과는 19절).

**마감**: 포스터 2026-09-09(수) 제출, 작성은 9/7(월)·9/8(화). 실험은 오늘 16:30에 끝나는 5 run이 마지막.
**제약**: 크레딧 약 58(13:30) → 5 run 뒤 약 28. VM 하나 시간당 9. 사용자는 Colab 셀을 실행하고 출력을 붙여 준다.

---

## 1. 연구 한 줄과 설정

논문 DSRL(Wagenmaker et al. 2025, arXiv:2506.15799)의 offline-to-online 학습에서 처음 수만 step 동안 성공률이 사전학습 정책보다 **떨어지는 dip**의 정체를 robomimic **Can**(보조 Square)에서 밝히고 처방을 비교한다.

- DSRL-NA: 고정 diffusion policy π_dp, 그 입력 노이즈를 고르는 SAC actor π_W(w∈[−1,1]^28, tanh), critic Q_A와 Q_W(노이즈 공간, Q_A에서 증류).
- 초기 rollout 24,016 env step(Square 32,016) 뒤 학습. env step 1당 grad step 1.25. 평가 100 에피소드, 5k 격자(일부 run 2.5k), 100k 이후 25k.
- SAC auto-α: 1.0 시작, **목표 엔트로피 0**(논문 설정), log α가 Adam step마다 lr(3e-4)만큼 감소. 정책 엔트로피가 목표에 닿으면 α는 그 엔트로피를 유지하는 값에 머문다. 초기 정책 엔트로피(28차원)는 약 17~18.
- π_dp 기준선(N(0,I) 노이즈, 500 에피소드×3): **0.405 ± 0.014**. 무작위 actor를 붙인 step-0 값(0.5~0.7)은 이보다 높다.
- 처리량: G4 VM(vCPU 48, RAM 176GB)에 run 9개까지(run당 RAM 17~19GB). 6개면 run당 약 18 env step/s → 150k에 2시간 반.

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

### 2.2 Can, α·엔트로피 축 (헤드라인)
| 조건 | n | 최저 | AUC | 100k | 129k |
|---|---|---|---|---|---|
| auto-α, 목표 0 (= baseline) | 5 | 0.24 | 0.40 | 0.48 | 0.53 |
| 고정 0.01 | 3 | 0.00 | 0.18 | 0.23 | 0.30 |
| 고정 0.03 | 3 | 0.00 | 0.24 | 0.37 | 0.41 |
| 고정 0.1 | 3 | 0.14 | 0.51 | 0.63 | 0.62 |
| **고정 0.3** | **5** | **0.47** (기준선 위, dip 없음) | **0.66** | **0.79** | **0.74** |
| 고정 1.0 | 3 | 0.03 | 0.25 | 0.26 | 0.15 |
| tent12 (auto-α, 목표 엔트로피 12, 초기 α 1.0) | 3 | 0.33 | 0.57 | 0.70 | 0.69 |
| **tent12i** (auto-α, 목표 12, 초기 α 0.3) | 3 | **0.46** (dip 없음) | 0.60 | 0.68 | 0.70 |
| alr_double / alr_half (α lr 2배 / 절반) | 3/3 | 0.28 / 0.27 | 0.48 / 0.39 | 0.53 / 0.35 | — |
| rs_025 / rs_2 (보상 ×0.25 / ×2) | 3/3 | 0.08 / 0.23 | 0.29 / 0.44 | 0.38 / 0.55 | — |
| hardq (critic 타깃에서 엔트로피 보너스 제거) | 3 | 0.10 | 0.27 | 0.31 | — |

### 2.3 Square (rollout 32k, 150k, 마지막 평가 127k)
| 조건 | 37k / 42k / 47k / 52k | AUC 37~100k | 100k | 127k (seed별) |
|---|---|---|---|---|
| square_baseline | 0.24 / **0.21** / 0.34 / 0.32 | 0.40 | 0.44 | 0.47 (0.35/0.62/0.43) |
| square_iql | 0.40 / 0.50 / 0.42 / 0.41 | 0.44 | 0.48 | 0.56 |
| square_mix_prefill | 0.30 / 0.03 / 0.14 / 0.35 | 0.43 | 0.63 | 0.62 |
| square_fixalpha_03 (고정 0.3) | 0.33 / 0.40 / 0.34 / 0.40 | 0.42 | 0.39 | 0.34 (0.09/0.32/0.60) |
| **square_tent12** (목표 엔트로피 12) | 0.34 / **0.41** / 0.46 / 0.51 | 0.43 | 0.34 | 0.40 (0.34/0.46/0.39) |

## 3. 해석 (포스터 문장의 뼈대)

1. **dip의 정체 = auto-α의 엔트로피 붕괴.** 목표 엔트로피 0인 auto-α는 학습 시작 15k 안에 정책 엔트로피를 17 → 0으로 무너뜨린다(`logp_mean` −17@26k → −6@30k → 0@40k, `mu_absmean` 0.2 → 0.8). 모든 auto-α 조건의 바닥이 그 창(Can 35k, Square 42k)에 있다. Q_W가 아직 부정확한데 actor가 한 점으로 몰리는 순간이다.
2. **엔트로피를 유지하면 dip이 사라진다 — 두 과제 모두.** Can: 고정 0.3(엔트로피 10~14 유지) 또는 목표 엔트로피 12 + 초기 0.3이면 평균곡선이 기준선 아래로 안 간다. Square: 목표 12면 42k dip이 없다(0.41 vs 0.21, 52k 0.51). 반대로 α=0.01·0.03 고정은 즉시 붕괴.
3. **하지만 최종 성능에 맞는 엔트로피 수준은 과제마다 다르다.** Can은 12에서 100k 0.79(baseline 0.48). Square는 12를 지키려면 α를 15까지 올려야 하고 |mu| 0.34에 머물러 127k 0.40(baseline 0.47). 고정 0.3은 Square에서 후반 붕괴(엔트로피 0 아래, seed 1 0.09): Square는 Q 스케일이 5배라 auto-α가 이미 0.2~0.3에 앉아 있어 0.3이 높은 α가 아니다. → **처방은 "숫자"가 아니라 "초기엔 엔트로피를 지키고, 뒤에 과제에 맞게 낮추기"**. 언제 낮출지(센서)가 future work.
4. **α 시계·Q 스케일은 방아쇠가 아니다.** α 감쇠 2배는 dip을 5k 앞당기지만 절반은 안 밀림. 보상 ×0.25~×2로 첫 하락이 격자 한 칸 안, hardq는 더 깊음. 기울기 비 `ratio_ge_gq`가 첫 하락 순간 0.48로 일정했던 건 auto-α 평형의 결과. 이 가설과 게이트(신호 vs 시계)는 **기각**. 한 줄로 보고.
5. **Q 스케일은 회복 속도를 정한다**(rs_2 39k, baseline 64k, rs_025 62k, hardq 81k).
6. **축 A**(iql·warmupc)는 dip을 얕게 하지만 최종 무영향. actor까지 로드하면 step 0 붕괴 후 최종 최고.
7. **축 B**(데모 리플레이)는 회복 10배·최종 0.9로 가장 큰 지렛대. 명시적 linear는 자연 감쇠보다 못함. 두 축은 안 쌓임.
8. **적응형의 남은 문제**: auto-α는 오차 크기와 무관하게 log α를 고정 속도로 움직여서 초기 1.0에서 내려오는 과도기(tent12의 34k dip)가 생긴다. 초기값 0.3으로 두면 해결(tent12i). 목표 엔트로피만 맞게 주면 auto-α 자체가 적응형 제어기다.

## 4. 선행 연구 (9/6 검색, 완전하지 않음. 포스터 문구는 "우리가 아는 한")
- DSRL 원논문: dip 언급 없음, 데모 리플레이 유지, 온도 논의 없음.
- **LP-DS** (arXiv 2606.01151, Simsir & Oguz 2026): DSRL의 noise가 prior 저밀도로 흘러가고 mode collapse → 섭동 크기에 Lagrangian trust region. Can·Square·Lift. SAC α·목표 엔트로피 언급 없음. **같은 현상(노이즈 공간 집중)을 다른 지렛대로 막음. 필수 인용.** 우리는 그 집중을 SAC 온도가 만든다는 것을 보이고 config 한 줄로 해결.
- SAC Flow (2509.25756): robomimic O2O에서 목표 엔트로피 0 그대로, dip 논의 없음 → "이 계열이 공유하는 설정의 비용을 아무도 안 봤다".
- 일반 O2O: WSRL(2412.07762), PORL(2505.16856), OCR(2412.18855)은 분포 이동 + 보수적 Q. Wang·White·White(2505.00913)는 "탐색이 오프라인 정책을 덮어씀" + 성능 추정으로 탐색 점진 허용(스케줄 future work 인용처). Three Regimes(2510.01460).
- 엔트로피 붕괴 개념: TES-SAC(2112.02852, 목표 엔트로피 스케줄), Meta-SAC(2007.01932), AEPO(2510.08141, LLM RFT). "알려진 현상을 DSRL에서 원인으로 확인"이 정확한 표현.

## 5. 구현돼 있는 것 (Hydra override, 기본값이면 상류와 동일)

| 키 | 뜻 |
|---|---|
| `variant` | baseline / iql / warmup / warmupc |
| `offline_mix.mode` | none / prefill / fixed(p0) / linear(p0→p1 until_env) |
| `train.ent_coef` | −1 auto, 양수면 고정, `auto_0.3`이면 auto-α 초기값 0.3(SB3 문법, 5df31e6부터 fingerprint 허용) |
| `train.target_ent` | auto-α 목표 엔트로피(논문 0). tent12 = 12, tent6 = 6 |
| `train.ent_coef_lr` | α 옵티마이저 lr 분리(−1 공유) |
| `train.reward_scale`, `train.critic_entropy_scale` | critic 타깃의 r 배율, 엔트로피 보너스 배율(0 = hard backup) |
| `gate.*` | 게이트(구현돼 있으나 폐기, 기본 off) |

`train_log.csv`(1,200 env step마다): `env_steps, ent_coef, logp_mean, mu_absmean, w_absmean, w_frac_sat, log_std_mean, qw_mean, qw_absmean, gq_norm, ge_norm, ratio_ge_gq, gate_open, gate_open_call, offline_p`. 진단 4열(`qw_absmean`~`ratio_ge_gq`)은 9/5 밤 이후 run에만. `eval_log.csv`: `env_steps, success_rate, avg_reward, mc_return, q_start`.
테스트 36개 통과. 최신 커밋은 `git log` 참조.

## 6. 지금 상태 (9/6 14:00)

- 76 run 완료. CSV 묶음은 로컬 `Downloads\csv_bundle (4).zip`, 그림·`metrics.csv`는 `Downloads\dsrl_figs_0906\`(success_adaptive, success_square, success_sweep, success_scale, diagnostics_*).
- **13:45 시작, 16:30 완료 예정 (G4 1대, 자동 반납 keepalive)**: `square_fixalpha_03_s{4,5}`(n=5로), `square_tent6_s{1,2,3}`(`train.target_ent=6`, 중간 엔트로피). 150k.
- 결과는 Drive `dsrl_project/logs/<exp_id>/{eval_log.csv,train_log.csv}`, `<exp_id>.out`.

## 7. 오늘 저녁 순서

### 7.1 완료 확인 + CSV 묶음 (GPU 불필요: 런타임 유형 CPU → 셀 0(Drive) → 아래)
```bash
%%bash
PROJ=/content/drive/MyDrive/dsrl_project
for E in square_fixalpha_03_s4 square_fixalpha_03_s5 square_tent6_s1 square_tent6_s2 square_tent6_s3; do
  printf "%-22s done=%s  last_eval=%s\n" $E "$(grep -c '\[done\]' $PROJ/logs/$E.out)" "$(tail -n 1 $PROJ/logs/$E/eval_log.csv | cut -d, -f2,5)"
done
cd $PROJ && rm -f csv_bundle.zip && zip -qr csv_bundle.zip logs -i "logs/*/eval_log.csv" "logs/*/train_log.csv" "logs/*.csv" && ls -la csv_bundle.zip
```
5개 `done=1`(127136) → 다운로드. zip 크기가 1.34MB보다 커야 새 파일(같으면 아직 안 만들어진 것).

### 7.2 로컬 분석 (PowerShell, 저장소 폴더)
```powershell
cd C:\Users\msp17\dsrl
git pull origin o2o
Expand-Archive -Force $HOME\Downloads\csv_bundle.zip $HOME\Downloads\csv_bundle
cd scripts
..\.venv\Scripts\python plot_results.py --logs $HOME\Downloads\csv_bundle\logs --out $HOME\Downloads\figs --axes "sweep=baseline,fixalpha,fixalpha_003,fixalpha_01,fixalpha_03,fixalpha_1;adaptive=baseline,iql,mix_prefill,fixalpha_03,tent12,tent12i;square=square_baseline,square_iql,square_mix_prefill,square_fixalpha_03,square_tent12,square_tent6;scale=baseline,rs_025,rs_2,hardq;critic=baseline,warmup,iql,warmupc,fixalpha;mix=baseline,mix_prefill,mix_fixed,mix_linear,iql_prefill"
```
`metrics.csv`의 `min_in_window`는 seed별 원시 최소(평균곡선 최소보다 낮게 나옴), `auc_window`는 0~100k. 위 표의 공통 격자 값과 정의가 달라 숫자가 조금 다르다.

### 7.3 판정
- **square_tent6**: 42k ≥ 0.33(dip 없음)이고 127k ≥ 0.47(baseline)이면 "적정 수준이 다를 뿐 규칙은 전이" → 포스터 6번 패널 마무리 문장. 42k는 좋은데 127k가 낮으면 tent12와 같은 결론(스케줄 필요). 42k도 낮으면 6은 너무 낮은 것.
- **square_fixalpha_03 n=5**: 127k 평균이 baseline 0.47보다 낮은 채 유지되면 "고정 α 전이 실패" 확정. seed 4·5가 0.6 근처면 "seed 편차 큼, 결론 보류"로 정직하게.
- `done=0`인 run은 같은 launch 명령 재실행(체크포인트 25k마다, resume 기본).

### 7.4 keepalive (run 띄운 뒤 필수. nohup은 Colab 눈에 활동이 아니라 유휴로 죽는다)
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
시각은 UTC(KST −9).

## 8. 포스터 (6 패널, Can 중심)

1. **문제**: DSRL O2O에서 baseline이 0.5 → 0.24(39k) → 84k에야 기준선 0.405 회복. `success_critic.png`.
2. **정체**: α와 `logp_mean` 곡선을 성공률에 겹침. auto-α가 엔트로피를 17 → 0으로 무너뜨리는 15k 창이 dip. "actor가 부정확한 Q_W의 한 점으로 몰리는 순간". Q 스케일 ×0.25~×2로 시점이 안 움직임(기각, 한 줄).
3. **critic 초기화(축 A)**: dip 얕게, 최종 무영향. actor 로드는 step 0 붕괴 후 최종 최고.
4. **데모 리플레이(축 B)**: 회복 47k, 최종 0.9. 명시적 linear는 자연 감쇠보다 못함. 두 축 안 쌓임. (RLPD 대칭 샘플링 = mix_fixed 0.5, Cal-QL·REDQ 인용)
5. **α 스윕 U자 + 고정 0.3 곡선(dip 없음, 100k 0.79, 5 seed) + 적응형 tent12i(dip 없음)**. 문구: "처방은 엔트로피 유지. 목표 엔트로피만 맞게 주면 auto-α가 적응형 제어기". `success_sweep.png`, `success_adaptive.png`.
6. **Square**: 예측대로 42k 바닥 재현. 고정 0.3은 후반 붕괴(Q 스케일 5배 → auto-α가 이미 0.3). 목표 12는 dip을 없애지만 후반에 너무 넓음. "초기엔 엔트로피를 지키고 뒤에 과제에 맞게 낮추는 스케줄"이 future work. tent6 결과로 마무리. `success_square.png`. LP-DS 인용.

## 9. 표현 주의
- α는 **무작위성**("고집" 아님). 높으면 퍼지고 낮으면 한 점.
- "α=0.3이 답"이라 쓰지 않는다. Square에서 틀렸다. "엔트로피 붕괴가 dip, 붕괴를 막으면 dip이 없다(두 과제), 최종에 맞는 수준은 과제마다 다르다"까지.
- Q 스케일 가설·게이트는 기각됐다고 명시. 숨기지 않는다.
- 평가 노이즈 ±0.1. 개별 seed 한 점으로 말하지 않는다. 3~5 seed 평균±SE. Square 127k 값은 seed 편차가 커서 특히 조심.
- alr_half(α를 오래 붙듦)가 나빴던 건 결국 0.05까지 내려가 엔트로피가 0으로 무너지기 때문. 붙드는 시간이 아니라 **어디까지 내려가느냐**가 문제. 고정 0.3이 최고인 것과 모순 아님.
- "우리가 아는 한" 없이 "처음"이라고 쓰지 않는다. LP-DS를 반드시 인용.

## 10. 운영 함정
- 노트북은 GitHub에서 열기(`msp0617/dsrl`, `o2o`, `colab/dsrl_colab_run_v3.ipynb`), 런타임 G4. Drive의 예전 사본은 torch 2.4.0(CUDA 오류). Colab에서 GitHub로 저장 금지.
- 새 VM: 셀 0 → 1 → 2 → 3 → 5b(캐시 복원) → 6 → 7 → 7b → 8 → 9(약 10분). Square run은 그 뒤 π_dp 배치 셀(노트북 6 아래, `=== SQUARE READY ===`) 필수.
- Hydra override는 config에 있는 키만. `--config-path`는 스크립트 위치 기준(`scripts/eval_base_policy.py`에는 안 붙임).
- 처리량: `python colab/throughput.py $PROJ/logs/<exp> --target 150000`.
- 영상: `python scripts/render_episode.py +policy=pi_dp|<exp_id> +episodes=2 +out=$PROJ/videos`.

## 11. 채팅에 붙이면 좋은 것
1. 7.1 완료 확인 출력, 7.2의 `metrics.csv` 해당 행, 그림은 설명으로.
2. `ps aux | grep "[t]rain_dsrl.py" | grep -o "exp_id=[a-z_0-9]*"` 결과와 남은 크레딧.
