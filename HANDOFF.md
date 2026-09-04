# DSRL offline-to-online 프로젝트 — 인수인계 (v3, 2026-09-04)

이 문서 하나로 새 세션(Claude Code든 claude.ai 채팅이든)이 작업을 이어갈 수 있게 쓴 것이다.
채팅에는 저장소 접근이 없으므로 **필요한 명령·숫자·판단 기준을 전부 여기에 넣었다.**
새 세션 첫 메시지: **"HANDOFF.md 붙여넣고, 섹션 2 '진행 중'부터 이어서"**.
코드 설명은 [O2O.md](O2O.md), 저장소는 https://github.com/msp0617/dsrl 브랜치 `o2o`.

---

## 0. 한 줄 요약

논문 DSRL(Wagenmaker et al. 2025, arXiv:2506.15799)의 offline-to-online 초기 성능 dip이
**critic 초기화** 때문인지, IQL로 critic을 미리 만들면 줄어드는지 robomimic Can에서 실험한다.
인프라·검토·사전학습은 끝났고, **본 실험 9 run 중 seed 1 세 개가 Colab G4 VM 하나에서 돌고 있다.**
남은 것은 처리량 판단 → seed 2, 3 (6 run) → 분석이다. 크레딧 만료는 약 2026-09-09.

---

## 1. 연구 내용

DSRL은 학습된 diffusion policy π_dp를 고정하고, π_dp에 넣을 초기 노이즈 w를 고르는
작은 정책 π_W를 RL로 학습한다. DSRL-NA는 critic이 둘이다.

- **Q_A(s, a)**: 실제 행동 청크(28차원 = 4스텝 × 7)의 가치. 환경 보상으로 학습. 코드 `model.critic`, `model.critic_target`
- **Q_W(s, w)**: 노이즈의 가치. Q_A를 증류해서 만든다(Algorithm 1 line 5). 코드 `model.critic_noise`
- π_W: `model.actor`. Q_W를 보고 w를 고른다.

**가설**: 시작 시 Q_A, Q_W가 무작위 → π_W가 엉터리 Q_W를 좇아 π_dp가 본 적 없는 w를 고름
→ 성공률이 π_dp 아래로 떨어지는 dip. 오프라인 데모로 critic을 미리 만들면 dip이 줄어드는가.

**세 변형** (config `variant`, 시작 critic 가중치 출처만 다름):

| variant | Q_A, Q_W 시작값 | actor 시작값 | α(엔트로피 계수) 시작값 |
|---|---|---|---|
| baseline | 무작위 | 무작위 | 1.0 |
| warmup | 데모로 DSRL 자신의 업데이트(Algorithm 1) 50k | 같이 학습된 것 로드 | 1.0 (`pretrain.load_ent_coef=True`일 때만 학습된 값 로드) |
| iql | 데모로 IQL 50k → Q_W로 증류 25k | 무작위 | 1.0 |

**논지의 핵심**: warmup의 Q_A 타깃 `r + γ Q̄_A(s', π_dp(s', π_W(s')))`는 무작위 actor에 의존한다.
IQL은 V(s)를 expectile 회귀로 데이터 안 행동들의 Q 상위쪽에 맞추고 타깃을 `r + γV(s')`로 써서
actor가 안 들어간다. Q_W는 오프라인 데이터에 w 라벨이 없어 Q_A → Q_W 증류를 경유한다.

**2026-09-03 검토에서 드러난 추가 메커니즘 (분석 때 반드시 같이 볼 것)**:
DSRL 설정은 α=1.0에서 시작하고 목표 엔트로피 0이다. 28차원 squashed Gaussian의 초기 log π ≈ −18이라
(a) 초반 수천 그래디언트 스텝은 actor 손실 `α·log π − Q_W`에서 엔트로피 항이 지배하고,
(b) critic 타깃에도 `−α·log π ≈ +18/청크`가 더해져 Q 값이 양수로 부풀며(hard Q는 −400~0),
(c) α는 log α가 그래디언트 스텝당 3e-4씩 내려가 학습 시작 후 ~1만 env step이면 0.1 근처가 된다.
실측(A100 처리량 run)에서 α는 1 → 0.11(학습 7천 env step), log π는 −18 → −1로 급격히 좁아졌고,
같은 구간(8,208 env step)에서 성공률이 0.53 → 0.35로 떨어졌다. **α 과도기 자체가 dip 후보다.**
critic 초기화가 dip을 줄이면 "Q_W 크기가 엔트로피 항을 이겨서"일 수도 있다. 이 구분이 분석의 핵심.
필요하면 `train.ent_coef=0.01`(고정 α) baseline을 4번째 조건으로 추가.

---

## 2. 현재 상태 (2026-09-04)

### 끝난 것
- 인프라: resume(2슬롯 체크포인트), CSV 로깅, env-step 단위 예산·평가 스케줄. Colab에서 검증됨.
- 오프라인 데이터: `$PROJ/offline/can_train_offline.npz` (61,856 청크, 12MB). 공개 train.npz와 상태 비트 일치 확인.
- `offline_pretrain.py` 정적 검토 완료(섹션 8). 버그 없음. 교란 2개 수정(커밋 20d414d):
  `pretrain.load_ent_coef`(기본 False), `pretrain.distill_steps=25000`.
- 노트북 설치 셀: GPU compute capability ≥ 12(G4 Blackwell)이면 torch 2.7.1 cu128, 아니면 2.4.0 (커밋 95de9c0).
- **A100 처리량**: 평가 포함 13.2 env step/s, 300k run당 6.3 h (학습 4.5 h + 평가 28회 ~1 h + 기타).
- **사전학습 6개 완료** (2026-09-04, G4): `$PROJ/logs/pretrain/{iql,warmup}_can_s{1,2,3}.pt` (각 228MB) + `_log.csv`.
  로그 수치: iql q_mean ≈ −100, v_mean이 q_mean 살짝 위(expectile 방향 맞음); warmup 끝 α ≈ 0.047, Q_W ≈ −47.
- **dip 관찰됨**: 처리량 run(`tput_can`, seed 0)에서 step 0 성공률 0.530 → 8,208 env step 0.350 (100 에피소드, SE ≈ 5%p).

### 진행 중
- Colab **G4 VM 하나**(RTX PRO 6000 Blackwell, vCPU 48)에서 본학습 seed 1 세 개가 백그라운드로 돌고 있음:
  `can_baseline_s1`(가장 먼저 시작), `can_iql_s1`, `can_warmup_s1`(마지막). 2026-09-04 새벽(KST) 시작.
- **아직 안 한 판단**: 한 VM에 3개를 얹었을 때 각 run 시간. 13번 진행확인 셀의 `h for the whole run`을 세 run에 대해 보고
  각 9 h 이하면 이 방식 유지(크레딧 1/3), 15 h 이상이면 세션을 나눈다.
- 잔여 크레딧: 정확한 값 미확인. 9/3 630 → 사전학습·테스트로 40~50 사용 추정 → **약 580~590**.
  9 run 비용: 한 VM 3개 얹기가 되면 약 300, 따로 돌리면 약 530(G4 9/h). A100(약 11.8/h)은 예산 초과.

### 남은 순서
1. 처리량 판단(위) → seed 2, 3 여섯 run 배치 결정.
2. baseline s1 평가가 4~5회 쌓이면 dip 모양 + `train_log.csv`의 `ent_coef` 곡선을 같이 확인.
3. 9 run 완료. 끝난 run은 `checkpoint/` 지워도 됨(CSV가 결과물).
4. 분석(섹션 7).

---

## 3. 실행 환경 (Colab)

- 노트북: `colab/dsrl_colab_run_v3.ipynb`. **반드시 GitHub에서 열 것**: 파일 → 노트 열기 → GitHub → `msp0617/dsrl`, 브랜치 `o2o`.
  Drive에 남은 예전 사본은 torch 2.4.0을 깔아서 G4에서 `CUDA error: no kernel image is available`로 죽는다.
- 런타임: **G4** (시간당 9 크레딧). vCPU 48, VRAM 96GB. 워크로드가 CPU·커널호출 바운드라(GPU fps 3~4) GPU 급은 무관.
  A100 VM은 vCPU 12, 시간당 약 11.8.
- 경로: 코드 `/content/dsrl`, conda env `dsrl`(py3.10), Drive `$PROJ=/content/drive/MyDrive/dsrl_project`
  - `$PROJ/dppo_log/` 공개 체크포인트, `$PROJ/offline/` 변환 데이터, `$PROJ/robomimic_raw/` hdf5
  - `$PROJ/logs/pretrain/` 사전학습 .pt, `$PROJ/logs/<exp_id>/` run별 `eval_log.csv`, `train_log.csv`, `checkpoint/`
  - `$PROJ/logs/<exp_id>.out` run의 stdout, `$PROJ/logs/pretrain_s{1,2,3}.out` 사전학습 stdout
- 새 VM마다: 노트북 섹션 0~9 실행(약 15분, 2~3 크레딧). 10(스모크), 11(처리량)은 건너뜀. 7b의 `run_bash` 헬퍼 셀은 꼭 실행.
- 섹션 9까지 끝나면 20번 검증 셀 출력에 `torch 2.7.1+cu128`, `compute cap (12, 0)`, `matmul ok True`가 있어야 한다.
- Pro+: 브라우저를 닫아도 런타임은 최대 24h 백그라운드 유지. **"런타임 삭제"는 도는 프로세스를 전부 죽인다.**
  본학습은 체크포인트(25k env step마다)에서 resume되지만, 사전학습은 끝나야 .pt가 써지므로 처음부터 다시다.

---

## 4. 저장소 지도

포크 https://github.com/msp0617/dsrl, 브랜치 **`o2o`**. 상류 ajwagen/dsrl은 `upstream` 리모트.
서브모듈(`dppo`, `stable-baselines3`)은 상류 것 그대로.

| 파일 | 역할 |
|---|---|
| `train_dsrl.py` | 온라인 학습. resume, variant 로드(`load_pretrained_weights`), 예산(env step), 시작 전 검사 |
| `utils.py` | `LoggingCallback`(CSV, env-step 평가 스케줄, resume 카운터), `collect_rollouts`, `load_offline_data`, π_dp 래퍼 |
| `o2o_utils.py` | 우리 추가분 전부: `DSRLResumable`, `ResumeCheckpointCallback`(2슬롯), `build_agent`, `SpacesOnlyEnv`, fingerprint, `load_pretrained_weights(load_ent_coef=False)` |
| `offline_pretrain.py` | 오프라인 사전학습. `pretrain.method=iql|warmup`. 시뮬레이터 불필요 |
| `scripts/make_offline_chunks.py` | robomimic hdf5 → 청크 npz. `--check_against`로 정규화 검증 |
| `scripts/test_*.py` | torch 없이 도는 테스트 (resume 9개, 데이터 6개) |
| `colab/dsrl_colab_run_v3.ipynb` | 실행 노트북 |
| `colab/patch_env.py` | robomimic site-packages 패치 (새 VM마다 1회, 섹션 9) |
| `colab/throughput.py` | `train_log.csv`에서 env step/s와 예상 시간 |
| `cfg/robomimic/dsrl_can.yaml` | 논문 Can 하이퍼파라미터 + 우리 키(`variant`, `exp_id`, `eval_schedule`, `pretrain` 등) |
| `O2O.md` | 코드 변경 설명, 변형 정의, 디스크·평가 비용 |

최근 커밋: 95de9c0 노트북 torch 선택 / 20d414d 검토·교란 수정 / dd3bbb3 이전 인수인계.

---

## 5. 명령 모음 (오늘 실제로 쓴 것)

노트북 셀 번호는 코드 셀 기준. 섹션 제목으로 찾는 게 안전하다.

```bash
# 본학습 띄우기: 섹션 "12b. 본 실험"의 %%bash 셀. VARIANT, SEED만 바꿔 실행. nohup 백그라운드.
#   iql/warmup은 $PROJ/logs/pretrain/${VARIANT}_can_s${SEED}.pt 를 자동으로 pretrain_path로 넘긴다.
#   같은 셀을 같은 VARIANT/SEED로 다시 실행하면 = resume.

# 살아 있는 본학습 확인
!ps aux | grep "[t]rain_dsrl.py" | grep -o "exp_id=[a-z_0-9]*"

# 사전학습 프로세스 수 (seed당 python+bash 래퍼라 2배로 잡힌다. 3개 돌면 6)
!ps aux | grep "[o]ffline_pretrain.py" | wc -l

# 로드·step-0 평가 확인 (띄운 뒤 3~5분)
!grep -h "\[pretrain\]\|\[eval\]\|\[budget\]" /content/drive/MyDrive/dsrl_project/logs/can_*_s1.out
#   기대: iql  -> [pretrain] iql: loaded critic, critic_target, critic_noise from ...
#         warmup -> [pretrain] warmup: loaded critic, critic_target, critic_noise, actor from ...  (log_ent_coef 없어야 함)
#         셋 다 [eval] env_steps=0 success_rate=...

# 진행 확인: 섹션 "13. 진행 확인" 첫 셀. EXP를 바꿔 실행. 마지막 줄 "... h for the whole run"이 처리량 판단 근거.
# 두 번째 셀(pandas/matplotlib)은 평가가 몇 번 쌓인 뒤 성공률 곡선.

# 로그 꼬리
!tail -n 30 /content/drive/MyDrive/dsrl_project/logs/can_baseline_s1.out

# 사전학습 결과 파일
!ls -lh /content/drive/MyDrive/dsrl_project/logs/pretrain/*.pt

# run 하나 죽이기 (pkill -f train_dsrl.py 는 자기 셸까지 죽인다)
!pkill -f "[c]an_iql_s1"
```

사전학습 6개 백그라운드 (필요할 때만. 이미 다 있음):
```bash
%%bash
source /usr/local/etc/profile.d/conda.sh && conda activate dsrl
source /content/env.sh
cd /content/dsrl
PROJ=/content/drive/MyDrive/dsrl_project
mkdir -p $PROJ/logs/pretrain
for SEED in 1 2 3; do
  nohup bash -c "
    for METHOD in iql warmup; do
      python offline_pretrain.py --config-path=cfg/robomimic --config-name=dsrl_can.yaml \
        pretrain.method=\$METHOD seed=$SEED \
        offline_data_path=$PROJ/offline/can_train_offline.npz log_dir=$PROJ/logs
    done" > $PROJ/logs/pretrain_s${SEED}.out 2>&1 &
done
```

예전 노트북 사본으로 설치해 버렸을 때 torch만 교체 (전체 재설치 불필요):
```bash
%%bash
source /usr/local/etc/profile.d/conda.sh && conda activate dsrl
python -m pip install -q "torch==2.7.1" "torchvision==0.22.1" --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; x=torch.randn(256,256,device='cuda'); print(torch.__version__, torch.cuda.get_device_capability(0), (x@x).sum().item()!=0)"
```
pip의 "dppo 0.8.0 requires torch==2.4.0" 경고는 무해.

run 이름 규칙: `can_{variant}_s{seed}`. 결과물은 `$PROJ/logs/<exp_id>/eval_log.csv`(열: wall_time, env_steps, sb3_timesteps,
deterministic, success_rate, avg_reward, episodes)와 `train_log.csv`(actor_loss, critic_loss, noise_critic_loss, ent_coef, ent_coef_loss 등).
**CSV가 결과물이고 가중치는 아니다.**

---

## 6. 다음 할 일과 판단 기준

1. **처리량 판단** (seed 1 세 개 띄운 뒤 30분): 진행확인 셀로 세 run의 `h for the whole run`.
   - 각 ≤ 9 h → 한 VM에 3개 방식 유지. seed 2 세 개는 seed 1이 끝나는 대로 같은 VM에, 또는 사본 노트북으로 VM 하나 더 열어 지금 바로.
     (사본: 파일 → 드라이브에 사본 저장 → 새 런타임 → 섹션 0~9 → 12b) 크레딧은 어디서 돌리든 같고 벽시계만 달라진다.
   - 각 ≥ 15 h → 얹은 이득 없음. 하나를 죽이고 다른 VM으로. (죽인 run은 체크포인트에서 resume 가능)
   - 판단 공식: 세 run의 env step/s 합이 13.2(혼자 돌 때)보다 크면 얹는 게 싸다.
2. **크레딧 확인**. 부족하면 seed 3의 세 run만 `train.total_env_steps=200000` (run당 −2 h). dip은 10만 step 안쪽이라 결론 영향 없음.
3. **baseline s1 초반 곡선 확인**: `eval_log.csv` 성공률이 step 0 값 아래로 내려갔다 회복하면 정상. 동시에 `train_log.csv`의 `ent_coef`가
   1 → 0.1 이하로 가는 시점과 dip 시점을 겹쳐 본다.
4. 9 run 완료 → 분석.

---

## 7. 분석 계획

final 성능만 보면 차이가 안 날 가능성이 높다. **dip 자체를 정량화**:
- 초기 N 평가의 평균 성공률, π_dp(step 0) 대비 regret, 학습 곡선 AUC, dip 깊이·지속 길이
- 3 seed 평균 ± 표준오차. 초반(10만 step까지 5천마다)은 100 에피소드라 점당 SE ≈ 5%p. 평활해서 볼 것. 이후는 2.5만마다 200 에피소드.
- x축은 `env_steps`(초기 rollout 24,016 포함). step 0 평가는 rollout 전에 한 번.
- **α 곡선(`train_log.csv` `ent_coef`)을 dip과 같은 축에** 그린다. dip이 α 과도기(학습 시작 후 ~1~3만 env step)와 겹치는지가 핵심.
- warmup은 actor도 로드하므로 step 0 성공률이 π_dp와 다를 수 있다(정의상). iql·baseline은 step 0가 π_dp 그대로.
- 교란 통제: `load_offline_data=False`(전부). α는 세 변형 모두 1.0에서 시작(2026-09-03 수정).
- 차이가 안 나도 "왜 DSRL은 오프라인 critic 초기화가 덜 필요한가"가 결과. 후보 설명: α 과도기, Q_W 학습분포(w~N(0,I) 무클리핑, ±3) vs actor 도달범위(tanh, ±1) 불일치.
- 여유가 생기면: 고정 α baseline(4번째 조건), 데이터 품질 축(변환 npz에 `quality` 라벨 있음).

---

## 8. `offline_pretrain.py` 정적 검토 결과 (2026-09-03 완료)

**조용히 틀리는 버그는 못 찾았다.** 통과 항목:
- `expectile_loss`: u = Q − V, 가중치 |τ − 1{u<0}|, τ=0.7이면 Q>V 샘플 0.7 → V가 위로. IQL 논문 식과 동일.
- V 타깃은 `critic_target`의 min. Q 타깃 `r + γ(1−d)V(s')`는 `no_grad`로 detach. 보상·γ·τ·lr·손실계수 온라인과 동일.
- 보상/종료: 온라인 청크 보상 = Σ(r−1) 4스텝 = 오프라인. 온라인 done은 300스텝 시간제한뿐(terminal로 저장). 성공 후 보상 0이라 데모 끝 terminal(0 부트스트랩)과 일치.
- `run_distill`: 온라인 `update_noise_critic`과 완전히 같은 경로. w ~ N(0,I) 무클리핑, `scale_action` 항등.
- `load_pretrained_weights`: 네트워크 모양은 obs 23·action 28에만 의존(n_envs 무관). fresh run에서 step-0 평가 직전에만 로드. 이후 재초기화 없음.
- 오프라인 청크 stride 1(겹침): MDP는 청크 단위로 같으므로 편향 없음.

수정한 교란: (1) warmup의 학습된 α 로드 → 옵션화(기본 안 함). (2) iql 증류 20k → 25k(warmup의 50000/20×10과 동일).
참고: `standard_gauss_init` 미사용, `log_std_init` gSDE 전용이라 무시됨 → step-0 actor는 순수 무작위.

---

## 9. 관찰된 수치 (다음 세션이 비교할 기준)

A100 처리량 run (`tput_can`, seed 0, rollout 3,200 + 학습 10,000 env step):
- step 0 성공률 0.530 (100 에피소드), 8,208 env step 0.350
- sb3 fps 3~4 (청크 단위), 평가 100 에피소드 ≈ 100 s, 체크포인트 580MB 3 s
- `ent_coef` 1.0 → 0.643(학습 1,200 env step) → 0.273(3,600) → 0.11(7,200). `ent_coef_loss` −8 → −19 → −1.3
- `actor_loss` −106 ~ −134 → Q_W ≈ +94 (엔트로피 보너스로 부풀어 양수). `critic_loss` 8~16, `noise_critic_loss` 50~110
- `ep_len_mean 75`(=300/4, 조기 종료 없음), `ep_rew_mean` −230 ~ −280

사전학습 (G4, 3개 동시):
- iql 50k ≈ 8분, 증류 25k ≈ 10분, warmup 50k ≈ 25분. seed당 총 ~45분
- iql: `q_mean` −77(11k) → −97(31k), `v_mean`이 `q_mean`보다 1~2 위. `value_loss` 3~11, `critic_loss` 5~45
- warmup(27.5k): `critic_loss` 0.75, `actor_loss` +48(Q_W ≈ −47), `noise_critic_loss` 2.5, `ent_coef` 0.047

---

## 10. 함정 모음

- **Drive의 예전 노트북 사본을 쓰면 G4에서 죽는다** (torch 2.4.0). GitHub에서 열거나 섹션 5의 torch 교체 셀.
- **"런타임 삭제"는 도는 것을 전부 죽인다.** 자러 갈 때 run을 돌려 둘 거면 브라우저만 닫는다(Pro+ 24h).
- 사전학습 .pt는 끝나야 써진다. 중간에 죽으면 `_log.csv`만 남고 처음부터.
- 셀에서 셸 명령은 `!` 앞에 붙여야 한다 (`!nproc`). 없으면 Python NameError.
- `ps | grep offline_pretrain | wc -l`은 bash 래퍼 때문에 2배로 센다.
- `pkill -f train_dsrl.py`는 자기 셸까지 죽인다. `pkill -f "[c]an_iql_s1"`처럼.
- Colab `%%bash`는 끝날 때까지 출력이 안 보인다. 긴 명령은 `run_bash` 헬퍼나 nohup + tail.
- `save_replay_buffer=False`인 run은 resume 거부(설계). 일회성은 `resume=False`.
- 커널 "세션 다시 시작"은 VM 디스크 유지(설치 불필요, Drive 마운트만). "런타임 삭제"는 전부 다시.
- 처리량은 GPU 종류보다 vCPU 수에 달렸다. T4 수치는 예산에 못 쓴다.
- pip `dppo requires torch==2.4.0` 경고는 무해.
- "Is instance: True" 반복 출력은 상류 디버그 프린트. 무해.

---

## 11. 집 데스크톱 (Windows) 세팅 — 완료됨

`C:\Users\msp17\dsrl`, `.venv` 있음(Python 3.14, numpy/gymnasium/pyyaml/h5py). git 사용자 정보는 저장소 로컬로 설정됨.
```powershell
.venv\Scripts\python scripts\test_resume_state.py        # 9 checks passed
.venv\Scripts\python scripts\test_make_offline_chunks.py # 6 checks passed
```
로컬에는 `dppo/log/` 체크포인트가 없어 torch가 필요한 스크립트는 못 돌린다. 코드 수정은 **커밋 + `git push origin o2o`** 해야 Colab에 반영된다.
이미 열린 Colab 세션은 `!cd /content/dsrl && git pull origin o2o`.

---

## 12. 채팅(claude.ai)에서 이어갈 때

저장소를 못 보니 이 문서와 함께 아래를 붙여 주면 바로 판단할 수 있다.
1. 진행확인 셀 출력(세 run의 `h for the whole run`, `eval_log.csv` 꼬리)
2. `!ps aux | grep "[t]rain_dsrl.py" | grep -o "exp_id=[a-z_0-9]*"`
3. 남은 크레딧

코드 수정이 필요해지면 Claude Code(이 폴더)에서 하고 푸시한다. 채팅에서는 판단·분석·명령 작성까지.
