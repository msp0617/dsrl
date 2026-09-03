# DSRL offline-to-online 프로젝트 — 인수인계 (v2, 2026-09-03)

이 문서 하나로 새 Claude Code 세션이 작업을 이어갈 수 있게 쓴 것이다.
새 세션 첫 메시지: **"HANDOFF.md 읽고 이어서 작업 준비해"**.
코드 설명은 [O2O.md](O2O.md)에 있다. 이 문서는 상태·결정·다음 할 일이다.

---

## 0. 한 줄 요약

논문 DSRL(Wagenmaker et al. 2025, arXiv:2506.15799)의 offline-to-online 초기 성능 dip이
**critic 초기화** 때문인지, IQL로 critic을 미리 만들면 줄어드는지 robomimic Can에서 실험한다.
인프라(resume, 로깅, 데이터, 오프라인 사전학습 스크립트)는 완성·검증됐고,
**남은 것은 A100 처리량 측정 → 본 실험 9 run → 분석**이다.

---

## 1. 연구 내용 (요약)

DSRL은 학습된 diffusion policy π_dp를 고정하고, π_dp에 넣을 초기 노이즈 w를 고르는
작은 정책 π_W를 RL로 학습한다. DSRL-NA는 critic이 둘이다.

- **Q_A(s, a)**: 실제 행동 청크(28차원)의 가치. 환경 보상으로 학습. 코드: `model.critic`, `model.critic_target`
- **Q_W(s, w)**: 노이즈의 가치. Q_A를 증류해서 만든다(Algorithm 1 line 5). 코드: `model.critic_noise`
- π_W: `model.actor`. Q_W를 보고 w를 고른다.

**가설**: 시작 시 Q_A, Q_W가 무작위 → π_W가 엉터리 Q_W를 좇아 π_dp가 본 적 없는 w를 고름
→ 성공률이 π_dp 아래로 떨어지는 dip. 오프라인 데모로 critic을 미리 만들면 dip이 줄어드는가.

**세 변형** (config `variant`, 시작 critic 가중치 출처만 다름):

| variant | Q_A, Q_W 시작값 | actor 시작값 |
|---|---|---|
| baseline | 무작위 | 무작위 |
| warmup | 데모로 DSRL 자신의 업데이트(Algorithm 1)를 k번 | 같이 학습된 것 로드 (α는 `pretrain.load_ent_coef=True`일 때만) |
| iql | 데모로 IQL k번 → Q_W로 증류 | 무작위 (옵션 `pretrain.actor_steps`) |

**논지의 핵심**: warmup의 Q_A 타깃은 `r + γ Q̄_A(s', π_dp(s', π_W(s')))`로 **무작위 actor에 의존**한다.
IQL은 V(s)를 expectile regression으로 데이터 안 행동들의 Q 상위쪽에 맞추고 타깃을 `r + γV(s')`로 써서
actor가 아예 안 들어간다. 이게 "오프라인 데이터를 쓰는 것"과 "IQL 방식으로 쓰는 것"의 차이.

이론 파트 주의(원문 HANDOFF v1에서): DSRL-NA는 Q_A를 데이터 행동과 π_dp 생성 행동에만 query하므로
CQL식 "OOD 과대평가 억제" 논지는 약하다. actor 의존 타깃의 불안정성이 더 정확한 논지.
Q_W는 오프라인 데이터에 w 라벨이 없어 직접 IQL 불가 → Q_A → Q_W 증류 경유.

---

## 2. 현재 상태 (2026-09-03)

### Colab에서 실제로 검증된 것
- 설치 (v3 노트북 0~9번), π_dp 로드, robomimic Can 환경 생성
- **체크포인트 저장 + 중단 → 재개**: `[resume] ... at 1200 env steps`, 카운터·엔트로피 계수·평가 스케줄·슬롯 교대 전부 이어짐
- **오프라인 데이터 변환**: robomimic can-mh hdf5(62,756 step, 300 demo)를 공개 정규화 통계로 재구성 →
  공개 `train.npz`와 상태가 **비트 단위 일치** (`max |diff| 0`). 온라인·오프라인 입력 공간 동일 증명 끝.
  결과: `$PROJ/offline/can_train_offline.npz` (61,856 청크, 15,464 슬롯)
- **`offline_pretrain.py`**: iql, warmup 모두 200스텝 테스트 끝까지 실행 (`test_iql.pt`, `test_warmup.pt`, 238MB)

### 아직 안 한 것 (순서대로)
1. **A100 처리량 측정** (노트북 11번) → 예산 확정. 이게 다음 할 일.
2. 전체 규모 사전학습 (`pretrain.steps=50000`, iql·warmup × seed 3개)
3. 본 실험 9 run (3 variant × 3 seed, 300k env step)
4. 분석 (섹션 7)

### 진행 중이던 것
- `offline_pretrain.py` 정적 검토는 2026-09-03 집 데스크톱에서 완료(섹션 8). 버그 없음.
  설계 교란 2개(α 로드 여부, 증류 스텝 수)는 config·코드에 반영했다. 50k 사전학습을 돌려도 된다.

---

## 3. 코드 정찰에서 확인된 사실 (계획을 바꾼 것 포함)

| 항목 | 사실 | 근거 |
|---|---|---|
| timestep 단위 | SB3 `total_timesteps`/`fps`는 청크 단위(vec step × n_envs). `LoggingCallback.total_timesteps`는 env 단위. 논문 x축은 env 단위. 우리 config의 예산·eval·ckpt 주기는 전부 env 단위 | `off_policy_algorithm.py:562`, `utils.py` |
| 초기 rollout | 1501 × 4 envs × 4 chunk = 24,016 env step = 논문 "initial steps 24000". 상류 코드의 4배 과소 카운트 버그는 고침 | `train_dsrl.py` |
| 관측 정규화 | 환경 래퍼(`robomimic_lowdim.py:74`)와 DPPO 데이터 전처리가 같은 식, 같은 `normalization.npz`. VecNormalize 없음 | 변환 스크립트 `--check_against`로 증명 |
| **`action_magnitude`(b_W=1.5)는 DSRL-NA에서 죽은 설정** | `dsrl_sac` 경로(`DiffusionPolicyEnvWrapper`, rollout 클리핑)에서만 쓰임. NA의 action space는 `[-1,1]^28`이라 scale/unscale이 항등. π_W는 tanh라 **w ∈ [-1,1]**. 그런데 Q_W 증류는 w ~ N(0,I)를 **자르지 않고** 뽑음(`dsrl.py:351`) → Q_W 학습 분포 ±3 vs actor 도달 범위 ±1 불일치 | `env_utils.py:199`, `utils.py:309`, `dsrl.py` |
| 공개 `train.npz`에 보상 없음 | diffusion policy 사전학습용. 보상은 robomimic hdf5에서 | `make_offline_chunks.py` |
| 데모 보상 구조 | 보상 1인 스텝 2.4%, 데모의 97% 지점에서 처음 등장(성공하면 곧 종료). 데모 끝을 terminal로 처리 — 온라인에서 성공 후 보상 0(shifted)이 계속되는 것과 일치 | 변환 출력 |
| 오프라인 단계는 시뮬레이터 불필요 | `SpacesOnlyEnv` + `build_agent`로 온라인과 같은 네트워크 생성. torch만 있으면 됨 | `o2o_utils.py` |
| eval 비용 | 1회 = num_evals 에피소드 × 300 step. 200 에피소드면 6만 env step. 초반 구간은 100 에피소드 | `eval_schedule.num_evals_early` |
| 체크포인트 크기 | 본 설정(3×2048)은 슬롯당 640MB(optimizer 상태가 대부분), 2슬롯 1.3GB/run. Drive 5TB라 문제 없음 | O2O.md |
| Hydra | chdir 안 함. `${logdir}`에 빈 `.hydra/` 폴더가 하나씩 생기는 건 무해 | |

---

## 4. 저장소 지도

포크 https://github.com/msp0617/dsrl, 브랜치 **`o2o`**. 상류 ajwagen/dsrl은 `upstream` 리모트.
서브모듈(`dppo`, `stable-baselines3`)은 상류 것 그대로(수정 안 함).

| 파일 | 역할 |
|---|---|
| `train_dsrl.py` | 온라인 학습. resume, variant 로드, 예산(env step), 시작 전 검사 |
| `utils.py` | `LoggingCallback`(CSV, env-step eval 스케줄, resume 카운터), `collect_rollouts`, `load_offline_data`, π_dp 래퍼 |
| `o2o_utils.py` | **우리 추가분 전부**: `DSRLResumable`, `ResumeCheckpointCallback`(2슬롯), `build_agent`(네트워크 모양의 단일 출처), `SpacesOnlyEnv`, fingerprint 검사, `load_pretrained_weights`, RNG 저장/복원 |
| `offline_pretrain.py` | 오프라인 사전학습. `pretrain.method=iql|warmup`. 시뮬레이터 없이 돎 |
| `scripts/make_offline_chunks.py` | robomimic hdf5 → 청크 npz. `--check_against`로 정규화 검증 |
| `scripts/test_*.py` | torch 없이 도는 테스트 (resume 상태 9개, 데이터 6개) |
| `colab/dsrl_colab_run_v3.ipynb` | **실행 노트북**. 설치 → 스모크 → 처리량 → 사전학습 → 본 실험 → 진행 확인 |
| `colab/patch_env.py` | robomimic site-packages 패치 (세션마다 1회) |
| `colab/throughput.py` | `train_log.csv`에서 env step/s와 예상 시간 |
| `cfg/robomimic/dsrl_can.yaml` | 논문 Can 하이퍼파라미터 + 우리 키(`variant`, `exp_id`, `eval_schedule`, `pretrain` 등) |
| `O2O.md` | 코드 변경 설명, 변형 정의, 디스크/평가 비용 |

Colab 쪽 경로: 코드 `/content/dsrl`, conda env `dsrl`(py3.10), Drive `$PROJ=/content/drive/MyDrive/dsrl_project`
(`dppo_log/` 공개 체크포인트, `offline/` 변환 데이터, `robomimic_raw/` hdf5, `logs/<exp_id>/` run별 CSV·체크포인트, `logs/pretrain/` 사전학습 .pt).

---

## 5. 실행 방법

전부 노트북 v3 셀에 있다. 핵심 명령만:

```bash
# 세션마다: 노트북 0~9번 (설치 10~15분). 그 뒤 헬퍼 셀(run_bash) 실행.

# 처리량 (11번): 실제 설정, rollout 3200 + 학습 10000 env step
python train_dsrl.py --config-path=cfg/robomimic --config-name=dsrl_can.yaml \
  exp_id=tput_can seed=0 resume=False log_dir=$PROJ/logs \
  train.init_rollout_steps=200 train.total_env_steps=13200 \
  eval_schedule.every_env_early=5000 ckpt_every_env_steps=5000 save_replay_buffer=False
python colab/throughput.py $PROJ/logs/tput_can --target 300000

# 사전학습 (12a): 시드별 1회, 온라인 run 3개가 재사용
python offline_pretrain.py --config-path=cfg/robomimic --config-name=dsrl_can.yaml \
  pretrain.method=iql seed=1 offline_data_path=$PROJ/offline/can_train_offline.npz log_dir=$PROJ/logs
# → $PROJ/logs/pretrain/iql_can_s1.pt

# 본 실험 (12b): 세션당 1 run, 백그라운드. 세션 죽으면 같은 명령 재실행 = resume
python train_dsrl.py --config-path=cfg/robomimic --config-name=dsrl_can.yaml \
  exp_id=can_iql_s1 seed=1 variant=iql pretrain_path=$PROJ/logs/pretrain/iql_can_s1.pt log_dir=$PROJ/logs
```

run 이름 규칙: `can_{variant}_s{seed}`. 결과는 `$PROJ/logs/<exp_id>/eval_log.csv`, `train_log.csv`.
**CSV가 결과물이고 가중치는 아니다.** 끝난 run의 `checkpoint/`는 지워도 된다.

---

## 6. 다음 할 일 (순서와 합격 기준)

1. **A100 처리량**: `throughput.py` 출력의 `env steps/s`. 300k run 한 개 시간 × 9 ÷ 병렬 세션 수가
   크레딧 만료(약 2026-09-09) 안에 들어와야 한다. 안 들어오면 `total_env_steps` 절단(초기 dip은 20만 안쪽) 또는 시드 2개.
2. **50k 사전학습** iql·warmup × seed {1,2,3}. GPU 아무거나. 30분/개.
3. **baseline seed 1 먼저** 띄워서 초기 dip이 실제로 보이는지 확인. 논문 Fig. 4의 Can 곡선처럼
   step 0 성공률 아래로 떨어졌다 회복하면 실험이 의미 있다. **dip이 안 보이면 멈추고 재설계**
   (예: b_W 대신 Q_W/actor 범위 불일치, utd, 초기 rollout 크기가 원인 후보).
4. 나머지 8 run. 세션 병렬로.
5. 분석 (섹션 7).

---

## 7. 분석 계획

final 성능만 보면 차이가 안 날 가능성이 높다(논문에서 offline 데이터 효과가 이미 큼). **dip 자체를 정량화**:
- 초기 N 에피소드 평균 성공률, π_dp(step 0) 대비 regret, 학습 곡선 AUC, dip 깊이·지속 길이
- 3 시드 평균 ± 표준오차. 초반 구간은 에피소드 100개라 점당 SE ≈ 5%p. 점을 평활해서 볼 것
- x축은 `env_steps` (CSV 열). 초기 rollout 24,016이 포함된 값

교란 통제: `load_offline_data`는 variant와 독립. 먼저 **False**로 돌려 critic 초기화 효과만 분리.
warmup은 actor·엔트로피 계수도 로드하고 iql은 안 한다(정의상). 필요하면 `pretrain.actor_steps>0`으로 iql에도 actor 사전학습을 붙인 네 번째 변형 가능.

차이가 안 나도 "왜 DSRL은 오프라인 critic 초기화가 덜 필요한가"가 결과다.
여유가 생기면 데이터 품질 축: 변환 npz에 조작자 품질 라벨(`quality`: worse/okay/better)이 들어 있다.

---

## 8. `offline_pretrain.py` 정적 검토 결과 (2026-09-03, 집 데스크톱에서 완료)

실행 경로 전부를 코드로 따라가 봤다. **조용히 틀리는 버그는 못 찾았다.** 50k 사전학습을 막을 이유는 없다.
단, 아래 "설계 교란" 두 개는 실험 시작 전에 결정이 필요하다.

### 통과한 항목
- `expectile_loss`: u = Q − V, 가중치 |τ − 1{u<0}|. τ=0.7이면 Q>V 샘플 가중치 0.7, Q<V는 0.3 → V가 위로 끌림. IQL 논문 식과 동일.
- V 타깃은 `critic_target`의 min(온라인 `critic_backup_combine_type`과 같은 결합). Q 타깃 `r + γ(1−d)V(s')`는 `no_grad` 안에서 계산되어 detach됨. 보상·γ·τ(polyak)·lr·손실 계수(0.5·Σmse) 전부 온라인 `dsrl.py:273-302`와 같음.
- 보상/종료 일관성: 온라인 청크 보상 = Σ(r−1) 4스텝 (`env_utils.py:84,177`) = 오프라인 `sum − 4·offset`. 온라인 done은 300스텝 시간제한뿐이고(`robomimic_lowdim.py:138`은 항상 False) `TimeLimit.truncated`를 안 넣어 terminal로 저장됨. 성공 후 shifted 보상이 0이므로 데모 끝 terminal(=0 부트스트랩)과 값이 일치.
- `run_distill`: 버퍼 샘플은 `to_torch`로 model.device에 올라옴. `scale_action`은 `[-1,1]^28`에서 항등(SpacesOnlyEnv, ActionChunkWrapper 둘 다). w ~ N(0,I) 무클리핑, 확산정책도 `cond["noise_action"]`을 그대로 씀(`diffusion.py:287`) → 온라인 `update_noise_critic`과 완전히 같은 경로.
- `load_pretrained_weights`: 네트워크 모양은 obs 23·action 28 차원에만 의존하므로 n_envs 1↔4 무관. state_dict 키는 같은 클래스라 일치. `critic_target`은 파일 것을 로드. `log_ent_coef`는 파일에 있을 때만(warmup) `.data.copy_`로 복사해 optimizer 파라미터 identity 유지. 로드는 `train_dsrl.py:225-233`의 fresh-run 분기, `evaluate` 직전에만 일어나고, 이후 `collect_rollouts`·`learn()`은 가중치를 건드리지 않음. resume 시엔 체크포인트 가중치가 우선(정상).
- `run_warmup`: `model.train()`이 쓰는 `_vec_normalize_env`·`logger`·`_current_progress_remaining` 전부 준비돼 있음. α도 같이 학습되고 저장됨.
- 오프라인 청크는 stride 1(겹침). MDP는 청크 단위(s_t, a_{t:t+4}, s_{t+4})로 같으므로 편향 없음.

### 설계 교란 (결정 필요)
1. **엔트로피 계수 α.** `ent_coef=auto`라 α=1.0에서 시작, `target_ent=0.0`. 28차원 squashed Gaussian의 초기 log π ≈ −18 nats라
   (a) 처음 수천 그래디언트 스텝 동안 actor 손실 `α·log π − Q_W`는 무작위 Q_W(크기 ~0.1)가 아니라 엔트로피 항이 지배 → π_W가 [-1,1]^28 균등분포 쪽으로 퍼짐. **이것 자체가 dip 후보 메커니즘**이고 critic 초기화와 무관하다.
   (b) critic 타깃에도 `−α·log π ≈ +18/청크`가 더해져 IQL의 hard Q(범위 [−400,0])와 스케일이 안 맞음. 상태만의 함수라 w에 대한 순위는 보존되지만 절대값은 수천 스텝 안에 덮어써진다.
   (c) α는 Adam으로 log α가 스텝당 3e-4씩 내려가 학습 시작 후 약 1만 env step이면 α≈0.02. **warmup은 이미 내려간 α를 로드하고 iql·baseline은 α=1에서 시작** → warmup이 좋게 나오면 critic 때문인지 α 때문인지 구분 못 함.
   → **반영함**: `pretrain.load_ent_coef`(기본 False)를 추가해 세 변형 모두 α=1에서 시작. α 궤적은 `train_log.csv`의 `ent_coef` 열에 있으니 분석 때 같이 볼 것.
2. **증류 스텝 수.** warmup은 `steps/utd × noise_critic_grad_steps` = 50000/20×10 = 25,000 증류 스텝, iql은 20,000이었다. → **반영함**: `pretrain.distill_steps=25000`.

### 참고 (버그 아님)
- `standard_gauss_init`은 `build_agent`가 안 넘겨 False. `log_std_init=0.0`은 gSDE 전용이라 무시됨. step-0 actor는 순수 무작위(mean·log_std 모두 Linear 기본 초기화, log_std clamp [−20,2]).
- 오프라인 단계에서 `hydra.run.dir=${logdir}` 때문에 `logs/robomimic-dsrl/<타임스탬프>/.hydra/`가 생김. 무해.
- 로컬(집)에는 `dppo/log/` 체크포인트가 없어 `offline_pretrain.py`를 돌려볼 수 없다. 실행 검증은 Colab에서 이미 됨(200스텝).

## 9. 집 데스크톱 세팅

```bash
# 1) 도구
#   git, gh (GitHub CLI), Claude Code, Python 3.10+ 아무거나
gh auth login          # msp0617 계정

# 2) 저장소 (서브모듈이 ssh 주소라 치환 필요)
git config --global url."https://github.com/".insteadOf "git@github.com:"
git clone --recurse-submodules -b o2o https://github.com/msp0617/dsrl.git
cd dsrl
git remote add upstream https://github.com/ajwagen/dsrl.git

# 3) 로컬 테스트용 venv (torch·mujoco 없이 도는 것만)
python3 -m venv .venv
.venv/bin/pip install numpy gymnasium pyyaml h5py
.venv/bin/python scripts/test_resume_state.py       # 9 checks passed
.venv/bin/python scripts/test_make_offline_chunks.py # 6 checks passed

# 4) Claude Code는 이 폴더에서 연다
claude
# 첫 메시지: "HANDOFF.md 읽고 이어서 작업 준비해"
```

Windows(PowerShell)라면 같은 순서를 이렇게. Claude Code를 WSL에 깔았다면 WSL에서 위 bash 명령 그대로.

```powershell
gh auth login
git config --global url."https://github.com/".insteadOf "git@github.com:"
git clone --recurse-submodules -b o2o https://github.com/msp0617/dsrl.git
cd dsrl
git remote add upstream https://github.com/ajwagen/dsrl.git
python -m venv .venv            # python이 없으면 py
.venv\Scripts\pip install numpy gymnasium pyyaml h5py
.venv\Scripts\python scripts\test_resume_state.py
.venv\Scripts\python scripts\test_make_offline_chunks.py
claude
```

- `.venv/`는 `.gitignore`에 넣어둠. 커밋하지 말 것.
- 코드 수정은 **커밋 + `git push origin o2o`** 해야 Colab에 반영된다. Colab은 클론으로 받는다.
  이미 열린 Colab 세션은 `!cd /content/dsrl && git pull origin o2o`.
- Colab 노트북은 `colab/dsrl_colab_run_v3.ipynb`를 올린다. Drive는 그대로.
- Claude Code 메모리는 폴더별이라 집에서는 비어 있다. 이 문서가 그 역할을 한다.

---

## 10. 함정 모음

- **Colab `%%bash`는 끝날 때까지 출력이 안 보인다.** 긴 명령은 노트북의 `run_bash` 헬퍼로.
- `pkill -f train_dsrl.py`는 자기 셸까지 죽인다. `pkill -f "[t]rain_dsrl.py"`.
- 스모크처럼 `num_evals`를 줄일 땐 `eval_schedule.num_evals_early`도 같이(코드에서 상한을 두긴 했다).
- `save_replay_buffer=False`인 run은 resume 거부된다(설계). 일회성 run은 `resume=False`.
- 커널 "세션 다시 시작"은 VM 디스크를 유지한다 → 설치 불필요, Drive 마운트 셀(2번)만. "런타임 삭제"는 전부 다시.
- 설치에서 robomimic은 git master가 아니라 **PyPI 0.3.0**. git master는 egl_probe 빌드 실패 + torch 2.14를 끌고 온다.
- 처리량은 **A100에서** 재야 한다. 환경 스텝이 CPU 바운드라 T4 수치는 예산에 못 쓴다.
