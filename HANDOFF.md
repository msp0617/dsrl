# DSRL offline-to-online 프로젝트 — 인수인계 (v3, 2026-09-04)

> **현재 상태는 [HANDOFF_2026-09-08.md](HANDOFF_2026-09-08.md)를 먼저 읽을 것.**
> 2026-09-08 기준 온라인 본 실험 121/121과 오프라인 사전학습 artifact 32/32가
> 완료됐으며, 신규 TD/CQL/Cal-QL 24개까지 종료됐다. 이 문서는 시간순 상세 기록으로 보존한다.

이 문서 하나로 새 세션(Claude Code든 claude.ai 채팅이든)이 작업을 이어갈 수 있게 쓴 것이다.
채팅에는 저장소 접근이 없으므로 **필요한 명령·숫자·판단 기준을 전부 여기에 넣었다.**
새 세션 첫 메시지: **"HANDOFF.md 붙여넣고, 섹션 2 '진행 중'부터 이어서"**.
코드 설명은 [O2O.md](O2O.md), 저장소는 https://github.com/msp0617/dsrl 브랜치 `o2o`.

---

## 0. 한 줄 요약

논문 DSRL(Wagenmaker et al. 2025, arXiv:2506.15799)의 offline-to-online 초기 성능 dip이
**critic 초기화** 때문인지, IQL로 critic을 미리 만들면 줄어드는지 robomimic Can에서 실험한다.
**2026-09-06 14:00 기준: 76 run 완료. dip = auto-α(목표 엔트로피 0)의 엔트로피 붕괴. 엔트로피를 12 근처에 유지하면 Can(고정 0.3 또는 목표 12 + 초기 0.3)과 Square(목표 12) 모두 dip이 사라진다(19.2, 19.8). 최종 성능에 맞는 엔트로피 수준은 과제마다 다르다(Square 12는 후반에 너무 넓음). Q 스케일 가설과 게이트는 기각(19.1). 마지막 5 run(`square_fixalpha_03_s{4,5}`, `square_tent6_s{1,2,3}`)이 16:30 완료 예정.**
결과 표·해석은 **섹션 15(축 A·B), 16(Square), 19(α·적응형)**. 월·화 포스터(마감 2026-09-09 수), 패널 개정안은 19.6.
잔여 크레딧 약 80(10:30) → 6 run 뒤 약 35. VM은 자동 반납 keepalive. 새 VM은 캐시 복원으로 10분(섹션 3).

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

### 진행 중 (2026-09-04 밤 22:00 KST 기준)
- **VM 1** (G4, vCPU 48, RAM 176GB): 본학습 9개 `can_{baseline,iql,warmup}_s{1,2,3}` 300k. run당 RAM 약 17GB, 9개에 155GB.
  9개 동시일 때 run당 약 7 env step/s. 오후에 유휴 판정으로 VM이 한 번 죽어 21:30에 체크포인트에서 resume함
  (baseline s1 199k, iql s1 149k, warmup s1·s2 124k, 나머지 99k). 예상 종료: seed 1 새벽 1~4시, seed 2·3 아침 6~7시.
- **VM 2** (G4, 노트북 사본): `can_fixalpha_s1`(100k) + `can_mix_prefill_s{1,2,3}`, `can_mix_fixed_s{1,2,3}`, `can_iql_linear_s1`(200k). 8개, RAM 138GB. 예상 종료 아침 5~6시.
- **VM 3** (G4, GitHub 노트북, 23:20 KST 시작): `can_baseline_s{4,5}`, `can_iql_s{4,5}`, `can_fixalpha_s{2,3}`, `can_warmupc_s{1,2,3}` 9개, 전부 150k(섹션 14). 사전학습 `iql_can_s{4,5}.pt` 생성 완료.
  warmupc step-0 = 0.69/0.52/0.30으로 같은 seed의 baseline(0.66/0.50/0.34)과 일치 → **warmup 붕괴는 actor 때문**임이 여기서 확인됨. 예상 종료 새벽 5시.
- 환경 캐시 `$PROJ/env_cache/dsrl_env.tar.gz`는 VM 3에서 완전한 설치(robomimic·torch 2.7.1 포함) 후 다시 저장함(약 5GB). 첫 저장본(3.0GB)은 설치가 덜 된 상태라 덮어씀.
- 세 VM 모두 **keepalive 셀**(섹션 3) 실행 중. 잔여 크레딧 23:00 기준 약 550, 세 VM 시간당 27. 아침까지 약 200 사용 예상.
- 아직 안 띄운 것: `can_mix_linear_s{1,2,3}` (VM 1 본학습이 끝나 RAM이 비면), π_dp 기준선(`eval_base_policy.py`), 선택 조건 `can_warmupc_s{1,2,3}`.

### 지금까지 관찰 (seed 1, 점당 ±5%p 이상의 노이즈)
- baseline s1: 0.66 → 29k에서 0.34, 49k까지 0.25~0.45, 64k부터 0.5~0.6, 179k 0.69, 204k 0.62~0.75. **dip 뚜렷.** s2는 dip이 거의 없고 s3는 낮게 시작(0.34)해 오름. dip은 seed 의존.
- iql s1: 0.62 → 29k 0.29, 34k부터 0.42, 44k 0.55. baseline보다 20~30k 빨리 회복. 154k 0.64~0.67.
- warmup: **세 seed 모두 step 0 성공률 0.01~0.03** (오프라인 actor가 π_dp를 망가뜨림), 5k 학습 뒤 0.48~0.70으로 회복. 가장 재현성 있는 현상.
- 같은 seed의 step-0 값이 run마다 다름(seed 1: 0.62~0.83). 무작위 actor + 평가 노이즈. **regret 기준선은 step 0가 아니라 π_dp+N(0,I) 평가값**이어야 함.
- 같은 정책의 100 에피소드 평가가 0.83 vs 0.71처럼 갈림(resume으로 같은 구간 두 번 평가됨). 평가 노이즈가 이론값보다 큼 → 분석 때 평활 필수.

### 9/5(토) 아침에 한 것 (07:20~09:00 KST)
- VM 1·2·3의 26 run 전부 `[done]` 확인. VM 2·3은 keepalive가 스스로 반납. VM 1은 붙들어 두고 재사용.
- `git pull` → π_dp 기준선 3 seed × 500 에피소드 → `base_policy_eval.csv` (0.440/0.394/0.382).
- `can_mix_linear_s{1,2,3}` 3개를 VM 1에 띄움(3개만 돌아 run당 15~20 env step/s, 12:00 완료 예정). keepalive(반납 없는 버전) 실행 중.
- `plot_results.py` 첫 실행 → `$PROJ/figures/` 7개 파일. 해석은 섹션 15.

### 남은 순서
1. 12:00 이후 VM 1에서 keepalive 정지 → `!cd /content/dsrl && git pull origin o2o` → 그림 셀 재실행(섹션 13 명령). linear가 축 B 그림에 추가되고 `at_129k` 열·진단 선 스타일이 반영됨.
2. 그림 세 장(`success_critic`, `success_mix`, `diagnostics_critic`)과 표를 보고 포스터 6패널 문안 확정(섹션 15의 해석 1~5가 초안).
3. VM 1 런타임 삭제(더 돌릴 run 없음). 끝난 run의 `checkpoint/`는 지워도 됨(CSV가 결과물, 640MB × 29).
4. 일요일: 포스터. 추가 run이 필요하면(예: prefill vs linear 차이가 애매할 때 seed 추가) 캐시 복원으로 VM 하나 열어 얹는다.

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
- **유휴 판정 주의 (9/4 오후에 실제로 당함).** nohup 백그라운드 프로세스는 Colab 눈에 "활동"이 아니다. 셀이 아무것도 실행 중이 아니면
  유휴로 판정돼 VM이 죽는다. run을 띄운 뒤 반드시 아래 **keepalive 셀**을 실행해 두고 탭을 열어 둔다(10분마다 한 줄, 다 끝나면 스스로 멈춤).
  다른 셀을 돌려야 하면 keepalive를 정지 → 셀 실행 → keepalive 재실행.
  ```python
  import subprocess, time
  while True:
      out = subprocess.run("ps aux | grep '[t]rain_dsrl.py' | grep -o 'exp_id=[a-z_0-9]*' | sed 's/exp_id=//' | tr '\\n' ' '",
                           shell=True, capture_output=True, text=True).stdout.strip()
      ram = subprocess.run("free -g | awk 'NR==2{print $3\\"/\\"$2}'", shell=True, capture_output=True, text=True).stdout.strip()
      print(time.strftime('%H:%M'), 'ram', ram, '|', out or '(none running)', flush=True)
      if not out:
          break
      time.sleep(600)
  ```
- **환경 캐시**: 설치 끝난 conda env가 `$PROJ/env_cache/dsrl_env.tar.gz`(torch 2.7.1 cu128 포함)에 저장돼 있다. 새 VM에서는
  **0(Drive) → 1(condacolab) → 2(Drive) → 3(클론) → 복원 셀(노트북 5b, 또는 아래) → 6 → 7 → 7b → 8 → 9**. 4~5번(15분)을 건너뛰어 3~5분.
  ```bash
  %%bash
  set -e
  CACHE=/content/drive/MyDrive/dsrl_project/env_cache
  mkdir -p /usr/local/envs && cd /usr/local/envs && rm -rf dsrl
  tar -xzf $CACHE/dsrl_env.tar.gz
  source /usr/local/etc/profile.d/conda.sh && conda activate dsrl
  python -c "import torch, robomimic; x=torch.randn(256,256,device='cuda'); print(torch.__version__, (x@x).sum().item()!=0)"
  ```
- VM 2는 노트북 **사본**(파일 → 드라이브에 사본 저장)으로 연다. 노트북 하나에 런타임 하나. 두 VM은 같은 Drive를 쓰고 결과는 같은 `$PROJ/logs/`에 쌓인다.
- VM 하나에 여러 run을 얹는 게 정답이다. run당 RAM 약 17GB, CPU 48개라 9개까지 얹어도 run당 약 7 env step/s. 크레딧은 VM당 시간당 9로 고정이므로 얹을수록 싸다.
- Drive의 예전 노트북 사본은 설치 셀이 torch 2.4.0을 깐다. 설치 후 torch만 교체하는 셀(섹션 5)을 돌리거나, 설치 셀의 마지막 torch 줄을 cu128로 바꿔서 실행.

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

현황 셀 (어느 VM에서든; 살아 있는 run, RAM, 최근 평가 4개, 남은 시간):
```python
run_bash(r'''
PROJ=/content/drive/MyDrive/dsrl_project
echo "alive: $(ps aux | grep '[t]rain_dsrl.py' | grep -o 'exp_id=[a-z_0-9]*' | sed 's/exp_id=//' | tr '\n' ' ')"
free -g | awk 'NR==2{print "ram used/total:", $3"/"$2, "GB"}'
for E in $(ps aux | grep '[t]rain_dsrl.py' | grep -o 'exp_id=[a-z_0-9]*' | sed 's/exp_id=//'); do
  echo "== $E =="
  [ -f $PROJ/logs/$E/eval_log.csv ] && cut -d, -f2,5 $PROJ/logs/$E/eval_log.csv | tail -n +2 | tail -n 4 | tr '\n' ' '; echo
  T=$([ "${E#can_mix_}" != "$E" ] || [ "${E#can_iql_linear}" != "$E" ] && echo 200000 || echo 300000)
  [ -f $PROJ/logs/$E/train_log.csv ] && python colab/throughput.py $PROJ/logs/$E --target $T | tail -n 1
done
''')
```

비율 실험 띄우기: 노트북(GitHub 버전) 12b 셀에 `MIX` 변수가 있다. `VARIANT`, `SEED`, `MIX` 세 줄만 바꿔 실행.
MIX≠none이면 200k, 이름은 `can_mix_<MIX>_s<seed>`(baseline) 또는 `can_<variant>_<MIX>_s<seed>`. 사본 노트북엔 없으니 GitHub 버전에서 복사.

고정 α 프로브:
```bash
python train_dsrl.py --config-path=cfg/robomimic --config-name=dsrl_can.yaml \
  exp_id=can_fixalpha_s1 seed=1 variant=baseline train.ent_coef=0.01 train.total_env_steps=100000 log_dir=$PROJ/logs
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
- resume한 run의 `eval_log.csv`에는 같은 `env_steps`가 두 번 나온다(체크포인트 이후 구간을 다시 돌기 때문). `plot_results.py`는 나중 행을 쓴다.
- `throughput.py`는 resume 직후 몇 행 동안 공백 때문에 시간이 부풀어 보였는데, 4616a56부터 마지막 재시작 이후 행만 잰다.
- 현황 셀(섹션 5 아래)은 `ps`로 살아 있는 run만 잡으므로 끝난 run은 목록에서 사라진다. 끝났는지는 `$PROJ/logs/<exp>.out` 끝의 `[done]`으로.
- 옛 12b 셀(사본 노트북)은 MIX가 없어 본학습만 띄운다. VM 2에서 그 셀을 돌리면 VM 1의 본학습과 같은 폴더에 두 프로세스가 쓰게 되니 **절대 금지**. VM 2에서는 MIX 셀만.

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

---

## 13. 2026-09-04 오후 추가: 비율 스케줄 + 진단 로깅 (작업 지시서 WORK_ORDER.md 구현)

포스터 마감 2026-09-09. 지시서의 두 가설: **H1** critic 부정확, **H2** actor가 N(0,I) prior 밖으로(α 과도기).
지시서 대비 바뀐 점(작성자 동의): `w_frac_gt2` → `w_frac_sat`(|w|>0.9). w는 tanh로 [-1,1]에 묶여 |w|>2가 불가능.
prior 이탈은 크기가 아니라 **분포가 좁아지고 ±1에 포화**하는 것으로 나타나며 tanh 이전 `mu_absmean`, `log_std_mean`이 직접 측정치.

### 구현된 것 (커밋 참조 `git log`)
- `offline_mix.mode` = none | prefill | fixed | linear (config, `o2o_utils.py`의 `ratio_at`, `mixed_sample`, `OfflineBuffer`, `OfflineRatioCallback`).
  `DSRLResumable.train()`이 상류 `train()`을 오버라이드해 샘플 두 곳(critic·actor 공용 배치, noise critic 배치)을 `mixed_sample`로.
  mode=none은 상류와 연산·난수 스트림이 동일. t=0은 학습 시작(초기 rollout 24,016 이후). resume 상태 없음. fingerprint에 mode·p0·p1·until 포함.
- **prefill의 정체**: D_off 15,464 슬롯이 버퍼(50k 슬롯)에 들어가고 밀려나지 않으므로 균등 샘플링에서 비율이 약 0.91(학습 시작) → 0.43(300k)으로 **자연 감쇠**.
  즉 논문 기본 세팅이 이미 암묵적 스케줄이고 linear는 기울기·도달점만 다른 것. 포스터 4번 패널 문구를 이렇게. `offline_p` 열이 prefill에서도 이 값을 기록.
- `train_log.csv` 새 열: `offline_p, w_absmean, w_std, w_frac_sat, mu_absmean, log_std_mean, logp_mean, qw_mean` (마지막 actor 스텝에서 계산, 추가 forward 1회).
- `eval_log.csv` 새 열: `mc_return`(평가 에피소드 실제 할인 리턴), `q_start`(같은 시작 상태의 Q_W). 격차 = Q 과대추정. **α < 0.1 이후에만 유효**(Q_W는 soft value).
  격차가 α 곡선을 따라 줄면 "부풀림의 원인이 엔트로피 보너스"라는 H2 증거.
- CSV 호환: 기존 파일은 자기 헤더의 열만 쓴다(resume된 옛 run에 새 열이 끼어들지 않음). 새 열은 새 run에서만.
- `scripts/test_offline_mix.py` 12개(torch 없이). `scripts/plot_results.py`: 축별 success(mean±SE)·diagnostics·qgap PNG + `metrics.csv`(step0, dip 깊이, 회복 시점, AUC 0~100k, final, π_dp 대비 regret). 합성 데이터로 검증 완료.
- 노트북 12b 셀에 `MIX` 변수(none|prefill|fixed|linear). MIX≠none이면 200k, exp_id는 `can_mix_<MIX>_s<seed>`(baseline) 또는 `can_<variant>_<MIX>_s<seed>`.

### Colab 스모크 (9 run 띄우기 전 필수, 세 개)
1. mode=none 회귀: 섹션 10 스모크 셀에 `exp_id=smoke_none`로 실행 → `train_log.csv`에 새 열이 채워지고 `offline_p`가 0인지.
2. linear resume: 같은 셀에 `exp_id=smoke_mix offline_mix.mode=linear offline_mix.p0=0.8 offline_mix.p1=0.1 offline_mix.until_env=800 offline_data_path=$PROJ/offline/can_train_offline.npz` 로 1200스텝, 다시 2000스텝 → `[resume]` 뒤 `offline_p`가 이어지는지(0.8 → 0.1 감소).
3. 진단 열이 NaN·상수가 아닌지 (`mu_absmean`, `log_std_mean`이 움직이는지).

### 실험 매트릭스 (지시서 섹션 4)
- 돌고 있음: `can_{baseline,warmup,iql}_s{1,2,3}` 300k (VM 1).
- 고정 α 프로브(코드 불필요, VM 2): `exp_id=can_fixalpha_s1 variant=baseline train.ent_coef=0.01 train.total_env_steps=100000`.
- 비율 축(스모크 통과 후, 토요일): baseline 200k × 3 seed × {prefill, fixed p0=0.5, linear 0.8→0.1 until 100k} = 9 run. 12b 셀 MIX로.
- 교차(1 seed): `can_iql_linear_s1` = VARIANT=iql MIX=linear.
- 선택: warmup critic-only `can_warmupc_s{1,2,3}`(섹션 2의 `pretrain.load_actor=False`), π_dp 기준선 `scripts/eval_base_policy.py`.

### 분석 (일요일)
`python scripts/plot_results.py --logs $PROJ/logs --out $PROJ/figures`. 기본 축이 `critic=baseline,warmup,iql,warmupc,fixalpha`, `mix=baseline,mix_prefill,mix_fixed,mix_linear,iql_linear`라 `--axes` 없이 되고, 바꾸려면 `--axes "critic=...;mix=..."`.

---

## 14. VM 3 (2026-09-04 밤, HANDOFF_VM3.md 검증 결과)

seed 1 결과로 헤드라인을 "actor를 로드하면 무너지고 critic만 로드하면 회복이 빨라진다"로 옮김. 이를 위해 VM 3에서 9 run(150k):
`can_baseline_s{4,5}`, `can_iql_s{4,5}`(사전학습 s4·s5 필요), `can_fixalpha_s{2,3}`, `can_warmupc_s{1,2,3}`.

### 검증한 것 (코드 기준)
- 사전학습 출력 경로: `offline_pretrain.py`는 `${log_dir}/pretrain/{method}_{env}_s{seed}.pt`와 `_log.csv`에 쓴다 → `iql_can_s4.pt`, `iql_can_s5.pt`. iql만 필요(warmup 사전학습 불필요).
- `pretrain.load_actor`(기본 True), `pretrain.load_ent_coef`(기본 False)는 config 키. warmupc = `variant=warmup pretrain.load_actor=False` → 로그에 `loaded critic, critic_target, critic_noise`만 나오고 α는 1.0에서 시작.
- 12b 셀(GitHub 버전, 커밋 이후)에 `STEPS`, `EXTRA_ARGS`, `EXP_TAG` 변수 추가. MIX=none일 때 `STEPS=150000`으로 예산 override 가능. 다만 VM 3은 아래 직접 명령이 더 간단.
- fingerprint(resume 충돌 검사): 네트워크 모양, n_envs, buffer_size, variant, load_actor, load_ent_coef, offline_mix에 **`train.ent_coef`, `train.target_ent` 추가**(이 커밋). `total_env_steps`는 일부러 제외 — 같은 exp_id로 예산만 늘려 이어 돌리는 게 정당한 용도. 즉 같은 exp_id를 다른 예산으로 resume하면 그냥 이어진다(사고 아님).
- `plot_results.py`: 조건별 seed 수가 달라도(5/3/3/5/3) 점마다 있는 seed로 평균·SE(ddof=1). 라벨은 `n=3-5`처럼 범위. 예산이 다른 run이 섞이면 150k 이후는 seed 수가 줄어든다. `metrics.csv`에 `at_150k`(모든 run이 갖는 지점) 추가.
- `base_policy_eval.csv`가 없으면 기준선·regret 없이 그림만 그린다(에러 없음). 있으면 평균을 점선으로, `regret_vs_pi_dp` = 기준선 − AUC(0~100k).

### VM 3 셀 (GitHub 노트북 → 0 → 1 → 2 → 3 → 5b 복원 → 6 → 7 → 7b → 8 → 9 → 아래)

사전학습 s4·s5 (백그라운드, seed당 약 20분, 둘 동시):
```bash
%%bash
source /usr/local/etc/profile.d/conda.sh && conda activate dsrl
source /content/env.sh
cd /content/dsrl
PROJ=/content/drive/MyDrive/dsrl_project
for SEED in 4 5; do
  nohup python offline_pretrain.py --config-path=cfg/robomimic --config-name=dsrl_can.yaml \
    pretrain.method=iql seed=$SEED \
    offline_data_path=$PROJ/offline/can_train_offline.npz log_dir=$PROJ/logs \
    > $PROJ/logs/pretrain_s${SEED}.out 2>&1 &
  echo "started iql pretrain seed $SEED (pid $!)"
done
```

사전학습이 필요 없는 7개:
```bash
%%bash
source /usr/local/etc/profile.d/conda.sh && conda activate dsrl
source /content/env.sh
cd /content/dsrl
PROJ=/content/drive/MyDrive/dsrl_project
CFG="--config-path=cfg/robomimic --config-name=dsrl_can.yaml"
COMMON="log_dir=$PROJ/logs train.total_env_steps=150000 offline_mix.mode=none load_offline_data=False"
launch () { EXP=$1; shift; nohup python train_dsrl.py $CFG exp_id=$EXP "$@" $COMMON > $PROJ/logs/$EXP.out 2>&1 & echo "started $EXP (pid $!)"; }
launch can_baseline_s4 seed=4 variant=baseline
launch can_baseline_s5 seed=5 variant=baseline
launch can_fixalpha_s2 seed=2 variant=baseline train.ent_coef=0.01
launch can_fixalpha_s3 seed=3 variant=baseline train.ent_coef=0.01
for S in 1 2 3; do
  launch can_warmupc_s$S seed=$S variant=warmup pretrain.load_actor=False pretrain_path=$PROJ/logs/pretrain/warmup_can_s$S.pt
done
```

`.pt` 두 개 확인(`ls $PROJ/logs/pretrain/iql_can_s4.pt iql_can_s5.pt`) 후 iql s4·s5:
```bash
%%bash
source /usr/local/etc/profile.d/conda.sh && conda activate dsrl
source /content/env.sh
cd /content/dsrl
PROJ=/content/drive/MyDrive/dsrl_project
CFG="--config-path=cfg/robomimic --config-name=dsrl_can.yaml"
COMMON="log_dir=$PROJ/logs train.total_env_steps=150000 offline_mix.mode=none load_offline_data=False"
launch () { EXP=$1; shift; nohup python train_dsrl.py $CFG exp_id=$EXP "$@" $COMMON > $PROJ/logs/$EXP.out 2>&1 & echo "started $EXP (pid $!)"; }
for S in 4 5; do launch can_iql_s$S seed=$S variant=iql pretrain_path=$PROJ/logs/pretrain/iql_can_s$S.pt; done
```

확인(5분 뒤): `grep -h "\[pretrain\]\|\[eval\] env_steps=0\|\[budget\]"`으로 warmupc 세 개에 actor가 없는지, fixalpha 로그의 `ent_coef`가 0.01인지, `[budget] ... target 150000`인지. 그다음 keepalive.

## 15. 9/5(토) 아침 결과 — 26 run 완료, 첫 분석

전부 `$PROJ/logs/` CSV에서 `plot_results.py`로 뽑은 값. 그림은 `$PROJ/figures/`. linear 3개는 12:00 KST 완료 예정(그 뒤 `git pull` → 그림 셀 재실행).

**π_dp 기준선** (`base_policy_eval.csv`, seed당 500 에피소드): 0.440 / 0.394 / 0.382, **평균 0.405 ± 0.014**.
무작위 actor를 붙인 step-0 값(0.5~0.8)이 이보다 높다: tanh로 좁혀진 w가 N(0,I)보다 π_dp에 유리하다. regret은 이 기준선으로 잰다.

| 조건 | n | step 0 | 최저(0~100k) | 회복 | AUC 0~100k | 최종 |
|---|---|---|---|---|---|---|
| baseline | 5 | 0.50 | 0.14 (평균곡선 0.24 @35k) | 84k | 0.43 | 0.64~0.75 |
| iql | 5 | 0.57 | 0.24 (평균곡선 0.45) | 73k | 0.50 | 0.65~0.75 |
| warmupc (critic만) | 3 | 0.50 | 0.32 | 42k | 0.48 | 0.58 @129k |
| warmup (actor 포함) | 3 | 0.02 | 0.29 | 61k | 0.44 | 0.80~0.82 |
| fixalpha (α=0.01) | 3 | 0.55 | 0.003 | 없음 | 0.18 | 0.23~0.30 @100~130k |
| mix_prefill (논문 기본) | 3 | 0.52 | 0.08 (35k 한 점) | 47k | 0.67 | 0.89 |
| mix_fixed 0.5 | 3 | 0.53 | 0.29 | 44k | 0.66 | 0.85 |
| iql_linear | 1 | 0.70 | 0.39 | 69k | 0.64 | 0.73 |

**해석 (포스터 문장)**
1. **dip의 정체**: α(`ent_coef`)는 학습 시작 후 5천 step 안에 1 → 0.1로 떨어지고, 모든 조건의 바닥은 35k(학습 1만 step)에 있다. α가 0.1 아래로 가며 actor가 Q_W의 argmax로 이동하는 순간이다. 직전 29k에서는 mix 조건이 0.77로 step 0보다 높다.
   actor 진단(`mu_absmean` 0.3→0.8, `w_frac_sat` 12→22%, `log_std` −0.2→−0.42)은 같은 창에서 **한 번 이동하고 이후 평평**. 이탈은 조건 무관하게 한 번이고, 해가 되느냐는 그때 Q_W 정확도가 정한다.
2. **fixalpha 붕괴 = H2 단독 기각**: α=0.01 고정이면 29k에 0.02로 즉시 붕괴, 130k에도 0.3. 엔트로피는 actor를 밖으로 미는 게 아니라 **엉터리 critic을 믿지 못하게 막는 보호막**.
   결정적 표: fixalpha와 warmupc는 actor 통계가 거의 같은데(mu 1.2~1.7, 포화 43~50%, log_std −0.75) 결과는 정반대(0.02~0.3 vs 0.5~0.6). 포화 자체는 원인이 아니고 **포화된 w가 맞는 Q_W의 argmax인지**가 원인. → **H1(critic 부정확)이 맞되 α 감쇠가 방아쇠.**
3. **축 A**: iql·warmupc는 dip을 얕고 짧게(최저 0.45/0.34, 회복 42~73k vs 0.24/84k), 최종은 무영향(0.75). warmup(actor 로드)은 step 0 붕괴(0.02, 3/3 seed) 후 최종 최고(0.82). warmupc의 step 0(0.69/0.52/0.30)가 같은 seed baseline과 일치 → 붕괴는 actor 때문.
4. **축 B**: 데모를 리플레이에 넣으면(prefill 0.91→0.43 자연 감쇠, fixed 0.5) 35k 급락은 못 막지만 회복이 55k로 10배 빠르고 최종 0.9. **축 B가 축 A보다 훨씬 큰 지렛대.** 논문 기본 세팅이 이미 암묵적 스케줄.
5. `qw_mean`: iql의 Q_W는 step 0 −145였다가 25k에 +70. 사전학습된 값 스케일은 첫 수천 update의 엔트로피 보너스에 덮이고 순위 구조만 남는다. 그래도 dip이 얕아지니 순위가 유효.

**주의**: 평가 노이즈가 커서 개별 seed 곡선은 ±0.1 흔들린다. 포스터는 3~5 seed 평균±SE로. 진단 그림의 fixalpha 선은 warmupc와 겹쳐 안 보였던 것(값은 있음) → 선 스타일 구분(커밋 참조).

### 9/5 오후 추가 결과 (CSV를 로컬로 받아 `plot_results.py`, `alpha_timing.py`로 분석)
CSV 묶음: Colab에서 `zip -r csv_bundle.zip logs -i "logs/*/eval_log.csv" "logs/*/train_log.csv" "logs/*.csv"` → 다운로드 → 로컬에서 스크립트 실행. Colab 그림 셀 불필요.

| 조건 | n | 최저 | 회복 | AUC 0~100k | 129k | 최종 |
|---|---|---|---|---|---|---|
| mix_linear 0.8→0.1 | 3 | **0.02** @35k | 51k | 0.62 | 0.75 | 0.81 |
| iql_prefill (두 축 교차) | 3 | 0.16 | 52k | 0.56 | 0.85 | 0.84 |
| alr_double (α lr 6e-4) | 3 | 0.14 @31.6k | 42k | 0.48 | — | 0.53 @100k |
| alr_half (α lr 1.5e-4) | 3 | 0.14 | 52k | 0.43 | — | 0.41 @100k |
| square_baseline | 3 | 0.19 @42k | 75k | 0.39 | — | 0.48 @127k |
| square_iql | 3 | 0.29 (평균곡선 0.37) | 72k | 0.44 | — | 0.49 @127k |

- **linear**: 명시적 0.8→0.1 스케줄은 prefill의 자연 감쇠(0.91→0.43)보다 **못하다**(AUC 0.62 vs 0.67, 최종 0.81 vs 0.89, 35k 급락 0.02로 가장 깊음). "논문 기본 세팅이 이미 좋은 스케줄"로 정리.
- **iql_prefill**: 두 축은 **쌓이지 않는다**. prefill 단독(0.67)보다 AUC가 낮고(0.56) 최종은 같다(0.84 vs 0.89). 축 B가 축 A를 흡수. 포스터 마지막 문장은 "데모가 리플레이에 있으면 critic 사전학습은 추가 이득이 없다".
- **Square 축 A 재현**: baseline 42k에서 0.21, iql은 dip이 거의 없음(평균곡선 최저 0.37). Can과 같은 방향·크기. 127k 최종은 둘 다 0.5 근처(미수렴, 150k 예산).
- **감쇠율 개입(`alpha_timing.py`)**: α<0.1 통과 double 27.6k / baseline 32k / half 40~42k. 기준선(0.405) 아래로 처음 내려간 시점 평균 **29.0k / 34.0k / 34.1k**, 바닥 평균 31.6k / 41k / 45k. double은 3 seed 모두 5k 앞당겨짐(확실). half는 첫 하락이 baseline과 같고 바닥만 늦음(약함). seed별 상관 0.47.
  해석: **α는 dip 시점을 당길 수는 있지만 밀지는 못한다.** 감쇠를 늦춰도 Q_W가 엔트로피 보너스로 부풀어(half는 +175까지) actor를 끌어당기므로 α=0.15 근처에서 이미 넘어간다. 스위치는 "α·|log π| 대 Q_W 크기"의 비이고 α는 그 한 축. half는 dip이 길고 100k 성능도 낮다(0.41) → α를 오래 붙드는 건 손해. 3단계 게이트 설계에 직접 영향: α_hi를 오래 유지하면 안 되고, 열리는 시점은 Q 스케일 신호로 잡아야 함.
- `diagnostics_alpha.png`: qw_mean 초기 부풀림이 half 175 > baseline 110 > double 55로 α 유지 시간에 비례. actor 통계(mu, 포화, log_std)는 세 조건이 같음.

### 9/5 오후 돌고 있는 것
- VM 1 (15:05 시작, 밤 9시 예상): `square_mix_prefill_s{1,2,3}`, `can_fixalpha_01_s{1,2,3}`(α=0.1 고정), `can_fixalpha_03_s{1,2,3}`(α=0.3 고정). 150k.
- VM 2 (10:30 시작, 오후 5~6시 예상): `can_iql_prefill_s{1,2,3}`(200k, 위 표는 129k까지 반영), `can_alr_{half,double}_s{1,2,3}`(150k, 위 표는 100k까지).
- VM 2가 비면 α 스윕 나머지: `can_fixalpha_003_s{1,2,3}`(0.03), `can_fixalpha_1_s{1,2,3}`(1.0). 셀은 섹션 14의 `launch` 형식으로 `train.ent_coef=0.03` / `=1.0`.
- 영상: `scripts/render_episode.py`로 `$PROJ/videos/`에 π_dp 2편, baseline s1(300k) 2편 생성됨. `+policy=<exp_id>`로 다른 run도 가능.

## 16. Square 일반성 확인 (9/5, HANDOFF_SQUARE.md 검증 결과)

**예측**: α 감쇠는 과제와 무관(그래디언트 스텝 함수)하므로 Square baseline도 학습 시작 후 약 1만 step, 즉 **env 42k 근처**(rollout 32,016 포함)에서 바닥. 포스터는 Can만으로 완성하고 Square는 나오면 패널 추가.

**검증에서 드러난 것**: `dsrl_square.yaml`이 상류 원본이라 `exp_id`, `variant`, `train.total_env_steps`, `offline_mix`, `eval_schedule`, `ckpt_every_env_steps`가 없고 `use_wandb: True`, `save_replay_buffer: False`, 버퍼 10M(4GB), 평가 주기 64k env step이었다.
지시서 명령을 그대로 돌리면 Hydra가 없는 키 override("Could not override 'exp_id'")로 즉시 죽거나, 돌아도 dip을 못 본다. → **커밋 a16f020**에서 config를 Can과 같은 키로 재작성(논문 Square 값은 유지: discount 0.999, `init_rollout_steps` 2001 = 32,016 env step, td100 정책 + DDIM 8, 에피소드 400 step). 평가 스케줄 5k/25k 동일. 진단 로깅은 config 무관하게 붙는다(`DSRLResumable.train`, `LoggingCallback`).
`plot_results.py`는 `<task>_<group>_s<seed>`를 인식해 `square_baseline`을 별도 축(`success_square.png`)에 그리고 기준선은 `base_policy_eval_square.csv`를 쓴다.

**π_dp 파일**: config는 `./dppo/log/robomimic-pretrain/square/.../state_3000.pt`와 `./dppo/log/robomimic/square/normalization.npz`를 기대한다. Drive에는 `dppo_log/dsrl_public_checkpoints/...` 아래에 있으므로 노트북 6번 아래에 추가한 **Square 배치 셀**(find → 상대경로에 복사)을 먼저 실행. `normalization.npz`가 Drive에 없으면 Square는 불가(공개 체크포인트 폴더에 같이 있어야 정상).

**순서 (VM 1, linear 3개 도는 중, RAM 여유 있음)**: keepalive 정지 → `git pull origin o2o` → Square 배치 셀 → 아래 → 5분 확인 → keepalive.
```bash
%%bash
source /usr/local/etc/profile.d/conda.sh && conda activate dsrl
source /content/env.sh
cd /content/dsrl
PROJ=/content/drive/MyDrive/dsrl_project
CFG="--config-path=cfg/robomimic --config-name=dsrl_square.yaml"
COMMON="log_dir=$PROJ/logs train.total_env_steps=150000 offline_mix.mode=none load_offline_data=False"
for S in 1 2 3; do
  nohup python train_dsrl.py $CFG exp_id=square_baseline_s$S seed=$S variant=baseline $COMMON \
    > $PROJ/logs/square_baseline_s$S.out 2>&1 &
  echo "started square_baseline_s$S (pid $!)"
done
```
5분 확인: `grep -h "Loaded base policy\|\[eval\] env_steps=0\|\[budget\]" $PROJ/logs/square_baseline_s?.out` → 경로에 `square`, `target 150000`, step-0 성공률. **step 0가 0.05 이하면 3개 죽이고(`pkill -f "[s]quare_baseline"`) Can에 집중.** 처리량이 느려 150k가 8시간을 넘길 것 같으면 100k로.
π_dp 기준선(선택): `python scripts/eval_base_policy.py --config-name=dsrl_square.yaml seed=1 num_evals=500 log_dir=$PROJ/logs` → `base_policy_eval_square.csv`.

**판정**: 바닥이 42k±5k면 예측 적중("메커니즘은 과제 무관"). 다른 곳이면 그대로 보고. dip 없음 + step 0 낮음이면 "헤드룸 부족"으로 한정. `ent_coef` 곡선을 겹쳐 α<0.1 시점을 표시.

### 9/5 09:30 진행: Square baseline 3개 띄움, 데이터도 확보
- `square_baseline_s{1,2,3}` VM 1에서 실행 중(150k). step-0 성공률 **0.57 / 0.57 / 0.45** → 헤드룸 충분.
- Square 오프라인 데이터 확보: robomimic square **mh** hdf5 → `make_offline_chunks.py` → `$PROJ/offline/square_train_offline.npz`
  (79,828 청크, 19,957 슬롯, 300 데모, 보상 1.9%). `--check_against` 공개 train.npz **비트 일치(max |diff| 0)** → DPPO Square 정책이 이 데이터로 학습됨.
- Square 사전학습 6개(iql·warmup × seed 1~3) VM 1에서 백그라운드 실행 중 → `$PROJ/logs/pretrain/{iql,warmup}_square_s{1,2,3}.pt` (약 1시간).
  버퍼 검사: prefill 150k = 2001 + 19,957 + 9,375 = 31,333 슬롯 < 50,000 ✓.

### Square 조건 확장 (baseline 첫 평가에서 dip이 보이면)
VM 2를 캐시 복원으로 열어(0→1→2→3→5b 복원→6→**Square 배치 셀**→7→7b→8→9) 아래 9개. RAM 9×18 ≈ 160GB.
```bash
%%bash
source /usr/local/etc/profile.d/conda.sh && conda activate dsrl
source /content/env.sh
cd /content/dsrl
PROJ=/content/drive/MyDrive/dsrl_project
CFG="--config-path=cfg/robomimic --config-name=dsrl_square.yaml"
COMMON="log_dir=$PROJ/logs train.total_env_steps=150000"
launch () { EXP=$1; shift; nohup python train_dsrl.py $CFG exp_id=$EXP "$@" $COMMON > $PROJ/logs/$EXP.out 2>&1 & echo "started $EXP (pid $!)"; }
for S in 1 2 3; do
  launch square_iql_s$S     seed=$S variant=iql    pretrain_path=$PROJ/logs/pretrain/iql_square_s$S.pt
  launch square_warmupc_s$S seed=$S variant=warmup pretrain.load_actor=False pretrain_path=$PROJ/logs/pretrain/warmup_square_s$S.pt
  launch square_mix_prefill_s$S seed=$S variant=baseline offline_mix.mode=prefill offline_data_path=$PROJ/offline/square_train_offline.npz
done
```
VM 1은 12:00에 linear가 끝나면 `square_warmup_s{1,2,3}`(actor 포함) 3개:
```bash
for S in 1 2 3; do
  launch square_warmup_s$S seed=$S variant=warmup pretrain_path=$PROJ/logs/pretrain/warmup_square_s$S.pt
done
```
(위 `launch` 정의와 CFG/COMMON을 같은 셀에 넣어서.) 확인: `[pretrain] ... loaded ...`에 warmupc는 actor 없음, `[budget] ... target 150000`.
`plot_results.py`는 `square_*` 그룹을 자동 인식하지만 기본 축은 `square=square_baseline`뿐이므로 확장 후엔
`--axes "critic=baseline,warmup,iql,warmupc,fixalpha;mix=baseline,mix_prefill,mix_fixed,mix_linear,iql_linear;square=square_baseline,square_iql,square_warmupc,square_warmup,square_mix_prefill"`.
Square π_dp 기준선: `python scripts/eval_base_policy.py --config-name=dsrl_square.yaml seed=$S num_evals=500 log_dir=$PROJ/logs` (seed 1~3) → `base_policy_eval_square.csv`.

## 17. Q 스케일 가설 검정 (NEXT_PHASE_v2, 9/5 저녁 구현)

감쇠율 실험의 비대칭(double은 5k 앞당김, half는 첫 하락 안 밀림)을 설명할 가설: 스위치는 α 단독이 아니라 **actor에 걸리는 두 기울기의 균형**.
주의: `qw_mean`의 +55~+175는 보상이 전부 ≤0인데도 양수이므로 critic 타깃의 엔트로피 보너스가 쌓인 **오프셋**이다. 오프셋은 w에 거의 무관해 argmax를 옮기지 않는다.
그래서 값의 비(`α|log π| / |Q_W|`)가 아니라 **기울기 노름의 비**를 잰다.

**구현 (커밋 참조, 전부 `DSRLResumable.train`·config·fingerprint)**
- `train.reward_scale` c: 타깃의 r만 c배(로깅된 보상·성공률은 그대로, `q_start`·`qw_mean`은 c배로 나옴).
- `train.critic_entropy_scale` β: 타깃의 `α·log π'` 보너스 배율. β=0이면 hard backup(오프셋 인플레이션 제거).
- `train_log.csv` 새 열: `qw_absmean`, `gq_norm`(‖∂(−Q_W)/∂u‖), `ge_norm`(‖∂(α log π)/∂u‖), `ratio_ge_gq`. u는 tanh 이전 샘플. 마지막 actor 스텝에서 backward 두 번 추가, 난수 소비 없음.
- 둘 다 1.0이면 상류와 동일 연산(곱셈을 건너뜀). `plot_results.py`에 `sweep`, `scale` 축과 3×3 진단 패널.

**스모크 (띄우기 전 필수, 아무 VM, 5분)**
```python
run_bash(r'''
PROJ=/content/drive/MyDrive/dsrl_project
git pull origin o2o | tail -n 1
COMMON="log_dir=$PROJ/logs env.n_envs=1 env.n_eval_envs=1 num_evals=1 eval_schedule.every_env_early=400 eval_schedule.early_until_env=100000 eval_schedule.num_evals_early=1 ckpt_every_env_steps=400 train.init_rollout_steps=50 train.utd=1 train.noise_critic_grad_steps=1 train.batch_size=32 train.layer_size=256 train.num_layers=2 train.buffer_size=20000 train.total_env_steps=1200 resume=False"
python train_dsrl.py --config-path=cfg/robomimic --config-name=dsrl_can.yaml exp_id=smoke_scale $COMMON train.reward_scale=2.0 train.critic_entropy_scale=0.0 2>&1 | grep "\[eval\]\|\[done\]\|Error"
head -n 1 $PROJ/logs/smoke_scale/train_log.csv | tr ',' '\n' | grep -n "gq_norm\|ratio_ge_gq\|qw_absmean"
cut -d, -f2,20-24 $PROJ/logs/smoke_scale/train_log.csv
''')
```
합격: `gq_norm`, `ge_norm`, `ratio_ge_gq`, `qw_absmean` 열이 있고 숫자(NaN 아님), `ratio_ge_gq`가 행마다 변함.

**매트릭스 (Can, baseline, 150k, 3 seed, 평가 2.5k 격자) — VM 1·2가 비는 밤 9시에**
| exp_id | override | 예측(비가 스위치라면) |
|---|---|---|
| `can_rs_025_s{1,2,3}` | `train.reward_scale=0.25` | 보상 기울기 ↓ → dip 뒤로 |
| `can_rs_05_s{1,2,3}` | `train.reward_scale=0.5` | 약간 뒤로 |
| `can_rs_2_s{1,2,3}` | `train.reward_scale=2.0` | dip 앞으로 |
| `can_hardq_s{1,2,3}` | `train.critic_entropy_scale=0` | 오프셋 인플레이션 제거. dip이 변하면 보너스가 원인 |
```bash
%%bash
source /usr/local/etc/profile.d/conda.sh && conda activate dsrl
source /content/env.sh
cd /content/dsrl
git pull origin o2o | tail -n 1
PROJ=/content/drive/MyDrive/dsrl_project
CFG="--config-path=cfg/robomimic --config-name=dsrl_can.yaml"
COMMON="log_dir=$PROJ/logs variant=baseline train.total_env_steps=150000 eval_schedule.every_env_early=2500"
launch () { EXP=$1; shift; nohup python train_dsrl.py $CFG exp_id=$EXP "$@" $COMMON > $PROJ/logs/$EXP.out 2>&1 & echo "started $EXP (pid $!)"; }
for S in 1 2 3; do
  launch can_rs_025_s$S seed=$S train.reward_scale=0.25
  launch can_rs_05_s$S  seed=$S train.reward_scale=0.5
  launch can_rs_2_s$S   seed=$S train.reward_scale=2.0
  launch can_hardq_s$S  seed=$S train.critic_entropy_scale=0.0
done
```
12개면 RAM 약 215GB라 **한 VM에 못 얹는다**. VM 1에 9개(rs 3종), VM 2에 3개(hardq) 또는 6/6.

**판정 (`alpha_timing.py` + `ratio_ge_gq`)**: 감쇠율 3조건 + 스케일 4조건 = 7조건에서 첫 하락 시점의 `ratio_ge_gq`가 일정하면 비가 스위치 → 제어 대상은 Q 스케일(타깃 정규화/adaptive reward scale).
hardq에서 dip이 얕아지면 제어기 없이 "초기 hard backup"이 방법. 둘 다 안 움직이면 두 번째 시계는 critic 학습 진행(데이터 양)이고 스케일은 결과.
**게이트(NEXT_PHASE 3단계)는 이 결과 전에는 설계하지 않는다.** → 부품 교체형으로 미리 구현해 둠(섹션 18). 결과가 신호·작동기·τ를 정한다.

### 9/5 밤 상태 (22:00 KST)
- 오후 15 run 전부 `[done]`: `square_mix_prefill_s{1,2,3}`(150k), `can_fixalpha_{01,03,003,1}_s{1,2,3}`(150k). CSV는 아직 로컬로 안 받음.
- α 스윕 첫인상(129k 성공률 seed 평균): 0.03 → 0.41, 0.1 → 0.63, 0.3 → **0.72**, 1.0 → 0.15. 고정 α는 0.3이 최적, 1.0은 무작위성 과다로 학습 안 됨. `alpha_hold` 작동기를 쓴다면 `gate.alpha_hi=0.3`.
- 22:00 VM 2에 **9 run 시작** (rs_05는 크레딧 때문에 뺌, 100k, 평가 2.5k 격자): `can_rs_025_s{1,2,3}`, `can_rs_2_s{1,2,3}`, `can_hardq_s{1,2,3}`. 코드 84dc2af. step-0 성공률은 조건과 무관하게 seed별로 같고(s1 0.71, s2 0.55, s3 0.37) 로깅된 보상은 −240 그대로 → reward_scale이 타깃에만 들어감을 확인. 자동 반납 keepalive 실행 중, 새벽 2~3시 완료 예상.
- **`ratio_ge_gq`·`gq_norm`·`ge_norm`·`qw_absmean` 열은 이 9 run에만 있다.** baseline·alr·fixalpha·Square는 진단 패치(21cef73) 전에 돌아서 없다. 따라서 "첫 하락 시 ratio" 비교는 rs_025 / rs_2 / hardq 세 조건(9점)으로 하고, baseline 스케일의 ratio는 rs_025와 rs_2 사이로 내삽하거나 게이트 signal run 자체(스케일 1, ratio 로깅됨)에서 읽는다.

### 9/6(일) 아침 순서
1. 새 VM(0 → 1 → 2 → 3 → 5b → 6 → 7 → 7b → 8 → 9). CSV 묶음 zip → 로컬.
   ```python
   run_bash(r'''
   cd /content/drive/MyDrive/dsrl_project
   rm -f csv_bundle.zip
   zip -qr csv_bundle.zip logs -i "logs/*/eval_log.csv" "logs/*/train_log.csv" "logs/*.csv"
   ls -la csv_bundle.zip
   ''')
   ```
2. 로컬 분석 (`.venv\Scripts\python`, `scripts/` 안에서):
   ```
   cd scripts
   ..\.venv\Scripts\python alpha_timing.py --logs <logs> --out <figs> --groups alr_double,baseline,alr_half,rs_025,rs_2,hardq
   ..\.venv\Scripts\python plot_results.py --logs <logs> --out <figs> --axes scale,sweep
   ```
   `alpha_timing.py`는 이제 run별로 `ratio_below_{3,1,0.3}`(ratio가 처음 그 값 아래로 간 env step, 3행 이동중앙값), `ratio_ge_gq_at_first_below`, `ent_coef_at_first_below`, `qw_absmean_at_first_below`를 내고, 마지막에 "첫 하락 시점에서 어느 양이 일정한가"를 log10 표준편차(spread)로 찍는다. `ratio_timing.png`도 씀.
   **판정**: (i) rs_025의 첫 하락이 baseline(34k)보다 뒤, rs_2가 앞이면 Q 스케일이 dip 시점을 움직인다. (ii) 그때 `ratio_ge_gq_at_first_below`의 spread가 `ent_coef_at_first_below`의 spread보다 작으면 비가 스위치 → τ = 그 중앙값. (iii) hardq의 dip이 얕으면 hard backup 자체가 처방이고 게이트 작동기는 `hard_backup`으로 확정.
3. τ 확정 → 섹션 18 본 실험 셀에 `TAU=` 넣고 `can_gate_sig_s{1..5}` (5 run, 100k, 약 4시간). keepalive 반납 없는 버전.
4. signal 5개의 `gate_open_call` 평균 N* → `can_gate_clk_s{1..5}` (`gate.signal=clock gate.clock_calls=N*`). 크레딧이 모자라면 clock은 3 seed.
5. 그림·표 재생성, 섹션 15 갱신, 포스터.

## 18. 게이트 구현 (9/5 저녁, `o2o_utils.GateController`)

config `gate:` (기본 `enabled: false`, 켜지 않으면 상류와 동일)
| 키 | 뜻 |
|---|---|
| `signal` | `ratio`: `ratio_ge_gq`(엔트로피 기울기/Q 기울기)가 τ 아래로 K번 연속이면 열림. `clock`: `clock_calls`번 update 뒤 무조건 열림(대조군) |
| `actuator` | 닫힌 동안 무엇을 붙드나. `hard_backup`: critic 타깃에서 엔트로피 보너스 제거(β=0), 열리면 β=`critic_entropy_scale`. `alpha_hold`: α=`alpha_hi` 고정, 열리면 auto-α가 `alpha_hi`에서 이어감 |
| `tau`, `K`, `clock_calls`, `alpha_hi` | 임계, 연속 횟수(train() 호출 = 16 env step 단위), 시계, 고정 α |
- 한 번 열리면 안 닫힘. 상태(`calls, streak, open, open_call`)는 `run_state.json`의 `gate`에 저장돼 resume에서 이어짐. 설정 전부 fingerprint.
- `train_log.csv`에 `gate_open`(0/1), `gate_open_call`(열린 update 번호, 닫혀 있으면 −1).
- 테스트: `test_offline_mix.py`에 상태기계 4개(K 연속, 위반 시 리셋·None 처리, clock, 상태 왕복·fingerprint), `test_resume_state.py`에 run_state 저장 1개.

**스모크 (Colab, 5분)** — 게이트가 열리고 resume에서 이어지는지
```python
run_bash(r'''
PROJ=/content/drive/MyDrive/dsrl_project
git pull origin o2o | tail -n 1
COMMON="log_dir=$PROJ/logs env.n_envs=1 env.n_eval_envs=1 num_evals=1 eval_schedule.every_env_early=400 eval_schedule.early_until_env=100000 eval_schedule.num_evals_early=1 ckpt_every_env_steps=400 train.init_rollout_steps=50 train.utd=1 train.noise_critic_grad_steps=1 train.batch_size=32 train.layer_size=256 train.num_layers=2 train.buffer_size=20000"
G="gate.enabled=true gate.signal=ratio gate.actuator=hard_backup gate.tau=20 gate.K=3"
rm -rf $PROJ/logs/smoke_gate
python train_dsrl.py --config-path=cfg/robomimic --config-name=dsrl_can.yaml exp_id=smoke_gate $COMMON $G train.total_env_steps=1200 2>&1 | grep "\[done\]\|Error"
python train_dsrl.py --config-path=cfg/robomimic --config-name=dsrl_can.yaml exp_id=smoke_gate $COMMON $G train.total_env_steps=2000 2>&1 | grep "\[resume\]\|\[done\]\|Error"
cut -d, -f2,11,20,24,25,26 $PROJ/logs/smoke_gate/train_log.csv
python -c "import json;print(json.load(open('$PROJ/logs/smoke_gate/checkpoint/run_state.json'))['gate'])"
''')
```
합격: `qw_mean`이 닫힌 동안 음수(hard backup), `gate_open`이 0에서 1로 바뀌고 `gate_open_call`이 고정값, resume 뒤에도 1 유지, `run_state.json`의 `gate.open` true.

**본 실험 (Q 스케일 결과 뒤, 100k, 5 seed, 평가 2.5k)**
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
TAU=1.0   # 7조건의 첫 하락 시 ratio_ge_gq로 정한다
for S in 1 2 3 4 5; do
  launch can_gate_sig_s$S seed=$S gate.enabled=true gate.signal=ratio gate.actuator=hard_backup gate.tau=$TAU gate.K=50
done
```
signal 5개의 `gate_open_call` 평균 N*이 나오면 대조군: `gate.signal=clock gate.clock_calls=N*`로 `can_gate_clk_s{1..5}`. 판정은 NEXT_PHASE.md 3단계 표(sig > clk이면 능동 조절 성립).
작동기를 `alpha_hold`로 바꾸려면 `gate.actuator=alpha_hold gate.alpha_hi=<α 스윕 최적>`.

π_dp 기준선 (9/5 아침 최우선, 빈 VM 어디서든, seed당 3~5분, 500 에피소드면 약 10분):
```python
run_bash(r'''
PROJ=/content/drive/MyDrive/dsrl_project
for S in 1 2 3; do
  python scripts/eval_base_policy.py seed=$S num_evals=500 log_dir=$PROJ/logs   # --config-path 붙이면 scripts/cfg를 찾아 실패
done
cat $PROJ/logs/base_policy_eval.csv
''')
```
판정 규칙: dip 시점에 `mu_absmean`·`w_frac_sat` 급등 + `ent_coef` 아직 높음 → H2. w 얌전한데 `q_start − mc_return` 부풀어 있음 → H1. p 높은 run에서 `mu_absmean` 낮게 유지 → 비율이 H2 경로로 작동.

## 19. 9/6(일) 결과 — Q 스케일 가설 기각, α 스윕이 헤드라인, 적응형(목표 엔트로피) 검증 중

전부 로컬에서 `plot_results.py`·`alpha_timing.py`·공통 5k 격자 재계산으로 뽑은 값. 그림은 로컬 scratch(`figs2/`, `figs3/`)에 있고 Drive `$PROJ/figures/`에는 아직 안 올림.

### 19.1 Q 스케일 9 run (rs_025 / rs_2 / hardq, 100k, 2.5k 격자, 9/5 22:00~9/6 01:10)
| 조건 | 첫 하락 | 바닥 | 회복 | 최저(공통 5k 격자) | AUC 24~100k | 100k |
|---|---|---|---|---|---|---|
| rs_025 (r×0.25) | 29.9k | 34.9k | 62k | 0.08 | 0.29 | 0.38 |
| baseline (5 seed, 5k 격자) | 34.0k | 41k | 64k | 0.24 | 0.40 | 0.48 |
| rs_2 (r×2) | 28.2k | 32.4k | 39k | 0.23 | 0.44 | 0.55 |
| hardq (β=0) | 28.2k | 35.7k | 81k | 0.10 | 0.27 | 0.31 |
- **기각**: 보상 8배 범위에서 첫 하락이 격자 한 칸(2.5k) 안. hardq는 더 깊고 회복 최악. Q_W 오프셋은 dip의 원인이 아님.
- `ratio_ge_gq`가 첫 하락 순간 0.48로 일정(spread 0.09)한 이유: **auto-α의 평형점**. 세 조건 모두 40k에 `logp_mean` −6 → 0(목표 엔트로피 도달), 그 뒤 α는 Q 크기에 비례(rs_025 0.012, baseline 0.05, rs_2 0.1~0.18)하고 비는 0.5에 머문다. 비는 방아쇠가 아니라 제어기의 결과.
- 회복은 Q 스케일에 크게 반응(rs_2 39k, rs_025 62k, hardq 81k): 100k 성능이 평형 α 순서(0.012 < 0.05 < 0.15)를 따른다.
- **게이트(섹션 18)는 접음.** 전제 기각 + 작동기(hard backup) 해로움 + "풀 시점"이 없음(아래).

### 19.2 고정 α 스윕 (150k, 5k 격자) — 헤드라인
| α | n | 최저(평균곡선) | AUC 24~100k | 100k | 129k |
|---|---|---|---|---|---|
| auto (baseline) | 5 | 0.24 | 0.40 | 0.48 | 0.53 |
| 0.01 | 3 | 0.00 | 0.18 | 0.23 | 0.30 |
| 0.03 | 3 | 0.00 | 0.24 | 0.37 | 0.41 |
| 0.1 | 3 | 0.14 | 0.51 | 0.63 | 0.62 |
| **0.3** | **5** | **0.47** (기준선 0.405 위) | **0.66** | **0.79** | **0.74** |
| 1.0 | 3 | 0.03 | 0.25 | 0.26 | 0.15 |
| (비교) iql | 5 | 0.34 | 0.49 | 0.54 | 0.56 |
| (비교) mix_prefill | 3 | 0.09 | 0.68 | 0.84 | 0.86 |
- α=0.3 고정은 **dip이 없고**(평균곡선이 기준선 아래로 안 감) 데모 없이 AUC가 prefill과 같고 100k에서 baseline보다 0.3 높다. 5 seed에서도 그대로(seed별 129k 0.74/0.66/0.78/0.78/0.74).
- **메커니즘**: auto-α(목표 엔트로피 0)는 학습 시작 15k 안에 정책 엔트로피를 17 → 0으로 무너뜨린다(`logp_mean` −17@26k → −6@30k → 0@40k, `mu_absmean` 0.2 → 0.8). **그 붕괴가 dip**. α=0.3이면 엔트로피 10~14에 머물고(|mu| 0.5) 붕괴가 없다. 0.1은 엔트로피 3~8로 중간, 1.0은 너무 퍼져 학습이 안 됨.
- 표현 주의: 0.3은 Can 5점 스윕의 최적이고 **Square로 전이되지 않는다**(19.4). 주장은 "α=0.3"이 아니라 "엔트로피를 무너뜨리지 말라"이며, 처방 후보는 목표 엔트로피(19.3).

### 19.3 적응형: auto-α + 목표 엔트로피 12 (`train.target_ent=12`, can_tent12, 3 seed, 150k)
| 조건 | 최저 | AUC | 100k | 129k |
|---|---|---|---|---|
| tent12 | 0.33 | 0.57 | 0.70 | 0.69 |
| fixalpha_03 | 0.47 | 0.66 | 0.79 | 0.74 |
- baseline·iql보다 확실히 낫지만 고정 0.3보다 못하다. α 궤적: 1.0 → 0.25(30k, 과도기 undershoot) → 엔트로피 12 도달 후 0.35~0.45로 회복 → 125k에 0.17. 34k의 dip(0.33)은 그 과도기.
- 그래서 오늘 `can_tent12i`: auto-α **초기값 0.3**(`train.ent_coef=auto_0.3`, SB3 문법, fingerprint는 5df31e6에서 문자열 허용) + 목표 12. 고정 0.3과 같아지면 Can 쪽 적응형 완성.

### 19.4 Square에 고정 0.3 (square_fixalpha_03, 3 seed, 150k) — 전이 실패
| 조건 | 최저 | AUC 37~100k | 100k | 127k |
|---|---|---|---|---|
| square_baseline | 0.21 @42k | 0.40 | 0.44 | 0.47 |
| square_iql | 0.37 | 0.44 | 0.48 | 0.56 |
| square_mix_prefill | 0.03 @42k | 0.43 | 0.63 | 0.62 |
| square_fixalpha_03 | 0.33 | 0.42 | 0.39 | **0.34** (0.09 / 0.32 / 0.60) |
- dip은 얕아지지만(0.33 vs 0.21) 80k 이후 무너진다. 이유: Square는 Q 스케일이 Can의 5배(Q_W 300~700 vs 30~100, `gq_norm` 4~5 vs 2)라 **auto-α가 스스로 0.2~0.3에 앉는다**(0.55 → 0.12@40k → 0.18~0.3). 즉 Square에서 0.3은 높은 α가 아니고, 엔트로피는 0 아래로 무너진다(seed 1 `logp_mean` +6.6, |mu| 0.96, 성공률 0.09). auto-α는 붕괴 시 α를 올려 엔트로피 0을 지키는데 고정 α는 그걸 못 한다.
- 결론: **숫자(α)는 전이 안 됨, 규칙(엔트로피 유지)은 전이 가능성 있음** → 오늘 `square_tent12`.

### 19.5 9/6 10:30 시작, 15:30 완료 예정 (G4, 6 run, 150k, 자동 반납 keepalive, 크레딧 80 → 약 35)
- `square_tent12_s{1,2,3}`: `train.target_ent=12`. 판정: 최저 ≥ 0.33이고 127k ≥ baseline 0.47이면 적응형 규칙이 과제 전이. 안 되면 "Can 한정"으로 보고.
- `can_tent12i_s{1,2,3}`: `train.ent_coef=auto_0.3 train.target_ent=12`. 판정: 최저 ≥ 0.45, 100k ≥ 0.75면 고정 0.3과 동급.
- 끝나면 CPU 런타임으로 zip(섹션 17 "9/6 아침 순서" 1번 셀) → 로컬:
  ```
  cd scripts
  ..\.venv\Scripts\python plot_results.py --logs <logs> --out <figs> --axes "sweep=baseline,fixalpha,fixalpha_003,fixalpha_01,fixalpha_03,fixalpha_1;adaptive=baseline,iql,mix_prefill,fixalpha_03,tent12,tent12i;square=square_baseline,square_iql,square_mix_prefill,square_fixalpha_03,square_tent12;scale=baseline,rs_025,rs_2,hardq"
  ..\.venv\Scripts\python alpha_timing.py --logs <logs> --out <figs> --groups baseline,fixalpha_03,tent12,tent12i
  ```
  공통 격자 표는 scratch의 `adaptive_check.py`(그룹 목록만 바꿔 실행).

### 19.6 포스터 패널 개정 (Can 중심, 6패널)
1. 문제: baseline dip(0.5 → 0.24@39k, 회복 84k), π_dp 기준선 0.405.
2. 정체: α 곡선 + `logp_mean` 곡선 겹침. auto-α가 엔트로피를 17 → 0으로 무너뜨리는 15k 창이 dip. Q 스케일 조작(×0.25~×2)은 시점을 못 움직임(한 줄, 기각 보고).
3. 축 A(critic 초기화): 얕게 하지만 최종 무영향.
4. 축 B(데모 리플레이): 회복 10배, 최종 0.9. linear는 자연 감쇠보다 못함. 두 축 안 쌓임.
5. **α 스윕 U자 + 고정 0.3 곡선**(dip 없음, 100k 0.79) + 적응형(목표 엔트로피 12) 곡선. "숫자가 아니라 엔트로피 유지가 처방".
6. Square: baseline dip 42k 재현. 고정 0.3은 전이 실패(후반 붕괴), 적응형 결과(오늘 오후)로 마무리.

### 19.7 커밋
93f750d alpha_timing ratio 열, 724b133 HANDOFF_CHAT, 1e62fd6·5df31e6 fingerprint `auto_<x>` 허용 + 테스트(20개 통과).

### 19.8 9/6 오후 결과 — 적응형 6 run (10:30~12:45)
**Can, `can_tent12i`(auto-α 초기값 0.3 + 목표 엔트로피 12), 3 seed, 공통 5k 격자**
| 조건 | 최저 | AUC 24~100k | 100k | 129k (seed별) |
|---|---|---|---|---|
| baseline | 0.24 | 0.40 | 0.48 | 0.53 |
| fixalpha_03 (n=5) | 0.47 | 0.66 | 0.79 | 0.74 (0.74/0.66/0.78/0.78/0.74) |
| tent12 | 0.33 | 0.57 | 0.70 | 0.69 (0.68/0.64/0.77) |
| **tent12i** | **0.46** (기준선 위) | 0.60 | 0.68 | 0.70 (0.68/0.61/0.81) |
- 초기값 0.3으로 두면 **dip이 사라진다**(평균곡선 29k 0.53, 34k 0.46, 기준선 0.405 아래로 안 감). 뒤는 고정 0.3보다 조금 낮지만 SE 안(129k 0.70±0.06 vs 0.74±0.02). α 궤적: 0.14@26k(초기 2.5k grad step 동안 감쇠) → 0.22@35k → 0.3~0.49@50~80k → 0.2~0.3@125k, 엔트로피 11~12 유지, |mu| 0.45.
- 결론(Can): "엔트로피를 12 근처에 유지"가 처방이고, 고정 α든 목표 엔트로피든 dip을 없앤다. auto-α의 초기 1.0은 불필요하게 높고, 그 감쇠 과도기가 tent12의 34k dip이었다.

**Square, `square_tent12`, 3 seed, 공통 5k 격자(37k~)**
| 조건 | 37k / 42k / 47k / 52k | AUC 37~100k | 100k | 127k (seed별) |
|---|---|---|---|---|
| square_baseline | 0.24 / **0.21** / 0.34 / 0.32 | 0.40 | 0.44 | 0.47 (0.35/0.62/0.43) |
| square_iql | 0.40 / 0.50 / 0.42 / 0.41 | 0.44 | 0.48 | 0.56 |
| square_mix_prefill | 0.30 / 0.03 / 0.14 / 0.35 | 0.43 | 0.63 | 0.62 |
| square_fixalpha_03 | 0.33 / 0.40 / 0.34 / 0.40 | 0.42 | 0.39 | 0.34 (0.09/0.32/0.60) |
| **square_tent12** | 0.34 / **0.41** / 0.46 / 0.51 | 0.43 | 0.34 | 0.40 (0.34/0.46/0.39) |
- **42k의 dip이 사라진다**(0.41 vs 0.21, 52k에 0.51로 그 시점 최고). 그러나 87k 이후 0.32~0.40으로 처지고 127k는 baseline보다 낮다. 엔트로피 12를 지키느라 α가 0.57 → 1(50k) → 5.5(80k) → 15~17(125k)까지 올라가고 |mu|는 0.34에 머문다. Square는 후반에 더 좁은 정책이 필요하다.
- **통합 결론**: dip = auto-α(목표 0)의 엔트로피 붕괴. 엔트로피를 높게 유지하면 **두 과제 모두** dip이 사라진다. 최종 성능에 맞는 엔트로피 수준은 과제마다 다르다(Can 12 OK, Square 12는 너무 넓음). 자연스러운 다음 단계는 "초기엔 높게, 뒤엔 낮추는" 목표 엔트로피 스케줄이고, 언제 낮출지가 원래 게이트가 물었던 "센서" 질문으로 돌아온다(future work).
- 13:45 시작(G4, 150k, 약 3시간, 자동 반납): `square_fixalpha_03_s{4,5}`(n=5로), `square_tent6_s{1,2,3}`(중간 엔트로피 6). tent6이 dip 없음 + 127k ≥ 0.47이면 "적정 수준이 다를 뿐 규칙은 전이"로 마무리.
- 그림: 로컬 scratch `figs4/`(success_adaptive, success_square, success_sweep, diagnostics_*). `Downloads\dsrl_figs_0906\`에 복사.

### 19.9 선행 연구 (9/6 검색, 완전하지 않음)
- DSRL 원논문(2506.15799): dip 언급 없음, 데모 리플레이 유지, 온도 논의 없음.
- **LP-DS**(2606.01151, Simsir & Oguz 2026): DSRL의 noise가 prior 저밀도로 흘러가고 mode collapse → 섭동 크기 ‖Δ‖²에 Lagrangian trust region. Can·Square·Lift. SAC α·목표 엔트로피 언급 없음. 같은 현상(노이즈 공간 집중)을 다른 지렛대로 막음. **필수 인용.**
- SAC Flow(2509.25756): robomimic O2O에서 목표 엔트로피 0 그대로, dip 논의 없음. Latent Policy Steering(2603.05296): 증류 critic의 손실 지적, 엔트로피 없음.
- 일반 O2O: WSRL(2412.07762)·PORL(2505.16856)·OCR(2412.18855)은 분포 이동+보수적 Q, Wang·White·White(2505.00913)는 "탐색이 오프라인 정책을 덮어씀" + 성능 추정으로 탐색 점진 허용(스케줄의 인용처). Three Regimes(2510.01460): stability-plasticity.
- 엔트로피 붕괴 개념: TES-SAC(2112.02852, 목표 엔트로피 스케줄), Meta-SAC(2007.01932), AEPO(2510.08141, LLM RFT).
- 포스터 문구: "우리가 아는 한 DSRL 계열에서 dip의 원인을 SAC 온도의 엔트로피 붕괴로 지목한 연구는 없다. LP-DS는 같은 현상을 trust region으로 막는다."

### 19.10 9/6 저녁 상태 (17:10)
- Square 5 run(`square_fixalpha_03_s{4,5}`, `square_tent6_s{1,2,3}`) 17:40 완료 예정, `can_tent6_s{1,2,3}`(100k, 15:45 시작) 18:30 예상. 크레딧 19:18 소진.
- 127k(마지막 평가, 이미 기록): fixalpha_03 n=5 → 0.09/0.32/0.60/0.52/0.30 = **0.37**(baseline 0.47), 5 seed 전부 엔트로피 0 아래로 붕괴(logp +1.7~+4.8). tent6 → 0.39/0.20/0.29 = **0.29**, 42k 0.28(dip 절반), 87k 이후 3 seed 모두 0.2~0.3. tent12 0.40.
- 해석: Square에서는 엔트로피를 붙들면(6이든 12든) 후반을 잃는다. 가설: Q 스케일이 학습 내내 커져 α가 1~15까지 올라가고 그 α가 critic 타깃 보너스를 부풀림 → 진단 그림으로 확인할 것. 처방은 "초기엔 지키고 뒤에 낮추는" 목표 엔트로피 스케줄. HANDOFF_CHAT.md(17:10판)에 전체 정리.
- GCE 이사 준비: `gce/` 스크립트 푸시(b3c43d8). 사용자는 GCP 계정 활성화·GPU 할당량 요청 중.

### 19.11 9/6 18:40 — 90 run 전부 완료, 최종 읽기
- `can_tent6`(auto-α 목표 6, 100k): 최저 0.22@29k, 34k에 0.45로 회복, AUC 0.52, 100k 0.61. α는 0.16 → 0.08(Can은 Q가 작아 엔트로피 6에 α 0.08이면 충분). `square_tent6`: 42k 0.28, 127k 0.29. `square_fixalpha_03` n=5: 127k 0.36(0.09/0.32/0.60/0.52/0.30).
- **용량-반응(두 과제 동일)**: 과도기 엔트로피 바닥 0 → Can 0.24(회복 84k)/Square 0.21(62k); ~6 → 0.22(34k)/0.28(52k); ~11 → dip 없음/없음. dip은 엔트로피가 떨어지는 동안 생기고 멈추면 회복; 바닥 6은 짧은 dip, 11 이상은 없음.
- **Square 후반 붕괴의 메커니즘(진단 확인)**: 엔트로피를 붙들면 α가 올라가고(tent6 → 1.0~1.8, tent12 → 15~17) critic 타깃의 α·log π' 보너스가 γ=0.999로 누적되어 `qw_mean`이 baseline 300~600 대비 tent6 1,700~2,800, tent12 23,000~24,000으로 폭주. critic 타깃이 보상이 아니라 보너스로 채워져 후반 정책이 무너진다. Can(γ=0.99, α ≤ 0.5)에서는 안 드러남.
- 다음 설계: actor 엔트로피 목표 + critic 타깃 보너스 제거(`train.critic_entropy_scale=0`, 구현돼 있음) 또는 목표 엔트로피 스케줄. 1순위 실험 `square_tent12 + critic_entropy_scale=0`.
- 그림 최종본 `Downloads\dsrl_figs_0906\`(figs5). HANDOFF_CHAT.md 17:10판을 18:40 결과로 갱신.

## 20. 9/7(월) — AUC 정의 통일, Square 인과 고리 배치, 동료 연구 비교, CQL/Cal-QL 계획

### 20.1 AUC·시간축 정의 통일 (`plot_results.py` 90975b4)
- 발견: 표의 "AUC 24~100k"가 **두 정의의 혼합**이었다. 15절(축 A·B)은 `plot_results.py`의 0~100k 적분(step-0 행을 x=0에 둔 채 포함 → 0→29k 직선이 창의 24%), 19절(α·Square)은 scratch `adaptive_check.py`의 첫 평가(29k/37k)~100k(step 0 제외). raw CSV는 초기 평가 `env_steps=0`, 첫 학습 후 평가 29,024(Square 37,024)이고 rollout 끝(24,016/32,016)에는 평가가 없다.
- 통일안(적용됨): **online step** = 초기 평가 0, 이후 `env_steps − rollout`. early AUC = seed별 online 0~75,984(Can)/0~67,984(Square) 사다리꼴(초기 평가 포함, 마지막 평가가 cutoff에 못 미치면 마지막 값 유지) → 평균±SE. 최저는 평균곡선을 공통 5k 격자(online 5,008·k)에서 읽음. `at_env129k`, `recovery_pi_dp_at` 열 추가. 그림은 `success_<axis>.png` + early 창 `success_<axis>_early.png`(마커·선종류 구분, 선을 음영 위에, mix 축에서 `iql_linear (n=1)` 제거·`iql_prefill` 추가).
- 값 변화(구→신): warmup 0.44→0.48, iql_prefill 0.56→0.60, warmupc 0.48→0.51, rs_025 0.29→0.32, hardq 0.27→0.30, mix_linear 0.62→0.64, fixalpha 0.01 0.18→0.15, 나머지 ±0.01. 순위 불변. "두 축 안 쌓임"·"linear < prefill"은 차이 0.04~0.08(SE 0.01~0.02)이므로 "prefill보다 낫지 않다"까지만. HANDOFF_CHAT §2, STATUS 표 갱신 완료. 로컬 그림 `~/Downloads/dsrl_figs_0907/`.
- 캡션: "Normalized early-online AUC over online steps 0–76k (Can) / 0–68k (Square): the initial evaluation is placed at online step 0 and each later evaluation at env steps minus the 24,016-step (32,016) initial rollout. Mean ± SE over seeds."

### 20.2 Square 인과 마지막 고리 배치 (Colab G4, 14:30 KST 시작, 150k, 7 run)
- `square_tent12_hq_s{1,2,3}`: `train.target_ent=12 train.critic_entropy_scale=0.0` (actor는 엔트로피 12 유지, critic 타깃의 α·log π′ 보너스 제거). `square_tent12_s{4,5}`, `square_baseline_s{4,5}`: 비교군·기준군 n=5. 자동 반납 keepalive. 완료 예상 19:30~20:00 KST. 결과는 Drive `$PROJ/logs/`.
- 판정(사전 고정): 42k 평균 ≥ 0.33 **그리고** 127k ≥ baseline(0.47, n=5로 갱신) → 인과 닫힘, 포스터 6번 패널에 처방 곡선 추가. dip 없음 + 127k < 0.47 → Q_W 폭주는 상관, 6번 패널을 "관찰"로 낮춤. 42k dip 복귀 → hard backup이 초반을 해침(Can hardq 방향), 처방 기각. 진단: `qw_mean`이 300~600에 머무는지(tent12 23,000), 엔트로피 12 유지되는지.
- 분석: `.venv/bin/python scripts/plot_results.py --logs ~/Downloads/logs --out <figs> --axes "square=square_baseline,square_tent12,square_tent12_hq,square_iql,square_mix_prefill"`.
- 이 실험의 위치: 헤드라인(dip = 엔트로피 붕괴, 두 과제 용량-반응)은 이 결과와 무관. 6번 패널만 바뀐다. 3 seed·Square 127k 편차(0.09~0.60)라 어느 쪽이든 "시사적"까지.

### 20.3 동료 연구(Cal-QL 사전학습, 자체 2D 과제)와의 비교 요점
- 동료: Q_A를 오프라인에서 ② plain TD / ③ CQL / ④ Cal-QL(max(Q, G_t) 바닥)로 사전학습, 온라인 코드 불변, 지표 T50/T80(시행 횟수), 3회 반복(매번 데모·π_dp 새로), 8차원 노이즈, G∈[0,1]. 결과: Cal-QL T50 2.7배 단축, plain 무효, CQL 3배 느림(Q −9 눈금 붕괴). 메커니즘 주장: Q_A 과대평가(+0.71) → Q_W → π_W.
- 우리와의 관계: (1) 동료의 질문은 우리의 **출발 가설**(critic 초기화)과 같고, 우리는 Can에서 critic 사전학습(iql·warmupc)이 dip을 얕게만 하고 시점·최종을 못 바꾼다고 결론 → 원인을 auto-α 엔트로피 붕괴로 옮김. (2) 동료 글에는 SAC 온도·목표 엔트로피·엔트로피가 변수로 없고(글의 α=1.0은 CQL 계수), dip(29% 아래 하락) 여부도 보고 안 됨(T50은 단조 상승과 dip을 구분 못 함). (3) 동료 ①(논문 그대로, 데모를 리플레이에)은 우리 `baseline`이 아니라 `mix_prefill`에 대응. (4) "plain 사전학습 무효"는 두 연구 일치. (5) 과대평가 주장은 우리 진단(`q_start − mc_return`은 α가 작아진 뒤에만 유효; 초기 Q_W는 엔트로피 보너스 오프셋 +55~+175로 덮임; 보상 ×0.25~×2·hard backup이 첫 하락 시점을 못 움직임)과 직접 맞대기 어려움 — 동료의 "critic 오차"가 soft Q를 어떻게 다뤘는지 불명. 워크플로 교차검증은 세션 한도로 미완(초안만).

### 20.4 축 A 확장: CQL / Cal-QL 사전학습 (**구현 완료 5bf75dc**, 아래 체크리스트는 구현에 반영됨. 실행 셀은 이 절 끝)
- 넣을 자리: `offline_pretrain.py` `run_iql` 옆에 `run_cql`(Cal-QL은 플래그) → 기존 `run_distill`로 Q_W 증류 → 같은 payload(critic, critic_target, critic_noise). `main`의 `method in ("iql","warmup")`, `o2o_utils.VARIANTS`, `check_pretrain_meta`(method == variant)에 `cql`, `calql` 등록. `train_dsrl.py`는 그대로.
- **"안 해 본 행동"은 반드시 π_dp를 통과**: w ~ N(0,I)(및 π_W) → `model.diffusion_policy(obs, w.reshape(-1, act_steps, action_dim), return_numpy=False)` → Q_A(s, a). `o2o_utils.py:176-181` 방식 그대로. w를 Q_A에 직접 넣으면 조용히 틀린다.
- **G_t가 데이터에 없다**: `can_train_offline.npz` 키는 states/states_next/actions/rewards/terminals/quality뿐. `make_offline_chunks.py` `build_chunks`의 궤적 루프에서 청크 단위 return-to-go `returns`를 추가로 저장해야 한다(stride 1, 청크 보상 = 4스텝 합 − 4·reward_offset ∈ [−4, 0], 할인은 **청크당 γ=0.99**(SB3 gamma, online과 동일; 스텝당 0.99⁴가 아님)). 데모는 전부 성공이라 G ∈ 약 [−200, 0]. Drive의 npz를 다시 만들어 올려야 함(`--check_against` 검증 포함). 동료의 G∈[0,1]·α=1.0 값은 그대로 쓰면 안 된다.
- **soft Q 문제(핵심 주의)**: 온라인 critic 타깃은 `r + γ(Q̄ − α·log π′)`이고 α는 1.0에서 시작한다. Cal-QL이 hard return에 맞춘 눈금은 25k 안에 엔트로피 보너스(+55~+175)에 덮인다(15절 iql: Q_W −145 → +70). 따라서 조건은 **{cql, calql} × {auto-α(논문), tent12i 또는 고정 0.3}** 으로 짜야 "보정이 살아남는 온도"에서의 효과를 볼 수 있다. auto-α 단독에서 효과가 없으면 그것이 곧 결과다(Can에서 동료 주장의 경계).
- CQL 계수 α_cql는 Q 스케일(≈ −100)에 맞춰 스윕 필요(동료 1.0은 Q∈[0,1] 기준). 로그 `q_mean`(데이터 행동)과 π_dp 샘플 행동의 Q 평균을 같이 찍어 눈금 붕괴(동료 ③의 −9)를 감시.
- 사전학습은 시뮬레이터 없이 GPU만 필요(`SpacesOnlyEnv`), 50k step. 사전학습 .pt 하나를 온라인 seed들이 공유.

**구현(5bf75dc)**: `pretrain.method=cql|calql`, `o2o_utils.VARIANTS`에 등록, `check_pretrain_meta`는 method==variant 그대로. 타깃 `r + γ Q̄_A(s′, π_dp(s′, w′))`, w′~N(0,I)(actor 무관). 벌점 `cql_alpha·(E_w Q_A(s, π_dp(s,w)) − Q_A(s, a_data))`, 노이즈 `cql_n_samples`개/상태, 반드시 π_dp 통과. calql은 벌점 안의 표본 Q를 `max(Q, G)`로(G = `returns`). 둘 다 `distill_steps`(25k)로 Q_W 증류, actor 저장 안 함(`actor_steps>0`이면 거부). 설정 기본값 `cql_alpha 5.0`(Q≈−100 눈금; 동료의 1.0은 Q∈[0,1] 기준), `cql_n_samples 4`, `cql_noise_clip 0`(N(0,I) 그대로, `update_noise_critic`과 동일; >0이면 ±clip). `make_offline_chunks.py --gamma 0.99`가 `returns`·`returns_gamma`를 씀(청크당 γ, 궤적 끝 자투리는 마지막 부분 청크). calql은 `returns` 없거나 γ 불일치면 시작 전 거부. 사전학습 로그에 `q_mean`(데이터 행동), `q_ood_mean`(표본 행동), `penalty`, `floored_frac`(바닥이 대신한 표본 비율). 테스트 44개 통과(`scripts/test_offline_pretrain.py` 7개 신규; 로컬 `.venv`에 torch CPU 설치됨).

**실행 (Colab, 환경 복원 뒤 `git pull origin o2o`)**
```bash
%%bash
# 1) 청크 npz를 returns 포함으로 다시 만든다 (hdf5는 $PROJ/robomimic_raw/ 아래). 상태 비트 일치(--check_against)까지 확인.
source /usr/local/etc/profile.d/conda.sh && conda activate dsrl && source /content/env.sh && cd /content/dsrl
PROJ=/content/drive/MyDrive/dsrl_project
H5=$(find $PROJ/robomimic_raw -name "low_dim_v141.hdf5" -path "*can*" | head -n 1); echo "hdf5: $H5"
python scripts/make_offline_chunks.py --load_path "$H5" \
  --normalization_path dppo/log/robomimic/can/normalization.npz --check_against dppo/log/robomimic/can/train.npz \
  --gamma 0.99 --save_path $PROJ/offline/can_train_offline.npz
python -c "import numpy as np; d=np.load('$PROJ/offline/can_train_offline.npz'); print(sorted(d.files), d['returns'].min(), d['returns'].mean(), float(d['returns_gamma']))"
```
기대: 키에 `returns`, `returns_gamma`가 있고 returns가 약 [−200, 0], 평균 −100 근처. (기존 run은 이 파일의 states/actions/rewards/terminals만 읽으므로 덮어써도 무방.)
```bash
%%bash
# 2) 스모크 (각 1~2분): 200 step 사전학습 + 100 step 증류가 [done]까지 가는지, calql의 floored_frac이 0과 1 사이인지
source /usr/local/etc/profile.d/conda.sh && conda activate dsrl && source /content/env.sh && cd /content/dsrl
PROJ=/content/drive/MyDrive/dsrl_project
for M in cql calql; do
  python offline_pretrain.py --config-path=cfg/robomimic --config-name=dsrl_can.yaml pretrain.method=$M seed=0 \
    pretrain.steps=200 pretrain.distill_steps=100 pretrain.log_every=50 pretrain.out_path=$PROJ/logs/pretrain/smoke_$M.pt \
    offline_data_path=$PROJ/offline/can_train_offline.npz log_dir=$PROJ/logs 2>&1 | grep "\[cql\]\|\[calql\]\|\[distill\]\|\[done\]\|Error\|Traceback"
done
```
```bash
%%bash
# 3) 본 사전학습: seed 1..3 × {cql, calql}, 50k + 증류 25k. 시뮬레이터 불필요. 스텝당 π_dp 호출이 5회(타깃 1 + 표본 4)라 warmup보다 느림.
source /usr/local/etc/profile.d/conda.sh && conda activate dsrl && source /content/env.sh && cd /content/dsrl
PROJ=/content/drive/MyDrive/dsrl_project; mkdir -p $PROJ/logs/pretrain
for SEED in 1 2 3; do
  nohup bash -c "for METHOD in cql calql; do
    python offline_pretrain.py --config-path=cfg/robomimic --config-name=dsrl_can.yaml pretrain.method=\$METHOD seed=$SEED \
      offline_data_path=$PROJ/offline/can_train_offline.npz log_dir=$PROJ/logs; done" > $PROJ/logs/pretrain_cql_s${SEED}.out 2>&1 &
done
```
결과 `$PROJ/logs/pretrain/{cql,calql}_can_s{1,2,3}.pt` + `_log.csv`. 판단: cql의 `q_ood_mean`이 `q_mean`보다 훨씬 아래로 계속 내려가며 둘 다 음수로 폭주하면 동료 ③의 눈금 붕괴(→ `cql_alpha` 낮춰 재실행); calql은 `floored_frac`이 0.2~0.8 사이에서 안정돼야 정상.
```bash
%%bash
# 4) 온라인 (축 A 통제: offline_mix.mode=none, load_offline_data=False = 기본값, actor·α 로드 없음). 150k, 5k 격자.
source /usr/local/etc/profile.d/conda.sh && conda activate dsrl && source /content/env.sh && cd /content/dsrl
PROJ=/content/drive/MyDrive/dsrl_project
CFG="--config-path=cfg/robomimic --config-name=dsrl_can.yaml"
COMMON="log_dir=$PROJ/logs train.total_env_steps=150000 offline_mix.mode=none load_offline_data=False"
launch () { EXP=$1; shift; nohup python train_dsrl.py $CFG exp_id=$EXP "$@" $COMMON > $PROJ/logs/$EXP.out 2>&1 & echo "started $EXP (pid $!)"; }
for S in 1 2 3; do
  launch can_cql_s$S   seed=$S variant=cql   pretrain_path=$PROJ/logs/pretrain/cql_can_s$S.pt
  launch can_calql_s$S seed=$S variant=calql pretrain_path=$PROJ/logs/pretrain/calql_can_s$S.pt
done
```
확인: `.out`에 `[pretrain] cql: loaded critic, critic_target, critic_noise from ...`(actor 없음), `[eval] env_steps=0`. 분석: `plot_results.py --axes "critic=baseline,iql,cql,calql,warmupc"`. 비교 기준(online AUC 0~76k): baseline 0.41, iql 0.49, warmupc 0.51. **권장 확장**: 같은 .pt로 `train.ent_coef=auto_0.3 train.target_ent=12`(tent12i)를 붙인 `can_calql_t12i_s{1,2,3}`도 띄워 "보정이 살아남는 온도"에서의 효과를 본다(§20.4 soft Q 주의).

**17:50~18:10 갱신 — 설계 실패 두 번과 현재 설계 (`CQL_NOTES_2026-09-07.md`에 로그와 함께 상세)**
- 시도 1(타깃 `r+γQ̄(s′,π_dp(s′,w′))` + 평균형 벌점): 벌점이 부트스트랩 분포와 같아 1/(1−γ)로 증폭 → Can cql Q −9,500(수익 −100). 시도 2(IQL V 타깃 + 평균형 벌점): 기울기가 상수라 Q_data가 (r+α)/(1−γ)=+100으로, Q_ood가 −8,000으로 → 자기 제한 없음. **시도 3(현재, 15a36ea)**: IQL V 타깃 + `α(logsumexp{Q_ood_1..4, Q_data} − Q_data)`(데이터 행동을 집합에 넣어 ≥0·자기 제한) + Cal-QL 바닥 `max(Q_ood, G)`.
- **이름·표기(GPT 검토 반영)**: 코드 키는 `cql`/`calql`이지만 정확히는 "IQL backup + anchored CQL(H)-style regularizer / + Cal-QL floor", 즉 **actor-free DSRL에 맞춘 CQL-style pretraining**. 포스터·문서에 이 한 줄을 반드시 밝힌다. `calql_prefill`은 "Cal-QL-pretrained critic + demo replay"이지 온라인에서도 regularizer를 쓰는 full Cal-QL이 아니다. `td`(59de4d9)는 prior 정책 FQE(`r+γQ̄(s′,π_dp(s′,w′))`, 벌점 없음) = 동료의 plain TD 칸.
- **온라인 전 오프라인 통과 기준** (`scripts/check_pretrain.py pretrain_path=<.pt>`, 4,096 상태 × 4 노이즈): `Q_data − G`가 G 눈금의 수십 이내, calibration gap `E_wQ_ood − G` ≈ 0 또는 양수(calql), `Q_ood − Q_data` 음수이되 같은 자릿수(cql). **Q_data > 0 또는 Q_ood < 10×G면 폐기**하고 α만 낮춰 1-seed 오프라인 스윕(Can α∈{0.1,0.5,1,5}, Square α∈{0.01,0.1,0.5,1}). 목적함수는 더 바꾸지 않는다. `floored_frac≈1`인데 raw Q_ood ≪ G면 "바닥은 작동, critic은 미교정"으로 기록.
- **우선순위**: ① Can td/cql/calql(리플레이 없음, 3 seed) ② Square hq 인과 ③ calql_t12i ④ calql_prefill ⑤ Square cql/calql은 오프라인 기준 통과 시에만 온라인.
- **반증 조건("Can에서는 critic보다 온도가 지배")**: 같은 auto-α에서 calql 단독이 엔트로피 붕괴가 그대로인데도 dip을 완전히 없앰; calibration gap이 엔트로피 지표보다 seed별 dip 깊이·시점을 더 잘 예측; calql 단독이 calql_t12i와 같은 성능; 잘 교정된 critic에서 엔트로피 개입 효과가 사라짐. 반대로 calql이 iql처럼 dip을 얕게만 하고 calql_t12i에서만 dip이 없어지면 "교정은 출발점을, 엔트로피 붕괴가 전이의 방아쇠를" 구조.
- 벌점 분포 = 증류 분포(N(0,I))는 버그가 아니라 목적에 부합(온라인 Q_W가 읽는 영역을 보수적으로). 순위 정보는 TD/IQL과 일반화에서 오고 CQL은 낮추기·Cal-QL은 그 한계만. 균등 상자 표본은 오늘 넣지 않음.

### 20.5 19:40 상태 — 사전학습 18개 완료, hq 인과 닫힘, 온라인 24 run 띄우는 중, 다른 기기에서 이어가기
- **Colab 세션 3개가 19:2x에 모두 끊김**(계정 한도 아님, 원인 불명). 잃은 것 없음: `$PROJ/logs/pretrain/{td,cql,calql}_{can,square}_s{1,2,3}.pt` 18개 완성(Can 18:58, Square 19:09, 증류 25k까지), hq 7 run 전부 `[done]`(127k가 마지막 평가).
- **hq 판정(127k)**: `square_tent12_hq` 0.45/0.51/0.52 = **0.49** vs baseline n=5 **0.40**(0.35/0.62/0.43/0.34/0.28) vs tent12 n=5 **0.37**(0.34/0.46/0.39/0.32/0.33). 42k는 0.33(판정선). 사전 기준(42k ≥ 0.33 그리고 127k ≥ 0.47) 충족 → **critic 타깃에서 엔트로피 보너스를 빼면 Square 후반 붕괴가 사라지고 baseline 위**. 인과 고리 닫힘(3 seed, 42k 경계 → "시사적"). 포스터 6번 패널.
- 사전학습 최종 로그(시도 3): Can `q_data` −90~−100(G −100), cql `q_ood` −145~−150, calql −135~−140(바닥 0.88); td `q ≈ q_ood ≈ −120`. Square td −170(G −150), cql −140/−150, calql −103/−110. 드리프트 없음. **prior 정책 가치 < G**(Can −20, Square −15): Cal-QL 바닥의 전제(정책 ≥ 행동정책)가 여기선 안 묶임.
- **온라인(새 VM 3대, 19:40~)**: VM1 = Can `can_{td,cql,calql}_s{1,2,3}` 150k(9 run, ≈01:50); VM2 = `can_calql_t12i_s{1,2,3}` + `can_calql_prefill_s{1,2,3}` 150k(≈00:00); VM3 = Square `square_{td,cql,calql}_s{1,2,3}` 100k(≈00:00). 셀은 `colab/vm2_can_td_cql.ipynb` 14번(`METHODS="td cql calql"`), `vm1_followup.ipynb` 3번, `vm3_square_cql.ipynb` 14번. keepalive가 끝나면 자동 반납.
- `scripts/check_pretrain.py`는 `--config-path`를 **주지 말 것**(상대경로가 scripts/ 기준으로 풀려 실패). 포스터 진단 숫자용이라 온라인 시작을 막지 않음.
- **포스터**: `poster/build_poster.py`(템플릿 PPTX를 채움, 와이어프레임 PNG 동반). 초안 v0는 채팅으로 전달(템플릿·초안 PPTX는 로고가 있어 저장소에 넣지 않음; 템플릿은 심포지엄 배포본, 초안은 스크립트로 재생성). 1막(critic 교정) 좌측·2막(엔트로피) 우측, `[tonight]` 자리 = 오늘 밤 결과.
- **다른 기기에서 이어가기**: 코드·문서·노트북은 전부 `origin/o2o`에 있고 결과는 Drive. 새 기기에서 `git clone -b o2o https://github.com/msp0617/dsrl.git` → 로컬 분석 환경 `python -m venv .venv && .venv/bin/pip install numpy pandas matplotlib torch h5py gymnasium nbformat python-pptx pymupdf pypdf` → Drive의 `csv_bundle.zip`을 `~/Downloads/logs/`에 풀기 → `HANDOFF.md` 20절부터. Claude Code 세션은 https://claude.ai/code/session_01PLS5zTNjh2cG5DtVqSPEy4 (웹에서 열람·Remote Control로 이어가기 가능한 경우) 또는 새 세션에 "HANDOFF.md 20.5부터 이어서".

### 20.6 기타
- Colab: G4(48 vCPU/176GB)는 Pro+에서만 보임. 이 워크로드는 RAM(run당 17GB)·CPU 바운드라 A100(83GB, 12 vCPU)은 유닛당 처리량이 절반 이하 → G4 배치를 돌리는 달에는 Pro+, 아니면 GCE.
- `STATUS_2026-09-07.md`: 팀 공유용 진행상황 보고서(초안, AUC는 갱신됨). 수치 검증 워크플로는 세션 한도로 미완 — 배포 전에 §2 표와 한 번 더 대조할 것.
- 로컬(macOS) 분석 환경: `.venv`(numpy/pandas/matplotlib), CSV는 `~/Downloads/logs/`(csv_bundle.zip 전개본, 98 run).

### 20.7 20:32 Drive 전수검사 — 온라인 24개는 실패한 것이 아니라 생성되지 않음
- `scripts/inspect_runs.py`로 Drive를 직접 조사한 결과 예정했던 `can_{td,cql,calql}_s{1,2,3}`, `can_calql_{t12i,prefill}_s{1,2,3}`, `square_{td,cql,calql}_s{1,2,3}`가 **24개 모두 없음**. launch 함수는 redirection으로 `.out`을 먼저 만들므로, `td` meta mismatch로 즉시 죽었더라도 흔적은 있어야 한다. 따라서 현재 판정은 **resume 0 / 처음부터 24 / 실패 run 폐기 0**이며, 19:40의 “띄우는 중”은 실제 launch 완료 상태가 아니었다.
- `td/cql/calql × {Can,Square} × seed 1..3` 사전학습 18개는 `.pt`와 `[done]`을 모두 확인했으므로 다시 돌리지 않는다. 유일한 `ERR`은 9/3의 오래된 `pretrain_test.out`이고 이번 배치와 무관하다.
- 기존 97개 완료 run이 검사기에서 `stale`로 잘못 보였다. 원인은 `[done]`을 `.out`의 물리적 마지막 5줄에서만 찾은 것. 검사기는 이제 마지막 유의미한 lifecycle event를 사용하고 TensorBoard 컨테이너 `logs/robomimic-dsrl/`도 run으로 세지 않는다.
- `td` mismatch 하나로 세 VM의 동시 종료를 설명할 수 없다. Can/Square의 cql·calql 프로세스는 남아 있어야 하고 t12i/prefill VM은 td와 무관하다. 세 VM이 함께 사라졌다면 compute unit 소진 또는 외부 종료 쪽이 더 일관된다. `write`는 마지막 파일 기록 시각이지 VM 사망 시각의 직접 증거는 아니다.
- 체크포인트 간격 25k는 **온라인 학습 추가분** 기준이다. 초기 rollout을 포함한 raw `env_steps`에서 첫 체크포인트는 Can 약 49,024, Square 약 57,024다. 그 전 `ckpt none`은 처음부터다. `[mrr]`은 세 파일의 존재를 뜻할 뿐이며, 실제 resume 때 손상 검사가 실패하면 이전 슬롯로 fallback한다. 유효 checkpoint 후보가 있으면 초기 critic이 이미 checkpoint 안에 있으므로 원래 pretrain `.pt`의 존재를 다시 강제하지 않도록 수정했다(처음 시작할 때만 필요).
- 재개 우선순위(hq 완료 반영): **Can 9 → calql_t12i 3 → calql_prefill 3 → Square 9**. 열린 Colab 사본의 셀 본문은 셀 안 `git pull`로 갱신되지 않으므로 최신 GitHub 노트북을 새로 열거나 온라인 셀이 실제로 `variant=$M`인지 확인한다.

### 20.8 혼동 방지용 신규 VM 실행판
- 기존 다목적 노트북 대신 `colab/vm1_new.ipynb`, `colab/vm2_new.ipynb`, `colab/vm3_new.ipynb`를 사용한다. 각 노트북은 0→9 순서의 실행 전용판이며 1번의 condacolab 재시작 뒤 2번부터 계속한다.
- VM1은 Can td/cql/calql 9개, VM2는 Can calql_t12i/prefill 6개, VM3은 Square td/cql/calql 9개만 띄운다. 모든 td launch는 `variant=td`이고 중복 프로세스는 다시 띄우지 않으며, 8번 자동 점검 뒤 9번 keepalive가 마지막 프로세스 종료 시 VM을 반납한다.
- VM3 6번은 Square 9개 메타와 4,096-state 오프라인 진단을 전수 출력한다. 진단의 REJECT는 기록하되 온라인 시작을 막지는 않고, 검사 명령 자체가 실패한 경우에만 셀이 실패한다.
## 21. 9/12(토) — 발표 뒤: 남은 질문 둘로 좁힘, `colab/vm_hq_can.ipynb`

- **발표(9/11) 피드백**: "critic을 왜 사전학습하나"에 답을 못 함; "RL은 원래 어렵다, 수맥부터 확인하라". 답은 저장소에 있었다: DSRL-NA에서 데모가 온라인 학습기에 닿는 통로는 critic뿐(w 라벨 없음·π_dp 고정 → π_W 지도학습 불가), 거의 공짜(시뮬레이터 불필요, seed 공유), O2O 문헌의 표준 처방(Cal-QL). 재보니 dip을 얕게 하고(IQL 0.24→0.34, warmupc 회복 84k→42k) 최종은 못 바꾸며, 눈금은 soft target(α=1, +18/청크)이 25k 안에 지운다(IQL Q_W −145 → +40@29k). 포스터 Objective가 "왜"를 말하지 않고 바로 사다리로 갔던 것이 맞은 이유. 다음 발표: 30초 답("유일한 오프라인 통로·거의 공짜·dip 얕게·눈금은 지워짐·방아쇠는 온도")과 주장 등급표(해결/시사/열림).
- **문헌 대조로 고친 표현**: "Cal-QL 바닥이 안 묶인다" → Can floor 사용률 97%(9/8 표), 깨진 건 전제(정책 ≥ 행동정책; prior 가치 < G). WSRL(Zhou, Peng, Li, Levine, Kumar, ICLR 2025)은 calibration이 아니라 warm-up + 앙상블이고 dip 일부는 불가피하다고 인정. RLPD(Ball, Smith, Kostrikov, Levine)는 "up to 2.5x", 앙상블·UTD 20 포함. LP-DS는 "완화"(ICML 2026). DSRL 원논문 실제 로봇 곡선(Fig. 7, 점당 10 rollout)은 3과제 중 2개가 초반에 BC 아래 — 저자 언급 없음.
- **seed 예산**: per-seed SD 0.1에서 d=0.1을 80% power로 → arm당 ~16. Can early AUC(SD 0.035~0.06) 3~6, Square 127k(SD ~0.13) ~25. Square 주장은 n을 늘려도 "시사적".
- **9/8 재분석 반영(`results/2026-09-08/README.md`)**: Square hq는 "닫힘"이 아니라 "지지"(tent12 대비 후반 +0.093 ± 0.020, 3/3; 초기 AUC −0.050 ± 0.040; 42k 0.33 < π_dp 0.494로 dip 잔존; seed-matched baseline 대비 후반 +0.018 ± 0.067). "두 과제 dip 제거"는 철회(Square π_dp 0.494). Cal-QL 단독·조합의 일관된 추가 이득 없음.
- **남은 질문 둘 → `colab/vm_hq_can.ipynb`** (`vm2_new` 판형 0→9, 코드 변경 없음, 6 run이면 G4 한 대 5~6시간):
  - 질문 1 (7번) `can_tent12i_hq_s{1,2,3}`: 엔트로피를 붙든 상태에서 β=0이 Can에서 무해한가. 상호작용 가설 명시(`can_hardq` β=0·목표 0은 AUC 0.298로 나빴음). 무해 = 최저 ≥ 0.405 그리고 129k ≥ 0.53 → 두 과제 공통 처방; 해로움 → hard target의 초반 비용 → β 스케줄(dip 창 soft, 이후 hard) 구현 근거. 그 전엔 스케줄을 구현하지 않는다.
  - 질문 2 (7번, 같은 셀) `can_prefill_t12i_s{1,2,3}`: prefill(129k 0.858, 최저 0.093) + t12i(최저 0.460, 129k 0.697)를 합치면 둘 다 얻는가. 둘 다 = 최저 ≥ 0.405 그리고 129k ≥ 0.80; 부분 = 후반 < 0.80(데모가 있어도 엔트로피 유지의 후반 비용); 실패 = 최저 < 0.405(prefill의 dip은 엔트로피 붕괴가 아님 → 엔트로피 설명의 경계).
  - 10번(선택) `square_tent12_hq_s{4,5}`: n=5 정렬용, 초기 약점은 못 없앰(README §9). 우선순위 낮음.
- 안 하는 것: Lift/Transport, 300k 연장, Cal-QL+hq, 목표 엔트로피 스케줄(미구현; 질문 1 결과 뒤 판단), LP-DS 비교.

## 22. 9/12(토) — vm_hq_can 결과: Q1 해로움, Q2 실패(사전 기준 그대로). 그림·표 `results/2026-09-12/`

숫자는 `scripts/plot_results.py`(9/7 정의) 값이며, 원본 CSV에서 두 번 독립 재계산해 ±0.005 안에서 일치(hardq만 격자 2,512 → 선형보간 기준 0.298 ± 0.025를 쓴다). 새 6 run은 `~/Downloads/csv_bundle (1).zip` → `~/Downloads/logs/`(136 run).

- **Q1 `can_tent12i_hq`(n=3)**: 최저 0.18@online 5k(seed 0.23/0.18/0.09) < 0.405, 129k 0.460 ± 0.048 < 0.53 → **해로움**. tent12i 대비 seed-matched 초기 AUC −0.204 ± 0.062(3/3, paired p 0.08), 129k −0.237 ± 0.105(3/3), 비용은 전 구간(0–20k −0.24, 20–50k −0.18, 50–76k −0.20; 격자 17점 모두 아래). baseline s1–3 대비는 −0.04 ± 0.04 / −0.01 ± 0.03 → **"baseline으로 되돌림"이지 "baseline 아래"가 아님**. hardq−baseline −0.134 ± 0.048(3/3)이 같은 방향의 독립 추정, tent12i_hq−hardq +0.093 ± 0.007(엔트로피 붕괴 비용만 회수).
- Q1 해석: hard target의 비용은 두 과제 공통(Square −0.05, Can −0.20), 후반 이득은 Square만. Can도 soft Q_W가 +255까지 부풀지만(α 최대 0.42 뒤 감소) Square는 폭주(α → 20, Q_W → 3e4) — 구분은 "부풀 때"가 아니라 **"폭주할 때"**. 미해결 교란: hard critic은 actor의 Q-gradient가 2–3배 작음(`gq_norm` 0.9–1.2 vs 2.6–3.0) → "나쁜 교사"와 "조용한 교사"를 못 가름(actor_lr ×2–3 1 run이면 분리). **β 스케줄은 구현하지 않는다**(Can에서 모든 구간 뒤짐).
- **Q2 `can_prefill_t12i`(n=3)**: 최저 0.097@5k(0.02/0.12/0.15) < 0.405 → **실패**(글자대로); 129k 0.870 ± 0.073, T80 3/3@20k, 초기 AUC 0.776 ± 0.007(Can 최고; mix_prefill 대비 +0.095 ± 0.024, 3/3, 5k 점을 빼면 +0.140). **dip은 제거가 아니라 이동**: 10–15k → 5–10k, 깊이 같음(0.093 vs 0.097), π_dp 아래 격자 2점 같음.
- Q2 해석: 5k dip은 auto-α 시작 과도(env 28.8k: α 0.196 → 0.099, 엔트로피 12 → 8–10 undershoot, |μ| 0.1 → 0.6; 평가 200 step 전) 3/3과 일치. 그러나 (i) 같은 과도가 tent12i에선 2/3 seed에서 dip 없음, (ii) **고정 α 0.1(과도 없음)도 5k에 0.08/0.34/0.01** — 즉 원인 변수는 "과도"가 아니라 **평가 직전의 α/엔트로피 수준(α≈0.1, H≈7–10, |μ|≈0.6이면 9/9 run이 ≤ 0.34)**이고 데모 90% 배치가 그 비용을 키운다. (iii) mix_prefill의 10–15k dip도 "엔트로피 붕괴 상태"가 아니라 **α 감쇠 + critic 타깃 재척도(Q_W +125 → −60, 13k step) 전이 구간**과 일치: 엔트로피는 env 34k에 0에 닿아 190k까지 0인데 성공률은 0.86까지 회복하고, calql_prefill은 같은 α/엔트로피 경로에서 dip 없음(10k 0.71/0.56/0.63). STATUS §3.4의 "dip은 엔트로피가 **떨어지는 동안**"과 일치하며, "붕괴 상태가 원인"으로 쓰면 틀림.
- 쓰면 안 되는 문장: "prefill_t12i가 둘 다 얻었다/dip을 없앴다"(실패; 최저는 Can 공동 최저), "hq는 baseline보다 나쁘다"(seed-matched 동급), "hard target에 3.7 SE의 초반 비용"(n=3 Welch, p≈0.035–0.08; 비용은 전 구간), "prefill의 dip은 엔트로피와 무관"(전이 구간과 일치; 경계가 드러난 것).
- **다음 실험(권고, 코드 변경 없음)** — 질문: "데모 배치에서 초기 α/엔트로피 수준을 0.3/13–16으로 붙들면 5k·10–15k dip이 모두 없어지는가".
  - 주 arm `can_prefill_fixa03_s{1,2,3}`: `offline_mix.mode=prefill offline_data_path=… train.ent_coef=0.3`. fixalpha_03은 과도 없음(α 0.300 전 행, 5k에 H 13–16, |μ| 0.33–0.43), Can 후반 엔트로피 13–15 유지(Square형 붕괴 없음).
  - 보조 arm `can_prefill_t12i_a015_s{1,2,3}`: `… train.ent_coef=auto_0.15 train.target_ent=12` — 과도기 뒤의 α에서 시작(undershoot 제거)해 "과도 vs 정상상태"를 가름.
  - **사전 판정은 최저 하나로 하지 않는다**(n=3에서 fixalpha_03 자체가 3-seed 부분집합 2/10에서 최저 < 0.405; "최저 ≥ 0.405 ∧ 129k ≥ 0.80"의 검정력 ≈ 0.6). 주 지표 = **online 5,008의 seed 평균**(≥ 0.405; fixalpha_03 s1–3 0.443, prefill_t12i 0.097, mix_prefill 0.78)과 seed-matched 차이; 보조 = 10–15k 평균(mix_prefill형 dip 유무), 초기 AUC(참조 0.71 = fixalpha_03 5k점 + mix_prefill 고원 / 0.78 = prefill_t12i 고원; 데모는 fixalpha_03 대비 초기 AUC를 안 올림 +0.003 ± 0.023), 129k와 104k/129k 평균 ± SE(예상 ~0.85, SE 0.05–0.07; 데모의 후반 이득은 fixalpha_03 대비 +0.133 ± 0.016로 견고). 선택: online 0–20k에 2,512 평가 격자(`eval_schedule`, hardq가 씀)로 dip 폭 해상.
