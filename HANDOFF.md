# DSRL offline-to-online 프로젝트 — 인수인계 (v3, 2026-09-04)

이 문서 하나로 새 세션(Claude Code든 claude.ai 채팅이든)이 작업을 이어갈 수 있게 쓴 것이다.
채팅에는 저장소 접근이 없으므로 **필요한 명령·숫자·판단 기준을 전부 여기에 넣었다.**
새 세션 첫 메시지: **"HANDOFF.md 붙여넣고, 섹션 2 '진행 중'부터 이어서"**.
코드 설명은 [O2O.md](O2O.md), 저장소는 https://github.com/msp0617/dsrl 브랜치 `o2o`.

---

## 0. 한 줄 요약

논문 DSRL(Wagenmaker et al. 2025, arXiv:2506.15799)의 offline-to-online 초기 성능 dip이
**critic 초기화** 때문인지, IQL로 critic을 미리 만들면 줄어드는지 robomimic Can에서 실험한다.
**2026-09-05 아침 기준: 29 run 중 26개 완료, `can_mix_linear_s{1,2,3}`만 VM 1에서 돌고 있다(12:00 KST 완료 예정).**
π_dp 기준선(0.405)과 첫 그림·`metrics.csv`가 나왔고 결과 해석은 **섹션 15**에 있다. 남은 것은 linear 반영 → 그림 재생성 → 포스터(일요일, 마감 2026-09-09 수).
잔여 크레딧 약 367. VM 2·3은 반납됨. 새 VM은 캐시 복원으로 10분(섹션 3).

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
