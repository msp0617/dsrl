# Compute Engine에서 Colab과 똑같이 돌리기

Colab 노트북의 경로(`/content/dsrl`, `/content/drive/MyDrive/dsrl_project`, `/usr/local/envs/dsrl`, `/content/env.sh`)를
VM에 그대로 만들어 두므로, HANDOFF의 모든 명령이 수정 없이 돈다. 노트북·keepalive는 필요 없다(VM은 유휴로 죽지 않음).

## 0. 한 번만: 계정·할당량
1. 콘솔 → "일반 계정 활성화"(무료 체험판 크레딧은 유지, 소진 뒤 과금). 결제·예산 알림 설정(₩300,000, 50/90/100%).
2. API: Compute Engine API 사용 설정.
3. IAM 및 관리자 → 할당량: `GPUs (all regions)` ≥ 1, 그리고 리전의 `NVIDIA L4 GPUs`(또는 `NVIDIA T4 GPUs`) ≥ 1 요청. 승인까지 몇 분~며칠.
4. 로컬(Windows)에 `gcloud` CLI 설치 후 `gcloud init`, `gcloud auth login`.

## 1. VM 만들기 (L4 1장, 32 vCPU, 128GB → run 6개)
```bash
gcloud compute instances create dsrl-1 \
  --zone=us-central1-a --machine-type=g2-standard-32 \
  --accelerator=type=nvidia-l4,count=1 --maintenance-policy=TERMINATE \
  --image-family=common-cu124-ubuntu-2204 --image-project=deeplearning-platform-release \
  --boot-disk-size=300GB --boot-disk-type=pd-balanced \
  --metadata=install-nvidia-driver=True
```
- Spot(60~80% 저렴, 선점되면 정지됨. 체크포인트에서 resume 가능): `--provisioning-model=SPOT --instance-termination-action=STOP` 추가.
- 서울(`asia-northeast3-a`)에도 L4가 있지만 할당량은 리전별이다. 없으면 `us-central1`.
- 더 큰 것: `g2-standard-48`(4×L4, 192GB, run 9개, 약 2배 요금). CPU 바운드 워크로드라 GPU 수는 무의미하고 vCPU·RAM만 본다.
- 첫 부팅 때 드라이버 설치 질문이 나오면 y. `nvidia-smi`가 되면 준비 끝.

접속: `gcloud compute ssh dsrl-1 --zone=us-central1-a`. 끄기: `gcloud compute instances stop dsrl-1 --zone=us-central1-a` (**정지해야 과금이 멈춘다**; 디스크는 유지).

## 2. 데이터 가져오기 (둘 중 하나)
**A. rclone (권장, 한 번 인증)** — VM에서:
```bash
git clone --recurse-submodules -b o2o https://github.com/msp0617/dsrl.git /content/dsrl 2>/dev/null || true
sudo apt-get install -y rclone && rclone config   # gce/sync_drive.sh 머리말의 순서대로. remote 이름 gdrive
bash /content/dsrl/gce/sync_drive.sh pull         # env_cache, dppo_log, offline, logs/pretrain (수 GB, 10분 안팎)
```
**B. 수동** — Drive 웹에서 `dsrl_project/env_cache/dsrl_env.tar.gz`, `dppo_log/`, `offline/`, `logs/pretrain/`을 받아
`gcloud compute scp --recurse <로컬폴더> dsrl-1:/data/dsrl_project/ --zone=us-central1-a`.

## 3. 세팅 (10분, 캐시 복원)
```bash
bash /content/dsrl/gce/setup_gce.sh          # 캐시가 없으면 --build (15분)
```
마지막에 `matmul ok True`와 `OK: robomimic-pretrain/can/...`, `OK: .../square/...`가 나와야 한다.

## 4. 실행
```bash
tmux new -s runs
cp /content/dsrl/gce/runs_example.txt ~/runs.txt && nano ~/runs.txt     # 돌릴 run 목록
bash /content/dsrl/gce/launch.sh ~/runs.txt --wait     # 띄우고, 기다리고, Drive에 CSV 올리고, VM 정지
# 다른 창에서 상태: bash /content/dsrl/gce/launch.sh --status
```
`--wait` 없이 띄우면 VM은 켜져 있으니 끝나고 직접 `stop`할 것. 선점·중단 뒤에는 같은 명령을 다시 실행하면 `[done]`이 아닌 run만 체크포인트에서 이어간다.
Colab 셀의 `run_bash(...)` 명령은 `tmux` 창에서 `cd /content/dsrl && source /usr/local/etc/profile.d/conda.sh && conda activate dsrl && source /content/env.sh` 뒤에 그대로 붙여 넣으면 된다.

## 5. 결과 받기
`sync_drive.sh push`가 `logs/<exp>/{eval_log,train_log}.csv`와 `.out`을 Drive에 올리므로, 로컬 분석은 지금처럼 Drive에서 zip을 받거나
`gcloud compute scp --recurse dsrl-1:/data/dsrl_project/logs ./logs --zone=us-central1-a`로 직접 받는다.

## 6. 비용 감각
g2-standard-32 온디맨드 약 $1.5~2/h(Spot 약 $0.5). ₩415k ≈ $300 → 온디맨드 150시간 이상. Colab G4(9 크레딧/h ≈ $0.9)보다 시간당 비싸지만 run 6~9개를 얹으면 run당 비용은 비슷하고, 유휴 종료가 없다.
