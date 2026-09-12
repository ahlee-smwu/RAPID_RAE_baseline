# HANDOFF — 새 Claude Code 세션용 작업 지시서

**이 문서를 읽는 너에게:** 너는 사용자의 GPU 서버에 로컬로 연결된 Claude Code
세션이다. 이 문서는 이전 세션(원격 컨테이너, GPU·데이터 없음)이 코드를 작성하고
넘긴 인수인계서다. 이전 세션은 **아무것도 실제로 실행해보지 못했다** — 합성
데이터 검증만 했다. 실제 데이터 위에서의 실행은 전부 네 몫이다.

사용자는 한국어로 소통한다. 보고도 한국어로 해라.

---

## 0. 시작하기 전에 — 반드시 먼저 할 것

### 0.1 코드를 읽어라

```bash
cd <repo>                      # RAE 레포 루트
git checkout train && git pull origin train
git log --oneline -8
```

브랜치는 **`train`**이다. `main`이나 `claude/fervent-cannon-bvpimq`에는 이 작업이
없다. 그 다음 이 순서로 읽어라:

| 파일 | 왜 |
|---|---|
| `learnable_eps/RUNBOOK.md` | 실행 절차와 정확한 명령어. 이 문서의 짝 |
| `learnable_eps/README_rapid.md` | 알고리즘·설계 근거·함정 |
| `learnable_eps/check_latents.py` | 1–2단계 도구 |
| `learnable_eps/gmm_doctor.py` | 3–5단계 도구 |
| `learnable_eps/train.py` | 학습 진입점 |

### 0.2 사용자에게 한 번에 물어볼 것

아래는 **서버에서 알아낼 수 없고 사용자만 아는 값**이다. 작업 중간에 하나씩
묻지 말고, 시작 전에 한 번에 물어라. 특히 2번은 재추출에 반드시 필요한데
`meta.json`에 저장되지 않는다.

1. **경로 확인**
   - latent 디렉터리 (예상: `/mnt/aisha/ahlee-rae`)
   - 기존 GMM pkl (예상: `gmm_imagenet_256/20_diag/gmm_clusters.pkl`)
   - ImageNet train 디렉터리
   - stage-1 가중치 `models/` 존재 여부
2. **원래 `extract_z.py`를 돌릴 때 쓴 플래그** — `--config`, `--data-path`,
   `--dtype`, `--batch-size`, `--classes-per-group`.
   `meta.json`에는 `dtype`/`world_size`/`classes_per_group`/`class_range`만
   있고 `--config`와 `--data-path`는 **없다.** 재추출하려면 이 둘이 필수다.
   (사용자 쉘 히스토리나 실행 스크립트에 남아 있을 수 있으니 먼저
   `history | grep extract_z` / `ls *.sh` 를 시도해보고, 없으면 물어라.)
3. **GPU 상황** — 5장 중 3장으로 학습한다고 했다. 나머지 2장을 누가 쓰는지,
   점유해도 되는지.
4. **디스크 여유** — `convert_gmm.py`가 약 16 GB(K=20)를 쓴다. 로컬 SSD여야
   한다. `df -h`로 확인하고 부족하면 알려라.

### 0.3 환경 확인

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
python -c "import sklearn, scipy, omegaconf, timm, transformers; print('deps ok')"
free -g                     # gmm_doctor / convert_gmm 는 큰 RAM을 쓴다
df -h <PRIOR 출력 예정 경로>
```

`scikit-learn`이 없으면 설치해라. 학습에 `scipy`는 필요 없다(OT 미사용).

---

## 1. 배경 — 무슨 일이 있었나

RAPID adaptive GMM prior를 RAE(DINOv2-B) stage-2에 이식했다. 코드는 완성되어
검증까지 끝났다(`verify_port.py` 21/21). 그런데 **학습을 돌리려다 데이터
문제가 드러났다.**

### 1.1 확인된 사실

`/mnt/aisha/ahlee-rae/group001/`에 `latents_rank002.dat`은 있는데
`labels_rank002.npy`와 `progress_rank002.json`이 없다.

`extract_z.py`의 쓰기 순서가 결정적이다:

```python
mm = np.memmap(path, mode="w+", shape=(local_n, *latent_shape))  # ① 전체 크기 선할당
for ...:                                                          # ② 인코딩 루프
    ...                                                           #    20배치마다 progress
json.dump({..., "completed": True}, ...)                          # ③ 완료 표시
np.save(label_path, labels)                                       # ④ 맨 마지막
```

→ `.dat`은 ①에서 최종 크기로 만들어지므로 **파일 크기는 아무것도 증명하지
않는다.** `.npy`가 없다 = 그 rank가 끝까지 못 갔다. `progress`조차 없으니
첫 20배치도 못 돌았다. 내용은 **0으로 채워져 있다.**

### 1.2 파생 문제 — GMM도 오염됐을 수 있다

`gmm_fit.py`의 `build_class_index`:

```python
label_path = group_dir / f"labels_rank{rank:03d}.npy"
if not label_path.exists():
    continue          # ← 경고 없이 조용히 건너뜀
```

GMM fit이 추출 실패 **이후에** 돌았다면, group001의 모든 클래스가 이미지의
2/3로만 적합됐고 **아무 에러도 나지 않았다.** 3단계에서 이걸 판정한다.

### 1.3 아직 확인 못 한 것

- **group001 외 다른 그룹도 깨졌는지** — 1단계에서 확인.
- **GMM fit이 추출 실패 전인지 후인지** — 3단계에서 확인. 전이라면 GMM은
  멀쩡하다.

---

## 2. 작업 순서

정확한 명령어는 `RUNBOOK.md`에 있다. 여기는 **판단 기준**을 적는다.
각 단계의 gate를 통과 못 하면 다음으로 넘어가지 마라.

### 1단계 — 추출 검점

```bash
python learnable_eps/check_latents.py --latent-path $LAT
```

- **exit 0 (`ALL GROUPS COMPLETE`)** → 3단계로.
- **exit 1** → 깨진 그룹 목록과 재추출 명령어가 출력된다. 2단계로.

### 2단계 — 재추출

스크립트가 그룹별 명령어를 찍어준다. **세 가지를 반드시 지켜라:**

1. `--nproc_per_node`는 그 그룹 `meta.json`의 `world_size`와 **같아야 한다.**
   rank 분할이 그룹 전체에 대한 `DistributedSampler`에서 나오므로, 값이 다르면
   **이미 정상인 rank까지 전부 어긋난다.**
2. **그룹 전체를 재추출한다.** 죽은 rank만 따로는 불가능하다(다른 rank가 같은
   process group에 있어야 한다).
3. `--dtype`, `--data-path`는 원래와 같아야 한다.

런치 전에 GPU 수를 확인해라. 원래 실패도 **보이는 GPU보다 많은 rank를 띄운
것**(`invalid device ordinal`)이 원인일 가능성이 높다.

**Gate:** 1단계를 다시 돌려 `ALL GROUPS COMPLETE`가 나올 때까지.

### 3·4·5단계 — GMM 감사와 부분 재fit

**한 명령으로 처리해라.** 감사를 내부에서 먼저 하고, 문제없으면 쓰지 않고
종료한다. 63 GB짜리 pickle을 두 번 읽지 않기 위해서다.

```bash
python learnable_eps/gmm_doctor.py --mode repair \
    --gmm $GMM --latent-path $LAT \
    --out <디렉터리>/gmm_clusters_repaired.pkl
```

판정 원리: `gmm_fit.py`가 저장한 `labels[cls]`의 길이 = **fit 당시 실제로 본
샘플 수**. 이걸 복구된 샤드의 현재 이미지 수와 비교한다.

**분기:**

- **`GMM IS COMPLETE`** → **4단계(패스)에 해당.** 재fit 불필요. 기존 `$GMM`을
  그대로 쓰고 5b로 간다. fit이 추출 실패 전에 돌았다는 뜻이다.
- **`REPAIRED — N classes`** → 낡은 클래스만 재fit됐다. 이후 `$GMM`을
  `_repaired.pkl`로 바꿔라. 정상 클래스는 바이트 단위로 그대로 복사되고,
  하이퍼파라미터는 기존 pkl에서 읽어 `gmm_fit.py`의 `process_class`를 그대로
  호출하므로 기존 클래스와 동일한 코드·설정으로 생성된다.

**주의:** 이 도구는 pickle을 통째로 읽는다. K=20·`--use-pca false`면 약
63 GB. `free -g`로 먼저 확인해라. 부족하면 실행하지 말고 사용자에게 보고해라.

**Gate:** `--mode audit`이 exit 0.

### 5b단계 — prior 체크포인트 변환

```bash
python learnable_eps/convert_gmm.py --gmm $GMM --out $PRIOR --latent-shape 768 16 16
```

`gmm_rae.pkl` + `means_fp16.npy` + `vars_fp16.npy`(합 약 16 GB)가 나온다.
**셋을 같은 디렉터리, 로컬 SSD에** 둬라.

출력되는 `global_sigma_scale`, `explained_variance_ratio`를 **기록해라.**
논문 표에 들어갈 값이다.

그 다음 config의 `prior.ckpt_path`를 실제 경로로 수정한다.

**Gate:** `python learnable_eps/verify_port.py` → `ALL CHECKS PASSED` (21/21).

### 6단계 — 학습

```bash
export CUDA_VISIBLE_DEVICES=$(python learnable_eps/pick_gpus.py --want 3)
```

`CUDA_VISIBLE_DEVICES` 설정 후 torch가 0,1,2로 재번호매김하므로
**`--nproc_per_node=3`** 이다(물리 번호와 무관).

**반드시 먼저 고쳐야 할 것:** `global_batch_size: 1024`는 3으로 나눠떨어지지
않는다(`1024 % 3 = 1`). `train.py`의 assert에 즉시 걸린다. 768 + `grad_accum_steps: 4`
(GPU당 micro batch 64)로 바꾸고, **baseline과 rapid config 양쪽에 똑같이**
적용해라. 다르면 실험 0/1 비교가 무의미해진다. OOM이면 `grad_accum_steps`만
올려라(`global_batch_size`는 고정).

실험 0(baseline)과 1(rapid)을 **같은 seed·epoch·batch**로 돌린다. config만
다르다.

---

## 3. 절대 하지 말 것

- **`src/` 아래 파일을 수정하지 마라.** 베이스라인은 원본 그대로여야 하고,
  그게 `prior.enable: false` 대조군이 유효한 이유다. `verify_port.py` 체크 0이
  `src/`를 grep해서 감시한다. 실험 코드는 전부 `learnable_eps/` 아래에 둔다.
- **추론 경로에 `z = z / z.std(...)`를 추가하지 마라.** 모델이 학습한 GMM/eps
  비율을 조용히 없앤다. 13~15차 실험 실패의 직접 원인이었다.
- **`w(t)`를 `q0*exp(-alpha*t)`로 되돌리지 마라.** 이 레포는 시간축이 반대라
  (노이즈가 `t=1`) `w(t) = q0*exp(-alpha*(1-t))`가 맞다. `verify_port.py`
  체크 3이 이 등가성을 수치로 고정한다.
- **prior 구성 코드를 bf16 autocast 안으로 옮기지 마라.** PCA 사영이 깨진다.
- **GMM `K`를 20 초과로 올리지 마라.**
- **실험마다 `experiment_name`을 바꿔라.** 같은 이름은 **자동 resume**이라,
  q0만 바꾸고 이름을 두면 새 실험이 아니라 이전 런의 연속이 된다.
- **`--compile`을 빼지 마라.** 없으면 `NotImplementedError`로 죽는다.
- **되돌릴 수 없는 작업 전에 사용자에게 확인받아라** — 특히 재추출은 기존
  `.dat`을 덮어쓴다.

---

## 4. 알려진 미해결 이슈

- **CFG 경로가 깨져 있다**(기존 베이스라인 버그, 이식과 무관).
  `src/train.py`가 `ys`를 정의 전에 참조하고, `src/eval/__init__.py:143`이
  `null_label`을 정의 없이 쓴다. 둘 다 `guidance.scale > 1.0`에서만 터진다.
  현재 config는 `scale: 1.0`이라 안 터지지만, **CFG 실험 전에 고쳐야 한다.**
  `src/` 수정이 필요하므로 사용자 동의를 먼저 받아라.
- **K=20의 통계적 타당성.** 클래스당 약 1300장 ÷ 20 = 컴포넌트당 65샘플로
  196,608차원 대각 공분산을 추정한 것이다. prior 성능이 안 나오면 **K=10
  재피팅이 1순위 후보**다(`fit_gmm_rae.py`가 재추출 없이 가능).
- **`lpf_alpha`의 −3 dB 유도식**이 power 비율인지 amplitude 비율인지 미확인.
  학습은 지시값 1.0을 쓴다. `estimate_lpf_alpha_minus3db()`는 진단 전용.
- **latent 경로엔 flip augmentation이 없다**(`extract_z.py`가 적용하지 않음).
  이미지 경로 베이스라인과 비교하면 교란 요인이므로 **양쪽 arm을 모두 latent
  경로로** 돌려라.

---

## 5. 보고 형식

각 단계마다: **통과/실패 + 근거 로그 3줄 이내.** 특히 아래는 반드시 보고해라.

| 시점 | 보고할 값 |
|---|---|
| 1단계 | 깨진 그룹/rank 목록, 사용 가능 이미지 수 |
| 3단계 | `GMM IS COMPLETE`인지, 아니면 낡은 클래스 수 |
| 5b | `global_sigma_scale`, `explained_variance_ratio` |
| 6단계 시작 100 step | `[data]` 줄의 group 수와 총 샘플 수 |
| 6단계 시작 100 step | **`var(z_init)`** — q0=0.5에서 ≈0.51 예상, 0.3 미만이면 q0 재검토 |
| 6단계 | baseline은 `[RAPID] disabled`, rapid는 `[RAPID] enabled` 확인 |

실패하면 추측해서 진행하지 말고 **멈추고 로그와 함께 보고해라.** 특히
1단계가 깨끗하지 않은데 학습을 시작하면 0으로 채워진 latent를 정상 데이터로
학습하게 된다.

---

## 6. 참고 — 이 이식의 핵심 사실

| | LightningDiT / RAPID | RAE (이 레포) |
|---|---|---|
| `compute_alpha_t(t)` (데이터) | `t` | `1 - t` |
| `compute_sigma_t(t)` (노이즈) | `1 - t` | `t` |
| **노이즈 끝** | **t = 0** | **t = 1** |
| `ut` | `x1 - x0` | `x0 - x1` |
| latent | 32×16×16 = 8,192-D | 768×16×16 = **196,608-D** |
| timestep shift | ~1 | `sqrt(196608/4096)` = **6.93** |

```
w(t) = q0 * exp(-decay_alpha * (1 - t))      # t=1(노이즈)에서 w=q0
x0_blended = w * x0_gmm + (1 - w) * eps
xt = (1 - t) * x1 + t * x0_blended
ut = x0_blended - x1
```

추론 초기값은 학습의 `t=1` 블렌드와 같아야 한다:
`z_init = q0*x0_gmm + (1-q0)*eps`. 세 진입점(`train.py` eval,
`sample_ddp.py`) 모두 `wrap_sampler_with_prior`를 거치며, 어디서도 정규화하지
않는다.

shift 때문에 학습 샘플의 약 87%가 `t>0.5` 구간에 몰린다(`E[w(t)]=0.41` vs
`q0=0.5`). LightningDiT보다 GMM이 훨씬 강하게 주입되므로 precision↑/recall↓
편향이 구조적으로 심하다 — **`q0=0.3` 대조군이 선택이 아니라 필수인 이유다.**

OT 매칭은 **사용자 지시로 미구현**이다. 추가하지 마라.
