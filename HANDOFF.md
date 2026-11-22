# HANDOFF — read this first (new local Mac session)

Project: Samsung PRISM worklet 26TS08, "A Cost-Aware Framework for Synthetic Data Detection".
Owner of this repo's **video** module. A teammate owns **audio** (already integrated internally;
fusion contract in `Samsung_PRISM_Video_Detection/docs/UNIFIED_PIPELINE.md`).
Branch: `claude/quirky-goodall-3ny0im`. Repo on the Mac:
`/Users/rudrakhale/Desktop/Samsung_PRISM Project/Samsung_PRISM_Project_1` (path has a space — always quote it).

**Decision: no more Google Colab.** Everything from here runs locally on the Mac (Apple Silicon → torch `mps`),
reading/writing data on Google Drive via **Google Drive for Desktop**
(`~/Library/CloudStorage/GoogleDrive-*/My Drive/prism_video/`). The user wants results, short answers,
and commands that work — diagnose before repeating a command.

---

## 1. What works today (verified)

| Model | Checkpoint | Result |
|---|---|---|
| Face-swap branch (ConvNeXt-Tiny, EXP-006) | `checkpoints/face_celebdf_identity.pt` | Celeb-DF identity-disjoint, video level through the served path: **AUC 0.9896**, recall 25/26, FPR 8/156 (all FPs from identities id21/id22) |
| General AI-video branch (VideoSpatialNet, EXP-004) | `checkpoints/best_model.pt` | GenVidBench, unseen generator t2vz: **AUC 0.9426** (1 epoch, frozen backbone). NOT yet re-verified through the served path |

- `src/inference/unified_detector.py`: one detector. General branch always runs; face branch only if ≥50% frames have a face; OR fusion; per-branch detail in `meta.branches`. Dashboard (`Samsung_PRISM_Video_Dashboard`, `docker compose up --build`) uses it via `config/config.yaml` (`pipeline: unified`).
- Honest capability: face swaps = verified; 2023-era text-to-video = trained but unverified in the served path; modern generators (Sora/Veo/Kling/Runway…) = **untested**.
- Checkpoint inventory + sha256: `Samsung_PRISM_Video_Detection/checkpoints/MANIFEST.md` (`best.pt` is an overwritten GenImage model — not usable).

## 2. The plan

Full plan: `Samsung_PRISM_Video_Detection/docs/TRAINING_ROADMAP.md`. Phases:
1. Fully AI-generated video detection + generator classification (GenVidBench) ← **in progress**
2. Faceless content (landscapes/objects/animals) per-slice eval + real-footage top-up
3. Modern generators (frozen test set first, then leave-one-generator-out)
4. Merge into one detector (calibrated thresholds, fusion, single bundle) → dashboard
5. Audio fusion

Phase 1 models: **B (recommended)** frozen CLIP ViT-L/14 per-frame features + small temporal head
(binary head + generator-attribution head); **A** ConvNeXt-Tiny + attention-pool, full fine-tune.
Pick by unseen-generator AUC; must beat 0.9426 on test-A.

Splits (leave-generator-out): train/val = pika, vc2, ms + vript reals; **test-A** = t2vz + held-out vript;
**test-B** = GenVidBench Pair2 (svd, musev, mora, cogvideo + hd_vg_130m reals), never trained on.
Manifests: `data/manifests/{train,val,test}.csv` (columns `video_id,video_path,label,generator,source,dataset,split`),
dated 19 Sep, also copied to Drive `prism_video/manifests/`. **Reuse them, do not regenerate** (keeps results comparable to EXP-004).

## 3. Phase 1 data state (as of 2026-11-22)

Drive `prism_video/frames/<generator or real-source>/shard-NNNNN.{tar,jsonl}` — 16 uniform frames/video,
short side 256px JPEG. A shard is complete only if its `.jsonl` exists. `_archives_done.json` lists finished archives.

| Archive (`jian-0/GenVidBench` on Hugging Face) | Status |
|---|---|
| Pair1 `ms.rar`, `pika.rar`, `t2vz.rar`, `vc2.rar` | **done** (sharded to Drive) |
| Pair1 `vript.rar` (49 GB, the real videos) | was running in Colab when we stopped. 1,536 of ~21.7k files failed to unpack (disk likely filled: 49 GB rar + 49 GB extracted on a 107 GB disk). **Check** `frames/real-vript/*.jsonl` count, `frames/failures_vript.jsonl`, and whether `GenVidBench/Pair1/vript.rar` is in `_archives_done.json`. If incomplete: re-run the shard builder (resumes; skips done ids). If many manifest rows are missing, extract only the needed files (`unrar x` with an `@listfile` of manifest filenames) to avoid the disk peak |
| Pair2 (test-B) | not started. First `scripts/build_genvidbench_testb.py --labels <Pair2 labels.txt> --out .../manifests/test_b.csv --per-class 2000`, then shard each Pair2 archive with `--manifest test_b.csv` |
| CLIP features → `prism_video/features/` | not started |

Known dataset quirks: some GenVidBench mp4s are truncated ("moov atom not found") — harmless, logged as failures.
`unar` returns non-zero if any member fails → don't treat as fatal; count extracted `.mp4` instead.

## 4. Scripts (all in `Samsung_PRISM_Video_Detection/scripts/`)

- `build_frame_shards.py --manifest <csv...> --video-root <dir containing Pair1/,Pair2/> --folder <archive name> --out <drive>/prism_video/frames` — resumable. Run Python with `-u` when piping output.
- `build_genvidbench_testb.py` — Pair2 labels → test_b.csv.
- `extract_clip_features.py --frames <drive>/frames --out <drive>/features` — uses cuda → **mps** → cpu; fp16; resumable; output `(n,16,768)` fp16 `.npy` + `.ids.json` per shard. **Never run yet.**
- `notebooks/Phase1_Data_Prep.ipynb` — the Colab version (abandoned; logic is the same).

## 5. Next steps (in order)

1. Mac setup: Google Drive for Desktop (set `prism_video` to "Available offline" or expect slow reads), Python venv with `torch torchvision transformers opencv-python numpy huggingface_hub`, `brew install unar` (or `unrar`). Download raw archives to a **local** temp dir with ≥100 GB free, never to Drive.
2. Finish vript (see table), then Pair2 test-B.
3. Run `extract_clip_features.py` on MPS.
4. **Write** `scripts/train_clip_head.py`: loads feature shards + jsonl labels, temporal attention head (binary + attribution), early-stop on val AUC, checkpoint every epoch, never overwrite (`--overwrite` guard like `train_general.py`). Eval on test-A and test-B: AUC, recall, FPR, per-generator recall → `reports/EXP-007-*.json` + `docs/experiments/EXP-007-*.md` (negatives included).
5. Candidate A (ConvNeXt full fine-tune) from the frame shards — slow on MPS; consider a smaller subset.
6. Plug the winner into `unified_detector.py` as the general branch; re-verify through `scripts/score_manifest.py` (reads dashboard config) before claiming numbers.

## 6. Gotchas we already hit

- MPS: MTCNN must stay on CPU; GradCAM used to move the model to CPU in place (fixed with try/finally in `predictor.py`).
- numpy must stay 2.x; `facenet-pytorch` installs with `--no-deps`.
- `checkpoints/*` is gitignored (MANIFEST.md is allow-listed). Never overwrite a checkpoint; record sha256.
- Mark any dataset claim not checked against real files as **(verify)**.
