# SAM3 × Falcon v1.5: runbook before the first long run

Run the steps in order. Each one says what to run, what "good" looks like and what to do if it isn't.
- Steps 0–2 need only CPU and the dataset mounts.
- Steps 3–4 need one GPU (step 3 also needs the PE weights).
- The rest need one 8-GPU node; step 10 needs two nodes. The implementation is described in
[`FALCON_V15_TRAINING.md`](FALCON_V15_TRAINING.md).

Everything below runs from the sam3 checkout, in the training image with the
tiiuae Energon fork and sam3 installed:

```bash
pip install -e ".[train]" torchmetrics
export BLEND=sam3/train/configs/falcon_v15/datasets/sam3_falcon_v15_seg.yaml
export RUNS=/mnt/falcon-perception/sam3_runs
```

## 0. CPU unit tests (5 min)

```bash
pytest -q test/test_falcon_schema.py test/test_falcon_energon.py test/test_falcon_energon_e2e.py \
  test/test_model_init.py test/test_matched_masks.py test/test_falcon_eval.py
pytest -q test/test_streaming_trainer.py   # ~6 min, ~12 GB RAM
```

**Good:** everything passes. `test_parity_with_torchtitan` runs only if `TORCHTITAN_INTERNAL_DIR` points at a torchtitan-internal checkout (or one sits next to sam3).

## 1. Scan the annotations (CPU, ~10–30 min)

```bash
python scripts/falcon_v15/scan_blend.py $BLEND --max-samples 2000 --workers 32 --out $RUNS/scan_blend.json
```

**Look at** the per-source table and the token table.

| column / line | good | if not |
|---|---|---|
| `!! no sample read` / `!! no mask query` | absent | Wrong path or a source without masks: remove it from the blend (`scripts/falcon_v15/make_seg_blend.py` output) or fix the data. |
| `w/mask%` | close to 100 for every source | A low value means many images carry only negatives or non-RLE data. Decide whether to keep the source. |
| `nonRLE` | 0 | Box-only values under `referring_segmentation`. The loader drops them; check the source with its owner. |
| `pos/img`, `neg/img` (p50/p95/max) | — | Sizes the step budget: with `max_queries_per_batch = 256`, an image at the p95 should fit many times over. Images above `max_queries_per_sample` (64) get split. |
| `>200 q%`, `>200 m%` | ≲ 1 % | Those queries are dropped (the decoder has 200 object queries). If a source loses much more, report it: it is a property of the comparison with the AR model. |
| token table `>256 %` | ~0 | Raise `trainer.model.text_context_length`. |
| token table `>32 %` | — | The share of prompts the released 32-token SAM3 would have truncated. Note it for the write-up. |

Prompt lengths are measured on the JSON text, before the loader strips a leading article, so they are an upper bound.

## 2. Loader on the real tars (CPU, ~10 min)

```bash
python scripts/falcon_v15/inspect_loader.py --batches 300 --render 8 --out $RUNS/loader_check
```

**Good:**
- The first batch arrives within a few minutes (buffer fill). After that, `wait p95` stays well below the expected step time (step 6 gives it; around 1–3 s).
- The budget-use line stays at or under the configured budgets, and queries per batch sit near `max_queries_per_batch`. Batches far below the budget mean packing is starved: raise `packing_buffer_size`.
- Images per source roughly follow the blend's repetitions.
- **Rendered PNGs:** every coloured mask lies on the object its text names, boxes are tight, the image is resized and padded with its masks, and negatives really are absent from the image. Non-Latin text draws as boxes with PIL's default font; check those prompts in the JSON.

**If it fails:**
- **Energon errors about sample fields or `image_decode`:** the fork's `NpyTxtJsonSample` differs from the assumption, which is `images` (list), `metadata` and `__subflavors__`. Adapt `Sam3FalconTaskEncoder.encode_sample`; `_decode_image` already accepts bytes, numpy and PIL.
- **Host RSS keeps growing over the run:** lower `trainer.data.train.num_workers` or `shuffle_buffer_size`. Multiply by 8 ranks per node for the real footprint; Falcon was killed at 1.7 TB.

## 3. PE weights carry over (1 GPU, 2 min)

Needs `facebookresearch/perception_models` installed (`pip install -e .` in its checkout).

```bash
python scripts/falcon_v15/check_pe_init.py --pe PE-Core-L14-336 --image <any photo>
```

**Good:**
- `vision, as mapped` patch cosine is high (≳ 0.9 on average) and clearly above `vision, rope axes swapped`. SAM3 drops the CLS token, so it is not 1.0.
- The text lines show a max abs diff of about 1e-5.

**If not:**
- **Swapped is higher:** the RoPE axis layouts differ. Permute the trunk's `freqs_cis` halves at init (in `model_init.load_pe_into_sam3`) and re-run.
- **Both are low:** debug the key mapping (`model_init._VISION_RULES`).
- **A PE API error in the script:** adapt the `forward_features` / `encode_text` calls to the installed perception_models version.

## 4. Numerics of the option flags (1 GPU, 5 min)

```bash
python scripts/falcon_v15/bench_train_step.py --mode numerics --queries 32 --images 4
```

**Good:** the `max rel diff` column against the fp32 reference.

| variant | expected |
|---|---|
| `batch_matching_over_layers` | ~0 |
| `masks_for_matched_queries_only` | ≲ 1e-5 |
| `target_sampling=nearest` | ~1e-2 on `loss_mask` / `loss_dice` (different sampling), ~0 elsewhere |
| `bf16 autocast`, `config (bf16, all flags)` | ~1e-2 |

**If not:** a flag that moves the box/class losses (which it should not touch) is a bug. Run that flag off, and send the output.

## 5. DDP: unused parameters and step overhead (8 GPUs, 10 min)

```bash
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node 8 scripts/falcon_v15/bench_train_step.py \
  --queries 128 --images 8 --matched-masks 1 --steps 5 --find-unused 0
```

**Good:** no "Expected to have finished reduction" error, and the step time is close to the single-GPU number for the same shape from step 6. The difference is the gradient all-reduce.

**If DDP names parameters that got no gradient:** add their patterns to `trainer.frozen_param_patterns` in the config, or set `trainer.distributed.find_unused_parameters: true` as a stopgap. Then re-run.

## 6. Memory / throughput sweep (1 GPU, 30–60 min)

```bash
python scripts/falcon_v15/bench_train_step.py --queries 64,128,256,384,512 --images 8,16 \
  --matched-masks 0,1 --out $RUNS/bench_sweep.jsonl
# activation checkpointing and compile at the most promising shapes:
python scripts/falcon_v15/bench_train_step.py --queries 256,384 --images 16 --matched-masks 1 \
  --ckpt-off none seg_head seg_head+text seg_head+text+geometry vit --out $RUNS/bench_ckpt.jsonl
python scripts/falcon_v15/bench_train_step.py --queries 256 --images 16 --matched-masks 1 \
  --compile 0,1 --warmup 10 --out $RUNS/bench_compile.jsonl
# a profile of the chosen setting, opened in chrome://tracing or Perfetto:
python scripts/falcon_v15/bench_train_step.py --queries 256 --images 16 --matched-masks 1 \
  --steps 3 --profile-dir $RUNS/profile
```

**Read the table:** `q/s` (queries per second per GPU) is the number to maximise; `alloc` / `reserv` are peak GiB.

**Pick** the setting with the best `q/s` whose `reserv` stays below about 160 GiB on a B200. Falcon saw the allocator thrash and a 3× slowdown past about 165 of 178 GiB. Then set in the config:
- `scratch.max_queries_per_batch` and `scratch.max_images_per_batch`
- `trainer.masks_for_matched_queries_only` (expected: on)
- `trainer.model.compile_vision`

Set `scratch.max_instances_per_batch` from the instance count you saw at that shape × 1.5.

If a `--ckpt-off` setting wins, put the same parts in `trainer.act_ckpt_off` (for example `[seg_head, text]`). The fusion encoder and decoder assert that checkpointing is on (`model/encoder.py`, `model/decoder.py`); relaxing that is the next lever if memory is left over.

**Also check in the profile:**
- gaps on the GPU stream around `Masks` / matcher calls (host syncs)
- time share of `TransformerEncoderFusion` (expected largest: it runs per query) vs the ViT

The synthetic batches use rectangles and random prompts. Re-check memory on real batches in step 8 (`Perf/max_mem_gib`).

## 7. Preemption and resume (8 GPUs, 20 min)

```bash
torchrun --nproc_per_node 8 -m sam3.train.train_streaming -c configs/falcon_v15/sam3_falcon_v15_seg.yaml \
  paths.experiment_log_dir=$RUNS/preempt_test paths.dataset_path=$BLEND \
  scratch.max_steps=400 trainer.checkpoint.save_every_steps=100000
# after ~60 steps, from another shell, signal ONE rank:
kill -TERM $(pgrep -f "sam3.train.train_streaming" | tail -1)
```

**Good:** within `log_freq` (10) steps every rank logs "Stopping after preemption checkpoint". Then `checkpoints/` holds `checkpoint.pt` and 8 `rank_state_*.pt`.

Relaunch the same command. **Good:** it logs `Resumed from ... at step N` with no "dataloader state is not restorable" warning, and the loss continues smoothly. Resuming with a different number of GPUs is expected to warn and restart the loader.

## 8. Short real run (1 node, ~2 h)

```bash
torchrun --nproc_per_node 8 -m sam3.train.train_streaming -c configs/falcon_v15/sam3_falcon_v15_seg.yaml \
  paths.experiment_log_dir=$RUNS/smoke_2k paths.dataset_path=$BLEND scratch.max_steps=2000
```

**Watch** TensorBoard (`$RUNS/smoke_2k/tensorboard`) or `logs/train_stats.jsonl`.
- `Losses/core_loss` goes down; `loss_mask`, `loss_dice`, `loss_ce`, `presence_loss` and `loss_semantic_seg` are all finite and moving.
- `Perf/data_wait_s_max` stays ≈ 0. If not, the loader is the bottleneck: raise `trainer.data.train.num_workers` while watching host RAM.
- `Perf/max_mem_gib` stays below the step-6 limit on real data.
- `Perf/queries_per_s_per_gpu` is close to the step-6 number.
- The per-worker encoder stats in the logs show low `sample_skipped_*` and `query_dropped_*` counts.

## 9. LR scale sweep (3 runs × 1 node, ~5k steps each)

The released configs finetune at `lr_scale` 0.1. Here the detector starts from scratch, so the scale is unknown.

```bash
for s in 0.5 1.0 2.0; do
  torchrun --nproc_per_node 8 -m sam3.train.train_streaming -c configs/falcon_v15/sam3_falcon_v15_seg.yaml \
    paths.experiment_log_dir=$RUNS/lr_$s paths.dataset_path=$BLEND scratch.lr_scale=$s \
    scratch.max_steps=5000 scratch.cooldown_steps=500
done
```

**Pick** the largest scale without loss spikes or divergence. The trainer stops on a non-finite loss.

## 10. Multi-node launch

On Nebius, reuse torchtitan-internal's `infra/launch_job_nebius.sh` and helm chart with the pod command below (one process per GPU). Keep `ulimit -n` at the hard limit, and stage the blend yaml under `/mnt/falcon-perception/configs/perception_datasets/` (the config's default `paths.dataset_path`).

```bash
ulimit -n $(ulimit -Hn)
torchrun --nnodes $NNODES --nproc_per_node 8 --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:29500 \
  -m sam3.train.train_streaming -c configs/falcon_v15/sam3_falcon_v15_seg.yaml \
  paths.experiment_log_dir=$RUNS/<run>
```

**Check:**
- The first step can take minutes while every rank fills its buffers. `trainer.distributed.timeout_mins` (30) covers it.
- Step time on 2 nodes ≈ 1 node: DDP all-reduce of ~850M parameters. If it is not, set `trainer.distributed.comms_dtype: bfloat16`.

## 11. Evaluation (separate job, on saved checkpoints)

Eval runs through the released epoch trainer in `mode: val`, with overlay configs in `configs/falcon_v15/eval/`.
- `falcon_model.yaml` swaps the model for `sam3.train.model_init.build_sam3_image_model_for_eval`, which has the run's trunk and text shape and loads `checkpoint.pt` strictly.
- Paths come from environment variables.
- If the run changed `vit_depth`, `vit_mlp_ratio` or `text_context_length`, set the same values in `falcon_model.yaml`. A mismatch fails loudly at load time.

**SA-Co/Gold** (does the model still do open-vocabulary concept segmentation?). Needs the released SA-Co/Gold GT and images.

```bash
export SAM3_EVAL_CHECKPOINT=$RUNS/<run>/checkpoints/checkpoint_<step>.pt
export SAM3_EVAL_OUT=$RUNS/<run>/eval_<step>
export SAM3_GOLD_GT_DIR=<gold GT dir> SAM3_METACLIP_IMG_DIR=<metaclip images> SAM3_SA1B_IMG_DIR=<sa1b images>
for s in metaclip_nps sa1b_nps crowded fg_food fg_sports attributes wiki_common; do
  python sam3/train/train.py -c configs/falcon_v15/eval/gold_$s.yaml --use-cluster 0 --num-gpus 8
done
```

Metrics are logged as `cgF1_eval_segm_*` / `cgF1_eval_bbox_*` in `$SAM3_EVAL_OUT/gold_*/logs/` (`val_stats.json`). Compare against the released SAM3 numbers on the same sets.

**Held-out Falcon pairs** (in-domain, per level and language; the comparison with the AR model).

1. Build the eval set once from the sources' `val` split tars (never the training tars):

   ```bash
   python scripts/falcon_v15/make_falcon_eval_set.py $BLEND --out /mnt/falcon-perception/sam3_eval/falcon_v15_heldout      --max-images-per-source 300
   ```

   It lists the sources without a val split. Prepare one with torchtitan-internal's `energon_prepare_perception.py --val-tar` **before training**, so the val tars are excluded from training through energon's split.

2. Evaluate and slice:

   ```bash
   export FALCON_EVAL_DIR=/mnt/falcon-perception/sam3_eval/falcon_v15_heldout
   python sam3/train/train.py -c configs/falcon_v15/eval/falcon_heldout.yaml --use-cluster 0 --num-gpus 8
   python scripts/falcon_v15/cgf1_by_slice.py --gt $FALCON_EVAL_DIR/gt.json      --pred $SAM3_EVAL_OUT/falcon_heldout/dumps/falcon_heldout/coco_predictions_segm.json      --by level lang source --out $SAM3_EVAL_OUT/falcon_heldout/cgf1_by_slice.json
   ```

Each slice is the positive pairs of that level, language or source plus the negatives asked on the same images; cgF1 includes the image-level MCC, which needs both. Use the same GT file for the AR model's predictions so both are scored identically.

**Good:** on the first checkpoints, cgF1 rises with training steps on both sets. A model trained only on Falcon data may score lower than the released SAM3 on SA-Co/Gold; report that as a result, not a bug.

## Results to send back

1. `scan_blend.json` and the printed tables from step 1.
2. The step-2 summary lines and 2–3 rendered PNGs.
3. The step-3 and step-4 printouts.
4. The `bench_*.jsonl` files, `top_ops_*.txt`, and the chosen budget.
5. TensorBoard scalars or `train_stats.jsonl` from steps 8 and 9.
6. `val_stats.json` per eval set and `cgf1_by_slice.json` (step 11).
