# Training SAM3 on the Falcon Perception v1.5 segmentation data

What is implemented, how to run it, and what still has to be checked on GPUs.
The review and the reasoning behind the design are in
[`LARGE_SCALE_TRAINING_PLAN.md`](LARGE_SCALE_TRAINING_PLAN.md).

## What it does

- **Init.** SAM3 is trained from scratch with PE vision and text encoders, like SAM3 itself, not from the released SAM3 checkpoint. Everything is trained.
- **Data.** The Falcon v1.5 perception tars (energon), segmentation subset only:
  - positives: `referring_segmentation` expression → RLE masks
  - negatives: `negative_expressions`
  - all levels and languages

  No box-only, pointing, exemplar or knowledge data.
- **SAM3 recipe.** Same losses, matcher, LR groups, layer decay and augmentation (random square resize 480–1008 plus random pad to 1008).
- **Text.** The encoder context is 256 tokens instead of 32.

## Files

| file | what |
|---|---|
| `sam3/train/data/falcon_schema.py` | Annotation-schema helpers, copied from torchtitan-internal `1ca4fa3` (attribute filters, alternatives, RLE → grid decode, bit packing), plus `parse_segmentation_queries`. Parity with torchtitan is tested. |
| `sam3/train/data/falcon_energon.py` | `Sam3FalconTaskEncoder` (energon), `Sam3EnergonLoader` (savable, per rank), `Sam3FalconBatch.to_batched_datapoint` (GPU-side mask unpack + normalize → SAM3 `BatchedDatapoint`). |
| `sam3/train/model_init.py` | `build_sam3_image_model_for_training` and PE → SAM3 weight mapping (`load_pe_into_sam3`, strict). |
| `sam3/train/streaming_trainer.py` | Step-based trainer: checkpoints with loader and RNG state, preemption save, no per-step host syncs, bf16 without GradScaler, side-stream prefetch. |
| `sam3/train/train_streaming.py` | torchrun entry point. |
| `sam3/train/configs/falcon_v15/sam3_falcon_v15_seg.yaml` | The run config. |
| `sam3/train/configs/falcon_v15/datasets/sam3_falcon_v15_seg.yaml` | Segmentation blend: 42 of the 47 v1.5 sources, same repetitions. Regenerate with `scripts/falcon_v15/make_seg_blend.py`. |
| `scripts/falcon_v15/check_pe_init.py` | GPU check that the PE features carry over (RoPE layout). |

### Changes to existing SAM3 code

All of these are needed for training. Inference outputs are unchanged.

- **`model/vitdet.py`, `Mlp.forward`:** the released code always called `perflib.fused.addmm_act`, an inference-only bf16 kernel that **raises under autograd**, so the ViT trunk could not be trained at all. With grad enabled it now takes `act(fc1(x))`.
- **`model/text_encoder_ve.py`:**
  - `VETextEncoder(pad_to_longest=...)` pads each batch to its longest prompt instead of the full context. It is off by default.
  - `TextTransformer._load_from_state_dict` extends a shorter checkpoint position table: old positions are kept, and new ones are copies of the last.
  - `TextTransformer.init_parameters()` adds open_clip init. `positional_embedding` and `text_projection` used to be left as `torch.empty`, which gives NaNs at random init.
- **`model/sam3_image.py`:** `batch_matching_over_layers` runs one matcher call for the final and the 5 aux decoder layers. It gives the same assignments and needs 1 device→host copy instead of 6. Off unless the trainer sets it.
- **`train/loss/loss_fns.py`:**
  - `Masks(target_sampling="nearest")`: point-sampled mask loss that reads the targets at the sampled points. It no longer gathers a full-resolution copy of every matched target (o2m repeats each one 4×).
  - The triton focal-loss kernels are only used on CUDA tensors.
- **`model/position_encoding.py`, `model/decoder.py`, `model/geometry_encoders.py`:** no hard-coded `cuda` device at build time, and no `pin_memory` on CPU. This only matters for the CPU tests.

## Run

1. **Environment.** The Falcon training image: the tiiuae Energon fork and `uv` lock. Then install sam3 with its train deps:

   ```bash
   pip install -e ".[train]" torchmetrics
   ```

2. **Blend.** Stage the blend where the pods can read it, and point `paths.dataset_path` at it:

   ```bash
   cp sam3/train/configs/falcon_v15/datasets/sam3_falcon_v15_seg.yaml /mnt/falcon-perception/configs/perception_datasets/
   ```

3. **Launch.** One process per GPU:

   ```bash
   ulimit -n $(ulimit -Hn)
   torchrun --nnodes $NNODES --nproc_per_node 8 --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:29500 \
     -m sam3.train.train_streaming -c configs/falcon_v15/sam3_falcon_v15_seg.yaml \
     paths.experiment_log_dir=/mnt/falcon-perception/sam3_runs/<run>
   ```

   On the Nebius cluster, reuse `infra/launch_job_nebius.sh` from torchtitan-internal with this command as the pod script. Keep the same `init` / `train` NCCL timeouts: the first buffer fill can take minutes.

**Resume.** Relaunching with the same `experiment_log_dir` resumes from `checkpoints/checkpoint.pt`, plus `rank_state_*.pt` for each rank's loader and RNG state. Loader state only restores with the same world size and `num_workers`, and an append-only blend. Otherwise model and optimizer resume and the loader starts fresh (a warning is logged).

**Knobs** (`scratch.*` in the config):
- Per-GPU budget: `max_queries_per_batch` (256), `max_instances_per_batch` (4000), `max_images_per_batch` (16).
- `max_queries_per_sample` (64): images with more queries are split.
- Instance cap: `max_instances_per_query` = 200 (the decoder's object queries). Larger queries are **dropped**, never subsampled.

## Design notes and deviations

**Vision tower.** SAM3's released trunk has width 1024, 32 blocks and MLP ratio 4.625. No public PE checkpoint matches that (PE configs: `core/vision_encoder/config.py` in perception_models).
- The closest public one, and the default, is **PE-Core-L14-336**: 24 blocks, MLP ratio 4.0, the same 336 px pretraining with a CLS position, abs pos-emb and 2D RoPE.
- So the trunk here is 24 blocks, with global attention in blocks 5/11/17/23 and 24×24 windows elsewhere, as SAM3 does on 32 blocks.
- PE-Core-L14-336's text tower matches SAM3's text encoder exactly.
- Other options: `PE-Spatial-L14-448` (vision only; pair it with the PE-Core text tower), or SAM3's own trunk shape (`vit_depth=32 vit_mlp_ratio=4.625`) from the SAM3 checkpoint (`sam3_checkpoint=`), which is not a PE init.

**Text context 256.** For prompts up to 32 tokens the output is identical to the 32-token encoder (tested). Positions 32–255 start as copies of position 31. The CLIP BPE vocabulary is English-centric: Arabic and other scripts tokenize into many byte-level tokens, which the longer context no longer truncates but which PE never trained on.

**Batching.**
- Per step, images are packed first-fit under the query, instance and image budgets (energon packing, one pack per rank step).
- An image with more than `max_queries_per_sample` queries is split into several samples. SAM3 queries are independent, so the only cost is the repeated ViT pass.
- `encode_sample` is always a generator, so energon restore keys never change layout.

**Exhaustive data only.**
- Every query is `is_exhaustive=True`.
- Instances that vanish at the sampled scale (0 pixels) are removed. A positive left with none is dropped; it is never turned into a negative.
- After article stripping, a file negative that equals a positive is dropped.

## Tests (CPU)

```bash
pytest test/test_falcon_schema.py test/test_falcon_energon.py test/test_falcon_energon_e2e.py \
  test/test_model_init.py test/test_streaming_trainer.py
```

**Schema** (`test_falcon_schema.py`):
- Parity with torchtitan-internal's functions, compiled from its source; set `TORCHTITAN_INTERNAL_DIR` or use a sibling checkout.
- RLE→grid equals a full decode followed by a nearest resize.
- Filters, alternatives and negatives.

**Encoder and batch** (`test_falcon_energon.py`):
- Masks, boxes, scale/pad and the instance cap.
- Split and packing budgets.
- The GPU-side batch equals SAM3's `collate_fn_api` on the same data, field by field.

**Text and init** (`test_model_init.py`):
- 32→256 context keeps outputs identical, and pad-to-longest keeps them identical.
- The PE mapping is strict: wrong depth raises.
- Batched matching equals per-layer matching.

**Energon** (`test_falcon_energon_e2e.py`): the encoder inside a real upstream-energon pipeline, using a stand-in sample type because `NpyTxtJsonSample` is fork-only.
- `energon prepare`d tar, 0 and 2 workers.
- Packing budgets hold.
- Save/restore gives the same next batches.

**Trainer** (`test_streaming_trainer.py`, about 6 min, needs about 12 GB RAM):
- The real config with a 2-block trunk and an in-memory loader: forward, all losses, backward and optimizer.
- Every trainable parameter gets a gradient; this checks the frozen list.
- Checkpoint, then resume, which restores step, weights and loader position.

## GPU checks still to do

Run these before a long run, roughly in this order. None of them could run in this environment (no GPU, no Energon fork, no cluster data).

1. **PE features carry over.** Run `scripts/falcon_v15/check_pe_init.py --image <img>` with perception_models installed.
   - "as mapped" patch cosine must be clearly above "rope axes swapped". If not, permute `freqs_cis`.
   - The text per-token diff must be about 1e-5.
   - The PE API calls in the script (`forward_features`, `encode_text`) may need adapting to the installed version.
2. **Loader on real tars.** Build `Sam3EnergonLoader` on the seg blend with the Energon fork. Upstream energon is covered by the e2e test; the fork-specific parts are:
   - `NpyTxtJsonSample` fields (`images`, `metadata`, `__subflavors__`)
   - `image_decode="pilrgb"` on npy image lists
   - `get_train_dataset` kwargs on the fork (`watchdog_timeout_seconds`)
   - packing with `batch_size=1`

   Render a few batches (masks + text) and compare with `scripts/dataloader_snapshot` in torchtitan-internal.
3. **Mask presence per source.** Check that every grounding source carries RLE values. Watch the encoder stats `non_rle_value` and `sample_skipped_no_query` per source in the first 1k steps (logged every 2000 samples per worker).
4. **Text-length scan.** Measure token length per level and language over the blend, and confirm 256 covers it. Set `strip_articles` to match the AR baseline's inputs.
5. **DDP unused parameters.** One multi-GPU step with `find_unused_parameters: false` and `TORCH_DISTRIBUTED_DEBUG=DETAIL`. If DDP names unused parameters, add them to `frozen_param_patterns`.
6. **Numerics.** Compare a few steps of loss between:
   - `batch_matching_over_layers` true and false (must be identical)
   - `target_sampling` `nearest` and `bilinear` (close)
   - bf16 vs fp32 autocast (close)
7. **Memory/throughput sweep** (Phase 0 of the plan). On one 8×B200 node, sweep `max_queries_per_batch` ∈ {128, 256, 512} × `max_images_per_batch` ∈ {8, 16}. Record:
   - `Perf/*` in TensorBoard: queries/s, step time, data wait, max memory
   - host RAM over 1k steps, at `num_workers` 8–12

   Pick the largest budget without allocator thrash (Falcon saw a 3× slowdown past about 165 GiB).
8. **Data wait.** `Perf/data_wait_s_max` should stay near 0. If not, raise `num_workers` (watch host RAM) or move tokenization into the workers.
9. **Preemption.** Send SIGTERM to one rank. All ranks save within `log_freq` steps and exit, and the relaunch resumes the loader position.
10. **LR sweep.** Sweep `scratch.lr_scale` ∈ {0.5, 1, 2} on one node for about 5k steps. The released configs finetune at 0.1. Here the detector starts from scratch, so the right scale is unknown.
11. **Evaluation.** Run the SA-Co/Gold configs (`configs/gold_image_evals/`) and a held-out Falcon slice on saved checkpoints as a separate job. The eval configs build the model with `build_sam3_image_model`, so point them at a trunk/text config that matches this run (24 blocks, context 256).

## Masks for matched queries only (flag, off by default)

`trainer.masks_for_matched_queries_only: true`.

- **Default path.** The seg head computes `pred_masks` [prompts, 400, 288, 288] for every object query (200 o2o + 200 o2m): about 66 MB bf16 per prompt, plus its gradient. The loss only ever uses the matched rows.
- **With the flag** (training only):
  - The seg head returns a `DeferredMasks` (`model/maskformer_segmentation.py`): the mask-embedded queries plus the per-prompt pixel embeddings.
  - `Masks` computes logits only for the matched (prompt, query) pairs, o2o and o2m, with one matmul per prompt that has matches.
  - Memory then scales with the number of matched instances, not with queries × 400.
  - Eval and inference are unchanged (dense masks).

**Tests** (`test/test_matched_masks.py`, CPU):
- `gather` equals rows of the dense masks, values and gradients.
- The `Masks` loss is equal for the nearest, bilinear and full-res variants.
- On the real tiny model with the real loss config, every loss term is equal, and gradients in the seg head and decoder match to about 1e-6 relative (fp32 rounding).

**GPU checks:**
1. A few steps with the flag on and off: identical loss curves under bf16 (close, not bitwise).
2. Peak memory at a fixed `max_queries_per_batch`.
3. Then re-run the budget sweep (check 7) with the flag on to find the new maximum queries per GPU.
4. The per-prompt grouping costs one device→host sync per `Masks` call (o2o and o2m). Check in the profiler that it doesn't stall the stream.

## Not done yet (planned efficiency work)

- **Masks at the sampled points only.** With `masks_for_matched_queries_only` (below), each matched query still gets a full 288² logit map before point sampling. Sampling the pixel embeddings at the points instead would cut that further.
- **o2m matcher.** It still runs once per layer in the loss.
- **Selective activation checkpointing and `torch.compile` of the trunk.** Use `compile_vision: true`, and measure.
- **Tokenization in the loader workers.**
