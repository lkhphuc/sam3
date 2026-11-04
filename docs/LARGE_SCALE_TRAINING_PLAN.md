# SAM3 large-scale finetuning on Falcon Perception v1.5 segmentation data — review and plan

Scope: finetune the SAM3 image model (text prompt → instance masks) on the
**segmentation subset** of the Falcon Perception v1.5 data, at multi-node scale,
reusing the v1.5 energon data and as much of its loader as makes sense. Out of
scope: box-only (DET), pointing, box/image exemplars, knowledge stream
(captions, VQA, interleaved, text), video.

References in the other repo are `torchtitan-internal:<path>:<line>`.

---

## 1. What the current SAM3 training stack does

The released trainer (`sam3/train/`) is built for few-shot finetuning on small
COCO-JSON datasets (RF100-VL, ODinW): `num_images: 100`, `train_batch_size: 1`,
epochs of ~1500 samples, checkpoint saving off. Its shape:

```
COCO json + image folder ─▶ Sam3ImageDataset (map-style, PIL)
   ─▶ CPU transforms per object (DecodeRle full-res, resize, pad to 1008², ToTensor, Normalize)
   ─▶ collate_fn_api (full-res bool masks, raw PIL images)
   ─▶ Trainer: DDP, epoch loop, fp32 masters + autocast, GradScaler
   ─▶ Sam3Image.forward: ViT(1008², 32 blocks) once per image; text encoder once per unique phrase;
      fusion encoder + decoder + seg head once per (image, phrase) query
   ─▶ Hungarian matching (in forward, final + 5 aux layers), Sam3LossWrapper (o2o + o2m)
```

The model side is sound. Every image is encoded once and shared by all its
queries (`img_ids` indexing in `_get_img_feats`), and the text encoder runs once
per unique phrase in the batch. The data path, trainer loop and loss are where
the cost goes at scale.

## 2. Review findings (ranked by expected impact)

### 2.1 Data path

| # | Finding | Where | Impact |
|---|---|---|---|
| D1 | Map-style dataset over a COCO JSON loaded into every worker, with a `DistributedSampler`. It can't stream 50M webdataset samples, and resume restarts the epoch (no loader state is ever saved: `train_dataset.load_checkpoint_state` is read but nothing writes it). | `train/data/torch_dataset.py:33`, `trainer.py:469` | blocker |
| D2 | RLE decoded at **original** resolution, then each mask is resized on the CPU one at a time with `torchvision.F.resize` (default **bilinear + antialias**, on a uint8 0/1 mask), then padded. The Falcon memory audit measured this exact pattern: 4000×6000 images with ~180 masks took 2.6–3.6 s and 3–4 GiB per sample. Decoding straight onto the target grid (`_decode_masks_resized`) took ~30 ms. | `transforms/segmentation.py` DecodeRle, `transforms/basic_for_api.py:212`; `torchtitan-internal:docs/V15_SEGMENTATION_MEMORY_AUDIT.md` §18 | worker-bound |
| D3 | Each target mask is a full `bool[1008,1008]` (1 MB). These masks are stacked in the collator, pickled through worker IPC, and copied to the GPU. At 8 images × 16 queries × ~10 instances, that is ~1.3 GB per step of host→device traffic. | `data/collator.py` (`with_seg_masks`) | IPC/H2D |
| D4 | `raw_images` (the **original** decoded PIL images) are carried in every `Datapoint` and returned through the collator, so every training batch pickles full-size images back to the main process. They are only needed for video eval. | `data/sam3_image_dataset.py:435`, `data/collator.py:200` | IPC waste |
| D5 | The batch is a dataclass, so DataLoader `pin_memory=True` silently does nothing, and `copy_data_to_device(..., non_blocking=True)` is effectively synchronous. Falcon hit the same issue and measured 297 ms/step (`torchtitan-internal:torchtitan/energon_datasets/falcon_taskencoder.py:150`). | `model/data_misc.py`, `trainer.py:508` | ~0.1–0.3 s/step |
| D6 | Images normalized to fp32 on the CPU (12 MB per 1008² image). Shipping uint8 and normalizing on the GPU is 4× less traffic. | `NormalizeAPI` | minor |
| D7 | Text is tokenized in the main process inside `forward` (Python BPE). | `model/vl_combiner.py:124` | minor; move to workers |

### 2.2 Trainer loop

| # | Finding | Where | Impact |
|---|---|---|---|
| T1 | **Dozens of GPU→CPU syncs per step:** `loss.item()` for the finiteness check, then `.item()` on **every** extra loss term. With 7 decoder outputs × o2o/o2m × (ce, presence, bbox, giou, mask, dice, accuracies), that is roughly 50–80 syncs per step. | `trainer.py:970-988` | large; the GPU idles between the per-query kernels |
| T2 | Epoch-based everything: `len(train_loader)` drives the schedule, checkpoints are saved only at epoch end, and `where` is computed from the epoch. A streaming run needs step-based max_steps, scheduler, checkpoints and eval. | `trainer.py:594-631`, `:781` | blocker |
| T3 | Checkpoints are saved by rank 0 with `torch.save` to one file, synchronously, with no dataloader state and no RNG state. | `trainer.py:343-404` | resume correctness |
| T4 | A GradScaler is constructed whenever AMP is enabled, including with bf16. With bf16 it adds an unscale, an inf check (a sync) and scale ops for no benefit. | `trainer.py:1109`, `:850` | small, free |
| T5 | `find_unused_parameters: True` in the configs. This is needed today because the geometry encoder (box/point prompts) gets no gradient on text-only data, but it adds a graph traversal every step and blocks `static_graph`. | configs | ~5–10 % |
| T6 | `torch.compile` is supported for the ViT trunk and pixel decoder (`compile_mode`), but `build_sam3_image_model(compile=...)` is not exposed in the train config. Compiling also forces `optimize_ddp=False` when act-ckpt is on. | `model_builder.py:604`, `model/vitdet.py:952` | to measure |
| T7 | Activation checkpointing is on for every component (ViT blocks, fusion encoder, decoder, boxRPB MLPs, seg head, text encoder). At 1008² and with B200 memory (178 GiB), part of it can likely be turned off: about 1/3 of fwd FLOPs come back as speed. | `model_builder.py` | to measure |

### 2.3 Loss and matcher

| # | Finding | Where | Impact |
|---|---|---|---|
| L1 | **Mask loss at full 1008² resolution in fp32:** predicted masks (288²) are bilinearly upsampled to the target size and focal + dice runs over 1M pixels per matched instance, for o2o and o2m (topk=4) matches. `Masks` already supports point-sampled loss (`num_sample_points`, `oversample_ratio`, `importance_sample_ratio`, as in Mask2Former), but the configs don't set it. | `loss/loss_fns.py:644-717` | large memory and compute; enables bigger batches |
| L2 | Matching runs separately for the final and each of 5 aux layers (`_compute_matching`), and the o2m matcher runs again per layer in the loss. Each call does `C.cpu().numpy()` (a sync) and one scipy `linear_sum_assignment` per query, in a Python loop. The matcher already has a `repeat_batch` path to do all layers in one call. | `model/sam3_image.py:607`, `train/matcher.py:615` | ~12 syncs + CPU-bound matching per step |
| L3 | `sum([[i] * len(src) ...], [])` (quadratic list build) and `.item()` in cumulative offsets inside the matcher. | `train/matcher.py:640-660` | small |
| L4 | `dice_loss` catches GPU OOM and silently recomputes on the CPU. At scale this hides a problem as a 100× slower step. | `loss/loss_fns.py:88` | remove |

### 2.4 Model cost structure (why queries per image is the main knob)

Rough forward FLOPs at 1008² (72×72 = 5184 image tokens):

- **ViT trunk** (32 blocks, d=1024, 28 windowed + 4 global): ~5 TFLOP per **image**.
- **Fusion encoder** (6 layers, d=256, full self-attention over 5184 tokens + cross-attention to text) is run **per query**, on that query's copy of the image features: ~0.25 TFLOP per **query**. The 6-layer decoder (≤400 queries with DAC) and the seg head add less.

So ~20 text queries per image cost about as much as the ViT. The number of
queries per image (positives + negatives) and the images per GPU are the two
throughput/memory knobs, and the loader must bound both per batch. The boxRPB
attention bias (`bs × 8 heads × 401 × 5184` per decoder layer) is a float mask.
It rules out flash attention in the decoder and is a memory term that scales
with queries.

## 3. Data: what we train on

Sources (from `torchtitan-internal:infra/config/training/datasets/perception_v15.yaml`):

- **Use:** the 37 grounding sources (`task: referring_segmentation`; ~49.8M samples in total per `V15_PERCEPTION_DATA_INVENTORY.md`), plus the domain ref-seg sources (`RefSeg_urban`, `powerline`, soybean `referring`, `vineyard_grape`, `merlot_mildew`) with their repetitions.
- **Exclude:** `exemplar_merged` (exemplar queries, no text), OmniCorpus, `cc_math`. `osm_det` is prepared but not in v1.5; it is a candidate to add.
- **Inside each JSON**, read only the `referring_segmentation` task dicts, plus `negative_expressions`, `expression_alternatives` and `expression_attributes`. Ignore `object detection`, `pointing` and exemplar objects, and `perception_caption`. Grounding files often carry a DET dict next to the ref-seg dict for the same phrase. Never fall back to it: it has boxes and no masks.
- **Still to verify on the cluster:** every grounding source actually has RLE values (the docs give no mask vs box-only breakdown). This is a one-off scan of a few shards per source.

Semantics to map onto SAM3's `FindQuery`:

| SAM3 field | From Falcon JSON |
|---|---|
| `query_text` | expression key, or an alternative drawn by `_apply_expression_attributes` (level/lang filter) |
| `object_ids_output` / masks | its list of RLEs (all decoded onto the 1008² grid) |
| boxes (for box L1/GIoU and the matcher) | tight box of each decoded mask, in normalized cxcywh |
| negative query | `negative_expressions` keys, with 0 targets |
| `is_exhaustive` | `True` by default (each ref-seg key lists all of its instances); a per-source subflavor overrides it; forced `False` when instances are subsampled (below) |

**Levels: all of them (decided).** The goal is to stress-test the SAM3 architecture against the autoregressive model on the same supervised data, so there is no level or lang filter. Consequences:

- **Text context is 32 BPE tokens including SOT/EOT** (`VETextEncoder` `context_length=32`, `model/text_encoder_ve.py:265`). Longer prompts are **silently truncated** (`model/tokenizer_ve.py:252`). Level 3–6 expressions and non-English text (the CLIP BPE is English-centric, so Arabic falls back to byte-level tokens) will exceed it. Before training, measure the token-length distribution per level and lang over the blend. If more than about 1 % is truncated, extend the context (e.g. 64): copy the 32 pretrained positions and initialise the rest from the last one. The encoder is causal, so existing positions are unaffected. Otherwise the comparison partly measures truncation, not the architecture.
- No horizontal flip in augmentations (left/right expressions).
- `clean_expression` strips a leading article, which is harmless for SAM3.

**Instance cap: drop, never subsample (decided).** The decoder has 200 o2o queries. A query with more than 200 instances is dropped (that query only; the image's other queries stay), and every kept query is `is_exhaustive=True`. Subsampling would be wrong here: the configs use `weak_loss: False`, so `is_exhaustive` is not even read, and the unsampled instances would be trained as background. Per the Falcon toml, dropping above 100 removed only about 1 % of masks, so the loss of data is small. **The AR comparison must drop the same queries**, or report those queries separately.

## 4. Loader design: a SAM3 energon TaskEncoder

Reuse energon and the tokenizer-free schema code from torchtitan. Don't reuse
`FalconTrainingTaskEncoder` itself: its packing, token sequence, `smart_resize`
grid and canvas padding are all Falcon-specific.

**Location.** Make `sam3/train/data/energon/` depend on
`torchtitan-internal` as an installed package, and import only:

- **Schema and filtering:** `dataset_utils.parse_attribute_filter`, `_apply_expression_attributes` (level/lang/alternatives), and `TaskConfig` / `clean_expression` from `task_processors`.
- **Masks:** `task_processors._decode_masks_resized` (RLE straight onto a target grid via `_rle_to_grid`; the 64–149× faster path), and `falcon_taskencoder.PackedMask`.
- **Runtime:** `disable_numpy_hugepages` and `raise_open_file_limit` from `energon_datasets.py`. Copy these two small functions rather than importing that module, which pulls in the tokenizer. Both fixed real hangs on B200 nodes.

This keeps one implementation of the annotation schema. Two small upstream
changes to torchtitan make the import cheap:

1. Make `spacy.load("en_core_web_sm")` in `task_processors.py:56` lazy. Today it runs at import in every worker even when rewrites are off.
2. Expose the schema helpers as public names. They are currently `_`-prefixed.

Same energon fork, same `.nv-meta`, same mounts and blend yaml format. The SAM3
blend is a copy of `perception_v15.yaml` with the non-segmentation sources
removed and `task: referring_segmentation` (no `tasks: all`).

**`Sam3TaskEncoder(DefaultTaskEncoder)`:**

- `encode_sample` (`@stateless(restore_seeds=True)`, **not** a generator) does, in order:
  1. Decode the image and convert it to RGB.
  2. Resize to 1008×1008. SAM3 was trained with a square resize (`RandomResizeAPI square: true`), so keep that; optionally add scale jitter later.
  3. Keep uint8 CHW.
  4. Parse the task dicts as above.
  5. Keep every positive expression and every file negative (no per-image cap).
  6. Shuffle the queries.
  7. Drop queries with more than 200 instances.
  8. Decode masks with `_decode_masks_resized(rles, (1008, 1008))` into `PackedMask`, and compute tight boxes from the masks.
  9. Raise `SkipSample` if no query is left, and count skips per source.
- Don't copy the Falcon `try/except → None` pattern: count and log skips instead of swallowing errors.
- `batch` / `encode_batch` build the model batch **in the worker**. The batch holds:
  - `img_batch uint8[B,3,1008,1008]` and the list of unique texts (plus token ids, once tokenization moves to the workers)
  - `img_ids`, `text_ids`, `num_boxes`, `boxes`, `boxes_padded`, `is_exhaustive`, `object_ids(_padded)`
  - masks as `uint8[N,1008,126]` bit-packed, with no `raw_images`
  - a `pin_memory()` method on the batch dataclass

  On the GPU, unpack the masks (or point-sample straight from bits) and normalize the image.
- **Batching: a query budget, not an image count.** Queries per image are uncapped, so the rank batch is filled to `max_queries_per_batch` (plus a ceiling on images).
  - An image with more queries than the budget is split into several samples of the same image. SAM3 queries are independent (each sees only the image and its own text, like Falcon's per-query attention), so splitting loses nothing; it only costs another ViT pass for that image.
  - The budget is set from the Phase 0 sweep: the largest that fits memory at good throughput.
  - Energon has no cross-rank balance, but a query budget also evens out per-rank step time.
- **Augmentation:** none to start (v1.5 runs with none, since `aug_prob_threshold` is unset). Add later:
  - SAM3-style random resize (min 480) + pad, on GPU or with nearest masks.
  - Photometric augs.
  - **No horizontal flip** while level ≥3 expressions are in the mix.

  Falcon's `apply_perception_augmentations` re-decodes at full resolution and leaves out-of-frame instances as empty positives. Don't reuse it as-is.
- **Resume:** save `EnergonSavableDataLoader.state_dict()` (per dp-rank) in the trainer checkpoint, and resume with the same world size and `num_workers`. Known caveats from `V15_DATALOADER_RESUME.md`:
  - Resume is not sample-exact.
  - Only append to the blend on resume.

  Not packing removes two of the three suspected sources of resume drift.
- **Host RAM:** Falcon OOM-killed a 1.7 TB node at 16 workers × 8 ranks. Decoded samples sit in the shuffle buffer, which is why masks stay bit-packed until the GPU. Start at 8–12 workers per rank and a modest `shuffle_buffer_size`, and measure on a full node, not one rank.

## 5. Implementation plan (phased, each phase measured)

**Phase 0 — Baseline and harness (1–2 days)**
- Add a synthetic-batch benchmark: B images × Q queries × K instances, real model, bf16. It reports img/s/GPU, queries/s, peak memory and a `torch.profiler` trace.
- Record the baseline on one B200 node with the current trainer for B ∈ {4, 8}, Q ∈ {8, 16, 32}. Every later change is judged against these numbers.

**Phase 1 — Trainer for streaming scale (2–3 days)**
- Step-based loop:
  - `max_steps`; the scheduler is driven by `step / max_steps`.
  - Checkpoint every N steps, plus on SIGTERM / preemption. The checkpoint holds model, optimizer, step, RNG and dataloader state.
  - Eval every M steps.
  - Keep the epoch path for the existing RF100/ODinW configs.
- Remove per-step syncs:
  - Accumulate loss components on the GPU and reduce/log every `log_freq` steps.
  - Do the NaN/Inf check on the GPU and act on it at log time, or skip the optimizer step via `torch.where`.
- bf16 without GradScaler. Keep fp32 master weights and AdamW (fused).
- Freeze or exclude the geometry encoder for text-only training, then set `find_unused_parameters=False` (and try `static_graph=True`).
- Prefetch the next batch to the GPU on a side stream; pinned batch.
- Checkpoint I/O: rank 0 writes to shared FS with a temp-file rename (as today), asynchronously. The model is ~850M params, so plain DDP is fine and FSDP isn't needed: params + grads + AdamW states are ~14 GB per GPU.

**Phase 2 — Energon SAM3 loader (3–5 days)**
- Build `Sam3TaskEncoder` and the SAM3 blend yaml as in §4, plus the two small torchtitan changes (lazy spaCy, public helpers).
- Unit tests on the `scripts/dataloader_snapshot` fixtures:
  - expression → masks alignment
  - negatives have 0 targets
  - level/lang filtering
  - the instance cap sets `is_exhaustive=False`
  - boxes match the masks
- A render script to spot-check batches (mask overlays + phrase).
- Loader-only throughput test on a full node: samples/s per rank must exceed the trainer's consumption with margin. Also measure worker CPU and host RAM over 1k steps.

**Phase 3 — Loss, matcher and model efficiency (3–4 days)**
- Point-sampled mask loss: `num_sample_points=12544`, `oversample_ratio=3.0`, `importance_sample_ratio=0.75`. Targets are sampled from the bit-packed masks without materializing the full float targets. Compare against full-res loss on a short run (mask AP / cgF1 within noise).
- **Masks only for matched queries.** The seg head computes `pred_masks` for all 400 queries per prompt (200 o2o + 200 o2m, at 288², `model/maskformer_segmentation.py:327-337`): ~66 MB bf16 per prompt, more in fp32 in the loss. The matcher uses only logits and boxes, so in training: run matching before the mask predictor, then compute mask logits only for matched o2o/o2m queries, ideally only at the sampled points. Memory per prompt then scales with #instances, not #queries. This is the main lever for many queries per GPU. Inference is unchanged.
- One matching call for the final + aux layers (`repeat_batch`) and for o2m. That is one D2H copy per step instead of ~12. Run the scipy calls in a thread pool, or try a GPU batched assignment. Fix L3 and drop the CPU OOM fallback (L4).
- Selective activation checkpointing:
  - Turn it off for the decoder, boxRPB and seg head first, then the fusion encoder; keep it on the ViT.
  - Keep each change if memory allows the target B×Q.
- `torch.compile` the ViT trunk (and the pixel decoder). Check DDP interaction (`optimize_ddp`) and recompiles from the variable query count. Keep the query count static per batch by padding to `max_queries_per_batch` if it helps.
- Attention kernels:
  - ViT: cuDNN/flash SDPA (FA3 path exists for Hopper).
  - Decoder: memory-efficient SDPA because of the float boxRPB bias. Store the bias in bf16.

**Phase 4 — Scale-out and recipe (ongoing)**
- Multi-node launch through the existing Nebius helm chart (`infra/launch_job_nebius.sh`, `vlm_nebius.yaml`) with a SAM3 entrypoint (`torchrun -m sam3.train.train_streaming ...`). Use the same mounts, `uv` lock + SAM3 extras, `ulimit -n`, and `init_timeout_seconds` / `train_timeout_seconds` for the first buffer fill.
- LR / batch:
  - Start from the released recipe's per-component LRs (transformer 8e-4, vision 2.5e-4, language 5e-5 at `lr_scale` 0.1, layer decay 0.9), with step-based warmup + inverse-sqrt/cosine.
  - Keep `scale_by_find_batch_size` (sqrt of the per-rank find batch).
  - Sweep `lr_scale` on 1 node before going wide.
- Eval, run as a separate job on saved checkpoints, not inline:
  - A held-out slice of each grounding source (level 0/1).
  - The SA-Co/Gold image configs already in `sam3/train/configs/gold_image_evals/`, to catch regression of the original capability.
  - Domain sets.

**Expected effect:**
- D1–D7 remove the loader bottleneck: decode at grid, no raw images, packed masks, pinned uint8 batches.
- T1, T4 and L2 remove almost all per-step host syncs.
- L1 and selective act-ckpt free memory for larger B×Q.

The numbers come from Phase 0; nothing here has been measured yet.

## 6. Decisions

Decided:

1. **Levels:** all levels and languages (architecture stress test vs AR). Text-length truncation must be measured first (§3).
2. **Queries per image:** as many as memory and throughput allow. A per-batch query budget, set by the Phase 0 sweep; all file negatives kept.
3. **Instances:** keep the default 200 queries; drop queries with more than 200 instances, never subsample. All data is exhaustive.
4. **Start:** the released SAM3 image checkpoint, full model (its PE-derived vision and text encoders, fusion encoder, decoder, seg head).

Open:

5. **Frozen or trained text encoder,** given that its training lr is 10× lower in the released recipe. Proposal: train it, since the AR baseline trains its text side.
6. **Where the encoder lives:** `sam3` depending on `torchtitan-internal` (proposal), or vendored helpers.
7. **Extending the text context** beyond 32 tokens, depending on the truncation scan.

### Cost of more than 200 object queries (for reference)

Per prompt, the per-query costs are:

- **Decoder:** self-attention O(Q²) is negligible at Q = 400. Cross-attention O(Q·5184) has a boxRPB bias of 8 × Q × 5184 per layer (~33 MB bf16 at Q = 400). FFNs are O(Q).
- **Mask head:** Q × 288² logits, plus the DAC o2m copy.
- **Hungarian matching:** on the CPU, growing super-linearly in Q.
- **New query embeddings:** queries beyond 200 start untrained.

Doubling to 400 roughly doubles decoder and mask-head memory and time per prompt, which directly cuts queries per GPU. With masks computed only for matched queries (Phase 3), the mask-head term no longer scales with Q. Not worth it for ~1 % of masks; revisit only if dense queries become a target.
