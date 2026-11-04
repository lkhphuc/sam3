"""cgF1 of a dumped prediction file, overall and per slice of the GT pairs.

The eval job (configs/falcon_v15/eval/falcon_heldout.yaml) writes
<dump_dir>/coco_predictions_segm.json (and _bbox). This script splits the GT
written by make_falcon_eval_set.py by the pair fields `level`, `lang` and
`source` (or any other field given in --by) and runs the cgF1 evaluator on
each slice. The predictions are shared: the evaluator only scores the pairs
present in each slice's GT.

    python scripts/falcon_v15/cgf1_by_slice.py --gt $FALCON_EVAL_DIR/gt.json \
        --pred $SAM3_EVAL_OUT/falcon_heldout/dumps/falcon_heldout/coco_predictions_segm.json \
        --by level lang source --out $SAM3_EVAL_OUT/falcon_heldout/cgf1_by_slice.json
"""

import argparse
import collections
import json
import os
import tempfile

KEYS = ("cgF1", "IL_MCC", "positive_micro_F1", "precision", "recall", "cgF1@0.5")


def evaluate(gt: dict, pairs: list, pred: str, iou_type: str) -> dict:
    from sam3.eval.cgf1_eval import CGF1Evaluator

    ids = {p["id"] for p in pairs}
    sub = dict(
        gt,
        images=pairs,
        annotations=[a for a in gt["annotations"] if a["image_id"] in ids],
    )
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(sub, f)
    try:
        res = CGF1Evaluator(gt_path=f.name, iou_type=iou_type).evaluate(pred)
    finally:
        os.unlink(f.name)
    prefix = f"cgF1_eval_{iou_type}_"
    return {k[len(prefix) :]: v for k, v in res.items() if k.startswith(prefix)}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--iou-type", default="segm", choices=["segm", "bbox"])
    ap.add_argument("--by", nargs="*", default=["level", "lang", "source"])
    ap.add_argument("--min-pairs", type=int, default=20, help="skip smaller slices")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.gt) as f:
        gt = json.load(f)
    results = {
        "all": {
            "n": len(gt["images"]),
            **evaluate(gt, gt["images"], args.pred, args.iou_type),
        }
    }
    negatives_by_image = collections.defaultdict(list)
    for p in gt["images"]:
        if p["num_instances"] == 0:
            negatives_by_image[p["file_name"]].append(p)
    for field in args.by:
        groups = collections.defaultdict(list)
        for p in gt["images"]:
            if p["num_instances"] > 0:
                groups[str(p.get(field, "none"))].append(p)
        for value, positives in sorted(groups.items()):
            if len(positives) < args.min_pairs:
                continue
            # A slice is its positive pairs plus the negatives asked on the same
            # images: cgF1 multiplies by the image-level MCC, which needs both.
            files = {p["file_name"] for p in positives}
            pairs = positives + [n for f in files for n in negatives_by_image[f]]
            results[f"{field}={value}"] = {
                "n": len(pairs),
                "n_pos": len(positives),
                **evaluate(gt, pairs, args.pred, args.iou_type),
            }
    cols = [k for k in KEYS if k in results["all"]]
    print(f"{'slice':40s} {'pairs':>6} " + " ".join(f"{c:>18s}" for c in cols))
    for name, r in results.items():
        print(
            f"{name[:40]:40s} {r['n']:6d} "
            + " ".join(f"{r.get(c, float('nan')):18.4f}" for c in cols)
        )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
