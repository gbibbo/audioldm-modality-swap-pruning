#!/usr/bin/env python3
"""RECOVERY-METRIC-AUDIT-1 — post-hoc metric-concordance diagnostic (CPU).

Computes, on the EXISTING frozen V1.1 outputs only, the metrics that establish the published
"recovery" (PANN top-10 event capture, KL, FAD, +FD) and assembles a concordance table against the
already-frozen CLAP / Human-CLAP numbers. This is OUTCOME-MOTIVATED / POST-HOC (see
docs/recovery_metric_audit_1.md). It does NOT and CANNOT change V1.1 PASS=FALSE. No gate/SESOI/
composite/majority-vote. No new audio, no new selection.

Provenance (frozen): PANNs Cnn14-16k (ckpt/Cnn14_16k_mAP=0.438.pth) via audioldm_eval's own
get_featuresdict + WaveDataset preprocessing (identical for all clips); clipwise_output=sigmoid(logits)
so top-10 capture ranking == logits ranking; KL = audioldm_eval.metrics.kl.calculate_kl formula
(softmax over logits, EPS 1e-6, mean over classes, KL(gen||gt)), VALIDATED against the library
aggregate; FD = audioldm_eval calculate_fid on 2048; FAD = VGGish (harritaylor/torchvggish).
Per-ytid: average the 2 replicates within ytid, bootstrap 96 ytids, B=10000, seed 20260828, NO gate.

Run:  OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/recovery_metric_audit_1.py \
          --out configs/research/recovery_metric_audit_1_result.json
"""
from __future__ import annotations
import argparse, csv, glob, json, os, re, sys, hashlib
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, os.getcwd())
import numpy as np
import torch

MANIFEST = "configs/research/reversal_v1_1_audiocaps_manifest.json"
LABEL_JSON = "data/dataset/metadata/audiocaps/datafiles/audiocaps_test_label.json"
LABELS_CSV = "artifacts/m0_baseline_reproduction/class_labels_indices.csv"
ZIP_AUDIOS = "data/dataset/audioset/zip_audios"
GEN_DIR = ("/teamspace/jobs/reversal-v11-gen-1/artifacts/audioldm-modality-swap-pruning/"
           "artifacts/icassp_gate0/reversal_v1_1_gen")
V1_RESULT = "configs/research/reversal_v1_1_result.json"
SYS_PREFIX = {"dense": "dense_noadapter",
              "pruned": "p1_pruned_ema_reconstructed_noadapter",
              "recovered": "p1_recovered_noadapter"}
SYSTEMS = ["dense", "pruned", "recovered"]
BOOT_SEED = 20260828
B = 10000
SR = 16000
EPS = 1e-6


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_mid_to_index():
    m = {}
    with open(LABELS_CSV) as fh:
        for row in csv.DictReader(fh):
            m[row["mid"]] = int(row["index"])
    return m


def load_gt_indices():
    """ytid -> frozenset of AudioSet class indices (from the label-string, identical across rows)."""
    mid2idx = load_mid_to_index()
    data = json.load(open(LABEL_JSON))["data"]
    by = {}
    for r in data:
        y = os.path.basename(r["wav"])
        y = y[1:] if y.startswith("Y") else y
        y = y[:-4] if y.endswith(".wav") else y
        idxs = frozenset(mid2idx[m] for m in r["labels"].split(",") if m in mid2idx)
        by.setdefault(y, idxs)  # first row; verified identical across rows upstream
    return by


def find_ref(ytid):
    for pat in (f"{ZIP_AUDIOS}/**/Y{ytid}.wav", f"{ZIP_AUDIOS}/**/{ytid}.wav"):
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0]
    raise SystemExit(f"reference wav not found for {ytid}")


def symlink_dir(scratch, name, files):
    d = os.path.join(scratch, name)
    os.makedirs(d, exist_ok=True)
    for src, bn in files:
        dst = os.path.join(d, bn)
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)
    return d


def get_features(helper, wav_dir):
    """audioldm_eval get_featuresdict -> {basename: {'logits':np[527], '2048':np[2048]}}."""
    from audioldm_eval.datasets.load_mel import WaveDataset
    loader = torch.utils.data.DataLoader(WaveDataset(wav_dir, SR), batch_size=1, num_workers=4)
    fd = helper.get_featuresdict(loader)
    names = [os.path.basename(x) for x in fd["file_path_"]]
    logits = fd["logits"].cpu().numpy()
    emb = fd["2048"].cpu().numpy()
    return {n: {"logits": logits[i], "2048": emb[i]} for i, n in enumerate(names)}


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def kl_pair(gen_logits, gt_logits):
    """Per-example KL contribution matching audioldm_eval's `kullback_leibler_divergence_softmax`.

    The library aggregate is  kl_div(log(softmax(gen)+EPS), softmax(gt), reduction='sum') / N
    = mean_i( sum_c elt_ic ), where kl_div(input=logQ, target=P) elementwise = P*(logP - input),
    P=softmax(gt), Q=softmax(gen). So the per-example term (whose mean over examples == the library
    aggregate) is the SUM over classes, i.e. the standard KL(softmax(gt) || softmax(gen)) in nats.
    Lower = better. Direction (target=gt) is the audioldm_eval/AudioLDM convention, reproduced as-is."""
    pg = softmax(gen_logits)
    pt = softmax(gt_logits)
    elt = pt * (np.log(pt) - np.log(pg + EPS))
    return float(elt.sum())


def boot_indices(n, B, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(B, n))


def ci(vec):
    return [float(np.percentile(vec, 2.5)), float(np.percentile(vec, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="configs/research/recovery_metric_audit_1_result.json")
    ap.add_argument("--scratch", default=os.path.join(
        os.environ.get("SCRATCH", "/tmp/claude-1000"), "metric_audit_1"))
    ap.add_argument("--skip-fad", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="smoke: cap #ytids (result NOT durable)")
    args = ap.parse_args()
    os.makedirs(args.scratch, exist_ok=True)

    man = json.load(open(MANIFEST))["prompts"]
    ytids = [p["ytid"] for p in man]          # frozen order, index == prompt_index
    if args.limit:
        ytids = ytids[:args.limit]
    n = len(ytids)
    assert args.limit or n == 96
    gt = load_gt_indices()
    gt_idx = [gt[y] for y in ytids]
    gt_sizes = np.array([len(g) for g in gt_idx], dtype=np.float64)
    print(f"[setup] {n} ytids; GT sizes min/median/max = "
          f"{int(gt_sizes.min())}/{int(np.median(gt_sizes))}/{int(gt_sizes.max())}")

    # reference symlinks
    ref_files = [(find_ref(y), f"Y{y}.wav") for y in ytids]
    ref_dir = symlink_dir(args.scratch, "refs", ref_files)

    # gen symlinks (all + per system/replicate)
    gen_all = []
    per_sysrep = {}
    for s in SYSTEMS:
        pref = SYS_PREFIX[s]
        for r in (0, 1):
            fl = []
            for pi in range(n):
                bn = f"{pref}_p{pi}_r{r}.wav"
                src = os.path.join(GEN_DIR, bn)
                if not os.path.exists(src):
                    raise SystemExit(f"missing gen wav {src}")
                fl.append((src, bn)); gen_all.append((src, bn))
            per_sysrep[(s, r)] = symlink_dir(args.scratch, f"gen_{s}_r{r}", fl)
    gen_all_dir = symlink_dir(args.scratch, "gen_all", gen_all)

    # ---- PANN feature extraction (identical preprocessing for all) ----
    from audioldm_eval import EvaluationHelper
    helper = EvaluationHelper(SR, torch.device("cpu"))
    print("[panns] extracting gen features (576) ...")
    feat_gen = get_features(helper, gen_all_dir)
    print("[panns] extracting ref features (96) ...")
    feat_ref = get_features(helper, ref_dir)

    ref_logits = {y: feat_ref[f"Y{y}.wav"]["logits"] for y in ytids}

    # ---- per-ytid metric grids: shape [n_sys][n=96][rep=2] ----
    cap_hits = {s: np.zeros((n, 2)) for s in SYSTEMS}   # |G_i ∩ P_i|
    kl_grid = {s: np.zeros((n, 2)) for s in SYSTEMS}
    for s in SYSTEMS:
        pref = SYS_PREFIX[s]
        for pi in range(n):
            for r in (0, 1):
                lg = feat_gen[f"{pref}_p{pi}_r{r}.wav"]["logits"]
                top10 = set(np.argsort(lg)[::-1][:10].tolist())
                cap_hits[s][pi, r] = len(top10 & set(gt_idx[pi]))
                kl_grid[s][pi, r] = kl_pair(lg, ref_logits[ytids[pi]])

    # per-ytid reduce (mean over replicates)
    cap_ytid = {s: cap_hits[s].mean(axis=1) for s in SYSTEMS}   # numerator per ytid
    kl_ytid = {s: kl_grid[s].mean(axis=1) for s in SYSTEMS}

    # ---- bootstrap (paired ytid indices) ----
    idx = boot_indices(n, B, BOOT_SEED)
    out = {"capture": {}, "kl": {}}

    # capture recall = sum(hits)/sum(GT sizes)
    cap_point = {s: float(cap_ytid[s].sum() / gt_sizes.sum()) for s in SYSTEMS}
    cap_boot = {s: (cap_ytid[s][idx].sum(1) / gt_sizes[idx].sum(1)) for s in SYSTEMS}
    for s in SYSTEMS:
        out["capture"][s] = {"recall": cap_point[s], "ci95": ci(cap_boot[s])}
    for a, b in [("recovered", "pruned"), ("recovered", "dense"), ("pruned", "dense")]:
        d = cap_boot[a] - cap_boot[b]
        out["capture"][f"delta_{a}_{b}"] = {
            "point": cap_point[a] - cap_point[b], "ci95": ci(d)}
    out["capture"]["frac_ytid_recovered_gt_pruned"] = float(
        np.mean(cap_ytid["recovered"] > cap_ytid["pruned"]))
    out["capture"]["frac_ytid_recovered_eq_pruned"] = float(
        np.mean(cap_ytid["recovered"] == cap_ytid["pruned"]))

    # KL (lower better)
    kl_point = {s: float(kl_ytid[s].mean()) for s in SYSTEMS}
    kl_boot = {s: kl_ytid[s][idx].mean(1) for s in SYSTEMS}
    for s in SYSTEMS:
        out["kl"][s] = {"kl": kl_point[s], "ci95": ci(kl_boot[s])}
    for a, b in [("recovered", "pruned"), ("recovered", "dense"), ("pruned", "dense")]:
        d = kl_boot[a] - kl_boot[b]
        out["kl"][f"delta_{a}_{b}"] = {"point": kl_point[a] - kl_point[b], "ci95": ci(d)}
    out["kl"]["frac_ytid_recovered_lt_pruned"] = float(
        np.mean(kl_ytid["recovered"] < kl_ytid["pruned"]))

    # ---- KL provenance validation vs audioldm_eval.calculate_kl (matched basenames) ----
    from audioldm_eval.metrics.kl import calculate_kl
    # build featuresdicts with SHARED basenames: dense r0 as "gen", ref as "gt"
    shared = [f"m{pi}.wav" for pi in range(n)]
    g_log = torch.tensor(np.stack([feat_gen[f"{SYS_PREFIX['dense']}_p{pi}_r0.wav"]["logits"]
                                   for pi in range(n)]))
    t_log = torch.tensor(np.stack([ref_logits[ytids[pi]] for pi in range(n)]))
    fd1 = {"file_path_": shared, "logits": g_log}
    fd2 = {"file_path_": shared, "logits": t_log}
    lib_kl, _, _ = calculate_kl(fd1, fd2, "logits", same_name=True)
    mine = float(np.mean([kl_pair(g_log[i].numpy(), t_log[i].numpy()) for i in range(n)]))
    out["kl_validation"] = {
        "audioldm_eval_softmax_aggregate": float(lib_kl["kullback_leibler_divergence_softmax"]),
        "my_mean_of_perpair": mine,
        "abs_diff": abs(float(lib_kl["kullback_leibler_divergence_softmax"]) - mine)}
    print(f"[validate] KL lib={lib_kl['kullback_leibler_divergence_softmax']:.6f} "
          f"mine={mine:.6f} |diff|={out['kl_validation']['abs_diff']:.2e}")

    # ---- FD (PANN-2048, distributional) via library calculate_fid ----
    from audioldm_eval.metrics.fid import calculate_fid
    def fdict(names):  # names: list of basenames present in feat_gen/ref
        src = feat_gen if names[0] in feat_gen else feat_ref
        return {"file_path_": names, "2048": torch.tensor(np.stack([src[nm]["2048"] for nm in names]))}
    ref_names = [f"Y{y}.wav" for y in ytids]
    fd_ref = {"file_path_": ref_names,
              "2048": torch.tensor(np.stack([feat_ref[nm]["2048"] for nm in ref_names]))}
    out["fd_pann2048"] = {}
    for s in SYSTEMS:
        pref = SYS_PREFIX[s]
        row = {}
        for r in (0, 1):
            names = [f"{pref}_p{pi}_r{r}.wav" for pi in range(n)]
            g = {"file_path_": names,
                 "2048": torch.tensor(np.stack([feat_gen[nm]["2048"] for nm in names]))}
            row[f"r{r}"] = float(calculate_fid(g, fd_ref, "2048")["frechet_distance"])
        row["mean"] = (row["r0"] + row["r1"]) / 2
        out["fd_pann2048"][s] = row
        print(f"[fd] {s}: r0={row['r0']:.4f} r1={row['r1']:.4f} mean={row['mean']:.4f}")

    # ---- FAD (VGGish, replicate-specific, descriptive) ----
    if not args.skip_fad:
        try:
            frechet = helper.frechet
            out["fad_vggish"] = {}
            for s in SYSTEMS:
                row = {}
                for r in (0, 1):
                    sc = frechet.score(per_sysrep[(s, r)], ref_dir, recalculate=True)
                    if not isinstance(sc, dict):
                        raise RuntimeError(f"frechet.score returned {sc}")
                    row[f"r{r}"] = float(sc["frechet_audio_distance"])
                row["mean"] = (row["r0"] + row["r1"]) / 2
                out["fad_vggish"][s] = row
                print(f"[fad] {s}: r0={row['r0']:.4f} r1={row['r1']:.4f} mean={row['mean']:.4f}")
            out["fad_vggish"]["note"] = ("replicate-specific VGGish FAD, descriptive; n=96 distributional "
                                         "— no CI claimed (kept descriptive per protocol section 5)")
        except Exception as e:
            out["fad_vggish"] = {"error": str(e)}
            print("[fad] FAILED:", e)

    # ---- reuse frozen CLAP / Human-CLAP for the concordance table ----
    v1 = json.load(open(V1_RESULT))
    out["clap_frozen"] = {"source": V1_RESULT,
                          "PRIMARY": v1.get("PRIMARY"), "SECONDARY_humanclap": v1.get("SECONDARY_humanclap")}

    payload = {
        "artifact": "recovery_metric_audit_1_result",
        "status": "POST-HOC METRIC-CONCORDANCE DIAGNOSTIC — cannot change V1.1 PASS=FALSE",
        "protocol_doc_sha256": sha_file("docs/recovery_metric_audit_1.md"),
        "provenance": {
            "panns_ckpt": "ckpt/Cnn14_16k_mAP=0.438.pth",
            "panns_ckpt_sha256": sha_file("ckpt/Cnn14_16k_mAP=0.438.pth"),
            "labels_csv_sha256": sha_file(LABELS_CSV),
            "label_json_sha256": sha_file(LABEL_JSON),
            "gen_dir": GEN_DIR, "gen_job": "reversal-v11-gen-1@5f2fa55",
            "sr": SR, "eps": EPS, "boot_seed": BOOT_SEED, "B": B,
            "event_family_analysis": "OMITTED — exact Singh 7-family mapping not recoverable from repo (no invented families)",
            "operating_point": "3.84s / DDIM50 / guidance2.5 / single-gen / n=96 (same as music arm); "
                               "NOT comparable in absolute terms to Singh 10s/200steps/full-test",
            "clip_length_note": "gen 3.84s vs ref up to 10s, identical preprocessing for all systems",
        },
        "n_ytids": n,
        "gt_event_sizes": {"min": int(gt_sizes.min()), "median": float(np.median(gt_sizes)),
                           "max": int(gt_sizes.max()), "sum": float(gt_sizes.sum())},
        "results": out,
    }
    payload["artifact_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in payload.items() if k != "artifact_sha256"},
                   ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    json.dump(payload, open(args.out, "w"), indent=2, ensure_ascii=False)
    print("\n=== SUMMARY ===")
    print("capture recall  dense/pruned/recovered:",
          {s: round(cap_point[s], 4) for s in SYSTEMS})
    print("  delta rec-pruned:", round(out["capture"]["delta_recovered_pruned"]["point"], 4),
          out["capture"]["delta_recovered_pruned"]["ci95"],
          "| frac rec>pruned:", round(out["capture"]["frac_ytid_recovered_gt_pruned"], 3))
    print("KL (lower=better) dense/pruned/recovered:",
          {s: round(kl_point[s], 4) for s in SYSTEMS})
    print("  delta rec-pruned:", round(out["kl"]["delta_recovered_pruned"]["point"], 4),
          out["kl"]["delta_recovered_pruned"]["ci95"])
    print("FD-2048 mean:", {s: round(out["fd_pann2048"][s]["mean"], 3) for s in SYSTEMS})
    if "fad_vggish" in out and "dense" in out["fad_vggish"]:
        print("FAD-VGGish mean:", {s: round(out["fad_vggish"][s]["mean"], 3) for s in SYSTEMS})
    print("wrote", args.out, "sha", payload["artifact_sha256"][:16])
    return 0


if __name__ == "__main__":
    sys.exit(main())
