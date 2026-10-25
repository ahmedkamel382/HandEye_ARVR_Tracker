#!/usr/bin/env python
"""Compile the project's two Random Forests to native C with Treelite + TL2cgen.

    pip install treelite tl2cgen scikit-learn joblib     # once, in the project's .venv
    python core_engine/tools/export_forests.py            # run from the repo root; exports BOTH forests
    python core_engine/tools/export_forests.py --only hand

  gaze : ai_pipeline/gaze_intent/gesture_model.pkl   (7 features, 4 classes)  -> core_engine/generated/gaze_forest/
  hand : assets/hand_gesture_rf.pkl                  (24 features, 3 classes) -> core_engine/generated/hand_forest/

Each output folder contains
    header.h, main.c, tu*.c   the forest as plain if/else C code (every global symbol is prefixed gaze_ / hand_
                              so both forests can be linked into one executable)
    parity_cases.csv          cases computed by scikit-learn ITSELF, including points exactly on split thresholds
                              (used by  core_engine --selftest-hand-forest)
    MODEL_INFO.txt            what was exported (sha256 of the pkl, library versions)

RE-RUN THIS EVERY TIME A .pkl IS RETRAINED, then delete the build folder and re-configure.
"""
import argparse
import ctypes
import datetime
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HAND_FEATURE_NAMES = (
    [f"ang_{n}" for n in ("thumb_cmc", "thumb_mcp", "thumb_ip", "index_mcp", "index_pip", "index_dip",
                          "middle_mcp", "middle_pip", "middle_dip", "ring_mcp", "ring_pip", "ring_dip",
                          "pinky_mcp", "pinky_pip", "pinky_dip")]
    + [f"thumb_to_{f}_tip" for f in ("index", "middle", "ring", "pinky")]
    + [f"wrist_to_{f}_tip" for f in ("thumb", "index", "middle", "ring", "pinky")]
)

SPECS = {
    "gaze": dict(rel_pkl=("ai_pipeline", "gaze_intent", "gesture_model.pkl"), out="gaze_forest", n_features=7,
                 prefix="gaze_", bundle=False),
    "hand": dict(rel_pkl=("assets", "hand_gesture_rf.pkl"), out="hand_forest", n_features=24,
                 prefix="hand_", bundle=True),
}

# global symbols of the code TL2cgen generates (see header.h); predict_unitN depends on --units
SYMBOL_RE = re.compile(r"\b(get_num_target|get_num_class|get_num_feature|get_threshold_type|get_leaf_output_type|"
                       r"predict_unit\d+|predict|postprocess)\b")


def fail(msg):
    print(f"\n[export_forests] ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load_model(path, spec):
    try:
        import joblib
    except ImportError:
        fail("joblib is not installed in this Python environment")
    if not os.path.exists(path):
        fail(f"model file not found: {path}")
    obj = joblib.load(path)
    if isinstance(obj, dict) and "model" in obj:                  # hand_gesture_rf.pkl = {"model", "feature_names"}
        names = list(obj.get("feature_names", []))
        if spec["n_features"] == 24 and names != HAND_FEATURE_NAMES:
            fail("the model's feature_names differ from hand_features.FEATURE_NAMES: the C++ feature code "
                 "(core_engine/include/hand_math.hpp) would feed it different inputs. Update both together.")
        model = obj["model"]
    else:
        model = obj
    for attr in ("predict", "predict_proba", "classes_", "n_features_in_"):
        if not hasattr(model, attr):
            fail(f"{path} does not look like a fitted scikit-learn classifier (missing .{attr})")
    if int(model.n_features_in_) != spec["n_features"]:
        fail(f"the model expects {model.n_features_in_} features, the engine computes {spec['n_features']}")
    classes = [int(c) for c in model.classes_]
    if classes != list(range(len(classes))):
        fail(f"class labels are {classes}; the engine expects 0..N-1 in order (state codes)")
    return model


def make_cases(model, n, nf, seed=2024):
    """Half uniform random points, half points placed on / 1 float32-ulp around real split thresholds."""
    rng = np.random.default_rng(seed)
    thr = [[] for _ in range(nf)]
    for est in getattr(model, "estimators_", [model]):
        tree = est.tree_
        for f, t in zip(tree.feature, tree.threshold):
            if f >= 0:
                thr[f].append(float(t))
    lo, hi = np.zeros(nf), np.ones(nf)
    for f in range(nf):
        if thr[f]:
            a, b = min(thr[f]), max(thr[f])
            span = max(b - a, 1e-6)
            lo[f], hi[f] = a - 0.25 * span, b + 0.25 * span
    n_uni = n // 2
    X = rng.uniform(lo, hi, size=(n, nf))
    usable = [f for f in range(nf) if thr[f]]
    if usable:
        for i in range(n_uni, n):
            f = usable[rng.integers(len(usable))]
            t = np.float32(thr[f][rng.integers(len(thr[f]))])
            X[i, f] = rng.choice([t, np.nextafter(t, np.float32(np.inf)), np.nextafter(t, np.float32(-np.inf))])
    X = X.astype(np.float32)                       # the engine rounds to float32, exactly like scikit-learn
    proba = model.predict_proba(X)
    k = np.argmax(proba, axis=1)
    return X, np.asarray(model.classes_)[k].astype(int), proba[np.arange(len(k)), k]


def write_cases(path, X, y, conf, with_conf):
    with open(path, "w", newline="\n") as fh:
        if with_conf:
            fh.write(f"# f0..f{X.shape[1] - 1} (float32), class, confidence -- generated by export_forests.py "
                     "from scikit-learn's own predict_proba()\n")
        else:
            fh.write(f"# f0..f{X.shape[1] - 1} (float32), label -- generated by export_forests.py "
                     "from scikit-learn's own predict()\n")
        for row, lab, c in zip(X, y, conf):
            line = ",".join("%.9g" % float(v) for v in row) + f",{int(lab)}"
            fh.write(line + (",%.17g\n" % float(c) if with_conf else "\n"))


def prefix_symbols(directory, prefix):
    """gaze_predict, hand_predict, ...: lets two generated forests coexist in one executable."""
    for f in os.listdir(directory):
        if f.endswith((".c", ".h")):
            p = os.path.join(directory, f)
            with open(p, "r", newline="") as fh:
                text = fh.read()
            text = SYMBOL_RE.sub(lambda m: prefix + m.group(1), text)
            with open(p, "w", newline="") as fh:
                fh.write(text)


def find_c_compiler():
    for cc in (os.environ.get("CC"), "cc", "gcc", "clang"):
        if cc and shutil.which(cc):
            return shutil.which(cc)
    return None


def verify_with_c_compiler(out_dir, shim, prefix, nf, X, y, conf, with_conf):
    """Compile generated code + the real engine shim into a shared library and compare every case."""
    cc = find_c_compiler()
    if cc is None or sys.platform.startswith("win"):
        print("  (no C compiler for a Python-side check: build core_engine, then run its --selftest option)")
        return None
    sources = [os.path.join(out_dir, f) for f in sorted(os.listdir(out_dir)) if f.endswith(".c")] + [shim]
    with tempfile.TemporaryDirectory() as tmp:
        lib = os.path.join(tmp, "forest_check.so")
        cmd = [cc, "-O1", "-shared", "-fPIC", f"-DFOREST_PREFIX={prefix}", "-I", out_dir, "-o", lib] + sources
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            fail("the generated C code did not compile:\n" + res.stderr[-2000:])
        dll = ctypes.CDLL(lib)
        proba = getattr(dll, f"{prefix}forest_proba")
        proba.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        proba.restype = ctypes.c_int
        bad = 0
        for row, lab, c in zip(X, y, conf):
            arr = (ctypes.c_float * nf)(*[float(v) for v in row])
            out = (ctypes.c_double * 16)()
            n = proba(arr, out, 16)
            k = max(range(n), key=lambda i: (out[i], -i)) if n else -1
            if k != int(lab) or (with_conf and abs(out[k] - float(c)) > 1e-9):
                bad += 1
        return bad


def export_one(which, args, repo, here):
    spec = SPECS[which]
    pkl = getattr(args, f"{which}_pkl") or os.path.join(repo, *spec["rel_pkl"])
    out_dir = os.path.join(repo, "core_engine", "generated", spec["out"])
    nf, prefix, with_conf = spec["n_features"], spec["prefix"], spec["bundle"]

    import sklearn
    import treelite
    import tl2cgen

    print(f"\n=== {which} forest ===\nmodel : {pkl}")
    model = load_model(pkl, spec)
    n_trees = len(getattr(model, "estimators_", [model]))
    print(f"        {type(model).__name__}, {n_trees} tree(s), classes {[int(c) for c in model.classes_]}")

    print("convert: Treelite -> TL2cgen C code")
    tl_model = treelite.sklearn.import_model(model)
    with tempfile.TemporaryDirectory() as tmp:
        gen = os.path.join(tmp, "gen")
        os.makedirs(gen)
        tl2cgen.generate_c_code(tl_model, gen, params={"parallel_comp": max(1, args.units)})
        os.makedirs(out_dir, exist_ok=True)
        for f in os.listdir(out_dir):                          # drop stale output, keep nothing we did not write
            if f.endswith((".c", ".h", ".json")) or f in ("MODEL_INFO.txt", "parity_cases.csv"):
                os.remove(os.path.join(out_dir, f))
        for f in os.listdir(gen):
            shutil.copy(os.path.join(gen, f), os.path.join(out_dir, f))
    if not os.path.exists(os.path.join(out_dir, "main.c")) or not os.path.exists(os.path.join(out_dir, "header.h")):
        fail("TL2cgen did not produce main.c / header.h (version change?). Check the installed tl2cgen.")
    prefix_symbols(out_dir, prefix)

    X, y, conf = make_cases(model, args.cases, nf)
    write_cases(os.path.join(out_dir, "parity_cases.csv"), X, y, conf, with_conf)

    sha = hashlib.sha256(open(pkl, "rb").read()).hexdigest()
    with open(os.path.join(out_dir, "MODEL_INFO.txt"), "w") as fh:
        fh.write(f"exported     : {datetime.datetime.now().isoformat(timespec='seconds')}\n"
                 f"pkl          : {os.path.basename(pkl)}  sha256 {sha}\n"
                 f"model        : {type(model).__name__}, {n_trees} trees, classes {[int(c) for c in model.classes_]}\n"
                 f"symbol prefix: {prefix}\n"
                 f"scikit-learn : {sklearn.__version__}\ntreelite     : {treelite.__version__}\n"
                 f"tl2cgen      : {tl2cgen.__version__}\nparity cases : {len(y)} (half on split thresholds)\n")
    print(f"wrote  : {out_dir}\n         {', '.join(sorted(os.listdir(out_dir)))}")

    print("verify : generated C vs scikit-learn")
    bad = verify_with_c_compiler(out_dir, os.path.join(here, "..", "src", "forest_shim.c"), prefix, nf, X, y, conf,
                                 with_conf)
    if bad is None:
        pass
    elif bad:
        fail(f"{bad} of {len(y)} cases disagree with scikit-learn. Do NOT use this build.")
    else:
        print(f"         PASS: all {len(y)} cases identical to scikit-learn")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=("gaze", "hand"), help="export just one forest (default: hand; the gaze models are .dll files now)")
    ap.add_argument("--gaze-pkl", default=None)
    ap.add_argument("--hand-pkl", default=None)
    ap.add_argument("--cases", type=int, default=6000, help="rows in each parity_cases.csv")
    ap.add_argument("--units", type=int, default=4, help="split the C code into this many files (MSVC compiles faster)")
    args = ap.parse_args()

    try:
        import sklearn  # noqa: F401
        import treelite  # noqa: F401
        import tl2cgen  # noqa: F401
    except ImportError as e:
        fail(f"{e}. Install the tools with:  pip install treelite tl2cgen scikit-learn joblib")
    except OSError as e:                                           # tl2cgen on Windows, plain venv
        fail(f"{e}\n  On Windows with a plain venv create the folder it looks for:  mkdir .venv\\Library\\bin")

    todo = [args.only] if args.only else ["hand"]      # the gaze models are .dll files now (ai_pipeline/gaze_intent/models); "--only gaze" exports the OLD 7-feature forest
    for which in todo:
        spec = SPECS[which]
        pkl = getattr(args, f"{which}_pkl") or os.path.join(repo, *spec["rel_pkl"])
        if which == "hand" and not args.only and not os.path.exists(pkl):
            print(f"\n=== hand forest ===\nSKIPPED: {pkl} not found. The engine will use the rough geometric fallback "
                  "(like hand_tracker.py does) - train the model first.")
            continue
        export_one(which, args, repo, here)
    print("\nNext: delete the build folder, re-configure, build, then run\n"
                    "      core_engine --selftest-hand-forest core_engine/generated/hand_forest/parity_cases.csv")


if __name__ == "__main__":
    main()
