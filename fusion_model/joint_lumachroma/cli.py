"""Command-line adapters shared with the runnable Colab notebook."""
import argparse
from pathlib import Path

from fusion_model.inference import select_device
from .config import CFG, HVI_COMMIT, METHODS, RET_COMMIT
from .datasets import dataset_configs, file_map
from .inference import all_predictions, predict_hvi, predict_ret
from .io_utils import digest, load_parameters, locked_json, read_rgb, save_png
from .metrics import rgb8_hvi_style, rgb8_ret_style
from .model_loader import load_models, resolve_weights, verify_source
from .protocol import protocol_identity, write_manifest


def parser_for(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--dataset-root", action="append", default=[], metavar="DATASET=PATH",
                        help="Override a dataset's root (repeatable); internal Train/Test layout is unchanged")
    parser.add_argument("--datasets", nargs="+", choices=list(CFG), default=list(CFG))
    parser.add_argument("--ret-repo", type=Path, required=True)
    parser.add_argument("--hvi-repo", type=Path, required=True)
    parser.add_argument("--ret-weights", type=Path, required=True)
    parser.add_argument("--hvi-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def setup(args):
    overrides = {}
    for entry in args.dataset_root:
        ds, separator, path = entry.partition("=")
        if not separator or ds not in CFG or not path or ds in overrides:
            raise ValueError("Use --dataset-root DATASET=PATH once per known dataset")
        overrides[ds] = Path(path)
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("Duplicate datasets are not allowed")
    configs = dataset_configs(args.data_root, overrides)
    verify_source(args.ret_repo, RET_COMMIT, "basicsr/models/archs")
    verify_source(args.hvi_repo, HVI_COMMIT, "net")
    return configs, resolve_weights(args.datasets, args.ret_weights, args.hvi_weights), select_device(args.device)


def fit_main(argv=None):
    from .train_coefficients import fit_experiment
    parser = parser_for("Fit all five fusion ablations using original Train pairs only")
    parser.add_argument("--fit-train-count", type=int, default=None)
    args = parser.parse_args(argv)
    configs, weights, device = setup(args)
    params = fit_experiment(configs, args.datasets, weights, args.ret_repo, args.hvi_repo, device,
                            args.output, args.fit_train_count)
    print("Train-only weights frozen:", params["joint"])
    return params


def evaluate_main(argv=None):
    from .evaluation import evaluate_experiment
    parser = parser_for("Evaluate frozen selected weights; no fitting or Test-based selection")
    parser.add_argument("--weights-json", type=Path, required=True)
    parser.add_argument("--smoke-max-test", type=int, default=None)
    parser.add_argument("--preview-count", type=int, default=3)
    parser.add_argument("--no-save-predictions", action="store_true")
    args = parser.parse_args(argv)
    configs, weights, device = setup(args)
    summary = evaluate_experiment(configs, args.datasets, weights, args.ret_repo, args.hvi_repo, device,
                                  args.output, args.weights_json, args.smoke_max_test,
                                  not args.no_save_predictions, args.preview_count)
    print("SMOKE" if args.smoke_max_test is not None else "FULL TEST")
    print(summary[["dataset", *[m + "_psnr_float" for m in METHODS]]].to_string(index=False))
    return summary


def run_main(argv=None):
    parser = parser_for("Enhance images using frozen joint regularized weights; no ground truth")
    # Inference selects one dataset-specific checkpoint pair explicitly.
    parser.add_argument("--dataset", choices=list(CFG), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--weights-json", type=Path, required=True)
    parser.add_argument("--compare", action="store_true", help="Save all seven methods")
    args = parser.parse_args(argv)
    args.datasets = [args.dataset]
    configs, weights, device = setup(args)
    params = load_parameters(args.weights_json, args.datasets)
    paths = ([args.input] if args.input.is_file() else
             [p for _, p in sorted(file_map(args.input).items())])
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError("Input stems collide after PNG conversion")
    args.output.mkdir(parents=True, exist_ok=False)
    lock = protocol_identity(configs, args.datasets, weights)
    lock.update(parameter_selection=params,
                inputs={p.name: digest(p) for p in paths}, compare=args.compare)
    locked_json(args.output / "inference_lock.json", lock)
    ret, hvi = load_models(weights[args.dataset], args.dataset, args.ret_repo, args.hvi_repo, device)
    methods = METHODS if args.compare else ["ycc_joint_regularized"]
    for path in paths:
        low = read_rgb(path)
        rr, hh = predict_ret(ret, low, device), predict_hvi(hvi, low, device)
        preds = all_predictions(rr, hh, params["local"][args.dataset], params["joint"])
        for method in methods:
            image = rgb8_ret_style(preds[method]) if method == "ret" else rgb8_hvi_style(preds[method])
            save_png(args.output / method / (path.stem + ".png"), image)
    write_manifest(args.output, lock, params, device, methods, stage="inference")
    print(f"Saved {len(paths)} enhanced images to {args.output}")
