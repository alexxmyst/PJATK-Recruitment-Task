"""Run with python -m catalogue_quality."""

import argparse
from pathlib import Path

from .pipeline import TrainingConfig, audit_data, predict_pipeline, train_pipeline

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data/dsc-pjatk-catalogue-quality-challenge"


def main():
    parser = argparse.ArgumentParser(description="Catalogue corruption detection pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Validate and describe the competition files")
    audit.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    audit.add_argument("--output-dir", type=Path, default=Path("outputs/audit"))
    train = commands.add_parser("train", help="Nested seller CV, saved ensemble, and submission")
    train.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    train.add_argument("--output-dir", type=Path, default=Path("outputs/catalogue_v1"))
    for name, default in [("folds", 5), ("inner-folds", 3), ("fold-seed", 42),
                          ("model-seed", 42), ("iterations", 450), ("depth", 5), ("threads", 4)]:
        train.add_argument("--" + name, type=int, default=default)
    train.add_argument("--learning-rate", type=float, default=.05)
    train.add_argument("--l2-leaf-reg", type=float, default=5.)
    predict = commands.add_parser("predict", help="Generate predictions from a saved ensemble")
    predict.add_argument("--model-dir", type=Path, required=True)
    predict.add_argument("--test", type=Path, default=DEFAULT_DATA / "test.csv")
    predict.add_argument("--sample-submission", type=Path)
    predict.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "audit":
            report = audit_data(args.data_dir, args.output_dir)
            print(f"Audited {report['train']['rows']} training and {report['test']['rows']} test rows")
        elif args.command == "train":
            config = TrainingConfig(**{name: getattr(args, name) for name in TrainingConfig.__dataclass_fields__})
            metrics = train_pipeline(args.data_dir, args.output_dir, config)
            print(f"Mean ROC-AUC: {metrics['mean_fold_roc_auc']:.6f}")
            print(f"OOF Average Precision: {metrics['oof']['average_precision']:.6f}")
        else:
            p = predict_pipeline(args.model_dir, args.test, args.output_dir, args.sample_submission)
            print(f"Predicted {len(p)} rows")
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(f"Artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
