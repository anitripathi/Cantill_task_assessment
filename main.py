"""Run the spend pipeline for one company from a YAML configuration."""
import argparse
from spendintel.pipeline import run


def main():
    parser = argparse.ArgumentParser(description="Config-driven procurement spend intelligence pipeline")
    parser.add_argument("--config", required=True, help="Company YAML configuration")
    parser.add_argument("--data-dir", required=True, help="Root directory containing configured source files")
    parser.add_argument("--output", required=True, help="Destination CSV path")
    parser.add_argument("--mode", default="auto", choices=["auto", "zero_shot"],
                        help="zero_shot ignores labels; use for an unseen company")
    parser.add_argument("--train-dir", default=None,
                        help="Original labelled data root when scoring A2/B2 blind-test records")
    parser.add_argument("--embedder", default="auto", choices=["auto", "e5", "tfidf"],
                        help="E5 when available, otherwise the local TF-IDF fallback")
    args = parser.parse_args()
    run(args.config, args.data_dir, args.output, args.mode, args.embedder, train_dir=args.train_dir)


if __name__ == "__main__":
    main()
