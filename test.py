"""Independent Canny V1 evaluation on all 100 official LOL-v2-real Test pairs."""

import argparse
import json
from pathlib import Path

from protocol_metrics import test_checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True, help='Real_captured directory')
    parser.add_argument('--output-dir', type=Path, required=True, help='New output directory')
    parser.add_argument('--config', type=Path, help='Model/Canny config for checkpoints without metadata')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args(argv)
    from train import resolve_device
    config = json.loads(args.config.read_text()) if args.config else None
    return test_checkpoint(args.checkpoint, args.dataset_root, args.output_dir,
                           resolve_device(args.device, False), config)


if __name__ == '__main__':
    main()
