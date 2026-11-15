#!/usr/bin/env python3
"""Create a deterministic flight-sequence hold-out list for all experiments."""

from __future__ import annotations

import argparse
from pathlib import Path


EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--validation-output", type=Path,
                        help="optional alternating half of held-out images for threshold selection")
    parser.add_argument("--test-output", type=Path,
                        help="optional remaining held-out images for final reporting")
    parser.add_argument("--every", type=int, default=10,
                        help="hold out one image from every N sorted images")
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    if args.every < 2:
        parser.error("--every must be at least 2")
    images = sorted(
        path for path in args.images_dir.iterdir() if path.suffix.lower() in EXTENSIONS
    )
    if len(images) < args.every:
        parser.error("not enough images for the requested hold-out interval")
    selected = [path.name for index, path in enumerate(images) if index % args.every == args.offset]
    if (args.validation_output is None) != (args.test_output is None):
        parser.error("provide both --validation-output and --test-output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(selected) + "\n")
    print("Held out {}/{} images: {}".format(len(selected), len(images), args.output))
    if args.validation_output is not None:
        validation = selected[::2]
        test = selected[1::2]
        if not validation or not test:
            parser.error("hold-out set is too small to divide into validation and test")
        args.validation_output.parent.mkdir(parents=True, exist_ok=True)
        args.test_output.parent.mkdir(parents=True, exist_ok=True)
        args.validation_output.write_text("\n".join(validation) + "\n")
        args.test_output.write_text("\n".join(test) + "\n")
        print("Validation: {} ({})".format(args.validation_output, len(validation)))
        print("Test: {} ({})".format(args.test_output, len(test)))


if __name__ == "__main__":
    main()
