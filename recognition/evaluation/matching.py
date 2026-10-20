"""Compare enrollment galleries and calibrate recognition thresholds offline."""

import os

from config import SIMILARITY_THRESHOLD
from .data import (
    create_evaluation_app,
    load_enrollment,
    load_labeled_images,
)
from .metrics import (
    calibrate_threshold,
    rates_at_threshold,
    score_samples,
)
from .gallery import (
    POSE_NAMES,
    build_gallery,
    pose_from_path,
)


def _roots_overlap(first, second):
    first = os.path.realpath(first)
    second = os.path.realpath(second)
    try:
        return os.path.commonpath((first, second)) in {first, second}
    except ValueError:
        return False


def evaluate_matching(args):
    roots = [args.enrollment_dir, args.calibration_dir, args.test_dir]
    for index, first in enumerate(roots):
        for second in roots[index + 1 :]:
            if _roots_overlap(first, second):
                raise ValueError(
                    "Enrollment, calibration, and test directories must be "
                    "separate and non-overlapping."
                )

    app = create_evaluation_app()
    calibration_samples = load_labeled_images(app, args.calibration_dir)
    test_samples = load_labeled_images(app, args.test_dir)
    for mode in args.enrollment_modes:
        enrollment_embeddings, enrollment_names, enrollment_paths = load_enrollment(
            app,
            args.enrollment_dir,
            mode,
        )
        enrolled_names = set(enrollment_names)
        print(f"\nEnrollment mode: {mode}; embeddings: {len(enrollment_names)}")
        for split_name, samples in (
            ("calibration", calibration_samples),
            ("test", test_samples),
        ):
            if not any(label in enrolled_names for label, _, _ in samples):
                raise RuntimeError(f"{split_name} data has no enrolled identities")
            if not any(label not in enrolled_names for label, _, _ in samples):
                raise RuntimeError(
                    f"{split_name} data needs at least one identity absent from enrollment"
                )

        for strategy in args.strategies:
            gallery_embeddings, gallery_names, matcher_strategy = build_gallery(
                enrollment_embeddings,
                enrollment_names,
                enrollment_paths,
                strategy,
            )
            calibration_scores = score_samples(
                gallery_embeddings,
                gallery_names,
                matcher_strategy,
                calibration_samples,
            )
            test_scores = score_samples(
                gallery_embeddings,
                gallery_names,
                matcher_strategy,
                test_samples,
            )
            threshold, calibration_rates = calibrate_threshold(
                calibration_scores,
                enrolled_names,
            )
            test_rates = rates_at_threshold(test_scores, enrolled_names, threshold)
            baseline_rates = rates_at_threshold(
                test_scores,
                enrolled_names,
                SIMILARITY_THRESHOLD,
            )

            print(
                f"{strategy}: calibrated threshold={threshold:.3f}; "
                f"calibration FAR/FRR="
                f"{calibration_rates['far']:.1%}/{calibration_rates['frr']:.1%}; "
                f"test FAR/FRR/ID rate="
                f"{test_rates['far']:.1%}/{test_rates['frr']:.1%}/"
                f"{test_rates['identification_rate']:.1%}; "
                f"test at {SIMILARITY_THRESHOLD:.2f}="
                f"{baseline_rates['far']:.1%}/{baseline_rates['frr']:.1%}/"
                f"{baseline_rates['identification_rate']:.1%}"
            )

    print("Test data was held out from enrollment and threshold calibration.")
