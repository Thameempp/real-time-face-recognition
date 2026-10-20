"""Matching scores and open-set evaluation metrics."""

import numpy as np

from recognition.recognizer import EmbeddingMatcher


def score_samples(gallery_embeddings, gallery_names, strategy, samples):
    matcher = EmbeddingMatcher(
        gallery_embeddings,
        gallery_names,
        strategy=strategy,
    )
    scored = []
    for label, embedding, path in samples:
        if embedding is None:
            scored.append((label, "Unknown", -1.0, path))
            continue
        predicted_name, score = matcher.match(embedding, threshold=-1.0)
        scored.append((label, predicted_name, score, path))
    return scored


def rates_at_threshold(scored, enrolled_names, threshold):
    known_samples = [row for row in scored if row[0] in enrolled_names]
    unknown_samples = [row for row in scored if row[0] not in enrolled_names]
    if not known_samples or not unknown_samples:
        raise RuntimeError(
            "Both known and unknown identities are required in calibration "
            "and test data. Unknown people should use labels not in enrollment."
        )

    false_rejections = sum(
        score < threshold or predicted_name != label
        for label, predicted_name, score, _ in known_samples
    )
    false_accepts = sum(
        predicted_name != "Unknown" and score >= threshold
        for _, predicted_name, score, _ in unknown_samples
    )
    correct_identifications = sum(
        predicted_name != "Unknown"
        and score >= threshold
        and predicted_name == label
        for label, predicted_name, score, _ in known_samples
    )

    return {
        "frr": false_rejections / len(known_samples),
        "far": false_accepts / len(unknown_samples),
        "identification_rate": correct_identifications / len(known_samples),
        "known_count": len(known_samples),
        "unknown_count": len(unknown_samples),
    }


def calibrate_threshold(scored, enrolled_names):
    scores = {float(row[2]) for row in scored}
    candidates = {-1.0, 1.0, *scores}
    candidates.update(
        float(np.nextafter(score, np.inf))
        for score in scores
        if score < 1.0
    )
    best = None
    for threshold in sorted(
        threshold for threshold in candidates if -1.0 <= threshold <= 1.0
    ):
        rates = rates_at_threshold(scored, enrolled_names, threshold)
        half_total_error = (rates["frr"] + rates["far"]) / 2.0
        candidate = (half_total_error, -threshold, rates)
        if best is None or candidate[:2] < best[:2]:
            best = candidate

    return -best[1], best[2]
