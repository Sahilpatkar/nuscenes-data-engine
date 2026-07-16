"""Great Expectations (GX 1.x) suites for the processed nuScenes metadata.

Checks: schema conformance (columns exist), null constraints, 2D boxes within image
bounds, valid category labels, valid visibility tokens, and per-scene sample counts.
Runs against the Parquet tables via an in-memory (ephemeral) GX context — no on-disk
GX project is required.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd
from great_expectations import ExpectationSuite

from nuscenes_data_engine.ingestion.categories import (
    ALL_NUSCENES_CATEGORIES,
    DETECTION_CLASSES,
)

logger = logging.getLogger("nuscenes_data_engine")

# All nuScenes camera images are 1600x900.
IMAGE_WIDTH = 1600
IMAGE_HEIGHT = 900
CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]
VISIBILITY_TOKENS = ["1", "2", "3", "4"]

# A nuScenes scene is ~40 keyframes; each keyframe yields 6 camera images.
MIN_IMAGES_PER_SCENE = 15 * 6
MAX_IMAGES_PER_SCENE = 60 * 6


def build_annotation_suite(context: Any) -> ExpectationSuite:
    """Expectation suite for the flattened annotations table (registered with ``context``)."""
    suite = context.suites.add(ExpectationSuite(name="annotations"))

    required = [
        "annotation_token",
        "sample_data_token",
        "channel",
        "category_name",
        "category_group",
        "visibility_token",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "bbox_area",
    ]
    for col in required:
        suite.add_expectation(gxe.ExpectColumnToExist(column=col))

    for col in [
        "annotation_token",
        "sample_data_token",
        "category_name",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
    ]:
        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=col))

    # Boxes within image bounds.
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="x_min", min_value=0, max_value=IMAGE_WIDTH)
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="x_max", min_value=0, max_value=IMAGE_WIDTH)
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="y_min", min_value=0, max_value=IMAGE_HEIGHT)
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="y_max", min_value=0, max_value=IMAGE_HEIGHT)
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="bbox_area", min_value=0, strict_min=True)
    )

    # Non-degenerate boxes: x_max > x_min and y_max > y_min.
    suite.add_expectation(
        gxe.ExpectColumnPairValuesAToBeGreaterThanB(column_A="x_max", column_B="x_min")
    )
    suite.add_expectation(
        gxe.ExpectColumnPairValuesAToBeGreaterThanB(column_A="y_max", column_B="y_min")
    )

    # Valid labels (null category_group is allowed — map expectations skip nulls).
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column="category_name", value_set=list(ALL_NUSCENES_CATEGORIES)
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(column="category_group", value_set=list(DETECTION_CLASSES))
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(column="visibility_token", value_set=VISIBILITY_TOKENS)
    )

    return suite


def build_sample_suite(context: Any) -> ExpectationSuite:
    """Expectation suite for the per-image (samples) table (registered with ``context``)."""
    suite = context.suites.add(ExpectationSuite(name="samples"))

    required = [
        "sample_data_token",
        "sample_token",
        "channel",
        "filename",
        "width",
        "height",
        "n_boxes",
        "scene_token",
    ]
    for col in required:
        suite.add_expectation(gxe.ExpectColumnToExist(column=col))

    for col in ["sample_data_token", "sample_token", "channel", "filename", "scene_token"]:
        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=col))

    suite.add_expectation(
        gxe.ExpectColumnDistinctValuesToBeInSet(column="width", value_set=[IMAGE_WIDTH])
    )
    suite.add_expectation(
        gxe.ExpectColumnDistinctValuesToBeInSet(column="height", value_set=[IMAGE_HEIGHT])
    )
    suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(column="channel", value_set=CAMERAS))
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="n_boxes", min_value=0))
    suite.add_expectation(gxe.ExpectColumnValuesToBeUnique(column="sample_data_token"))

    return suite


def _validate_frame(
    context: Any, df: pd.DataFrame, suite: ExpectationSuite, asset_name: str
) -> bool:
    """Validate a dataframe against a suite using the given ephemeral GX context."""
    batch = (
        context.data_sources.add_pandas(f"{asset_name}_src")
        .add_dataframe_asset(name=asset_name)
        .add_batch_definition_whole_dataframe(f"{asset_name}_batch")
        .get_batch(batch_parameters={"dataframe": df})
    )
    result = batch.validate(suite)
    for r in result.results:
        if not r.success:
            logger.warning("FAILED: %s", r.expectation_config.type)
    logger.info(
        "[%s] %d/%d expectations passed",
        asset_name,
        sum(r.success for r in result.results),
        len(result.results),
    )
    return bool(result.success)


def _check_per_scene_counts(samples: pd.DataFrame) -> bool:
    """Each scene should have a plausible number of camera images."""
    counts = samples.groupby("scene_token").size()
    out_of_range = counts[(counts < MIN_IMAGES_PER_SCENE) | (counts > MAX_IMAGES_PER_SCENE)]
    if len(out_of_range):
        logger.warning(
            "per-scene image counts out of range for %d scene(s): %s",
            len(out_of_range),
            out_of_range.to_dict(),
        )
        return False
    logger.info(
        "per-scene image counts OK (%d scenes, %d..%d images each)",
        len(counts),
        int(counts.min()),
        int(counts.max()),
    )
    return True


def validate_dataset(processed_dir: Path) -> bool:
    """Run all suites against the processed dataset; return True if everything passes."""
    samples = pd.read_parquet(processed_dir / "samples.parquet")
    annotations = pd.read_parquet(processed_dir / "annotations.parquet")

    context = gx.get_context(mode="ephemeral")
    ok_samples = _validate_frame(context, samples, build_sample_suite(context), "samples")
    ok_annotations = _validate_frame(
        context, annotations, build_annotation_suite(context), "annotations"
    )
    ok_counts = _check_per_scene_counts(samples)

    passed = ok_samples and ok_annotations and ok_counts
    logger.info("Validation %s", "PASSED" if passed else "FAILED")
    return passed
