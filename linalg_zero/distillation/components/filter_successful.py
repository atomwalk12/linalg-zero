"""
Custom filtering step for distillation pipeline that filters rows based on boolean conditions.
This step removes failed examples from the pipeline to prevent downstream failures.
"""

from typing import Any

from distilabel.steps import StepInput, StepOutput
from distilabel.steps.base import Step
from distilabel.typing import StepColumns
from pydantic import Field
from typing_extensions import override


class FilterSuccessful(Step):
    """
    Marks rows for skipping based on boolean success conditions while preserving all data.

    This step preserves all rows but adds skip markers for failed examples,
    allowing downstream components to skip processing while keeping data for analysis.

    Attributes:
        filter_columns: Dictionary mapping column names to expected boolean values.
                       Rows not meeting conditions will be marked for skipping.
        log_filtered: Whether to log information about filtered rows.
        preserve_data: If True, keeps all rows but adds skip markers (default: True).
    """

    filter_columns: dict[str, bool] = Field(
        default_factory=dict, description="Dictionary of column names and their expected boolean values for filtering"
    )
    log_filtered: bool = Field(default=True, description="Whether to log statistics about filtered rows")
    preserve_data: bool = Field(
        default=True, description="If True, keeps all rows but adds skip markers. If False, filters out failed rows."
    )

    @property
    @override
    def inputs(self) -> StepColumns:
        """Accept all input columns - filtering is applied based on filter_columns."""
        return []  # Accept any inputs

    @property
    @override
    def outputs(self) -> StepColumns:
        """Output the same columns as input, just with filtered rows."""
        return []  # Pass through all columns

    @override
    def process(self, *inputs: StepInput) -> "StepOutput":  # noqa: C901
        """
        Filter rows based on boolean conditions in filter_columns.

        Args:
            inputs: Batch of input data as list of dictionaries

        Yields:
            Filtered batch (all rows if preserve_data=True, only successful if False)
        """
        # Since this step has only one previous step, inputs[0] contains the data
        input_batch = inputs[0]

        if not self.filter_columns:
            # No filtering conditions specified, pass through all data
            yield input_batch
            return

        result_batch = []
        filtered_count = 0

        for row in input_batch:
            # Check if row meets all filter conditions
            should_skip = False
            for column_name, expected_value in self.filter_columns.items():
                if column_name not in row:
                    # Column missing, consider as failed condition
                    should_skip = True
                    break

                # Check if the actual value matches expected value
                actual_value = row[column_name]
                if actual_value != expected_value:
                    should_skip = True
                    break

            if self.preserve_data:
                # Keep all rows but add skip marker for failed ones
                row_copy = row.copy()
                row_copy["skip_downstream_processing"] = should_skip
                result_batch.append(row_copy)
                if should_skip:
                    filtered_count += 1
            else:
                if not should_skip:
                    result_batch.append(row)
                else:
                    filtered_count += 1

        # Log filtering statistics
        if self.log_filtered and (filtered_count > 0 or len(input_batch) > 0):
            total_input = len(input_batch)
            kept_count = len(result_batch) if not self.preserve_data else len(result_batch) - filtered_count
            filter_rate = (filtered_count / total_input * 100) if total_input > 0 else 0

            if self.preserve_data:
                self._logger.info(
                    f"Marked {filtered_count}/{total_input} rows for skipping ({filter_rate:.1f}%). "
                    + f"Preserved all {len(result_batch)} examples for analysis."
                )
            else:
                self._logger.info(
                    f"Filtered {filtered_count}/{total_input} rows ({filter_rate:.1f}%). "
                    + f"Kept {kept_count} successful examples."
                )

            # Log details about filter conditions
            if filtered_count > 0:
                condition_str = ", ".join([f"{k}={v}" for k, v in self.filter_columns.items()])
                self._logger.debug(f"Filter conditions: {condition_str}")

        yield result_batch


class FilterExecutionSuccessful(FilterSuccessful):
    """
    Convenience class for filtering based on execution success.

    Pre-configured to filter rows where keep_row_after_execution_check=True.
    """

    def __init__(self, **kwargs: Any):
        # Set default filter condition for execution success
        if "filter_columns" not in kwargs:
            kwargs["filter_columns"] = {"keep_row_after_execution_check": True}
        super().__init__(**kwargs)


class FilterSemanticSuccessful(FilterSuccessful):
    """
    Convenience class for filtering based on semantic verification success.

    Pre-configured to filter rows where keep_row_after_semantic_check=True.
    """

    def __init__(self, **kwargs: dict):
        # Set default filter condition for semantic success
        if "filter_columns" not in kwargs:
            kwargs["filter_columns"] = {"keep_row_after_semantic_check": True}
        super().__init__(**kwargs)
