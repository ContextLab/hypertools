"""Which drawn run came from which source dataset, and from which of its rows.

`hue=`/`cluster=` replace the one-trace-per-dataset correspondence with one
trace per contiguous same-category RUN (`_shared.helpers.segment_by_run`).
Everything that has to reason about a DATASET after that -- the animation's
reveal clock, a forecast's history, which trace a forecast attaches to --
needs the inverse mapping, in ORIGINAL row indices rather than run-local ones.

It also needs to distinguish two spans that are easy to conflate and were:

* the rows a run OWNS (`run_span`), which is what a forecast history is
  sliced from, and
* the source parameters a run DRAWS (`draw_span`), which is one longer for a
  bridged run because `patch_lines` appends the next run's first observation
  so the polyline is continuous across a colour change.

Using the owned span to pace the drawn geometry mis-times every category
boundary by one vertex, and leaves an observation on screen that the reveal
schedule says is not visible.

Deliberately pure and free of plotting imports: it is the thing both backends
and the forecast schedule agree on, so it must be testable without drawing
anything.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class TraceOwnership:
    """Run -> dataset -> original rows, for one figure.

    Attributes
    ----------
    dataset_by_run : tuple of int
        Each run's source input-dataset index. Dense, zero-based, and in
        first-appearance order (validated in `from_segments`), because
        `final_run_by_dataset` is indexed by dataset.
    source_rows_by_run : tuple of tuple of int
        Each run's OWNED original row indices, numbered WITHIN ITS OWN
        DATASET (dataset 1's first row is 0, not the global stacked offset).
        Excludes the bridge row `patch_lines` appends: that row is owned by
        the run it came from and merely duplicated here for rendering.
    bridged_by_run : tuple of bool
        Whether `patch_lines` appended the NEXT run's first observation to
        this run's drawn array.
    final_run_by_dataset : tuple of int
        Each dataset's LAST run -- the one holding its final observation, and
        so the trace a static forecast continues.
    """

    dataset_by_run: tuple
    source_rows_by_run: tuple
    bridged_by_run: tuple
    final_run_by_dataset: tuple

    @property
    def n_runs(self):
        return len(self.dataset_by_run)

    @property
    def n_datasets(self):
        return len(self.final_run_by_dataset)

    def runs_of(self, dataset):
        """Every run this dataset produced, in order."""
        return tuple(r for r, d in enumerate(self.dataset_by_run)
                     if d == dataset)

    def row_count(self, dataset):
        """How many original rows this dataset has."""
        return sum(len(self.source_rows_by_run[r])
                   for r in self.runs_of(dataset))

    def run_span(self, run):
        """``(first_row, n_owned_rows)`` in this run's dataset's numbering."""
        rows = self.source_rows_by_run[run]
        return rows[0], len(rows)

    def draw_span(self, run):
        """Source-parameter span of the run's DRAWN polyline.

        ``n_owned_rows - 1`` vertices' worth of trajectory, plus 1 when
        `patch_lines` bridged it -- the drawn line reaches the NEXT run's
        first observation, and the reveal must reach it at the same moment
        that observation becomes visible, or the boundary shows a vertex the
        schedule denies.
        """
        return len(self.source_rows_by_run[run]) - 1 + int(
            self.bridged_by_run[run])

    def run_holding(self, dataset, row):
        """The run OWNING original `row` of `dataset`."""
        for r in self.runs_of(dataset):
            if row in self.source_rows_by_run[r]:
                return r
        raise ValueError(
            f"dataset {dataset} has no row {row} (it has "
            f"{self.row_count(dataset)}).")

    @classmethod
    def from_segments(cls, seg_dataset, seg_lengths, seg_bridge):
        """Build from `segment_by_run`'s outputs and PRE-`patch_lines` lengths.

        `patch_lines` appends the NEXT run's first point to every bridged run,
        so a drawn run's array is one row longer than the span it owns.
        Passing post-bridge lengths here would hand every bridged run one row
        of its neighbour's data -- exactly the double-counting that makes a
        forecast history wrong by one observation per category boundary.
        Pass `seg_bridge` straight through from `segment_by_run` rather than
        re-deriving it from run positions: `plot._regroup_categorical_lines`
        turns it into `patch_lines`' `breaks` set, so it is the only record of
        what was actually bridged.
        """
        seg_dataset = [int(d) for d in seg_dataset]
        seg_lengths = [int(n) for n in seg_lengths]
        seg_bridge = [bool(b) for b in seg_bridge]
        if len(seg_dataset) != len(seg_lengths):
            raise ValueError(
                f"one length per run is needed; got {len(seg_dataset)} run "
                f"dataset(s) and {len(seg_lengths)} length(s).")
        if len(seg_bridge) != len(seg_dataset):
            raise ValueError(
                f"one bridge flag per run is needed; got {len(seg_bridge)} "
                f"for {len(seg_dataset)} run(s).")
        bad = [r for r, n in enumerate(seg_lengths) if n <= 0]
        if bad:
            raise ValueError(
                f"every run holds at least one observation (a run exists "
                f"because an observation started it); run(s) {bad} have no "
                f"rows.")

        seen_order = []
        for d in seg_dataset:
            if d not in seen_order:
                seen_order.append(d)
        # `final_run_by_dataset` is INDEXED by dataset and `runs_of` is called
        # with `range(n_datasets)`, so ids must be dense, zero-based and in
        # order. `segment_by_run` always emits them that way; this class is
        # independently callable, and sparse ids fail silently rather than
        # loudly (entry 0 would describe some other dataset).
        if seen_order and seen_order[0] != 0:
            raise ValueError(
                f"dataset indices must start at 0 and be dense; the first "
                f"one seen is {seen_order[0]}.")
        if seen_order != list(range(len(seen_order))):
            raise ValueError(
                f"dataset indices must appear in order 0, 1, 2, ...; got "
                f"{seen_order}.")
        for d in seen_order:
            runs = [r for r, dd in enumerate(seg_dataset) if dd == d]
            if runs != list(range(runs[0], runs[-1] + 1)):
                raise ValueError(
                    f"a dataset's runs must be consecutive (segment_by_run "
                    f"emits them dataset by dataset); dataset {d} owns runs "
                    f"{runs}.")

        rows_by_run, next_row = [], {}
        for d, n in zip(seg_dataset, seg_lengths):
            start = next_row.get(d, 0)
            rows_by_run.append(tuple(range(start, start + n)))
            next_row[d] = start + n

        final = {}
        for r, d in enumerate(seg_dataset):
            final[d] = r                       # last write wins
        for d, r in final.items():
            if seg_bridge[r]:
                raise ValueError(
                    f"a dataset's last run cannot be bridged (patch_lines is "
                    f"given a break at every dataset boundary); run {r} is "
                    f"the last run of dataset {d} and is marked bridged.")
        return cls(tuple(seg_dataset), tuple(rows_by_run), tuple(seg_bridge),
                   tuple(final[d] for d in sorted(final)))

    @classmethod
    def identity(cls, dataset_lengths):
        """The UNREGROUPED case: one run per dataset, holding all its rows.

        Every consumer takes the same code path whether or not `hue=` split
        anything, so the regrouped path cannot quietly rot while the common
        one stays green.
        """
        lengths = [int(n) for n in dataset_lengths]
        return cls.from_segments(list(range(len(lengths))), lengths,
                                 [False] * len(lengths))
