"""
mfp_dmanp.py
========================
Interactive CLI tool for slicing, sorting, merging, generating, appending,
deleting, modifying, filtering, renaming, casting, deduplicating, and
computing new columns on CSV / Excel / JSON data files.

Usage:
    python mfp_dmanp.py [datafile]

Author : Swarnadeep Seth
Version: 1.0.3
"""

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------
import os
import sys
import warnings
from collections import deque
from typing import Optional

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore")

__version__ = "1.0.3"

# Maximum number of undo snapshots kept in memory.
_UNDO_LIMIT = 20

# File extensions recognised by load() / save().
_READERS: dict[str, object] = {
    ".csv":  pd.read_csv,
    ".xlsx": pd.read_excel,
    ".xls":  pd.read_excel,
    ".json": pd.read_json,
}
_WRITERS: dict[str, str] = {
    ".csv":  "to_csv",
    ".xlsx": "to_excel",
    ".xls":  "to_excel",
    ".json": "to_json",
}

# Actions the REPL recognises, with one-line descriptions.
ACTION_HELP: dict[str, str] = {
    # ── Inspection ──────────────────────────────────────────────────────────
    "show":       "Print the current DataFrame.",
    "head":       "Print the first N rows  (default 10).",
    "tail":       "Print the last N rows   (default 10).",
    "properties": "Column index, dtypes, NaN counts, and summary statistics.",
    "props":      "Alias for 'properties'.",
    "counts":     "Frequency count of unique values in a column.",
    # ── Transformation ───────────────────────────────────────────────────────
    "filter":     "Keep rows matching a pandas query expression.",
    "slice":      "Keep rows at positions [start, end).",
    "sort":       "Sort by a column (asc / desc).",
    "rename":     "Rename one or more columns  (old:new, ...).",
    "cast":       "Change a column's dtype  (int / float / str / datetime).",
    "addcol":     "Add a new column from an expression (uses df.eval).",
    "modify":     "Replace a specific value inside a column.",
    "delete":     "Drop columns (by name) or rows (by integer index).",
    "del":        "Alias for 'delete'.",
    "dedup":      "Remove duplicate rows, optionally over chosen columns.",
    "fillna":     "Fill empty / NaN cells in a column with a given value.",
    "dropna":     "Drop rows that have empty / NaN in a column (or any col).",
    # ── I/O ──────────────────────────────────────────────────────────────────
    "load":       "Load a new CSV / Excel / JSON file, replacing current data.",
    "generate":   "Build a numeric x/y table from an expression.",
    "gen":        "Alias for 'generate'.",
    "append":     "Append rows from a second file.",
    "merge":      "Merge with a second file on a shared column.",
    "save":       "Write the current DataFrame to a file (CSV / Excel / JSON).",
    # ── History ──────────────────────────────────────────────────────────────
    "undo":       "Revert the last mutating operation.",
    "redo":       "Re-apply the last undone operation.",
    # ── Meta ─────────────────────────────────────────────────────────────────
    "help":       "List all available actions.",
    "exit":       "Quit the program.",
    "q":          "Alias for 'exit'.",
}

_BANNER = f"""
{'=' * 70}
  MFP Data Manipulator  v{__version__}
  CSV · Excel · JSON  |  Filter · Cast · Dedup · Undo/Redo · and more
{'=' * 70}
Type  help  for a list of commands.
"""


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class MFPDataManipulator:
    """
    Wraps a pandas DataFrame and exposes named operations for interactive
    data exploration and cleaning.

    Design rules
    ------------
    - Every mutating method calls ``self._save_snapshot()`` *before* making
      changes so that ``undo`` can restore the previous state.
    - Every mutating method returns ``self.df`` so callers can print or
      inspect the result immediately.
    - I/O helpers are format-aware: the file extension drives the
      reader / writer, so CSV, Excel, and JSON all work transparently.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, datafile: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        datafile:
            Path to a data file (CSV / Excel / JSON) to load on startup.
            When *None* an empty DataFrame is created so the user can
            start from a ``generate`` or ``load`` call.
        """
        self._undo_stack: deque[pd.DataFrame] = deque(maxlen=_UNDO_LIMIT)
        self._redo_stack: deque[pd.DataFrame] = deque(maxlen=_UNDO_LIMIT)

        self.datafile: Optional[str] = None
        self.df = pd.DataFrame()

        if datafile:
            self._load_file(datafile, announce=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_snapshot(self) -> None:
        """Push a deep copy of the current DataFrame onto the undo stack."""
        self._undo_stack.append(self.df.copy(deep=True))
        self._redo_stack.clear()   # new mutation invalidates redo history

    def _load_file(self, path: str, announce: bool = True) -> None:
        """
        Read *path* into ``self.df`` using the appropriate pandas reader.

        Raises
        ------
        FileNotFoundError
            When the file does not exist.
        ValueError
            When the file extension is not supported.
        """
        abs_path = os.path.abspath(path)
        ext = os.path.splitext(abs_path)[1].lower()
        if ext not in _READERS:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported: {', '.join(_READERS)}"
            )
        reader = _READERS[ext]
        self.datafile = abs_path
        self.df = reader(abs_path).replace(np.nan, "", regex=True)
        if announce:
            print(
                f"Loaded '{abs_path}' — "
                f"{len(self.df):,} rows × {len(self.df.columns)} cols."
            )
            print(self.df)

    # ------------------------------------------------------------------
    # Inspection  (non-mutating — no snapshot needed)
    # ------------------------------------------------------------------

    def show(self) -> pd.DataFrame:
        """Print and return the current DataFrame."""
        print(self.df)
        return self.df

    def head(self, n: int = 10) -> pd.DataFrame:
        """
        Print and return the first *n* rows.

        Parameters
        ----------
        n : int  Number of rows to display (default 10).
        """
        print(self.df.head(int(n)))
        return self.df.head(int(n))

    def tail(self, n: int = 10) -> pd.DataFrame:
        """
        Print and return the last *n* rows.

        Parameters
        ----------
        n : int  Number of rows to display (default 10).
        """
        print(self.df.tail(int(n)))
        return self.df.tail(int(n))

    def properties(self) -> pd.DataFrame:
        """
        Display a column-index table, NaN / empty counts, and summary
        statistics.  Returns the ``describe()`` DataFrame.
        """
        col_table = pd.DataFrame({
            "Index":  range(len(self.df.columns)),
            "Column": self.df.columns,
            "dtype":  self.df.dtypes.values,
        })
        print("\nColumns:")
        print(col_table.to_string(index=False))

        nan_counts = self.df.replace("", np.nan).isnull().sum()
        print("\nNaN / empty counts:")
        print(nan_counts.to_string())

        summary = self.df.describe(include="all")
        print("\nSummary statistics:")
        print(summary)
        return summary

    def counts(self, col: str) -> pd.Series:
        """
        Print and return the frequency of each unique value in *col*,
        sorted most-frequent first.

        Parameters
        ----------
        col : str  Column name.
        """
        result = self.df[col].value_counts(dropna=False)
        print(result.to_string())
        return result

    # ------------------------------------------------------------------
    # Transformation operations  (mutating)
    # ------------------------------------------------------------------

    def filter(self, query: str) -> pd.DataFrame:
        """
        Keep only rows that satisfy a pandas query expression.

        Parameters
        ----------
        query : str
            A boolean expression understood by ``DataFrame.query()``.
            Examples: ``'age > 30'``,  ``'city == "Roanoke"'``,
            ``'score >= 90 and grade == "A"'``.

        Notes
        -----
        Column names with spaces must be wrapped in backticks:
        ``'`first name` == "Alice"'``.
        """
        self._save_snapshot()
        before = len(self.df)
        self.df = self.df.query(query).reset_index(drop=True)
        print(f"Filter kept {len(self.df):,} of {before:,} rows.")
        return self.df

    def slice(self, start: int, end: int) -> pd.DataFrame:
        """
        Keep rows in the half-open interval [start, end).

        Parameters
        ----------
        start, end : int  Row positions (0-based).
        """
        self._save_snapshot()
        self.df = self.df.iloc[int(start):int(end)]
        return self.df

    def sort(self, col: str, order: str = "asc") -> pd.DataFrame:
        """
        Sort the DataFrame by *col*.

        Parameters
        ----------
        col   : str  Column name.
        order : str  ``'asc'`` / ``'ascending'`` or ``'desc'`` / ``'descending'``.

        Raises
        ------
        ValueError  When *order* is not recognised.
        """
        order = order.strip().lower()
        if order in {"asc", "ascending"}:
            ascending = True
        elif order in {"desc", "descending"}:
            ascending = False
        else:
            raise ValueError(f"Unknown sort order '{order}'. Use 'asc' or 'desc'.")

        self._save_snapshot()
        self.df = self.df.sort_values(by=[col], ascending=ascending)
        return self.df

    def rename(self, mapping: str) -> pd.DataFrame:
        """
        Rename one or more columns.

        Parameters
        ----------
        mapping : str
            Comma-separated ``old:new`` pairs.
            Example: ``'first name:first_name, Last Name:last_name'``

        Raises
        ------
        ValueError  When a pair is malformed.
        KeyError    When an old column name is not found.
        """
        pairs: dict[str, str] = {}
        for token in mapping.split(","):
            token = token.strip()
            if ":" not in token:
                raise ValueError(
                    f"Bad rename pair '{token}'. Expected format: old:new"
                )
            old, new = token.split(":", 1)
            pairs[old.strip()] = new.strip()

        missing = [k for k in pairs if k not in self.df.columns]
        if missing:
            raise KeyError(f"Column(s) not found: {missing}")

        self._save_snapshot()
        self.df = self.df.rename(columns=pairs)
        print(f"Renamed: {pairs}")
        return self.df

    def cast(self, col: str, dtype: str) -> pd.DataFrame:
        """
        Change the dtype of *col*.

        Parameters
        ----------
        col   : str  Column name.
        dtype : str  Target type: ``'int'``, ``'float'``, ``'str'``,
                     or ``'datetime'``.

        Raises
        ------
        ValueError  When *dtype* is not a supported keyword or conversion fails.
        """
        dtype = dtype.strip().lower()
        self._save_snapshot()
        try:
            if dtype == "int":
                self.df[col] = pd.to_numeric(self.df[col], errors="raise").astype(int)
            elif dtype == "float":
                self.df[col] = pd.to_numeric(self.df[col], errors="raise").astype(float)
            elif dtype == "str":
                self.df[col] = self.df[col].astype(str)
            elif dtype == "datetime":
                self.df[col] = pd.to_datetime(self.df[col], errors="raise")
            else:
                raise ValueError(
                    f"Unknown dtype '{dtype}'. Use: int, float, str, datetime."
                )
        except Exception as exc:
            self._undo_stack.pop()   # roll back the snapshot — nothing changed
            raise exc

        print(f"Column '{col}' cast to {dtype}.")
        return self.df

    def addcol(self, name: str, expr: str) -> pd.DataFrame:
        """
        Add a new column derived from an expression over existing columns.

        Parameters
        ----------
        name : str  Name for the new column.
        expr : str  A ``DataFrame.eval()``-compatible expression.
                    Examples: ``'price * qty'``, ``'score / score.max()'``.
        """
        self._save_snapshot()
        self.df[name] = self.df.eval(expr)
        print(f"Column '{name}' added.")
        return self.df

    def modify(self, col: str, old_val: str, new_val: str) -> pd.DataFrame:
        """
        Replace occurrences of *old_val* with *new_val* in column *col*.

        Parameters
        ----------
        col     : str  Target column name.
        old_val : str  Value to search for (exact match).
        new_val : str  Replacement value.
        """
        self._save_snapshot()
        self.df[col] = self.df[col].replace(old_val, new_val)
        return self.df

    def delete(self, targets: str) -> pd.DataFrame:
        """
        Drop rows or columns.

        Parameters
        ----------
        targets : str
            Comma-separated list of **integer row indices** *or*
            **column names** — not both at once.

        Logic
        -----
        If every token is a plain integer → treat as row indices.
        Otherwise → treat as column names.
        """
        tokens = [t.strip() for t in targets.split(",")]
        self._save_snapshot()
        if all(t.lstrip("-").isnumeric() for t in tokens):
            self.df = self.df.drop(index=[int(t) for t in tokens])
        else:
            self.df = self.df.drop(columns=tokens)
        return self.df

    def dedup(self, cols: Optional[str] = None) -> pd.DataFrame:
        """
        Remove duplicate rows.

        Parameters
        ----------
        cols : str | None
            Comma-separated column names to consider.  When *None* (or
            empty string), all columns are used.
        """
        self._save_snapshot()
        before = len(self.df)
        subset = (
            [c.strip() for c in cols.split(",")]
            if cols and cols.strip()
            else None
        )
        self.df = self.df.drop_duplicates(subset=subset).reset_index(drop=True)
        print(f"Removed {before - len(self.df):,} duplicate row(s). {len(self.df):,} remain.")
        return self.df

    def fillna(self, col: str, value: str) -> pd.DataFrame:
        """
        Fill empty / NaN cells in *col* with *value*.

        Parameters
        ----------
        col   : str  Column name.
        value : str  Replacement value.  Automatically converted to int or
                     float when the string represents a number.
        """
        self._save_snapshot()
        try:
            typed_value: object = float(value) if "." in value else int(value)
        except (ValueError, AttributeError):
            typed_value = value

        self.df[col] = self.df[col].replace("", np.nan)
        self.df[col] = self.df[col].fillna(typed_value)
        print(f"Filled NaN / empty values in '{col}' with {typed_value!r}.")
        return self.df

    def dropna(self, col: Optional[str] = None) -> pd.DataFrame:
        """
        Drop rows that contain empty / NaN values.

        Parameters
        ----------
        col : str | None
            When given, only rows where *col* is empty / NaN are dropped.
            When *None* / empty string, any row with at least one empty /
            NaN cell is dropped.
        """
        self._save_snapshot()
        before = len(self.df)
        tmp = self.df.replace("", np.nan)
        if col and col.strip():
            self.df = tmp.dropna(subset=[col.strip()]).reset_index(drop=True)
            scope = f"column '{col.strip()}'"
        else:
            self.df = tmp.dropna().reset_index(drop=True)
            scope = "any column"
        print(f"Dropped {before - len(self.df):,} row(s) with NaN / empty in {scope}.")
        return self.df

    # ------------------------------------------------------------------
    # I/O operations
    # ------------------------------------------------------------------

    def load(self, path: str) -> pd.DataFrame:
        """
        Replace the current DataFrame by loading a new file.

        Parameters
        ----------
        path : str  Path to a CSV / Excel / JSON file.

        Notes
        -----
        The REPL calls ``_dirty_check`` before invoking this; calling it
        directly skips the unsaved-changes prompt.
        """
        self._save_snapshot()
        self._load_file(path, announce=True)
        return self.df

    def generate(self, xr: str, expr: str) -> pd.DataFrame:
        """
        Replace the current DataFrame with a numeric x/y table.

        Parameters
        ----------
        xr   : str  Range string ``'start:end'`` (inclusive on both ends).
        expr : str  Python expression in terms of *x* and *np*.
                    Example: ``'5*x**2 / np.exp(x)'``

        Security note
        -------------
        ``eval`` is used intentionally for a numeric DSL.
        Do **not** expose this to untrusted input.
        """
        self._save_snapshot()
        lo, hi = (int(v) for v in xr.split(":"))
        x = np.linspace(lo, hi, hi - lo + 1)
        y = eval(expr)  # noqa: S307
        self.df = pd.DataFrame({"x": x, "y": y})
        return self.df

    def append(self, datafile: str) -> pd.DataFrame:
        """
        Append rows from *datafile* to the current DataFrame.

        Parameters
        ----------
        datafile : str  Path to a CSV / Excel / JSON file.
        """
        self._save_snapshot()
        ext = os.path.splitext(datafile)[1].lower()
        reader = _READERS.get(ext, pd.read_csv)
        df2 = reader(datafile).replace(np.nan, "", regex=True)
        before = len(self.df)
        self.df = pd.concat([self.df, df2], ignore_index=True)
        print(f"Appended {len(self.df) - before:,} rows.")
        return self.df

    def merge(
        self,
        other_file: str,
        on_column: str,
        how: str = "inner",
    ) -> pd.DataFrame:
        """
        Merge ``self.df`` with a second file on a shared column.

        Parameters
        ----------
        other_file : str  Path to the second file (CSV / Excel / JSON).
        on_column  : str  Column name present in both DataFrames.
        how        : str  ``'inner'``, ``'outer'``, ``'left'``, or ``'right'``.
        """
        how = how.strip().lower()
        try:
            ext = os.path.splitext(other_file)[1].lower()
            reader = _READERS.get(ext, pd.read_csv)
            df2 = reader(other_file).replace(np.nan, "", regex=True)
            self._save_snapshot()
            self.df = pd.merge(self.df, df2, on=on_column, how=how)
        except FileNotFoundError:
            print(f"[ERROR] File not found: '{other_file}'")
        except KeyError:
            print(f"[ERROR] Column '{on_column}' not found in one of the files.")
        return self.df

    def save(self, filename: str) -> str:
        """
        Save the current DataFrame to a file.

        The format is determined by *filename*'s extension (CSV / Excel /
        JSON).  The file is written next to the originally loaded file,
        or to the current working directory if no file was loaded.

        Parameters
        ----------
        filename : str  Output file name or relative path.

        Returns
        -------
        str  Absolute path of the saved file.
        """
        directory = (
            os.path.dirname(self.datafile)
            if self.datafile
            else os.getcwd()
        )
        save_path = os.path.join(directory, filename)
        ext = os.path.splitext(filename)[1].lower()
        writer_name = _WRITERS.get(ext, "to_csv")
        writer = getattr(self.df, writer_name)

        if writer_name == "to_csv":
            writer(save_path, index=False)
        elif writer_name == "to_excel":
            writer(save_path, index=False)
        elif writer_name == "to_json":
            writer(save_path, orient="records", indent=2)

        print(f"Saved → {save_path}")
        return save_path

    # ------------------------------------------------------------------
    # Undo / redo
    # ------------------------------------------------------------------

    def undo(self) -> pd.DataFrame:
        """
        Revert the last mutating operation.

        Raises
        ------
        IndexError  When the undo stack is empty.
        """
        if not self._undo_stack:
            raise IndexError("Nothing to undo.")
        self._redo_stack.append(self.df.copy(deep=True))
        self.df = self._undo_stack.pop()
        print(f"Undone. DataFrame is now {len(self.df):,} rows × {len(self.df.columns)} cols.")
        return self.df

    def redo(self) -> pd.DataFrame:
        """
        Re-apply the last undone operation.

        Raises
        ------
        IndexError  When the redo stack is empty.
        """
        if not self._redo_stack:
            raise IndexError("Nothing to redo.")
        self._undo_stack.append(self.df.copy(deep=True))
        self.df = self._redo_stack.pop()
        print(f"Redone. DataFrame is now {len(self.df):,} rows × {len(self.df.columns)} cols.")
        return self.df


# ---------------------------------------------------------------------------
# REPL helpers
# ---------------------------------------------------------------------------

def _print_help() -> None:
    """Print a formatted, sectioned table of all available actions."""
    sections: dict[str, list[str]] = {
        "Inspection":     ["show", "head", "tail", "properties", "props", "counts"],
        "Transformation": ["filter", "slice", "sort", "rename", "cast", "addcol",
                           "modify", "delete", "del", "dedup", "fillna", "dropna"],
        "I/O":            ["load", "generate", "gen", "append", "merge", "save"],
        "History":        ["undo", "redo"],
        "Meta":           ["help", "exit", "q"],
    }
    for section, actions in sections.items():
        print(f"\n  {section}")
        print("  " + "-" * 44)
        for a in actions:
            if a in ACTION_HELP:
                print(f"    {a:<14}  {ACTION_HELP[a]}")
    print()


def _require_data(dm: MFPDataManipulator, action: str) -> bool:
    """Return *True* when ``dm.df`` is non-empty; print an error otherwise."""
    if dm.df.empty:
        print(
            f"[ERROR] No data loaded. "
            f"Run 'load' or 'generate' before '{action}'."
        )
        return False
    return True


def _dirty_check(dm: MFPDataManipulator) -> bool:
    """
    Prompt for confirmation when the undo stack is non-empty (i.e. the user
    has made changes that may not have been saved).

    Returns *True* if it is safe to proceed, *False* if the user cancelled.
    """
    if dm._undo_stack:
        ans = input(
            "  You may have unsaved changes. Continue anyway? (y / n): "
        ).strip().lower()
        return ans in {"y", "yes"}
    return True


def _run_repl(dm: MFPDataManipulator) -> None:
    """
    Enter the interactive command loop for *dm*.

    Each iteration reads one action keyword, prompts for its arguments,
    dispatches to the corresponding method, and prints the result.
    Exceptions raised by method calls are caught and displayed without
    crashing the loop.
    """
    while True:
        try:
            action = input("\nAction> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nInterrupted — exiting.")
            sys.exit(0)

        try:
            # ----------------------------------------------------------------
            # Meta
            # ----------------------------------------------------------------
            if action in {"exit", "q"}:
                if dm._undo_stack:
                    ans = input(
                        "  You may have unsaved changes. Quit anyway? (y / n): "
                    ).strip().lower()
                    if ans not in {"y", "yes"}:
                        continue
                print("Goodbye.")
                sys.exit(0)

            elif action == "help":
                _print_help()

            # ----------------------------------------------------------------
            # Inspection
            # ----------------------------------------------------------------
            elif action == "show":
                if not _require_data(dm, action): continue
                dm.show()

            elif action == "head":
                if not _require_data(dm, action): continue
                n = input("  Number of rows (default 10): ").strip()
                dm.head(int(n) if n else 10)

            elif action == "tail":
                if not _require_data(dm, action): continue
                n = input("  Number of rows (default 10): ").strip()
                dm.tail(int(n) if n else 10)

            elif action in {"properties", "props"}:
                if not _require_data(dm, action): continue
                dm.properties()

            elif action == "counts":
                if not _require_data(dm, action): continue
                col = input("  Column name: ").strip()
                dm.counts(col)

            # ----------------------------------------------------------------
            # I/O  (load / generate allowed without existing data)
            # ----------------------------------------------------------------
            elif action == "load":
                if not _dirty_check(dm): continue
                path = input("  File path (CSV / Excel / JSON): ").strip()
                dm.load(path)

            elif action in {"generate", "gen"}:
                if not _dirty_check(dm): continue
                xr   = input("  x range (e.g. 0:10): ").strip()
                expr = input("  y expression (e.g. 5*x**2/np.exp(x)): ").strip()
                print(dm.generate(xr, expr))

            elif action == "append":
                if not _require_data(dm, action): continue
                path = input("  File to append: ").strip()
                print(dm.append(path))

            elif action == "merge":
                if not _require_data(dm, action): continue
                other = input("  File to merge with: ").strip()
                col   = input("  Column to merge on: ").strip()
                how   = input("  Merge type (inner / outer / left / right): ").strip()
                print(dm.merge(other, col, how))

            elif action == "save":
                if not _require_data(dm, action): continue
                filename = input(
                    "  Output file name (e.g. output.csv / .xlsx / .json): "
                ).strip()
                dm.save(filename)

            # ----------------------------------------------------------------
            # Transformation
            # ----------------------------------------------------------------
            elif action == "filter":
                if not _require_data(dm, action): continue
                query = input("  Query expression (e.g. age > 30): ").strip()
                print(dm.filter(query))

            elif action == "slice":
                if not _require_data(dm, action): continue
                start = input("  Start index: ").strip()
                end   = input("  End index  : ").strip()
                print(dm.slice(start, end))

            elif action == "sort":
                if not _require_data(dm, action): continue
                col   = input("  Column to sort by: ").strip()
                order = input("  Order (asc / desc): ").strip()
                print(dm.sort(col, order))

            elif action == "rename":
                if not _require_data(dm, action): continue
                mapping = input(
                    "  Rename pairs, comma-separated (old:new, ...): "
                ).strip()
                print(dm.rename(mapping))

            elif action == "cast":
                if not _require_data(dm, action): continue
                col   = input("  Column name: ").strip()
                dtype = input("  Target dtype (int / float / str / datetime): ").strip()
                print(dm.cast(col, dtype))

            elif action == "addcol":
                if not _require_data(dm, action): continue
                name = input("  New column name: ").strip()
                expr = input("  Expression (e.g. price * qty): ").strip()
                print(dm.addcol(name, expr))

            elif action == "modify":
                if not _require_data(dm, action): continue
                col     = input("  Column name : ").strip()
                old_val = input("  Old value   : ").strip()
                new_val = input("  New value   : ").strip()
                print(dm.modify(col, old_val, new_val))

            elif action in {"delete", "del"}:
                if not _require_data(dm, action): continue
                targets = input(
                    "  Column names or row indices (comma-separated): "
                ).strip()
                print(dm.delete(targets))

            elif action == "dedup":
                if not _require_data(dm, action): continue
                cols = input(
                    "  Columns to check (comma-separated, or Enter for all): "
                ).strip()
                print(dm.dedup(cols if cols else None))

            elif action == "fillna":
                if not _require_data(dm, action): continue
                col   = input("  Column name: ").strip()
                value = input("  Fill value : ").strip()
                print(dm.fillna(col, value))

            elif action == "dropna":
                if not _require_data(dm, action): continue
                col = input(
                    "  Column name (or Enter to drop rows with any NaN): "
                ).strip()
                print(dm.dropna(col if col else None))

            # ----------------------------------------------------------------
            # Undo / redo
            # ----------------------------------------------------------------
            elif action == "undo":
                dm.undo()

            elif action == "redo":
                dm.redo()

            # ----------------------------------------------------------------
            # Unknown action
            # ----------------------------------------------------------------
            else:
                print(f"[WARN] Unknown action '{action}'. Type 'help' for options.")

        except (KeyError, ValueError, IndexError, TypeError) as exc:
            print(f"[ERROR] {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    """
    Parse CLI arguments, optionally load a datafile, and start the REPL.

    Parameters
    ----------
    argv : list[str] | None
        Overrides ``sys.argv[1:]``; useful for tests that need to pass
        arguments without patching the global.
    """
    if argv is None:
        argv = sys.argv[1:]

    print(_BANNER)

    if argv:
        try:
            dm = MFPDataManipulator(argv[0])
        except (FileNotFoundError, ValueError) as exc:
            print(f"[ERROR] {exc}")
            sys.exit(1)
    else:
        while True:
            path = input(
                "File to load (CSV / Excel / JSON), or Enter to skip: "
            ).strip()
            if not path:
                dm = MFPDataManipulator()
                print("No file loaded — run 'generate' or 'load' to get started.")
                break
            try:
                dm = MFPDataManipulator(path)
                break
            except (FileNotFoundError, ValueError) as exc:
                print(f"[ERROR] {exc}  Please try again.")

    _run_repl(dm)


if __name__ == "__main__":
    main()
