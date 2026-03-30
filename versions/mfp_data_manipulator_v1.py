"""
mfp_data_manipulator.py
========================
Interactive CLI tool for slicing, sorting, merging, generating, appending,
deleting, modifying, and inspecting data in CSV files.

Usage:
    python mfp_data_manipulator.py [datafile.csv]

Author : <your name>
Version: 2.0.0
"""

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------
import os
import sys
import warnings
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

__version__ = "2.0.0"

# Actions the REPL recognises and a short help string for each
ACTION_HELP: dict[str, str] = {
    "properties": "Show column index, NaN counts, and summary statistics.",
    "props":      "Alias for 'properties'.",
    "slice":      "Keep only rows [start, end).",
    "sort":       "Sort by a column (asc / desc).",
    "merge":      "Merge with a second CSV on a common column.",
    "generate":   "Generate a numeric x/y dataset from an expression.",
    "gen":        "Alias for 'generate'.",
    "append":     "Append rows from a second CSV.",
    "delete":     "Drop columns (by name) or rows (by integer index).",
    "del":        "Alias for 'delete'.",
    "modify":     "Replace a value inside a column.",
    "save":       "Write the current DataFrame to a CSV file.",
    "show":       "Print the current DataFrame.",
    "help":       "List all available actions.",
    "exit":       "Quit the program.",
    "q":          "Alias for 'exit'.",
}

_BANNER = f"""
{'=' * 70}
  MFP Data Manipulator  v{__version__}
  Slice · Sort · Merge · Generate · Append · Delete · Modify
{'=' * 70}
Type  help  for a list of commands.
"""


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class MFPDataManipulator:
    """
    Wraps a pandas DataFrame and exposes named operations that can be
    chained in a REPL loop or called programmatically.

    All mutating methods update ``self.df`` in-place and return the
    updated DataFrame so callers can inspect or chain results easily.
    """

    # ------------------------------------------------------------------
    # Construction / loading
    # ------------------------------------------------------------------

    def __init__(self, datafile: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        datafile:
            Path to a CSV file to load.  When *None* an empty DataFrame
            is created so the user can start from a ``generate`` call.
        """
        if datafile:
            self.datafile: Optional[str] = os.path.abspath(datafile)
            self.df = pd.read_csv(self.datafile).replace(np.nan, "", regex=True)
            print(f"Loaded '{self.datafile}' — {len(self.df):,} rows × {len(self.df.columns)} cols.")
            print(self.df)
        else:
            self.datafile = None
            self.df = pd.DataFrame()

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def show(self) -> pd.DataFrame:
        """Print and return the current DataFrame."""
        print(self.df)
        return self.df

    def properties(self) -> pd.DataFrame:
        """
        Display:
        - a numbered column-index table
        - NaN counts per column
        - pandas ``describe()`` summary statistics

        Returns
        -------
        pd.DataFrame
            The ``describe()`` summary.
        """
        # Column index table
        col_table = pd.DataFrame({
            "Index": range(len(self.df.columns)),
            "Column": self.df.columns,
            "dtype": self.df.dtypes.values,
        })
        print("\nColumns:")
        print(col_table.to_string(index=False))

        # NaN counts (replace "" back with NaN just for counting)
        nan_counts = self.df.replace("", np.nan).isnull().sum()
        print("\nNaN / empty counts:")
        print(nan_counts.to_string())

        # Summary statistics
        summary = self.df.describe(include="all")
        print("\nSummary statistics:")
        print(summary)
        return summary

    # ------------------------------------------------------------------
    # Transformation operations
    # ------------------------------------------------------------------

    def slice(self, start: int, end: int) -> pd.DataFrame:
        """
        Keep rows in the half-open interval [start, end).

        Parameters
        ----------
        start, end : int
            Row positions (0-based).
        """
        self.df = self.df.iloc[int(start):int(end)]
        return self.df

    def sort(self, col: str, order: str = "asc") -> pd.DataFrame:
        """
        Sort the DataFrame by *col*.

        Parameters
        ----------
        col   : str   Column name to sort by.
        order : str   ``'asc'``/``'ascending'`` or ``'desc'``/``'descending'``.

        Raises
        ------
        ValueError
            When *order* is not a recognised keyword.
        """
        order = order.strip().lower()
        if order in {"asc", "ascending"}:
            ascending = True
        elif order in {"desc", "descending"}:
            ascending = False
        else:
            raise ValueError(f"Unknown sort order '{order}'. Use 'asc' or 'desc'.")

        self.df = self.df.sort_values(by=[col], ascending=ascending)
        return self.df

    def generate(self, xr: str, expr: str) -> pd.DataFrame:
        """
        Replace the current DataFrame with a numeric x/y table.

        Parameters
        ----------
        xr   : str   Range string ``'start:end'`` (inclusive on both ends).
        expr : str   A Python expression in terms of *x* and *np*.
                     Example: ``'5*x**2 / np.exp(x)'``

        Returns
        -------
        pd.DataFrame  Columns: ``x``, ``y``.

        Security note
        -------------
        ``eval`` is used intentionally here for scientific/exploratory use.
        Do **not** expose this method to untrusted input.
        """
        lo, hi = (int(v) for v in xr.split(":"))
        x = np.linspace(lo, hi, hi - lo + 1)
        y = eval(expr)  # noqa: S307  (intentional for numeric DSL)
        self.df = pd.DataFrame({"x": x, "y": y})
        return self.df

    def append(self, datafile: str) -> pd.DataFrame:
        """
        Append rows from *datafile* to the current DataFrame.

        Parameters
        ----------
        datafile : str  Path to a CSV file whose schema matches ``self.df``.
        """
        df2 = pd.read_csv(datafile).replace(np.nan, "", regex=True)
        self.df = pd.concat([self.df, df2], ignore_index=True)
        return self.df

    def merge(
        self,
        other_file: str,
        on_column: str,
        how: str = "inner",
    ) -> pd.DataFrame:
        """
        Merge ``self.df`` with a second CSV on a shared column.

        Parameters
        ----------
        other_file : str  Path to the second CSV.
        on_column  : str  Column name present in both DataFrames.
        how        : str  ``'inner'``, ``'outer'``, ``'left'``, or ``'right'``.
        """
        how = how.strip().lower()
        try:
            df2 = pd.read_csv(other_file).replace(np.nan, "", regex=True)
            self.df = pd.merge(self.df, df2, on=on_column, how=how)
        except FileNotFoundError:
            print(f"[ERROR] File not found: '{other_file}'")
        except KeyError:
            print(f"[ERROR] Column '{on_column}' not found in one of the files.")
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
        If every token is numeric → treat as row indices.
        Otherwise → treat as column names.
        """
        tokens = [t.strip() for t in targets.split(",")]
        if all(t.lstrip("-").isnumeric() for t in tokens):
            row_indices = [int(t) for t in tokens]
            self.df = self.df.drop(index=row_indices)
        else:
            self.df = self.df.drop(columns=tokens)
        return self.df

    def modify(self, col: str, old_val: str, new_val: str) -> pd.DataFrame:
        """
        Replace occurrences of *old_val* with *new_val* in column *col*.

        Parameters
        ----------
        col     : str  Target column name.
        old_val : str  Value to search for.
        new_val : str  Replacement value.
        """
        self.df[col] = self.df[col].replace(old_val, new_val)
        return self.df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filename: str) -> str:
        """
        Save the current DataFrame to a CSV file.

        The file is written next to the originally loaded file when a
        datafile was provided; otherwise it is written to the current
        working directory.

        Parameters
        ----------
        filename : str  Output file name (not a full path).

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
        self.df.to_csv(save_path, index=False)
        print(f"Saved → {save_path}")
        return save_path


# ---------------------------------------------------------------------------
# REPL helpers
# ---------------------------------------------------------------------------

def _print_help() -> None:
    """Print a formatted table of available actions."""
    print(f"\n{'Action':<14}  Description")
    print("-" * 60)
    for action, description in ACTION_HELP.items():
        print(f"  {action:<12}  {description}")
    print()


def _require_data(dm: MFPDataManipulator, action: str) -> bool:
    """Return *True* if ``dm.df`` has data; print an error and return *False* otherwise."""
    if dm.df.empty:
        print(f"[ERROR] No data loaded. Load a file or run 'generate' before '{action}'.")
        return False
    return True


def _run_repl(dm: MFPDataManipulator) -> None:
    """
    Enter the interactive command loop for *dm*.

    Each iteration reads one action keyword, prompts for its arguments,
    executes the corresponding method, and prints the result.
    """
    while True:
        try:
            action = input("\nAction> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nInterrupted — exiting.")
            sys.exit(0)

        # ----------------------------------------------------------------
        # Exit
        # ----------------------------------------------------------------
        if action in {"exit", "q"}:
            print("Goodbye.")
            sys.exit(0)

        # ----------------------------------------------------------------
        # Help
        # ----------------------------------------------------------------
        elif action == "help":
            _print_help()

        # ----------------------------------------------------------------
        # Inspection
        # ----------------------------------------------------------------
        elif action == "show":
            dm.show()

        elif action in {"properties", "props"}:
            dm.properties()

        # ----------------------------------------------------------------
        # Data generation (allowed without a pre-loaded file)
        # ----------------------------------------------------------------
        elif action in {"generate", "gen"}:
            xr   = input("  x range (e.g. 0:10): ").strip()
            expr = input("  y expression (e.g. 5*x**2/np.exp(x)): ").strip()
            print(dm.generate(xr, expr))

        # ----------------------------------------------------------------
        # Operations that require existing data
        # ----------------------------------------------------------------
        elif action == "slice":
            if not _require_data(dm, action):
                continue
            start = input("  Start index: ").strip()
            end   = input("  End index  : ").strip()
            print(dm.slice(start, end))

        elif action == "sort":
            if not _require_data(dm, action):
                continue
            col   = input("  Column to sort by: ").strip()
            order = input("  Order (asc / desc): ").strip()
            print(dm.sort(col, order))

        elif action == "merge":
            if not _require_data(dm, action):
                continue
            other  = input("  File to merge with: ").strip()
            col    = input("  Column to merge on: ").strip()
            how    = input("  Merge type (inner / outer / left / right): ").strip()
            print(dm.merge(other, col, how))

        elif action == "append":
            file_to_append = input("  File to append: ").strip()
            print(dm.append(file_to_append))

        elif action in {"delete", "del"}:
            if not _require_data(dm, action):
                continue
            targets = input("  Column names or row indices (comma-separated): ").strip()
            print(dm.delete(targets))

        elif action == "modify":
            if not _require_data(dm, action):
                continue
            col     = input("  Column name : ").strip()
            old_val = input("  Old value   : ").strip()
            new_val = input("  New value   : ").strip()
            print(dm.modify(col, old_val, new_val))

        elif action == "save":
            filename = input("  Output file name: ").strip()
            dm.save(filename)

        # ----------------------------------------------------------------
        # Unknown action
        # ----------------------------------------------------------------
        else:
            print(f"[WARN] Unknown action '{action}'. Type 'help' for options.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    """
    Parse CLI arguments, load (or skip loading) a datafile, and start the REPL.

    Parameters
    ----------
    argv : list[str] | None
        Override ``sys.argv[1:]`` in tests.
    """
    if argv is None:
        argv = sys.argv[1:]

    print(_BANNER)

    # Optional positional argument: path to a CSV file
    if argv:
        datafile = argv[0]
        try:
            dm = MFPDataManipulator(datafile)
        except FileNotFoundError:
            print(f"[ERROR] File not found: '{datafile}'")
            sys.exit(1)
    else:
        # Interactive prompt when no argument is supplied
        while True:
            datafile = input("CSV file to load (or press Enter to skip): ").strip()
            if not datafile:
                dm = MFPDataManipulator()
                print("No file loaded — you can generate data from scratch.")
                break
            try:
                dm = MFPDataManipulator(datafile)
                break
            except FileNotFoundError:
                print(f"[ERROR] File not found: '{datafile}'. Please try again.")

    _run_repl(dm)


if __name__ == "__main__":
    main()