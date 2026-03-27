#!/usr/bin/env python3
"""
mfp_dmanp_help.py — MFP Data Manipulator Manual Page
================================================
Import and call ``show()`` to print the full manual, or call individual
section printers for targeted help.

Usage from CLI:
    python mfp_dmanp_help.py                  # full manual
    python mfp_dmanp_help.py overview         # startup & file formats
    python mfp_dmanp_help.py inspection       # show, head, tail, properties, counts
    python mfp_dmanp_help.py filter           # filter / slice / sort
    python mfp_dmanp_help.py transform        # rename, cast, addcol, modify, delete
    python mfp_dmanp_help.py clean            # dedup, fillna, dropna
    python mfp_dmanp_help.py io               # load, generate, append, merge, save
    python mfp_dmanp_help.py history          # undo / redo
    python mfp_dmanp_help.py tips             # query syntax, eval expressions, gotchas
"""

import os
import sys

# ---------------------------------------------------------------------------
# ANSI colour helpers  (auto-disabled on non-TTY / Windows cmd)
# ---------------------------------------------------------------------------

_NO_COLOR = not sys.stdout.isatty() or os.name == "nt"

def _c(code: str, text: str) -> str:
    return text if _NO_COLOR else f"\033[{code}m{text}\033[0m"

H1   = lambda t: _c("1;36",  t)   # bold cyan    — top-level heading
H2   = lambda t: _c("1;33",  t)   # bold yellow  — section heading
H3   = lambda t: _c("1;37",  t)   # bold white   — sub-heading / command name
KW   = lambda t: _c("1;32",  t)   # bold green   — keyword / argument token
EX   = lambda t: _c("90",    t)   # dark grey    — example / prompt line
NOTE = lambda t: _c("35",    t)   # magenta      — note / tip
WARN = lambda t: _c("1;31",  t)   # bold red     — warning


# ---------------------------------------------------------------------------
# Reusable layout helpers
# ---------------------------------------------------------------------------

def _rule(char: str = "─", width: int = 70) -> str:
    return _c("90", char * width)

def _h1(title: str) -> None:
    print()
    print(H1("┌" + "─" * 68 + "┐"))
    print(H1(f"│  {title:<66}│"))
    print(H1("└" + "─" * 68 + "┘"))

def _h2(title: str) -> None:
    print()
    print(H2(f"  {title}"))
    print(_c("33", "  " + "─" * (len(title) + 2)))

def _h3(cmd: str, synopsis: str) -> None:
    print(f"\n  {H3(cmd)}")
    print(f"    {synopsis}")

def _args(*rows: tuple[str, str]) -> None:
    """Print an aligned argument table."""
    w = max(len(r[0]) for r in rows) + 2
    for name, desc in rows:
        print(f"      {KW(name):<{w + 9}}  {desc}")

def _ex(*lines: str) -> None:
    """Print example prompt lines."""
    for line in lines:
        print(EX(f"    Action> {line}") if not line.startswith("  ") else EX(line))

def _note(text: str) -> None:
    print(f"    {NOTE('Note:')} {text}")

def _warn(text: str) -> None:
    print(f"    {WARN('Warning:')} {text}")

def _blank() -> None:
    print()


# ---------------------------------------------------------------------------
# Section: Overview
# ---------------------------------------------------------------------------

def section_overview() -> None:
    _h1("MFP Data Manipulator  v3.0.0  —  Overview")

    print(f"""
  {H2('What it does')}

  MFP Data Manipulator is an interactive command-line tool for exploring
  and cleaning tabular data without writing any code.  Load a file, type
  commands at the prompt, inspect results immediately, and save when done.

  {H2('Starting the program')}

    {EX('python mfp_data_manipulator.py')}            # interactive file prompt
    {EX('python mfp_data_manipulator.py data.csv')}   # load a file directly

  On startup you will see the  {KW('Action>')}  prompt.  Type a command name
  and press Enter.  The program will ask for any required arguments one
  at a time.

  {H2('Supported file formats')}""")

    rows = [
        (".csv",  "Comma-separated values  (read + write)"),
        (".xlsx", "Excel workbook          (read + write, requires openpyxl)"),
        (".xls",  "Legacy Excel            (read + write, requires xlrd)"),
        (".json", "JSON array of objects   (read + write)"),
    ]
    w = max(len(r[0]) for r in rows)
    for ext, desc in rows:
        print(f"    {KW(ext):<{w + 9}}  {desc}")

    print(f"""
  The format is detected automatically from the file extension for every
  {KW('load')}, {KW('save')}, {KW('append')}, and {KW('merge')} operation.

  {H2('Command anatomy')}

  Every command follows the same pattern:

    1. Type the command name at  {KW('Action>')}  and press Enter.
    2. The program prompts for each argument in turn.
    3. The result is printed immediately.
    4. The change is recorded on the undo stack — type  {KW('undo')}  to revert.

  {NOTE('Tip:')} Type  {KW('help')}  at the prompt for a compact command listing.
    Type  {KW('q')}  or  {KW('exit')}  to quit (you will be warned about unsaved changes).""")
    _blank()


# ---------------------------------------------------------------------------
# Section: Inspection
# ---------------------------------------------------------------------------

def section_inspection() -> None:
    _h1("Inspection Commands")

    print(f"""
  These commands are read-only — they never modify the DataFrame and do
  not push to the undo stack.""")

    # show ─────────────────────────────────────────────────────────────────
    _h2("show")
    _h3("show", "Print the entire DataFrame to the terminal.")
    _blank()
    _ex("show")
    _note("For large files use  head  or  tail  instead to avoid flooding the terminal.")

    # head ─────────────────────────────────────────────────────────────────
    _h2("head")
    _h3("head", "Print the first N rows.")
    _args(("N", "Number of rows to show.  Default: 10  (press Enter to accept)."))
    _blank()
    _ex("head", "  Number of rows (default 10): 20")

    # tail ─────────────────────────────────────────────────────────────────
    _h2("tail")
    _h3("tail", "Print the last N rows.")
    _args(("N", "Number of rows to show.  Default: 10."))
    _blank()
    _ex("tail", "  Number of rows (default 10): 5")

    # properties / props ───────────────────────────────────────────────────
    _h2("properties  (alias: props)")
    _h3("properties", "Display a full diagnostic summary of the current DataFrame.")
    print(f"""
    Output includes three sections:

      {KW('Columns')}       — index number, name, and dtype for every column.
      {KW('NaN counts')}    — number of empty / NaN cells per column.
      {KW('Statistics')}    — pandas describe() covering count, mean, std, min,
                       quartiles, max for numeric columns; top/freq for others.""")
    _blank()
    _ex("properties")
    _ex("props")

    # counts ───────────────────────────────────────────────────────────────
    _h2("counts")
    _h3("counts", "Show the frequency of each unique value in a column.")
    _args(("Column name", "Name of the column to count."))
    _blank()
    _ex("counts", "  Column name: status")
    _note("Results are sorted most-frequent first.  NaN / empty values are included.")
    _blank()


# ---------------------------------------------------------------------------
# Section: Filter / Slice / Sort
# ---------------------------------------------------------------------------

def section_filter() -> None:
    _h1("Row Selection — filter, slice, sort")

    # filter ───────────────────────────────────────────────────────────────
    _h2("filter")
    _h3("filter", "Keep only rows that satisfy a boolean expression.")
    _args(
        ("Query expression",
         "A pandas query string (see Tips section for full syntax)."),
    )
    print(f"""
    {H3('Examples')}""")
    examples = [
        ("filter", "age > 30",                     "rows where age is over 30"),
        ("filter", 'city == "Roanoke"',             "exact string match"),
        ("filter", "score >= 90 and grade == \"A\"","compound condition"),
        ("filter", "price != 0",                   "exclude zero-price rows"),
        ("filter", "`first name` == \"Alice\"",    "column names with spaces need backticks"),
    ]
    for cmd, expr, comment in examples:
        print(f"    {EX(f'Action> {cmd}')}")
        print(f"    {EX(f'  Query expression: {expr}')}{_c('90', f'   # {comment}')}")
        _blank()

    _note("filter resets the row index after removing rows.")
    _warn("filter is permanent until you run  undo.  Use  head  to preview first.")

    # slice ────────────────────────────────────────────────────────────────
    _h2("slice")
    _h3("slice", "Keep rows at integer positions [start, end)  (end is excluded).")
    _args(
        ("Start index", "First row to keep (0-based)."),
        ("End index",   "Row after the last one to keep."),
    )
    _blank()
    _ex("slice",
        "  Start index: 0",
        "  End index  : 100")
    _note("Equivalent to df.iloc[start:end].  Use filter for condition-based selection.")

    # sort ─────────────────────────────────────────────────────────────────
    _h2("sort")
    _h3("sort", "Sort the DataFrame by a single column.")
    _args(
        ("Column to sort by", "Column name."),
        ("Order",             "asc  (ascending) or  desc  (descending)."),
    )
    _blank()
    _ex("sort", "  Column to sort by: price", "  Order (asc / desc): desc")
    _note("Aliases  ascending  and  descending  are also accepted.")
    _blank()


# ---------------------------------------------------------------------------
# Section: Transform
# ---------------------------------------------------------------------------

def section_transform() -> None:
    _h1("Transformation Commands")

    # rename ───────────────────────────────────────────────────────────────
    _h2("rename")
    _h3("rename", "Rename one or more columns in a single operation.")
    _args(
        ("Rename pairs",
         "Comma-separated  old:new  pairs."),
    )
    _blank()
    _ex("rename",
        "  Rename pairs, comma-separated (old:new, ...): first name:first_name, Last Name:last_name")
    _note("All old names are validated before any renaming takes place.")

    # cast ─────────────────────────────────────────────────────────────────
    _h2("cast")
    _h3("cast", "Change the data type of a column.")
    _args(
        ("Column name",   "The column to convert."),
        ("Target dtype",  "One of:  int  |  float  |  str  |  datetime"),
    )
    print(f"""
    {H3('dtype reference')}

      {KW('int')}       Converts via pd.to_numeric then casts to Python int.
               Fails loudly if any value cannot be converted.
      {KW('float')}     Converts via pd.to_numeric then casts to float64.
      {KW('str')}       Converts every value to a Python string (never fails).
      {KW('datetime')}  Parses dates with pd.to_datetime.  Many formats are
               recognised automatically (ISO 8601, US, EU, etc.).
    """)
    _ex("cast",
        "  Column name: purchase_date",
        "  Target dtype (int / float / str / datetime): datetime")
    _note("If conversion fails, the DataFrame is NOT modified (the snapshot is rolled back).")

    # addcol ───────────────────────────────────────────────────────────────
    _h2("addcol")
    _h3("addcol", "Add a new column computed from existing columns.")
    _args(
        ("New column name", "Name for the derived column."),
        ("Expression",      "A pandas eval expression over existing column names."),
    )
    print(f"""
    {H3('Expression examples')}

      {KW('price * qty')}                    multiply two columns
      {KW('revenue - cost')}                 subtract
      {KW('score / score.max()')}            normalise to [0, 1]
      {KW('(a + b) / 2')}                    average of two columns
      {KW('tax_rate * subtotal + subtotal')} compound expression
    """)
    _ex("addcol",
        "  New column name: total",
        "  Expression (e.g. price * qty): price * qty")
    _note("Uses DataFrame.eval() — column names with spaces must be wrapped in backticks.")

    # modify ───────────────────────────────────────────────────────────────
    _h2("modify")
    _h3("modify", "Replace a specific value anywhere in a column.")
    _args(
        ("Column name", "Column to search."),
        ("Old value",   "Exact value to find (string comparison)."),
        ("New value",   "Replacement value."),
    )
    _blank()
    _ex("modify",
        "  Column name : status",
        "  Old value   : PNDG",
        "  New value   : PENDING")
    _note("Only exact matches are replaced.  For pattern-based replacement, use  cast  to str first.")

    # delete / del ─────────────────────────────────────────────────────────
    _h2("delete  (alias: del)")
    _h3("delete", "Drop rows (by index) or columns (by name).")
    _args(
        ("Targets",
         "Comma-separated integers to drop rows,  OR  column names to drop columns."),
    )
    print(f"""
    {H3('Row deletion')}  — supply integer row indices:
    """)
    _ex("delete", "  Column names or row indices (comma-separated): 0, 5, 12")
    print(f"""
    {H3('Column deletion')}  — supply column names:
    """)
    _ex("delete", "  Column names or row indices (comma-separated): notes, internal_id")
    _warn("You cannot mix row indices and column names in the same call.")
    _blank()


# ---------------------------------------------------------------------------
# Section: Clean
# ---------------------------------------------------------------------------

def section_clean() -> None:
    _h1("Data Cleaning — dedup, fillna, dropna")

    # dedup ────────────────────────────────────────────────────────────────
    _h2("dedup")
    _h3("dedup", "Remove duplicate rows, keeping the first occurrence.")
    _args(
        ("Columns (optional)",
         "Comma-separated column names to compare.  Press Enter to use all columns."),
    )
    _blank()
    _ex("dedup", "  Columns to check (comma-separated, or Enter for all): ")
    print(EX("    → uses all columns (full-row duplicates)"))
    _blank()
    _ex("dedup", "  Columns to check (comma-separated, or Enter for all): email")
    print(EX("    → keeps the first row for each unique email address"))
    _note("The row index is reset after deduplication.")

    # fillna ───────────────────────────────────────────────────────────────
    _h2("fillna")
    _h3("fillna", "Fill empty / NaN cells in one column with a fixed value.")
    _args(
        ("Column name", "Column containing the missing values."),
        ("Fill value",  "Replacement value.  Automatically cast to int or float\n"
                        "                     when the string is a valid number."),
    )
    _blank()
    _ex("fillna",
        "  Column name: age",
        "  Fill value : 0")
    _blank()
    _ex("fillna",
        "  Column name: country",
        "  Fill value : Unknown")
    _note("Both empty strings ('') and NaN are treated as missing.")

    # dropna ───────────────────────────────────────────────────────────────
    _h2("dropna")
    _h3("dropna", "Drop rows that contain missing values.")
    _args(
        ("Column name (optional)",
         "Drop rows where this specific column is empty / NaN.\n"
         "                          Press Enter to drop rows missing in ANY column."),
    )
    _blank()
    _ex("dropna", "  Column name (or Enter to drop rows with any NaN): ")
    print(EX("    → drops every row that has at least one empty / NaN cell"))
    _blank()
    _ex("dropna", "  Column name (or Enter to drop rows with any NaN): email")
    print(EX("    → drops only rows where the email column is empty / NaN"))
    _note("The row index is reset after dropping.")
    _blank()


# ---------------------------------------------------------------------------
# Section: I/O
# ---------------------------------------------------------------------------

def section_io() -> None:
    _h1("I/O Commands — load, generate, append, merge, save")

    # load ─────────────────────────────────────────────────────────────────
    _h2("load")
    _h3("load", "Replace the current DataFrame by loading a new file.")
    _args(
        ("File path", "Path to a CSV, Excel (.xlsx / .xls), or JSON file."),
    )
    _blank()
    _ex("load", "  File path (CSV / Excel / JSON): /data/customers.xlsx")
    _note("If you have unsaved changes you will be prompted to confirm before loading.")
    _note("The previous state is pushed to the undo stack, so  undo  recovers it.")

    # generate / gen ───────────────────────────────────────────────────────
    _h2("generate  (alias: gen)")
    _h3("generate", "Build a numeric x/y DataFrame from a mathematical expression.")
    _args(
        ("x range",     "Integer range in the form  start:end  (both inclusive)."),
        ("y expression","A Python/NumPy expression in terms of  x  and  np."),
    )
    print(f"""
    {H3('Expression examples')}

      {KW('x**2')}                  square
      {KW('np.sin(x)')}             sine wave
      {KW('5*x**2 / np.exp(x)')}   damped parabola
      {KW('np.log(x + 1)')}        shifted log  (safe for x = 0)
      {KW('np.where(x > 5, 1, 0)')}step function
    """)
    _ex("generate",
        "  x range (e.g. 0:10): 0:100",
        "  y expression (e.g. 5*x**2/np.exp(x)): np.sin(x / 10)")
    _warn("generate replaces the current DataFrame.  Use  undo  to recover previous data.")

    # append ───────────────────────────────────────────────────────────────
    _h2("append")
    _h3("append", "Append rows from a second file to the bottom of the current DataFrame.")
    _args(
        ("File to append", "Path to a file whose columns match the current DataFrame."),
    )
    _blank()
    _ex("append", "  File to append: new_orders.csv")
    _note("The row index is reset after appending.  Column names must match exactly.")

    # merge ────────────────────────────────────────────────────────────────
    _h2("merge")
    _h3("merge", "Join the current DataFrame with a second file on a shared column.")
    _args(
        ("File to merge with", "Path to the second file (any supported format)."),
        ("Column to merge on", "Column name that exists in both DataFrames."),
        ("Merge type",         "inner | outer | left | right"),
    )
    print(f"""
    {H3('Merge type reference')}

      {KW('inner')}   Keep only rows where the key exists in BOTH tables.
      {KW('outer')}   Keep all rows from both tables; fill gaps with NaN.
      {KW('left')}    Keep all rows from the left (current) table.
      {KW('right')}   Keep all rows from the right (new) table.
    """)
    _ex("merge",
        "  File to merge with: products.csv",
        "  Column to merge on: product_id",
        "  Merge type (inner / outer / left / right): left")

    # save ─────────────────────────────────────────────────────────────────
    _h2("save")
    _h3("save", "Write the current DataFrame to a file.")
    _args(
        ("Output file name",
         "File name with extension (.csv / .xlsx / .json).\n"
         "                    Saved next to the loaded file, or in the current\n"
         "                    working directory if no file was loaded."),
    )
    _blank()
    _ex("save", "  Output file name (e.g. output.csv / .xlsx / .json): cleaned_data.csv")
    _ex("save", "  Output file name (e.g. output.csv / .xlsx / .json): report.xlsx")
    _note("The format is determined entirely by the extension you choose.")
    _note("Saving does NOT clear the undo stack.")
    _blank()


# ---------------------------------------------------------------------------
# Section: History (undo / redo)
# ---------------------------------------------------------------------------

def section_history() -> None:
    _h1("History — undo & redo")

    print(f"""
  Every command that modifies the DataFrame automatically saves a snapshot
  of the previous state before making any change.  Up to {KW('20')} snapshots are
  kept in memory at a time.

  {NOTE('Non-mutating commands')} (show, head, tail, properties, counts) do NOT push
  to the undo stack.
    """)

    # undo ─────────────────────────────────────────────────────────────────
    _h2("undo")
    _h3("undo", "Revert the last mutating operation.")
    print(f"""
    Pops the most recent snapshot off the undo stack and restores it as
    the current DataFrame.  The reverted state is pushed onto the redo
    stack so it can be re-applied with  {KW('redo')}.
    """)
    _ex("undo")
    print(EX("    Undone. DataFrame is now 1 200 rows × 8 cols."))

    # redo ─────────────────────────────────────────────────────────────────
    _h2("redo")
    _h3("redo", "Re-apply the last undone operation.")
    print(f"""
    Pops the most recent entry off the redo stack.  Any new mutating
    command clears the redo stack entirely (standard editor behaviour).
    """)
    _ex("redo")
    print(EX("    Redone. DataFrame is now 900 rows × 8 cols."))

    print(f"""
  {H3('Typical workflow')}

    {EX('Action> filter')}
    {EX('  Query expression: age > 18')}
    {EX('    Filter kept 820 of 1 000 rows.')}

    {EX('Action> undo')}
    {EX('    Undone. DataFrame is now 1 000 rows × 5 cols.')}

    {EX('Action> redo')}
    {EX('    Redone. DataFrame is now 820 rows × 5 cols.')}

  {NOTE('Tip:')} The exit / load / generate commands warn you if the undo stack
  is non-empty, indicating you may have unsaved changes.
    """)


# ---------------------------------------------------------------------------
# Section: Tips
# ---------------------------------------------------------------------------

def section_tips() -> None:
    _h1("Tips, Query Syntax & Common Patterns")

    # Filter / query syntax ────────────────────────────────────────────────
    _h2("filter — query expression syntax")
    print(f"""
  The  {KW('filter')}  command uses pandas  DataFrame.query()  under the hood.
  The full expression language is a subset of Python with these rules:

  {H3('Comparison operators')}
    {KW('>')}  {KW('>=')}  {KW('<')}  {KW('<=')}  {KW('==')}  {KW('!=')}

  {H3('Logical operators')}
    {KW('and')}   {KW('or')}   {KW('not')}

  {H3('String values')}  — use double or single quotes inside the expression:
    {EX('Action> filter')}
    {EX('  Query expression: city == "New York"')}

  {H3('Column names with spaces')}  — wrap in backticks:
    {EX('Action> filter')}
    {EX('  Query expression: `first name` == "Alice" and `zip code` == "24060"')}

  {H3('Membership test')}  — check if a value is in a list:
    {EX('Action> filter')}
    {EX('  Query expression: status in ["open", "pending"]')}

  {H3('Negation')}:
    {EX('Action> filter')}
    {EX('  Query expression: status not in ["closed", "cancelled"]')}

  {H3('Null / empty check')}  — find non-empty rows:
    {EX('Action> filter')}
    {EX('  Query expression: email != ""')}

  {NOTE('Tip:')} Run  {KW('cast')}  to convert a column to the right dtype before
  filtering on numeric comparisons — CSV columns are often read as strings.
    """)

    # addcol / generate expression syntax ─────────────────────────────────
    _h2("addcol & generate — expression syntax")
    print(f"""
  {KW('addcol')}  uses  DataFrame.eval()  which understands all standard Python
  arithmetic operators over column names.

  {KW('generate')}  uses Python's built-in  eval()  with  x  (a NumPy array) and
  the  np  module in scope.

  {H3('addcol examples')}
    {EX('price * qty')}                multiply two columns → total revenue
    {EX('(high + low) / 2')}          midpoint
    {EX('revenue - cost')}            profit
    {EX('score / score.max()')}       normalise to [0, 1]
    {EX('tax * (subtotal + shipping)')}

  {H3('generate examples')}
    {EX('x**2')}                      parabola
    {EX('np.sqrt(x)')}                square root
    {EX('np.sin(x) * np.exp(-x/10)')} damped sine
    {EX('np.log1p(x)')}               natural log of (x + 1)
    {EX('np.where(x % 2 == 0, 1, -1)')}alternate sign by parity

  {NOTE('Tip:')} For  generate, always wrap potentially unsafe domains:
  e.g.  {KW('np.log(x + 1e-9)')}  instead of  {KW('np.log(x)')}  to avoid -inf at 0.
    """)

    # cast datetime ────────────────────────────────────────────────────────
    _h2("Working with dates")
    print(f"""
  Step 1 — cast the column to datetime:
    {EX('Action> cast')}
    {EX('  Column name: order_date')}
    {EX('  Target dtype (int / float / str / datetime): datetime')}

  Step 2 — filter using ISO 8601 strings (pandas parses them automatically):
    {EX('Action> filter')}
    {EX('  Query expression: order_date >= "2024-01-01"')}

  Step 3 — add a derived year column:
    {EX('Action> addcol')}
    {EX('  New column name: year')}
    {EX("  Expression: order_date.dt.year")}

  {NOTE('Tip:')} After  cast datetime, the column supports  .dt  accessor
  expressions inside  addcol  (e.g.  order_date.dt.month).
    """)

    # rename patterns ──────────────────────────────────────────────────────
    _h2("Bulk rename pattern")
    print(f"""
  Rename multiple columns in one command by separating pairs with commas:
    {EX('Action> rename')}
    {EX('  Rename pairs: First Name:first_name, Last Name:last_name, ZIP Code:zip_code')}

  Handy after loading files that have inconsistent column name formatting.
    """)

    # Common cleaning workflow ─────────────────────────────────────────────
    _h2("Common cleaning workflow")
    steps = [
        ("load",       "Load the raw file."),
        ("properties", "Inspect dtypes, spot NaN columns."),
        ("cast",       "Fix columns read as wrong dtype (e.g. dates, numbers)."),
        ("rename",     "Standardise column names."),
        ("dedup",      "Remove duplicate rows."),
        ("fillna",     "Fill known-default columns (e.g. quantity → 0)."),
        ("dropna",     "Remove rows missing critical fields (e.g. id, email)."),
        ("filter",     "Discard out-of-range or invalid rows."),
        ("addcol",     "Compute derived columns (e.g. total = price * qty)."),
        ("save",       "Write the clean file."),
    ]
    w = max(len(s[0]) for s in steps)
    for i, (cmd, desc) in enumerate(steps, 1):
        print(f"    {i:>2}.  {KW(cmd):<{w + 9}}  {desc}")
    _blank()

    print(f"  {NOTE('Tip:')} Between each step, run  {KW('undo')}  if the result looks wrong,")
    print(f"  then try a different approach.  Run  {KW('head 5')}  to spot-check quickly.")
    _blank()


# ---------------------------------------------------------------------------
# Master index
# ---------------------------------------------------------------------------

_SECTIONS: dict[str, tuple[str, object]] = {
    "overview":   ("Overview & startup",                  section_overview),
    "inspection": ("Inspection — show, head, tail, ...",  section_inspection),
    "filter":     ("Row selection — filter, slice, sort", section_filter),
    "transform":  ("Transformation — rename, cast, ...",  section_transform),
    "clean":      ("Data cleaning — dedup, fillna, ...",  section_clean),
    "io":         ("I/O — load, generate, save, ...",     section_io),
    "history":    ("History — undo & redo",               section_history),
    "tips":       ("Tips, query syntax & patterns",       section_tips),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def show(section: str = "") -> None:
    """
    Print the manual.

    Parameters
    ----------
    section : str
        Optional section key.  When empty (default) the full manual is
        printed.  Pass one of the keys listed in ``_SECTIONS`` to print
        only that section.

    Raises
    ------
    KeyError
        When *section* is not a recognised key.
    """
    if section:
        key = section.strip().lower()
        if key not in _SECTIONS:
            valid = ", ".join(_SECTIONS)
            raise KeyError(f"Unknown section '{key}'. Valid: {valid}")
        _SECTIONS[key][1]()
        return

    # Full manual — print a table of contents first
    _h1("MFP Data Manipulator  —  Full Manual")
    print(f"\n  {H2('Table of contents')}\n")
    for key, (title, _) in _SECTIONS.items():
        print(f"    {KW(f'python mfp_dmanp_help.py {key}'):<52}  {title}")
    _blank()

    for _, (_, printer) in _SECTIONS.items():
        printer()
        print(_rule())


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    arg = sys.argv[1].strip().lower() if len(sys.argv) > 1 else ""
    if arg in {"-h", "--help"}:
        print(__doc__)
        sys.exit(0)
    try:
        show(arg)
    except KeyError as exc:
        print(f"Error: {exc}")
        print(f"Usage: python mfp_dmanp_help.py [{' | '.join(_SECTIONS)}]")
        sys.exit(1)
