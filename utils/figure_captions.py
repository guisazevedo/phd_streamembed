# IMPORTS *****************************************************************************************
# External libraries
import textwrap


# Main function ***********************************************************************************
def split_with_notes(
    row,
    notes: list[tuple[int, int, str]],
    width_ratios: list[float],
    row_width_in: float,
    plot_height: float,
    note_height: float,
    fontsize: float = 8.5,
    color: str = "#52514e",
):
    """
    Split one figure row into a plot area (top) and a caption area (bottom) and write the captions.

    Each caption is wrapped to the width of the columns it spans, so it sits under its panel(s).
    The column widths follow `width_ratios`, which should be the ratios used for the panels.

    * Arguments:
    row: The matplotlib SubFigure (one row of the figure) to split
    notes: Captions as (first_column, last_column_exclusive, text), one per panel or group of panels
    width_ratios: The relative width of each panel column in this row
    row_width_in: The width of the row in inches (used to wrap the text)
    plot_height: The relative height of the plot area
    note_height: The relative height of the caption area
    fontsize: The caption font size
    color: The caption text colour

    Returns the top SubFigure, where the caller draws the panels.
    """

    # STEP 1: Split the row and build a column grid for the captions
    plots, bottom = row.subfigures(2, 1, height_ratios=[plot_height, note_height])
    grid = bottom.add_gridspec(1, len(width_ratios), width_ratios=width_ratios)

    # STEP 2: Wrap each caption to its columns (about 15 characters per inch at 8.5 pt) and draw it
    for first, last, text in notes:
        share = sum(width_ratios[first:last]) / sum(width_ratios)
        chars = max(20, int((row_width_in * share - 0.5) * 15 * 8.5 / fontsize))
        ax = bottom.add_subplot(grid[0, first:last])
        ax.axis("off")
        paragraphs = text.split("\n")
        wrapped = "\n\n".join(textwrap.fill(p.strip(), chars) for p in paragraphs)
        ax.text(0, 1, wrapped, va="top", ha="left", fontsize=fontsize, color=color, linespacing=1.35)

    return plots
