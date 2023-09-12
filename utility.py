import os
import matplotlib as mpl
import matplotlib.font_manager as font_manager
from pathlib import Path


def load_matplotlib_local_fonts(font_path: str, font_size: int) -> None:
    """Load a local font file and update the Matplotlib font configuration"""
    # Construct a Path object from the provided font_path
    font_path_obj = Path(os.path.join(os.path.dirname(__file__), font_path))

    # Check if the file exists
    assert font_path_obj.exists(), "Font file does not exist"

    # Add the font to Matplotlib's font manager
    font_manager.fontManager.addfont(str(font_path_obj))
    prop = font_manager.FontProperties(fname=font_path_obj)  # type: ignore

    # Update the default font family and font size
    mpl.rc("font", family="sans-serif")
    mpl.rcParams.update(
        {
            "font.size": font_size,
            "font.sans-serif": [prop.get_name()],
        }
    )
