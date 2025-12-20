import matplotlib.pyplot as plt

import config

# Subdirectory for plots within artifacts
PLOTS_SUBDIR = "plots"


def is_notebook() -> bool:
    """Check if code is running in a Jupyter notebook."""
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        shell_name = shell.__class__.__name__
        if shell_name == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell_name == "TerminalInteractiveShell":
            return False  # Terminal IPython
        else:
            return False
    except (ImportError, NameError):
        return False


def show_or_save_plot(name: str, always_store: bool = False) -> None:
    """
    Show or save a matplotlib plot depending on the execution environment.

    Args:
        name: The name for the plot file (without extension).
        always_store: If True, always save the plot to disk regardless of environment.
    """
    if always_store or not is_notebook():
        plots_dir = config.ARTIFACTS_DIR / PLOTS_SUBDIR
        plots_dir.mkdir(parents=True, exist_ok=True)

        plot_path = plots_dir / f"{name}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {plot_path}")

    if is_notebook():
        plt.show()
    else:
        plt.close()
