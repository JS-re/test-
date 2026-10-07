import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from shiny import App, Inputs, Outputs, Session, render, ui

# Set global aesthetic theme for plots
sns.set_theme(style="darkgrid")

# UI Definition
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("Control Panel"),
        ui.hr(),
        # Correlation Strength Slider
        ui.input_slider(
            "r_target",
            "Target Correlation (r):",
            min=-0.99,
            max=0.99,
            value=0.65,
            step=0.01,
        ),
        # Relationship Preset Buttons
        ui.label("Relationship Presets:"),
        ui.layout_columns(
            ui.input_action_button(
                "btn_pos", "Positive (+0.75)", class_="btn-sm btn-outline-success"
            ),
            ui.input_action_button(
                "btn_neg", "Negative (-0.75)", class_="btn-sm btn-outline-danger"
            ),
            col_widths=(6, 6),
        ),
        ui.layout_columns(
            ui.input_action_button(
                "btn_zero", "Zero (0.00)", class_="btn-sm btn-outline-secondary"
            ),
            ui.input_action_button(
                "btn_resample",
                "Resample 🎲",
                class_="btn-sm btn-outline-primary",
            ),
            col_widths=(6, 6),
        ),
        ui.hr(),
        # Sample Size Slider
        ui.input_slider(
            "n_samples",
            "Sample Size (N):",
            min=30,
            max=1000,
            value=200,
            step=10,
        ),
        ui.hr(),
        # Visual Feature Toggles
        ui.input_checkbox("show_trend", "Show OLS Trend Line", value=True),
        ui.input_checkbox("show_ci", "Show 95% Confidence Band", value=True),
        ui.input_checkbox("show_density", "Show Density Contours", value=False),
        width=340,
    ),
    ui.h2("Bivariate Scatter Plot & Trend Analysis"),
    ui.p("Interactive correlation and linear regression explorer powered by Shiny for Python."),
    # Summary Statistics Dashboard Cards
    ui.layout_columns(
        ui.card(
            ui.card_header("Pearson's r"),
            ui.output_text("stat_r"),
        ),
        ui.card(
            ui.card_header("R² (Variance Explained)"),
            ui.output_text("stat_r2"),
        ),
        ui.card(
            ui.card_header("OLS Slope (β₁)"),
            ui.output_text("stat_slope"),
        ),
        ui.card(
            ui.card_header("p-value"),
            ui.output_text("stat_p"),
        ),
        col_widths=(3, 3, 3, 3),
    ),
    # Interactive Plot Output
    ui.card(
        ui.card_header("Interactive Scatter Plot"),
        ui.output_plot("scatter_plot", height="520px"),
    ),
    title="Bivariate Analysis Explorer",
)


# Server Logic
def server(input: Inputs, output: Outputs, session: Session):

    # Reactive Effect: Handle Preset Buttons
    @session.react()
    def _handle_presets():
        if input.btn_pos() > 0:
            ui.update_slider("r_target", value=0.75)
        if input.btn_neg() > 0:
            ui.update_slider("r_target", value=-0.75)
        if input.btn_zero() > 0:
            ui.update_slider("r_target", value=0.00)

    # Reactive Data Generation via Cholesky Decomposition
    @session.reactive
    def data():
        # Trigger re-generation when Resample button is clicked
        input.btn_resample()

        r = input.r_target()
        n = input.n_samples()

        # Generate standard normal variables
        z1 = np.random.normal(0, 1, n)
        z2 = np.random.normal(0, 1, n)

        # Inject correlation r using Cholesky transform
        x = z1
        y = r * z1 + np.sqrt(max(0, 1 - r**2)) * z2

        # Standardize X and Y (Z-scores)
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)

        # Calculate OLS linear regression metrics
        res = stats.linregress(x, y)

        return {
            "x": x,
            "y": y,
            "r_emp": res.rvalue,
            "r2": res.rvalue**2,
            "slope": res.slope,
            "intercept": res.intercept,
            "p_val": res.pvalue,
        }

    # Render Summary Metric Outputs
    @output
    @render.text
    def stat_r():
        return f"{data()['r_emp']:.3f}"

    @output
    @render.text
    def stat_r2():
        return f"{data()['r2']:.3f}"

    @output
    @render.text
    def stat_slope():
        return f"{data()['slope']:.3f}"

    @output
    @render.text
    def stat_p():
        p = data()["p_val"]
        return "< 0.001" if p < 0.001 else f"{p:.4f}"

    # Render Main Scatter Plot
    @output
    @render.plot
    def scatter_plot():
        d = data()
        fig, ax = plt.subplots(figsize=(9, 6))

        # Optional Density Contours (KDE)
        if input.show_density():
            sns.kdeplot(
                x=d["x"],
                y=d["y"],
                ax=ax,
                cmap="mako",
                alpha=0.4,
                fill=True,
                thresh=0.05,
            )

        # Scatter plot with optional trend line & 95% CI
        if input.show_trend():
            sns.regplot(
                x=d["x"],
                y=d["y"],
                ax=ax,
                ci=95 if input.show_ci() else None,
                color="#2b5c8f",
                scatter_kws={"alpha": 0.65, "s": 40, "edgecolor": "none"},
                line_kws={"color": "#e03131", "linewidth": 2.5},
            )
        else:
            sns.scatterplot(
                x=d["x"],
                y=d["y"],
                ax=ax,
                color="#2b5c8f",
                alpha=0.65,
                s=40,
            )

        # Zero reference lines
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        # Titles and Labels
        equation = f"Y = {d['slope']:.2f}X + {d['intercept']:.2f}"
        ax.set_title(
            f"Bivariate Normal Distribution (N = {len(d['x'])})\nModel Equation: {equation}",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("Variable X (Standardized Z-Score)", fontsize=11)
        ax.set_ylabel("Variable Y (Standardized Z-Score)", fontsize=11)
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)

        return fig


# Run App
app = App(app_ui, server)
