from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
from app_header import render_page_header
from gw2ml.data.loaders import list_items, load_gw2_series_batch
from gw2ml.evaluation.adf import perform_adf_test
from gw2ml.utils import get_logger

logger = get_logger("item_analysis_app")

CACHE_TIMEOUT = 60 * 60 * 24 * 7  # 7 days in seconds

@st.cache_data(ttl=CACHE_TIMEOUT, persist="disk", show_spinner=True,show_time=True)
def cached_list_items(limit: int = 30000):
    return list_items(limit=limit)

@st.cache_data(ttl=CACHE_TIMEOUT, persist="disk", show_spinner=True,show_time=True)
def available_history_columns(limit: int = 30):
    #TODO:  should ideally be taken from db
    # Another note: does this matter at the moment because we need to retrieve everything anyways?
    # Currently only mattters for what to show. 
    return ["buy_quantity", "sell_quantity","buy_unit_price", "sell_unit_price"]

@st.cache_data(ttl=CACHE_TIMEOUT, show_spinner=False)
def cached_perform_adf_test(series: pd.Series) -> dict:
    """Cache the ADF test results to prevent re-calculation on UI changes."""
    return perform_adf_test(series)

@st.cache_data(ttl=CACHE_TIMEOUT, persist="disk", show_spinner=True,show_time=True)
def cached_load_gw2_series_batch(item_ids, days_back, value_column):
    return load_gw2_series_batch(
        item_ids=list(item_ids),  # streamlit cache needs hashable types
        days_back=days_back,
        value_column=value_column
    )

def render_item_analysis_tab() -> None:
    """Render the item analysis tab content."""
    logger.info("Rendering Item Analysis tab")
    render_page_header("GW2ML: Item Analysis")
    with st.sidebar:
        st.header("Item Analysis Configuration")
        history_days = st.slider(
            "History Range (days)",
            min_value=1,
            max_value=365,
            value=30,
            help="History Data Load N Days Back"
        )
        resample_rule = st.selectbox(
            "Resample Interval",
            options=["5min", "15min", "1h", "4h", "12h", "1d"],
            index=2,
            help="Reducing frequency improves performance, OInly used for Hisotry Plot. ADF etc will still use all data within histroy range."
        )
        show_history = st.checkbox("Show history graph", value=False)
        show_distribution = st.checkbox("Show distribution (pairplot)", value=False)
        show_adf = st.checkbox("Augmented Dickey Fuller", value=False)
        st.info("Configure item analysis parameters here")

        if st.button("Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache cleared!")

    st.subheader("Item Analysis")

    # Use a cached version to avoid database hammering
    items = cached_list_items()

    # Find the default item object for the multi-select

    default_ids = {}
    default_items = [
        item for item in items
        if str(item.get("item_id")) in default_ids
    ]

    selected_items = st.multiselect(
        "Search for items",
        options=items,
        default=default_items,
        format_func=lambda x: f"[{x['item_id']}] {x['item_name']}",
        help="Type to search and select one or more items"
    )

    if selected_items:
        # Store a list of IDs in session state
        st.session_state['selected_item_ids'] = [str(item["item_id"]) for item in selected_items]

    # Placeholder content - you can customize this
    if selected_items:
        if show_history:
            render_history_graph(selected_items, history_days, resample_rule)

        if show_adf:
            render_adf_analysis(selected_items, history_days)

        if show_distribution:
            render_distribution_analysis(selected_items, history_days)





def render_history_graph(selected_items: list[dict], days_back: int, resample_rule: str = "1H") -> None:
    """Render a history graph for the selected items."""
    st.write("---")
    st.subheader("Historical Data")

    cols = available_history_columns()
    selected_col = st.selectbox("Select metric to plot", options=cols, index=2)

    item_ids = tuple(item["item_id"] for item in selected_items)

    with st.spinner("Loading historical data..."):
        batch_data = cached_load_gw2_series_batch(
            item_ids=item_ids,
            days_back=days_back,
            value_column=selected_col
        )

    if not batch_data:
        st.warning("No historical data found for the selected items.")
        return

    # Prepare data for plotting
    plot_data = {}
    for item_id, gw2_series in batch_data.items():
        # Convert Darts TimeSeries to pandas Series for easier plotting in Streamlit
        # and use the item name for the column header
        df = gw2_series.series.to_dataframe()
        
        # Resample to prevent browser lag (mean value for the period)
        if resample_rule:
            df = df.resample(resample_rule).mean()
            
        plot_data[gw2_series.item_name or str(item_id)] = df.iloc[:, 0]

    if plot_data:
        plot_df = pd.DataFrame(plot_data)
        st.line_chart(plot_df)
    else:
        st.info("No data available to plot.")


def render_distribution_analysis(selected_items: list[dict], days_back: int) -> None:
    """Render a pairplot for each selected item to show relationships between metrics."""
    st.write("---")
    st.subheader("Metric Correlation Analysis (Seaborn Pairplot)")

    cols = available_history_columns()

    with st.spinner("Preparing correlation data..."):
        for item in selected_items:
            item_id = item["item_id"]
            item_name = item.get("item_name", str(item_id))

            # Combine all metrics for this specific item into one DataFrame
            item_df = pd.DataFrame()

            for col in cols:
                batch_data = cached_load_gw2_series_batch(
                    item_ids=(item_id,),
                    days_back=days_back,
                    value_column=col
                )

                if item_id in batch_data:
                    series_df = batch_data[item_id].series.to_dataframe()
                    series_df.columns = [col]
                    if item_df.empty:
                        item_df = series_df
                    else:
                        item_df = item_df.join(series_df, how="outer")

            if not item_df.empty:
                # Drop rows with NaN to ensure the pairplot works correctly
                plot_df = item_df.dropna()

                if not plot_df.empty:
                    st.write(f"#### {item_name}")
                    sns.set_theme(
                        style="whitegrid",
                        font_scale=1.1,
                        rc={
                            "text.color": "white",
                            "axes.labelcolor": "white",
                            "xtick.color": "white",
                            "ytick.color": "white",
                            "grid.color": "#444444" # Darker grid so it doesn't clash with white text
                        }
                    )

                # Create the pairplot
                    # diag_kind='kde' adds density plots on the diagonal
                    fig = sns.pairplot(
                        plot_df,
                        vars=cols,
                        diag_kind='kde',
                        height=2.5,
                        aspect=1,
                        palette="viridis",
                        plot_kws={'alpha': 0.9, 's': 30, 'edgecolor': 'white', 'linewidth': 0.5, 'color': '#21918c'},
                        diag_kws={'fill': True, 'alpha': 0.9, 'color': '#21918c'}
                    )
                    #fig.tight_layout()


                # Make backgrounds transparent
                    fig.figure.patch.set_alpha(0.0)
                    for ax in fig.axes.flatten():
                        if ax is not None:
                            ax.set_facecolor((1, 1, 1, 0.1))
                            # Clean up labels to be more readable
                            ax.set_xlabel(ax.get_xlabel().replace('_', ' ').title())
                            ax.set_ylabel(ax.get_ylabel().replace('_', ' ').title())

                    fig.fig.suptitle(f"Correlation Matrix: {item_name}", y=1.02, fontsize=12)

                    plot_col, _ = st.columns([1, 2])
                    with plot_col:
                        st.pyplot(fig.fig, use_container_width=True)
                else:
                    st.info(f"Not enough overlapping data points for {item_name} to create a matrix.")

            else:
                st.info(f"No data available for {item_name}")


def render_adf_analysis(selected_items: list[dict], days_back: int) -> None:
    """Render Augmented Dickey-Fuller test results for selected items."""
    st.write("---")
    st.subheader("Stationarity Analysis (Augmented Dickey-Fuller Test)")

    cols = available_history_columns()
    selected_col = st.selectbox("Select metric for ADF test", options=cols, index=2, key="adf_metric_select")

    item_ids = tuple(item["item_id"] for item in selected_items)

    with st.spinner("Performing ADF tests..."):
        batch_data = cached_load_gw2_series_batch(
            item_ids=item_ids,
            days_back=days_back,
            value_column=selected_col
        )

    if not batch_data:
        st.warning("No historical data found for the selected items.")
        return

        # Create an empty placeholder for the table
    results_table = st.empty()
    all_results = []

    for item_id, gw2_series in batch_data.items():
        item_name = gw2_series.item_name or str(item_id)
        # Convert Darts TimeSeries to pandas Series
        series = gw2_series.series.to_series()

        # Use the cached version of the ADF test
        adf_res = cached_perform_adf_test(series)

        if "error" in adf_res:
            row = {
                "Item": item_name,
                "Status": "Error",
                "Details": adf_res["error"],
                "Stationary": None, "P-Value": None, "ADF Statistic": None, "Lags Used": None, "Observations": None
            }
        else:
            row = {
                "Item": item_name,
                "Stationary": "✅ Yes" if adf_res["is_stationary"] else "❌ No",
                "P-Value": f"{adf_res['p_value']:.4f}",
                "ADF Statistic": f"{adf_res['adf_statistic']:.4f}",
                "Lags Used": adf_res["used_lag"],
                "Observations": adf_res["n_obs"]
            }

        all_results.append(row)
        # Update the table in real-time
        results_table.table(pd.DataFrame(all_results))


    with st.expander("About Augmented Dickey-Fuller Test"):
            st.write("""
            The Augmented Dickey-Fuller (ADF) test is a common statistical test used to determine whether a given time series is stationary or not. 
            
            - **Stationary series**: Has constant mean, variance, and autocorrelation over time. Most time series models (like ARIMA) require the data to be stationary.
            - **Null Hypothesis (H0)**: The series has a unit root (is non-stationary).
            - **Alternative Hypothesis (H1)**: The series is stationary.
            - **Interpretation**: If the P-Value is less than 0.05, we reject the null hypothesis and conclude the series is likely stationary.
            """)


def main() -> None:
    """Main entry point when running item_analysis_app.py standalone."""
    st.set_page_config(page_title="GW2ML Item Analysis", layout="wide")
    render_item_analysis_tab()


if __name__ == "__main__":
    main()
