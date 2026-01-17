from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib as mpl
import seaborn as sns
from app_header import render_page_header
from gw2ml.data import DatabaseClient, get_prices
from gw2ml.data.loaders import list_items, load_gw2_series_batch
from gw2ml.evaluation.plot import implot
from gw2ml.evaluation.transform import reindex
from gw2ml.evaluation.adf import perform_adf_test
from gw2ml.utils import get_logger

logger = get_logger("item_analysis_app")

CACHE_TIMEOUT = 60 * 60 * 24 * 7  # 7 days in seconds
VALUE_COLUMNS = [
    "buy_unit_price",
    "sell_unit_price",
    "buy_quantity",
    "sell_quantity",
]
RESAMPLE_OPTIONS = ["5min", "15min", "30min", "1h", "4h", "12h", "1d"]

@st.cache_data(ttl=CACHE_TIMEOUT, persist="disk", show_spinner=True,show_time=True)
def cached_list_items(limit: int = 30000):
    return list_items(limit=limit)

@st.cache_data(ttl=CACHE_TIMEOUT, persist="disk", show_spinner=True,show_time=True)
def available_history_columns(limit: int = 30):
    #TODO:  should ideally be taken from db
    # Another note: does this matter at the moment because we need to retrieve everything anyways?
    # Currently only mattters for what to show. 
    return VALUE_COLUMNS

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


@st.cache_data(ttl=CACHE_TIMEOUT, persist="disk", show_spinner=True, show_time=True)
def cached_load_prices_raw(item_id: int, days_back: int) -> pd.DataFrame:
    client = DatabaseClient.from_env()
    return get_prices(client, item_id=int(item_id), last_days=int(days_back), order="ASC")

def render_item_analysis_tab(
    selected_items: list[dict] | None = None,
    *,
    show_header: bool = True,
) -> None:
    """Render the item analysis tab content."""
    logger.info("Rendering Item Analysis tab")
    if show_header:
        render_page_header("GW2ML: Item Analysis")
    with st.sidebar:
        st.header("Item Analysis Configuration")
        history_days = st.number_input(
            "History range (days back)",
            min_value=1,
            max_value=365,
            value=30,
            step=1,
            help="Use the last X days for analysis.",
        )
        resample_labels = st.multiselect(
            "Resample intervals",
            options=RESAMPLE_OPTIONS,
            default=["5min"],
            help="Reducing frequency improves performance; only used for history plots.",
        )
        selected_metrics = st.multiselect(
            "Metrics",
            options=available_history_columns(),
            default=available_history_columns(),
            help="Select metrics used across analysis views.",
        )
        show_history = st.checkbox("Show history graph", value=False)
        show_distribution = st.checkbox("Show distribution (pairplot)", value=False)
        show_adf = st.checkbox("Augmented Dickey Fuller", value=False)
        show_price_boxplots = st.checkbox("Show metric boxplots (mean/median, fluctuations)", value=False)
        show_week_heatmap = st.checkbox("Show calendar-week heatmap", value=False)
        show_profitability = st.checkbox("Show profitability indicator", value=False)
        show_basic_stats = st.checkbox("Show basic statistics", value=True)

        if st.button("Clear Cache", width="stretch"):
            st.cache_data.clear()
            st.success("Cache cleared!")

    if show_header:
        st.subheader("Item Analysis")

    if selected_items is None:
        # Use a cached version to avoid database hammering
        items = cached_list_items()

        default_items = []
        stored_ids = st.session_state.get("selected_item_ids", [])
        if stored_ids and "global_item_select" not in st.session_state:
            default_items = [item for item in items if str(item.get("item_id")) in stored_ids]

        selected_items = st.multiselect(
            "Search for items",
            options=items,
            default=default_items,
            format_func=lambda x: f"[{x['item_id']}] {x['item_name']}",
            help="Type to search and select one or more items"
        )
        st.session_state["selected_item_ids"] = [str(item["item_id"]) for item in selected_items]
    elif not selected_items:
        st.info("Select one or more items above to run analysis.")
        return

    if selected_items:
        # Store a list of IDs in session state
        st.session_state['selected_item_ids'] = [str(item["item_id"]) for item in selected_items]

    # Placeholder content - you can customize this
    if st.button("Run analysis"):
        if not selected_items:
            st.error("Please select at least one item to analyze.")
            return
        if not selected_metrics:
            st.error("Please select at least one metric.")
            return

        for item in selected_items:
            item_id = item["item_id"]
            item_label = item.get("item_name", str(item_id))
            with st.expander(f"[{item_id}] {item_label}", expanded=False):
                if show_basic_stats:
                    render_basic_statistics([item], history_days, selected_metrics)

                if show_history:
                    render_history_graph([item], history_days, resample_labels, selected_metrics)

                if show_adf:
                    render_adf_analysis([item], history_days, selected_metrics)

                if show_distribution:
                    render_distribution_analysis([item], history_days, selected_metrics)

                if show_price_boxplots:
                    render_price_boxplots([item], history_days, selected_metrics)

                if show_week_heatmap:
                    render_calendar_week_heatmap([item], history_days, selected_metrics)

                if show_profitability:
                    render_profitability_indicator([item], history_days, resample_labels)





def render_history_graph(
    selected_items: list[dict],
    days_back: int,
    resample_labels: list[str],
    metrics: list[str],
) -> None:
    """Render a history graph for the selected items."""
    st.write("---")
    st.subheader("Historical Data")

    if not resample_labels:
        st.info("Select at least one resample interval to show history.")
        return
    if not metrics:
        st.info("Select at least one metric to show history.")
        return

    item_ids = tuple(item["item_id"] for item in selected_items)

    with st.spinner("Loading historical data..."):
        metric_batches = {
            metric: cached_load_gw2_series_batch(
                item_ids=item_ids,
                days_back=days_back,
                value_column=metric,
            )
            for metric in metrics
        }

    if not metric_batches:
        st.warning("No historical data found for the selected items.")
        return

    metric_tabs = st.tabs(metrics)
    for metric, metric_tab in zip(metrics, metric_tabs):
        with metric_tab:
            batch_data = metric_batches.get(metric, {})
            if not batch_data:
                st.info(f"No historical data found for {metric}.")
                continue
            resample_tabs = st.tabs([label for label in resample_labels])
            for resample_label, tab in zip(resample_labels, resample_tabs):
                with tab:
                    plot_data = {}
                    for item_id, gw2_series in batch_data.items():
                        # Convert Darts TimeSeries to pandas Series for easier plotting in Streamlit
                        # and use the item name for the column header
                        df = gw2_series.series.to_dataframe()

                        # Resample to prevent browser lag (mean value for the period)
                        if resample_label:
                            df = df.resample(resample_label).mean()

                        plot_data[gw2_series.item_name or str(item_id)] = df.iloc[:, 0]

                    if plot_data:
                        plot_df = pd.DataFrame(plot_data)
                        st.line_chart(plot_df)
                    else:
                        st.info("No data available to plot.")


def render_distribution_analysis(selected_items: list[dict], days_back: int, metrics: list[str]) -> None:
    """Render a pairplot for each selected item to show relationships between metrics."""
    st.write("---")
    st.subheader("Metric Correlation Analysis (Seaborn Pairplot)")

    if len(metrics) < 2:
        st.info("Select at least two metrics to render a pairplot.")
        return

    item_ids = tuple(item["item_id"] for item in selected_items)

    with st.spinner("Preparing correlation data..."):
        metric_batches = {
            metric: cached_load_gw2_series_batch(
                item_ids=item_ids,
                days_back=days_back,
                value_column=metric
            )
            for metric in metrics
        }

    show_item_header = len(selected_items) > 1
    for item in selected_items:
        item_id = item["item_id"]
        item_name = item.get("item_name", str(item_id))

        # Combine all metrics for this specific item into one DataFrame
        item_df = pd.DataFrame()
        for col in metrics:
            batch_data = metric_batches.get(col, {})
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
                if show_item_header:
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
                    vars=metrics,
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

                st.pyplot(fig.fig, width="stretch")
            else:
                st.info(f"Not enough overlapping data points for {item_name} to create a matrix.")

        else:
            st.info(f"No data available for {item_name}")


def render_basic_statistics(selected_items: list[dict], days_back: int, metrics: list[str]) -> None:
    """Render basic statistics for selected metrics."""
    st.write("---")
    st.subheader("Basic Statistics")

    if not metrics:
        st.info("Select at least one metric to compute statistics.")
        return

    item_ids = tuple(item["item_id"] for item in selected_items)
    with st.spinner("Loading data for statistics..."):
        metric_batches = {
            metric: cached_load_gw2_series_batch(
                item_ids=item_ids,
                days_back=days_back,
                value_column=metric,
            )
            for metric in metrics
        }

    show_item_header = len(selected_items) > 1
    for item in selected_items:
        item_id = item["item_id"]
        item_name = item.get("item_name", str(item_id))
        if show_item_header:
            st.write(f"#### {item_name}")

        rows = []
        for metric in metrics:
            batch_data = metric_batches.get(metric, {})
            if item_id not in batch_data:
                continue
            series_df = batch_data[item_id].series.to_dataframe()
            values = series_df.iloc[:, 0].dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "metric": metric,
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "count": int(values.count()),
                }
            )

        if rows:
            stats_df = pd.DataFrame(rows)
            st.dataframe(stats_df, width="stretch", hide_index=True)
        else:
            st.info("No data available to compute statistics.")


def render_price_boxplots(selected_items: list[dict], days_back: int, metrics: list[str]) -> None:
    """Render boxplots of metric distributions and fluctuations."""
    st.write("---")
    st.subheader("Metric Boxplots by Weekday (Mean/Median + Fluctuations)")

    if not metrics:
        st.info("Select at least one metric for the boxplots.")
        return

    item_ids = tuple(item["item_id"] for item in selected_items)
    with st.spinner("Loading metric data..."):
        metric_batches = {
            metric: cached_load_gw2_series_batch(
                item_ids=item_ids,
                days_back=days_back,
                value_column=metric,
            )
            for metric in metrics
        }

    if not metric_batches:
        st.warning("No price data available for the selected items.")
        return

    weekday_order = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

    show_item_header = len(selected_items) > 1
    for item in selected_items:
        item_id = item["item_id"]
        item_name = item.get("item_name", str(item_id))

        if show_item_header:
            st.write(f"#### {item_name}")
        for metric in metrics:
            batch_data = metric_batches.get(metric, {})
            if item_id not in batch_data:
                st.info(f"No {metric} data available for {item_name}.")
                continue

            series_df = batch_data[item_id].series.to_dataframe()
            series_df.columns = [metric]
            series_df = series_df.sort_index()
            series_df["fetched_at_5m"] = pd.to_datetime(series_df.index).floor("5min")
            series_df = series_df.groupby("fetched_at_5m").last()
            series_df["weekday"] = series_df.index.day_name()

            metric_df = series_df[[metric, "weekday"]].dropna()
            if metric_df.empty:
                st.info(f"No {metric} data available for {item_name}.")
                continue

            fig_price = px.box(
                metric_df,
                x="weekday",
                y=metric,
                category_orders={"weekday": weekday_order},
                points="outliers",
            )
            fig_price.update_traces(boxmean=True)
            fig_price.update_layout(
                title=f"{metric} by weekday",
                xaxis_title="weekday",
                yaxis_title=metric,
                showlegend=False,
            )
            st.plotly_chart(fig_price, width="stretch")

            changes = metric_df.copy()
            changes["pct_change"] = changes[metric].pct_change(fill_method=None)
            changes = changes.replace([np.inf, -np.inf], np.nan).dropna(subset=["pct_change"])
            if changes.empty:
                st.info(f"Not enough data to compute {metric} fluctuations.")
                continue

            fig_change = px.box(
                changes,
                x="weekday",
                y="pct_change",
                category_orders={"weekday": weekday_order},
                points="outliers",
            )
            fig_change.update_traces(boxmean=True)
            fig_change.update_layout(
                title=f"{metric} fluctuations by weekday",
                xaxis_title="weekday",
                yaxis_title="pct change",
                showlegend=False,
            )
            fig_change.update_yaxes(tickformat=".1%")
            st.plotly_chart(fig_change, width="stretch")


def render_calendar_week_heatmap(selected_items: list[dict], days_back: int, metrics: list[str]) -> None:
    """Render a calendar-week heatmap for a selected item/metric."""
    st.write("---")
    st.subheader("Calendar-Week Heatmap")

    item_labels = {
        f"[{item['item_id']}] {item.get('item_name', item['item_id'])}": item["item_id"]
        for item in selected_items
    }
    if not item_labels:
        st.info("Select at least one item to render a heatmap.")
        return
    if not metrics:
        st.info("Select at least one metric to render a heatmap.")
        return

    if len(item_labels) == 1:
        selected_label = next(iter(item_labels))
    else:
        selected_label = st.selectbox(
            "Select item for heatmap",
            options=list(item_labels.keys()),
            key="week_heatmap_item",
        )

    item_id = item_labels[selected_label]
    raw_df = cached_load_prices_raw(item_id, days_back)
    if raw_df.empty:
        st.warning("No data available for the selected item.")
        return

    metric_tabs = st.tabs(metrics)
    for metric, metric_tab in zip(metrics, metric_tabs):
        with metric_tab:
            with st.spinner("Loading heatmap data..."):
                batch_data = cached_load_gw2_series_batch(
                    item_ids=(item_id,),
                    days_back=days_back,
                    value_column=metric,
                )

            if item_id not in batch_data:
                st.warning("No data available for the selected item.")
                continue

            series_df = batch_data[item_id].series.to_dataframe()
            series_df.columns = [metric]
            series_df = series_df.dropna()
            if series_df.empty:
                st.info("No data available to build a heatmap.")
                continue

            df = series_df.copy()
            iso = df.index.isocalendar()
            df["iso_year"] = iso.year
            df["iso_week"] = iso.week
            df["weekday"] = df.index.day_name()
            df["week_label"] = df["iso_year"].astype(str) + "-W" + df["iso_week"].astype(str).str.zfill(2)
            df["week_key"] = df["iso_year"] * 100 + df["iso_week"]

            pivot = df.pivot_table(
                index="week_label",
                columns="weekday",
                values=metric,
                aggfunc="mean",
            )
            if pivot.empty:
                st.info("No data available to build a heatmap.")
                continue

            weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            week_order = (
                df[["week_label", "week_key"]]
                .drop_duplicates()
                .sort_values("week_key")["week_label"]
                .tolist()
            )
            pivot = pivot.reindex(index=week_order, columns=weekday_order)

            fig = px.imshow(
                pivot,
                aspect="auto",
                color_continuous_scale="Viridis",
                labels={"x": "weekday", "y": "calendar week", "color": metric},
            )
            fig.update_layout(xaxis_title="", yaxis_title="")
            st.plotly_chart(fig, width="stretch")

            st.markdown("##### Notebook-style weekly heatmap (raw data)")
            if metric not in raw_df.columns:
                st.info(f"No raw {metric} data available for the selected item.")
                continue

            raw_metric = raw_df[["fetched_at", metric]].dropna()
            if raw_metric.empty:
                st.info("Not enough raw data to build the notebook-style heatmap.")
                continue

            series = raw_metric.set_index("fetched_at")[metric]
            series = series.sort_index()
            series.index = pd.to_datetime(series.index, utc=True)

            series_5m = series.groupby(series.index.floor("5min")).last()
            if len(series_5m.index) < 2:
                st.info("Not enough raw data to build the notebook-style heatmap.")
                continue

            regular_index = pd.date_range(
                series_5m.index.min(),
                series_5m.index.max(),
                freq="5min",
                tz=series_5m.index.tz,
            )
            series_regular = series_5m.reindex(regular_index)
            if series_regular.dropna().empty:
                st.info("Not enough raw data to build the notebook-style heatmap.")
                continue

            with mpl.rc_context(
                {
                    "text.color": "black",
                    "axes.labelcolor": "black",
                    "xtick.color": "black",
                    "ytick.color": "black",
                }
            ):
                fig = implot(
                    series_regular,
                    title=f"{metric} (5-min buckets, UTC)",
                    scale=False,
                    cmap="viridis",
                    label=metric,
                )
            try:
                _, steps_per_week, complete_ts = reindex(series_regular)
            except Exception:
                steps_per_week = 0
            if steps_per_week:
                total_weeks = len(complete_ts) // steps_per_week
                if total_weeks > 0:
                    if total_weeks > 52:
                        stride = 4
                    elif total_weeks > 26:
                        stride = 2
                    else:
                        stride = 1
                    y_positions = list(range(0, total_weeks, stride))
                    y_labels = []
                    for pos in y_positions:
                        ts = complete_ts[pos * steps_per_week]
                        iso = ts.isocalendar()
                        y_labels.append(f"{iso.year}-{iso.week:02d}")
                    fig.axes[0].set_yticks(y_positions, labels=y_labels)
            for ax in fig.axes:
                ax.set_facecolor("white")
                ax.tick_params(colors="black")
                ax.title.set_color("black")
                ax.xaxis.label.set_color("black")
                ax.yaxis.label.set_color("black")
            fig.patch.set_facecolor("white")
            fig.tight_layout()
            st.pyplot(fig, width="stretch")


def render_adf_analysis(selected_items: list[dict], days_back: int, metrics: list[str]) -> None:
    """Render Augmented Dickey-Fuller test results for selected items."""
    st.write("---")
    st.subheader("Stationarity Analysis (Augmented Dickey-Fuller Test)")

    item_ids = tuple(item["item_id"] for item in selected_items)

    if not metrics:
        st.info("Select at least one metric for the ADF test.")
        return

    for metric in metrics:
        with st.spinner(f"Performing ADF tests for {metric}..."):
            batch_data = cached_load_gw2_series_batch(
                item_ids=item_ids,
                days_back=days_back,
                value_column=metric,
            )

        if not batch_data:
            st.warning(f"No historical data found for metric: {metric}.")
            continue

        st.markdown(f"#### {metric}")
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
                    "Stationary": None, "P-Value": None, "ADF Statistic": None, "Lags Used": None,
                    "Observations": None,
                }
            else:
                row = {
                    "Item": item_name,
                    "Stationary": "✅ Yes" if adf_res["is_stationary"] else "❌ No",
                    "P-Value": f"{adf_res['p_value']:.4f}",
                    "ADF Statistic": f"{adf_res['adf_statistic']:.4f}",
                    "Lags Used": adf_res["used_lag"],
                    "Observations": adf_res["n_obs"],
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


def _build_profitability_summary(combined: pd.DataFrame) -> go.Figure | None:
    if combined.empty:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=combined.index,
            y=combined["buy"],
            mode="lines",
            name="current buy",
            line=dict(color="#ff7f0e"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=combined.index,
            y=combined["sell"],
            mode="lines",
            name="current sell",
            line=dict(color="#2ca02c"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=combined.index,
            y=combined["max_buy"],
            mode="lines",
            name="max buy for profit (sell*0.85)",
            line=dict(color="#2ca02c", dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=combined.index,
            y=combined["breakeven_sell"],
            mode="lines",
            name="required sell for current buy (buy/0.85)",
            line=dict(color="#ff7f0e", dash="dot"),
        )
    )
    fig.update_layout(
        title="Buy/Sell with required thresholds",
        xaxis_title="timestamp",
        yaxis_title="price",
        legend_title="series",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(b=120),
    )
    return fig


def _build_profitability_detail_plots(combined: pd.DataFrame) -> tuple[go.Figure, go.Figure] | None:
    if combined.empty:
        return None

    y_series = pd.concat(
        [
            combined["buy"],
            combined["sell"],
            combined["breakeven_sell"],
            combined["max_buy"],
        ],
        axis=0,
    ).dropna()
    y_range = None
    if not y_series.empty:
        y_min = float(y_series.min())
        y_max = float(y_series.max())
        pad = (y_max - y_min) * 0.05 if y_max != y_min else max(1.0, y_max * 0.01)
        y_range = [y_min - pad, y_max + pad]

    sell_profit = combined["sell"].where(combined["sell"] >= combined["breakeven_sell"])
    sell_loss = combined["sell"].where(combined["sell"] < combined["breakeven_sell"])

    buy_profit = combined["buy"].where(combined["buy"] <= combined["max_buy"])
    buy_loss = combined["buy"].where(combined["buy"] > combined["max_buy"])

    fig_sell = go.Figure()
    fig_sell.add_trace(
        go.Scatter(
            x=combined.index,
            y=combined["breakeven_sell"],
            mode="lines",
            name="required sell for current buy (buy/0.85)",
            line=dict(color="#ff7f0e", dash="dot"),
        )
    )
    fig_sell.add_trace(
        go.Scatter(
            x=combined.index,
            y=sell_profit,
            mode="lines",
            name="profit zone",
            line=dict(color="rgba(0,0,0,0)"),
            fill="tonexty",
            fillcolor="rgba(44, 160, 44, 0.2)",
        )
    )
    fig_sell.add_trace(
        go.Scatter(
            x=combined.index,
            y=combined["breakeven_sell"],
            mode="lines",
            name="required sell for current buy (buy/0.85)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
        )
    )
    fig_sell.add_trace(
        go.Scatter(
            x=combined.index,
            y=sell_loss,
            mode="lines",
            name="loss zone",
            line=dict(color="rgba(0,0,0,0)"),
            fill="tonexty",
            fillcolor="rgba(214, 39, 40, 0.2)",
        )
    )
    fig_sell.add_trace(
        go.Scatter(
            x=combined.index,
            y=combined["sell"],
            mode="lines",
            name="current sell",
            line=dict(color="#2ca02c"),
        )
    )
    fig_sell.update_layout(
        title="Sell view: sell vs. required sell (buy/0.85)",
        xaxis_title="timestamp",
        yaxis_title="price",
        legend_title="series",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(b=120),
    )
    if y_range:
        fig_sell.update_yaxes(range=y_range)

    fig_buy = go.Figure()
    fig_buy.add_trace(
        go.Scatter(
            x=combined.index,
            y=combined["max_buy"],
            mode="lines",
            name="max buy for profit (sell*0.85)",
            line=dict(color="#2ca02c", dash="dot"),
        )
    )
    fig_buy.add_trace(
        go.Scatter(
            x=combined.index,
            y=buy_profit,
            mode="lines",
            name="profit zone",
            line=dict(color="rgba(0,0,0,0)"),
            fill="tonexty",
            fillcolor="rgba(44, 160, 44, 0.2)",
        )
    )
    fig_buy.add_trace(
        go.Scatter(
            x=combined.index,
            y=combined["max_buy"],
            mode="lines",
            name="max buy for profit (sell*0.85)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
        )
    )
    fig_buy.add_trace(
        go.Scatter(
            x=combined.index,
            y=buy_loss,
            mode="lines",
            name="loss zone",
            line=dict(color="rgba(0,0,0,0)"),
            fill="tonexty",
            fillcolor="rgba(214, 39, 40, 0.2)",
        )
    )
    fig_buy.add_trace(
        go.Scatter(
            x=combined.index,
            y=combined["buy"],
            mode="lines",
            name="current buy",
            line=dict(color="#ff7f0e"),
        )
    )
    fig_buy.update_layout(
        title="Buy view: buy vs. max buy (sell*0.85)",
        xaxis_title="timestamp",
        yaxis_title="price",
        legend_title="series",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(b=120),
    )
    if y_range:
        fig_buy.update_yaxes(range=y_range)
    return fig_sell, fig_buy


def render_profitability_indicator(
    selected_items: list[dict],
    days_back: int,
    resample_labels: list[str],
) -> None:
    st.write("---")
    st.subheader("Profitability indicator")

    if not resample_labels:
        st.info("Select at least one resample interval to render profitability.")
        return

    item_ids = tuple(item["item_id"] for item in selected_items)
    with st.spinner("Loading buy/sell price data..."):
        buy_batch = cached_load_gw2_series_batch(
            item_ids=item_ids,
            days_back=days_back,
            value_column="buy_unit_price",
        )
        sell_batch = cached_load_gw2_series_batch(
            item_ids=item_ids,
            days_back=days_back,
            value_column="sell_unit_price",
        )

    show_item_header = len(selected_items) > 1
    for item in selected_items:
        item_id = item["item_id"]
        item_name = item.get("item_name", str(item_id))
        if item_id not in buy_batch or item_id not in sell_batch:
            st.info(f"Missing buy/sell data for {item_name}.")
            continue

        if show_item_header:
            st.write(f"#### {item_name}")
        resample_tabs = st.tabs([label for label in resample_labels])
        for resample_label, tab in zip(resample_labels, resample_tabs):
            with tab:
                buy_df = buy_batch[item_id].series.to_dataframe()
                sell_df = sell_batch[item_id].series.to_dataframe()
                buy_df.columns = ["buy"]
                sell_df.columns = ["sell"]
                if resample_label:
                    buy_df = buy_df.resample(resample_label).mean()
                    sell_df = sell_df.resample(resample_label).mean()
                combined = buy_df.merge(sell_df, left_index=True, right_index=True, how="outer").sort_index()
                combined["breakeven_sell"] = combined["buy"] / 0.85
                combined["max_buy"] = combined["sell"] * 0.85
                combined = combined.dropna(how="all")
                if combined.empty:
                    st.info("Not enough data to render profitability charts.")
                    continue

                fig_summary = _build_profitability_summary(combined)
                if fig_summary is not None:
                    st.plotly_chart(fig_summary, width="stretch")
                detail_figs = _build_profitability_detail_plots(combined)
                if detail_figs is not None:
                    fig_sell, fig_buy = detail_figs
                    st.plotly_chart(fig_sell, width="stretch")
                    st.plotly_chart(fig_buy, width="stretch")


def main() -> None:
    """Main entry point when running item_analysis_app.py standalone."""
    st.set_page_config(page_title="GW2ML Item Analysis", layout="wide")
    render_item_analysis_tab()


if __name__ == "__main__":
    main()
