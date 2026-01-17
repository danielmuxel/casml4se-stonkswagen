from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity

from app_header import render_page_header
from gw2ml.data import DatabaseClient, get_prices
from gw2ml.data.database_queries import get_items
from gw2ml.data.loaders import list_items
from gw2ml.utils import get_logger

logger = get_logger("item_finder_app")

CACHE_TIMEOUT = 60 * 60 * 24 * 7  # 7 days
RARITY_ORDER = ["Junk", "Basic", "Fine", "Masterwork", "Rare", "Exotic", "Ascended", "Legendary"]
ITEM_TYPES = [
    "Armor",
    "Back",
    "Bag",
    "Consumable",
    "Container",
    "CraftingMaterial",
    "Gathering",
    "Gizmo",
    "Key",
    "MiniPet",
    "Tool",
    "Trait",
    "Trinket",
    "Trophy",
    "UpgradeComponent",
    "Weapon",
]
RESAMPLE_OPTIONS = ["Raw (no resample)", "5min", "15min", "30min", "1h", "4h", "12h", "1d"]
DENDROGRAM_SCALE_OPTIONS = [
    "Raw distance",
    "Distance x100",
    "Min-max (0-1)",
]
SIMILARITY_INPUT_OPTIONS = [
    "Price levels",
    "Price returns (%)",
]


@st.cache_data(ttl=CACHE_TIMEOUT, persist="disk", show_spinner=True)
def cached_list_items(limit: int = 30000):
    return list_items(limit=limit)


@st.cache_data(ttl=CACHE_TIMEOUT, persist="disk", show_spinner=True)
def cached_get_items_full(limit: int = 30000) -> pd.DataFrame:
    """Get full item details from database."""
    client = DatabaseClient.from_env()
    return get_items(client, limit=limit)


@st.cache_data(ttl=CACHE_TIMEOUT, persist="disk", show_spinner=False)
def cached_get_prices(item_id: int, days_back: int) -> pd.DataFrame:
    """Get price history for an item."""
    client = DatabaseClient.from_env()
    return get_prices(client, item_id=int(item_id), last_days=int(days_back), order="ASC")


def _load_price_data_parallel(
    item_ids: list[int],
    days_back: int,
    resample_freq: str = "1h",
) -> dict[int, pd.Series]:
    """Load price data for multiple items in parallel."""

    def load_single(item_id: int) -> tuple[int, pd.Series | None]:
        try:
            df = cached_get_prices(item_id, days_back)
            if df.empty or "sell_unit_price" not in df.columns:
                return item_id, None
            df = df.copy()
            df["fetched_at"] = pd.to_datetime(df["fetched_at"], format="ISO8601")
            df = df.set_index("fetched_at")
            price_series = df["sell_unit_price"].sort_index()
            if resample_freq == "Raw (no resample)":
                return item_id, price_series
            hourly_avg = price_series.resample(resample_freq).mean()
            return item_id, hourly_avg
        except Exception as e:
            logger.warning(f"Failed to load prices for item {item_id}: {e}")
            return item_id, None

    results = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        for item_id, series in executor.map(load_single, item_ids):
            if series is not None:
                results[item_id] = series

    return results


def _compute_similarity_matrix(
    hourly_df: pd.DataFrame,
    *,
    input_mode: str = "Price levels",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cosine similarity and distance matrices."""
    filled = hourly_df.sort_index().ffill().bfill()
    if input_mode == "Price returns (%)":
        series_df = filled.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    else:
        series_df = filled
    price_matrix = series_df.T.values
    similarity_matrix = cosine_similarity(price_matrix)
    distance_matrix = 1 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0)
    return similarity_matrix, distance_matrix


def _plot_dendrogram(
    distance_matrix: np.ndarray,
    item_labels: list[str],
    color_threshold: float = 0.02,
    y_limit: float = 0.15,
    y_axis_title: str = "Distance (1 - Cosine Similarity)",
) -> go.Figure:
    """Create a dendrogram plot using hierarchical clustering."""
    condensed_dist = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_dist, method="ward", optimal_ordering=True)

    # Use scipy dendrogram to get coordinates
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors

    fig_mpl, ax = plt.subplots(figsize=(14, 8))
    dend = dendrogram(
        linkage_matrix,
        color_threshold=color_threshold,
        labels=item_labels,
        ax=ax,
        no_plot=False,
    )
    plt.close(fig_mpl)

    # Convert to plotly
    fig = go.Figure()
    for i, (x_coords, y_coords, color) in enumerate(
        zip(dend["icoord"], dend["dcoord"], dend["color_list"])
    ):
        try:
            line_color = mcolors.to_hex(color)
        except ValueError:
            line_color = "#1f77b4"
        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="lines",
                line=dict(color=line_color, width=1.5),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Add x-axis labels
    fig.update_layout(
        title="Hierarchical Clustering Dendrogram (Cosine Similarity)",
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(5, len(item_labels) * 10, 10)),
            ticktext=dend["ivl"],
            tickangle=90,
            title="Item ID",
        ),
        yaxis=dict(title=y_axis_title, range=[0, y_limit]),
        height=600,
        margin=dict(b=220, l=60, r=20, t=60),
    )
    return fig, linkage_matrix


def _get_clusters(
    linkage_matrix: np.ndarray,
    item_ids: list[int],
    n_clusters: int,
) -> dict[int, list[int]]:
    """Get cluster assignments from linkage matrix."""
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust")
    clusters: dict[int, list[int]] = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(item_ids[i])
    return clusters


def _compute_profitability_metrics(
    item_ids: list[int],
    days_back: int,
) -> pd.DataFrame:
    """Compute profitability metrics for items."""
    metrics = []

    for item_id in item_ids:
        try:
            df = cached_get_prices(item_id, days_back)
            if df.empty:
                continue

            buy_prices = df["buy_unit_price"].dropna()
            sell_prices = df["sell_unit_price"].dropna()

            if buy_prices.empty or sell_prices.empty:
                continue

            # Current prices (most recent)
            current_buy = float(buy_prices.iloc[-1])
            current_sell = float(sell_prices.iloc[-1])

            # Average prices
            avg_buy = float(buy_prices.mean())
            avg_sell = float(sell_prices.mean())

            # Spread and margin calculations (15% TP fee)
            current_spread = current_sell - current_buy
            current_margin = (current_sell * 0.85) - current_buy
            current_margin_pct = (current_margin / current_buy * 100) if current_buy > 0 else 0

            avg_spread = avg_sell - avg_buy
            avg_margin = (avg_sell * 0.85) - avg_buy
            avg_margin_pct = (avg_margin / avg_buy * 100) if avg_buy > 0 else 0

            # Volatility (coefficient of variation)
            buy_volatility = float(buy_prices.std() / buy_prices.mean() * 100) if buy_prices.mean() > 0 else 0
            sell_volatility = float(sell_prices.std() / sell_prices.mean() * 100) if sell_prices.mean() > 0 else 0

            # Volume (quantity)
            avg_buy_qty = float(df["buy_quantity"].mean()) if "buy_quantity" in df.columns else 0
            avg_sell_qty = float(df["sell_quantity"].mean()) if "sell_quantity" in df.columns else 0

            # Absolute price change over the period (last vs first)
            abs_buy_change = float(abs(buy_prices.iloc[-1] - buy_prices.iloc[0])) if len(buy_prices) > 1 else 0
            abs_sell_change = float(abs(sell_prices.iloc[-1] - sell_prices.iloc[0])) if len(sell_prices) > 1 else 0

            metrics.append({
                "item_id": item_id,
                "current_buy": current_buy,
                "current_sell": current_sell,
                "current_spread": current_spread,
                "current_margin": current_margin,
                "current_margin_pct": current_margin_pct,
                "avg_buy": avg_buy,
                "avg_sell": avg_sell,
                "avg_spread": avg_spread,
                "avg_margin": avg_margin,
                "avg_margin_pct": avg_margin_pct,
                "buy_volatility_pct": buy_volatility,
                "sell_volatility_pct": sell_volatility,
                "avg_buy_qty": avg_buy_qty,
                "avg_sell_qty": avg_sell_qty,
                "avg_qty_min": min(avg_buy_qty, avg_sell_qty),
                "abs_buy_change": abs_buy_change,
                "abs_sell_change": abs_sell_change,
            })
        except Exception as e:
            logger.warning(f"Failed to compute metrics for item {item_id}: {e}")

    return pd.DataFrame(metrics)


def render_item_finder_tab(
    selected_items: list[dict] | None = None,
    *,
    show_header: bool = True,
) -> None:
    """Render the item finder tab with filtering and similarity analysis."""
    if show_header:
        render_page_header("GW2ML: Item Finder")

    with st.sidebar:
        st.header("Item Finder Configuration")

        st.subheader("Filters")
        filter_by_type = st.multiselect(
            "Item Type",
            options=ITEM_TYPES,
            default=[],
            help="Filter by item type (e.g., Weapon, Armor).",
        )

        filter_by_rarity = st.multiselect(
            "Rarity",
            options=RARITY_ORDER,
            default=[],
            help="Filter by rarity.",
        )

        level_range = st.slider(
            "Level Range",
            min_value=0,
            max_value=80,
            value=(0, 80),
            help="Filter by item level.",
        )

        min_vendor_value = st.number_input(
            "Min Vendor Value",
            min_value=0,
            value=0,
            help="Minimum vendor value filter.",
        )

        st.subheader("Similarity Analysis")
        days_back = st.number_input(
            "History (days)",
            min_value=1,
            max_value=180,
            value=30,
            help="Days of price history for similarity analysis.",
        )

        resample_freq = st.selectbox(
            "Resample Frequency",
            options=RESAMPLE_OPTIONS,
            index=0,  # default to "Raw (no resample)"
            help="Resample interval for price data.",
        )

        similarity_input = st.selectbox(
            "Similarity Input",
            options=SIMILARITY_INPUT_OPTIONS,
            index=0,  # default to "Price levels"
            help="Use price levels for similarity (matches simple_correlation notebook).",
        )

        n_clusters = st.slider(
            "Number of Clusters",
            min_value=2,
            max_value=20,
            value=5,
            help="Number of clusters for hierarchical grouping.",
        )

        dendrogram_scale = st.selectbox(
            "Dendrogram Scale",
            options=DENDROGRAM_SCALE_OPTIONS,
            index=0,
            help="Scale dendrogram distances for readability (does not change clustering).",
        )

        st.subheader("Profitability")
        min_margin_pct = st.number_input(
            "Min Margin %",
            min_value=-100.0,
            max_value=100.0,
            value=0.0,
            step=0.5,
            help="Filter items with margin percentage above this value.",
        )

        max_items_analysis = st.number_input(
            "Max Items for Analysis",
            min_value=10,
            max_value=500,
            value=100,
            help="Limit items for similarity/profitability analysis.",
        )

    if show_header:
        st.subheader("Item Finder")

    # Load items with full details
    with st.spinner("Loading items..."):
        items_df = cached_get_items_full(limit=30000)

    if items_df.empty:
        st.warning("No items found in database.")
        return

    # Apply filters
    filtered_df = items_df.copy()

    if filter_by_type:
        filtered_df = filtered_df[filtered_df["type"].isin(filter_by_type)]

    if filter_by_rarity:
        filtered_df = filtered_df[filtered_df["rarity"].isin(filter_by_rarity)]

    filtered_df = filtered_df[
        (filtered_df["level"] >= level_range[0]) & (filtered_df["level"] <= level_range[1])
    ]

    if min_vendor_value > 0:
        filtered_df = filtered_df[filtered_df["vendor_value"] >= min_vendor_value]

    # Only tradeable items
    filtered_df = filtered_df[filtered_df["is_tradeable"] == True]

    st.info(f"Found {len(filtered_df)} items matching filters.")

    # Display filter summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Items", len(items_df))
    with col2:
        st.metric("Filtered Items", len(filtered_df))
    with col3:
        st.metric("Tradeable", len(filtered_df[filtered_df["is_tradeable"] == True]))
    with col4:
        if not filtered_df.empty:
            st.metric("Unique Types", filtered_df["type"].nunique())

    # Quick selection buttons to populate the global item selector
    st.markdown("---")
    st.subheader("Quick Selection from Filtered Items")
    st.caption("These buttons will update the 'Search for items' selector at the top of the page.")

    quick_col1, quick_col2, quick_col3 = st.columns(3)

    with quick_col1:
        if st.button("Select Top 50 by Name"):
            top_ids = [str(row["id"]) for _, row in filtered_df.head(50).iterrows()]
            st.session_state["selected_item_ids"] = top_ids
            st.session_state["apply_item_selection"] = True
            st.rerun()

    with quick_col2:
        if st.button("Select Random 50"):
            sample_df = filtered_df.sample(min(50, len(filtered_df)))
            random_ids = [str(row["id"]) for _, row in sample_df.iterrows()]
            st.session_state["selected_item_ids"] = random_ids
            st.session_state["apply_item_selection"] = True
            st.rerun()

    with quick_col3:
        if st.button("Clear Selection"):
            st.session_state["selected_item_ids"] = []
            st.session_state["apply_item_selection"] = True
            st.rerun()

    # Show how many items are currently selected
    if selected_items:
        st.success(f"{len(selected_items)} item(s) selected for analysis.")
    else:
        st.info("Use the 'Search for items' selector at the top or the quick selection buttons above to select items.")

    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Profitability Analysis",
        "Similarity Analysis",
        "Filtered Items Table",
        "Type/Rarity Distribution",
    ])

    selected_ids = {int(item["item_id"]) for item in selected_items} if selected_items else set()
    display_df = filtered_df
    if selected_ids:
        display_df = filtered_df[filtered_df["id"].isin(selected_ids)]

    with tab1:
        st.markdown("### Profitability Analysis")
        if st.button("Run Profitability Analysis", key="run_profit"):
            if not selected_items:
                st.warning("Select at least one item for analysis.")
            else:
                item_ids = [int(item["item_id"]) for item in selected_items[:max_items_analysis]]
                item_names = {int(item["item_id"]): item["item_name"] for item in selected_items}

                with st.spinner(f"Computing profitability metrics for {len(item_ids)} items..."):
                    metrics_df = _compute_profitability_metrics(item_ids, int(days_back))

                if metrics_df.empty:
                    st.warning("No price data available for selected items.")
                else:
                    # Add item names
                    metrics_df["item_name"] = metrics_df["item_id"].map(item_names)

                    # Filter by margin
                    profitable = metrics_df[metrics_df["current_margin_pct"] >= min_margin_pct]

                    st.success(f"Found {len(profitable)} items with margin >= {min_margin_pct}%")

                    # Sort by margin
                    profitable = profitable.sort_values("current_margin_pct", ascending=False)

                    # Display metrics
                    display_cols = [
                        "item_id",
                        "item_name",
                        "current_buy",
                        "current_sell",
                        "current_margin",
                        "current_margin_pct",
                        "avg_margin_pct",
                        "buy_volatility_pct",
                        "sell_volatility_pct",
                        "avg_buy_qty",
                        "avg_sell_qty",
                    ]
                    st.dataframe(
                        profitable[display_cols].style.format({
                            "current_buy": "{:.0f}",
                            "current_sell": "{:.0f}",
                            "current_margin": "{:.0f}",
                            "current_margin_pct": "{:.2f}%",
                            "avg_margin_pct": "{:.2f}%",
                            "buy_volatility_pct": "{:.2f}%",
                            "sell_volatility_pct": "{:.2f}%",
                            "avg_buy_qty": "{:.0f}",
                            "avg_sell_qty": "{:.0f}",
                        }),
                        width="stretch",
                        hide_index=True,
                    )

                    # Scatter plot: margin vs volatility
                    if len(profitable) > 1:
                        st.markdown("#### Margin % vs Volatility")
                        st.caption(
                            "How this is calculated: current_margin_pct = ((current_sell * 0.85) - current_buy) "
                            "/ current_buy * 100; sell_volatility_pct = std(sell_unit_price) / "
                            "mean(sell_unit_price) * 100; size = avg_sell_qty."
                        )
                        fig = px.scatter(
                            profitable,
                            x="sell_volatility_pct",
                            y="current_margin_pct",
                            size="avg_sell_qty",
                            color="current_margin_pct",
                            hover_name="item_name",
                            hover_data=["item_id", "current_buy", "current_sell"],
                            color_continuous_scale="RdYlGn",
                            size_max=40,
                        )
                        fig.update_layout(
                            xaxis_title="Sell Volatility %",
                            yaxis_title="Current Margin %",
                        )
                        fig.update_traces(marker=dict(sizemin=6, sizemode="area"))
                        st.plotly_chart(fig, width="stretch")
                        st.caption(
                            "Why this matters: margin shows potential profit after fees, volatility signals risk, "
                            "and bubble size reflects liquidity."
                        )

                    # Opportunity plot: absolute price change vs margin (size = volume)
                    opportunity_df = profitable[
                        (profitable["avg_buy_qty"] > 0) & (profitable["avg_sell_qty"] > 0)
                    ].copy()
                    if len(opportunity_df) > 1:
                        st.markdown("#### Potential Opportunities")
                        st.caption(
                            "Opportunity plot: abs_price_change = max(|last_buy - first_buy|, "
                            "|last_sell - first_sell|); current_margin_pct as above; "
                            "size = min(avg_buy_qty, avg_sell_qty)."
                        )
                        opportunity_df["abs_price_change"] = opportunity_df[
                            ["abs_buy_change", "abs_sell_change"]
                        ].max(axis=1)
                        fig_op = px.scatter(
                            opportunity_df,
                            x="abs_price_change",
                            y="current_margin_pct",
                            size="avg_qty_min",
                            color="current_margin_pct",
                            hover_name="item_name",
                            hover_data=["item_id", "current_buy", "current_sell", "avg_buy_qty", "avg_sell_qty"],
                            color_continuous_scale="RdYlGn",
                            size_max=40,
                        )
                        fig_op.update_layout(
                            xaxis_title="Absolute Price Change (last vs first)",
                            yaxis_title="Current Margin %",
                        )
                        fig_op.update_traces(marker=dict(sizemin=6, sizemode="area"))
                        st.plotly_chart(fig_op, width="stretch")
                        st.caption(
                            "Why this matters: large price moves plus healthy margin and volume can indicate "
                            "tradable momentum with enough depth."
                        )

                    # Store for use in other tabs
                    st.session_state["profitability_metrics"] = metrics_df

    with tab2:
        st.markdown("### Cosine Similarity Analysis")
        st.markdown("""
        This analysis finds items with similar price movement patterns using cosine similarity.
        Items that move together might be related (same crafting chains, same market forces).
        """)

        if st.button("Run Similarity Analysis", key="run_sim"):
            if not selected_items:
                st.warning("Select at least 2 items for similarity analysis.")
            elif len(selected_items) < 2:
                st.warning("Need at least 2 items for similarity analysis.")
            else:
                item_ids = [int(item["item_id"]) for item in selected_items[:max_items_analysis]]
                item_names = {int(item["item_id"]): item["item_name"] for item in selected_items}

                with st.spinner(f"Loading price data for {len(item_ids)} items..."):
                    price_data = _load_price_data_parallel(item_ids, int(days_back), str(resample_freq))

                if len(price_data) < 2:
                    st.warning("Not enough price data. Need at least 2 items with data.")
                else:
                    st.success(f"Loaded price data for {len(price_data)} items.")

                    # Build hourly DataFrame
                    hourly_df = pd.DataFrame(price_data)

                    with st.spinner("Computing similarity matrix..."):
                        similarity_matrix, distance_matrix = _compute_similarity_matrix(
                            hourly_df,
                            input_mode=similarity_input,
                        )

                    # Similarity heatmap
                    item_labels = [str(item_names.get(col, col)) for col in hourly_df.columns]
                    heatmap_matrix = similarity_matrix
                    heatmap_title = "Cosine Similarity Matrix"
                    heatmap_color_label = "Similarity"
                    fig_heatmap = px.imshow(
                        heatmap_matrix,
                        x=item_labels,
                        y=item_labels,
                        color_continuous_scale="Viridis",
                        title=heatmap_title,
                        labels=dict(color=heatmap_color_label),
                        range_color=(
                            float(np.min(heatmap_matrix)),
                            float(np.max(heatmap_matrix)),
                        ),
                    )
                    fig_heatmap.update_layout(height=600)
                    st.plotly_chart(fig_heatmap, width="stretch")

                    # Dendrogram
                    st.markdown("#### Hierarchical Clustering")
                    id_labels = [
                        f"{item_names.get(int(col), col)} [{col}]"
                        for col in hourly_df.columns
                    ]
                    distance_matrix_plot = distance_matrix
                    dendrogram_y_label = "Distance (1 - Cosine Similarity)"
                    if dendrogram_scale == "Distance x100":
                        distance_matrix_plot = distance_matrix * 100.0
                        dendrogram_y_label = "Distance x100"
                    elif dendrogram_scale == "Min-max (0-1)":
                        mask = ~np.eye(distance_matrix.shape[0], dtype=bool)
                        off_diag = distance_matrix[mask]
                        if off_diag.size:
                            d_min = float(np.min(off_diag))
                            d_max = float(np.max(off_diag))
                        else:
                            d_min = float(np.min(distance_matrix))
                            d_max = float(np.max(distance_matrix))
                        if d_max > d_min:
                            distance_matrix_plot = (distance_matrix - d_min) / (d_max - d_min)
                            distance_matrix_plot = np.clip(distance_matrix_plot, 0, 1)
                        else:
                            distance_matrix_plot = np.zeros_like(distance_matrix)
                        np.fill_diagonal(distance_matrix_plot, 0.0)
                        dendrogram_y_label = "Scaled Distance (0-1)"
                    max_dist = float(np.max(distance_matrix_plot)) if distance_matrix_plot.size else 0.0
                    y_limit_plot = max(max_dist * 1.05, 0.01)
                    fig_dend, linkage_mat = _plot_dendrogram(
                        distance_matrix_plot,
                        id_labels,
                        color_threshold=0.02,
                        y_limit=y_limit_plot,
                        y_axis_title=dendrogram_y_label,
                    )
                    st.plotly_chart(fig_dend, width="stretch")

                    # Cluster assignments
                    st.markdown("#### Cluster Assignments")
                    clusters = _get_clusters(
                        linkage_mat,
                        list(hourly_df.columns),
                        int(n_clusters),
                    )

                    for cluster_id, cluster_items in sorted(clusters.items()):
                        with st.expander(f"Cluster {cluster_id} ({len(cluster_items)} items)"):
                            cluster_info = []
                            for item_id in cluster_items:
                                cluster_info.append({
                                    "item_id": item_id,
                                    "item_name": item_names.get(item_id, str(item_id)),
                                })
                            st.dataframe(
                                pd.DataFrame(cluster_info),
                                width="stretch",
                                hide_index=True,
                            )

                            # Add button to select cluster items
                            if st.button(f"Select Cluster {cluster_id} Items", key=f"select_cluster_{cluster_id}"):
                                st.session_state["selected_item_ids"] = [str(i) for i in cluster_items]
                                st.session_state["apply_item_selection"] = True
                                st.rerun()

                    # Store similarity data
                    st.session_state["similarity_matrix"] = similarity_matrix
                    st.session_state["similarity_items"] = list(hourly_df.columns)

    with tab3:
        st.markdown("### Filtered Items Table")
        st.dataframe(
            display_df[["id", "name", "type", "rarity", "level", "vendor_value"]].head(500),
            width="stretch",
            hide_index=True,
        )

    with tab4:
        st.markdown("### Type and Rarity Distribution")

        col1, col2 = st.columns(2)

        with col1:
            type_counts = display_df["type"].value_counts()
            fig_type = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title="Distribution by Type",
            )
            st.plotly_chart(fig_type, width="stretch")

        with col2:
            # Sort rarity by order
            rarity_counts = display_df["rarity"].value_counts()
            rarity_order_present = [r for r in RARITY_ORDER if r in rarity_counts.index]
            rarity_counts = rarity_counts.reindex(rarity_order_present)

            # Convert to DataFrame for plotly
            rarity_df = pd.DataFrame({
                "rarity": rarity_counts.index.tolist(),
                "count": rarity_counts.values.tolist(),
            })
            fig_rarity = px.bar(
                rarity_df,
                x="rarity",
                y="count",
                title="Distribution by Rarity",
                color="rarity",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_rarity.update_layout(
                xaxis_title="Rarity",
                yaxis_title="Count",
                showlegend=False,
            )
            st.plotly_chart(fig_rarity, width="stretch")

        # Level distribution
        fig_level = px.histogram(
            display_df,
            x="level",
            nbins=20,
            title="Distribution by Level",
            color_discrete_sequence=["#1f77b4"],
        )
        fig_level.update_layout(
            xaxis_title="Level",
            yaxis_title="Count",
        )
        st.plotly_chart(fig_level, width="stretch")


def main() -> None:
    """Main entry point when running item_finder_app.py standalone."""
    st.set_page_config(page_title="GW2ML Item Finder", layout="wide")
    render_item_finder_tab()


if __name__ == "__main__":
    main()
