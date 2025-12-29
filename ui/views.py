import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
from features.airports import AIRPORT_COORDINATES


def render_map_view(data):
    st.header("Where to?")

    df_preds = data["predictions"]

    if df_preds.empty:
        st.warning("No prediction data available.")
        return

    # User inputs
    origins = sorted(set(df_preds["departure_from"]))
    if not origins:
        st.warning("No origins found in predictions.")
        return

    col1, col2 = st.columns(2)
    with col1:
        default_origins = ["Stockholm"] if "Stockholm" in origins else [origins[0]]
        selected_origins = st.multiselect(
            "Departures",
            origins,
            default=default_origins,
            help="Compare multiple departure airports on the same map.",
        )
    with col2:
        # Default to tomorrow or first available date in predictions
        available_dates = sorted(set(df_preds["availability_start"]))
        default_date = date.today() + timedelta(days=1)
        if default_date not in available_dates and available_dates:
            default_date = available_dates[0]

        selected_date = st.date_input("Date", default_date)

    min_probability = st.slider(
        "Minimum probability",
        min_value=0,
        max_value=100,
        value=40,
        step=5,
        help="Hide destinations with predicted probability below the selected threshold.",
    )
    min_probability_value = min_probability / 100

    if not selected_origins:
        st.info("Select at least one departure airport to compare.")
        return

    # Filter data
    # Ensure selected_date is compared correctly (it comes as date from st.date_input)
    filtered = df_preds[
        (df_preds["departure_from"].isin(selected_origins))
        & (df_preds["availability_start"] == selected_date)
    ].copy()

    filtered = filtered[filtered["predicted_probability"] >= min_probability_value]

    if filtered.empty:
        st.info(
            f"No destinations meet the {min_probability}% threshold on {selected_date}."
        )
        return

    # Add coordinates
    filtered["lat"] = filtered["departure_to"].map(
        lambda x: AIRPORT_COORDINATES.get(x, (None, None))[0]
    )
    filtered["lon"] = filtered["departure_to"].map(
        lambda x: AIRPORT_COORDINATES.get(x, (None, None))[1]
    )

    # Drop missing coords
    missing_coords = filtered[filtered["lat"].isna()]
    if not missing_coords.empty:
        missing_destinations = ", ".join(sorted(set(missing_coords["departure_to"])))
        st.caption(f"Missing coordinates for: {missing_destinations}")

    plot_data = filtered.dropna(subset=["lat", "lon"])
    if plot_data.empty:
        st.error("No valid destination coordinates found.")
        return

    # Plot destinations with probability gradient and origin symbols
    fig = px.scatter_mapbox(
        plot_data,
        lat="lat",
        lon="lon",
        hover_name="departure_to",
        hover_data={
            "predicted_probability": ":.2f",
            "departure_from": True,
            "lat": False,
            "lon": False,
        },
        color="predicted_probability",
        color_continuous_scale=["red", "yellow", "green"],
        range_color=[0, 1],
        size_max=16,
        zoom=3,
        mapbox_style="carto-positron",
    )

    origin_points = []
    for airport in selected_origins:
        lat, lon = AIRPORT_COORDINATES.get(airport, (None, None))
        if lat is None or lon is None:
            continue
        origin_points.append({"departure_from": airport, "lat": lat, "lon": lon})

    if origin_points:
        origin_df = pd.DataFrame(origin_points)
        fig.add_trace(
            go.Scattermapbox(
                lat=origin_df["lat"],
                lon=origin_df["lon"],
                mode="markers+text",
                text=origin_df["departure_from"],
                textposition="top center",
                marker=dict(size=18, color="rgba(33, 150, 243, 0.85)"),
                name="Departure airports",
            )
        )

        fig.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            legend=dict(title="Departures"),
            coloraxis_colorbar=dict(title="Probability"),
        )

        st.plotly_chart(fig, width="stretch")

    summary = (
        filtered.sort_values("predicted_probability", ascending=False)
        .groupby("departure_from", group_keys=False)
        .head(5)
        .copy()
    )

    if not summary.empty:
        summary["Probability"] = summary["predicted_probability"].map(
            lambda val: f"{val:.0%}"
        )
        summary_display = summary[
            ["departure_from", "departure_to", "Probability"]
        ].rename(
            columns={
                "departure_from": "From",
                "departure_to": "To",
            }
        )
        st.subheader("Route comparison highlights")
        st.dataframe(summary_display, use_container_width=True)
        st.caption("Top 5 destinations per departure ordered by predicted probability.")


def render_calendar_view(data):
    st.header("When?")

    df_preds = data["predictions"]
    df_hist = data["history"]

    all_routes = pd.concat(
        [
            df_preds[["departure_from", "departure_to"]],
            df_hist[["departure_from", "departure_to"]],
        ],
    ).drop_duplicates()

    origins = sorted(set(all_routes["departure_from"]))
    col1, col2 = st.columns(2)
    with col1:
        origin = st.selectbox("Departure", origins, key="cal_origin")

    # Filter destinations based on origin
    destinations = sorted(
        set(all_routes[all_routes["departure_from"] == origin]["departure_to"])
    )
    with col2:
        destination = st.selectbox("Destination", destinations, key="cal_dest")

    # Prepare data for grid
    # Combine history and predictions
    # History: is_available (True) -> 1.0 probability (conceptually) or distinct marker?
    # Actually, history is boolean. Predictions are float.
    # We can create a "score" column.

    hist_route = df_hist[
        (df_hist["departure_from"] == origin) & (df_hist["departure_to"] == destination)
    ].copy()

    pred_route = df_preds[
        (df_preds["departure_from"] == origin)
        & (df_preds["departure_to"] == destination)
    ].copy()

    # Create a full date range to display (e.g. min to max available in loaded data)
    dates_hist = hist_route["availability_start"].tolist()
    dates_pred = pred_route["availability_start"].tolist()

    if not dates_hist and not dates_pred:
        st.info("No data available for this route.")
        return

    all_dates = sorted(set(dates_hist + dates_pred))
    min_date = min(all_dates)
    max_date = max(all_dates)

    # Create calendar grid dataframe
    # We want rows = weeks, cols = days (Mon-Sun)
    # But filtering by month is better.

    current_month = date.today().replace(day=1)
    # Show current month and next month?
    # Or just show the 60-day window as a continuous stream?
    # A simple timeline heatmap is effective:
    # X = Date, Y = "Availability"
    # Let's do a Heatmap Calendar: X=Day, Y=Month
    grid_data = []
    # Fill history
    for _, row in hist_route.iterrows():
        grid_data.append(
            {
                "date": row["availability_start"],
                "type": "History",
                "score": 1.0
                if row["is_available"]
                else 0.0,  # Availabilities table only has available rows usually
                "label": "Available",
            }
        )

    # Fill predictions
    for _, row in pred_route.iterrows():
        # If date overlaps with history (unlikely if 'data_generated' is recent, but possible), history should win?
        # Usually predictions are for future.
        grid_data.append(
            {
                "date": row["availability_start"],
                "type": "Prediction",
                "score": row["predicted_probability"],
                "label": f"{row['predicted_probability']:.0%}",
            }
        )

    df_grid = pd.DataFrame(grid_data)

    if df_grid.empty:
        st.write("No data.")
        return

    # Remove duplicates (prefer History over Prediction if same date)
    # Sort by type (History first?) No, just drop duplicates on date.
    # Actually, if we have both, it's interesting. But for now, let's just keep one.
    # History usually implies we know the truth.
    df_grid["type_rank"] = df_grid["type"].replace({"History": 1, "Prediction": 2})
    df_grid = df_grid.sort_values("type_rank").drop_duplicates(
        subset=["date"], keep="first"
    )
    # Visualization
    # Create a unified timeline
    df_grid["day_of_week"] = df_grid["date"].apply(
        lambda d: d.strftime("%a")
    )  # Mon, Tue...
    df_grid["week_of_year"] = df_grid["date"].apply(lambda d: d.isocalendar()[1])
    df_grid["iso_year"] = df_grid["date"].apply(lambda d: d.isocalendar()[0])
    # Create a unique Y axis identifier for the week (Year-Week)
    df_grid["week_id"] = df_grid.apply(
        lambda r: f"{r['iso_year']}-W{r['week_of_year']:02d}", axis=1
    )

    # Sort dates
    df_grid = df_grid.sort_values("date")

    # Plotly Heatmap
    # X: Day of Week (ordered)
    # Y: Week ID (ordered desc)

    fig = go.Figure(
        data=go.Heatmap(
            x=df_grid["day_of_week"],
            y=df_grid["week_id"],
            z=df_grid["score"],
            text=df_grid["label"],
            hoverongaps=False,
            colorscale=["red", "yellow", "green"],
            zmin=0,
            zmax=1,
            xgap=2,
            ygap=2,
        )
    )

    days_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig.update_xaxes(categoryorder="array", categoryarray=days_order)
    fig.update_yaxes(
        autorange="reversed"
    )  # Latest weeks at bottom usually, but standard calendar is top-down.
    # Standard calendar: Earlier dates at top. So 'reversed' implies descending order of Y?
    # Y axis strings sort alphanumerically: 2025-W01, 2025-W02.
    # Reversed means W02 is below W01. That's correct for a calendar.

    fig.update_layout(
        title=f"{origin} to {destination}",
        xaxis_title="Day of Week",
        yaxis_title="Week",
        height=400,
    )

    st.plotly_chart(fig, width="stretch")
    # Legend/Key
    st.caption(
        "Green: High Probability / Available. Red: Low Probability. Cells show prediction %."
    )
