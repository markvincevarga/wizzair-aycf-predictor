import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
from streamlit_calendar import calendar
from features.airports import AIRPORT_COORDINATES


def get_color_for_probability(prob: float) -> str:
    """Generate a hex color code for a probability value (0-1).

    Uses a red-yellow-green gradient where:
    - 0.0 = Red (#dc3545)
    - 0.5 = Yellow (#ffc107)
    - 1.0 = Green (#28a745)

    Args:
        prob: Probability value between 0 and 1.

    Returns:
        Hex color code string.
    """
    prob = max(0.0, min(1.0, prob))

    if prob < 0.5:
        # Interpolate between red and yellow
        t = prob * 2
        r = int(220 + (255 - 220) * t)
        g = int(53 + (193 - 53) * t)
        b = int(69 + (7 - 69) * t)
    else:
        # Interpolate between yellow and green
        t = (prob - 0.5) * 2
        r = int(255 + (40 - 255) * t)
        g = int(193 + (167 - 193) * t)
        b = int(7 + (69 - 7) * t)

    return f"#{r:02x}{g:02x}{b:02x}"


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

    hist_route = df_hist[
        (df_hist["departure_from"] == origin) & (df_hist["departure_to"] == destination)
    ].copy()

    pred_route = df_preds[
        (df_preds["departure_from"] == origin)
        & (df_preds["departure_to"] == destination)
    ].copy()

    dates_hist = hist_route["availability_start"].tolist()
    dates_pred = pred_route["availability_start"].tolist()

    if not dates_hist and not dates_pred:
        st.info("No data available for this route.")
        return

    # Build calendar events
    events = []
    seen_dates = set()

    # Add history events (these take priority)
    for _, row in hist_route.iterrows():
        event_date = row["availability_start"]
        if event_date in seen_dates:
            continue
        seen_dates.add(event_date)

        is_available = row.get("is_available", True)
        events.append(
            {
                "title": "Available" if is_available else "Unavailable",
                "start": event_date.isoformat(),
                "end": event_date.isoformat(),
                "backgroundColor": "#28a745" if is_available else "#dc3545",
                "borderColor": "#28a745" if is_available else "#dc3545",
            }
        )

    # Add prediction events
    for _, row in pred_route.iterrows():
        event_date = row["availability_start"]
        if event_date in seen_dates:
            continue
        seen_dates.add(event_date)

        prob = row["predicted_probability"]
        color = get_color_for_probability(prob)
        events.append(
            {
                "title": f"{prob:.0%}",
                "start": event_date.isoformat(),
                "end": event_date.isoformat(),
                "backgroundColor": color,
                "borderColor": color,
            }
        )

    if not events:
        st.write("No data.")
        return

    # Determine initial date for calendar view
    all_dates = sorted(seen_dates)
    initial_date = date.today().isoformat()
    if all_dates:
        initial_date = min(all_dates).isoformat()

    calendar_options = {
        "initialView": "dayGridMonth",
        "initialDate": initial_date,
        "headerToolbar": {
            "left": "prev,next today",
            "center": "title",
            "right": "",
        },
        "firstDay": 1,  # Monday
        "height": 500,
    }

    calendar(events=events, options=calendar_options, key="availability_calendar")

    st.caption(
        "Green: Available / High probability. Yellow: Medium probability. "
        "Red: Low probability. Past dates show actual availability."
    )
