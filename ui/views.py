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
            "Leaving from",
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


def render_performance_view(db):
    """Render the performance view showing prediction metrics over time.
    
    Args:
        db: DatabaseWrapper instance for fetching data.
    """
    from storage.predictions import Predictions
    from storage.availabilities import Availabilities
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    st.header("Performance")
    st.caption("Prediction performance analysis over the last 30 days")
    
    # Define date range
    today = date.today()
    start_date = today - timedelta(days=30)
    end_date = today - timedelta(days=1)  # Yesterday, since we need actual outcomes
    
    preds_repo = Predictions(db)
    avail_repo = Availabilities(db)
    
    # Fetch all predictions for targets in the date range (without deduplication)
    df_preds = preds_repo.get_all_predictions_for_target_range(start_date, end_date)
    
    # Fetch actual availabilities for the date range
    df_avail = avail_repo.get_recent_availabilities(start_date, end_date)
    
    if df_preds.empty:
        st.warning("No prediction data available for the last 30 days.")
        return
        
    if df_avail.empty:
        st.warning("No availability data available for the last 30 days.")
        return
    
    # Check for required column
    if 'predicted_available' not in df_preds.columns:
        st.error(f"Missing 'predicted_available' column. Available columns: {list(df_preds.columns)}")
        return
    
    df_preds = df_preds.copy()
    df_avail = df_avail.copy()
    
    # Optional filters for departure and destination
    all_departures = sorted(df_preds['departure_from'].unique())
    all_destinations = sorted(df_preds['departure_to'].unique())
    
    col1, col2 = st.columns(2)
    with col1:
        selected_departure = st.selectbox(
            "Filter by departure",
            options=["All"] + all_departures,
            index=0,
            key="performance_departure"
        )
    with col2:
        # Filter destination options based on selected departure
        if selected_departure != "All":
            available_destinations = sorted(
                df_preds[df_preds['departure_from'] == selected_departure]['departure_to'].unique()
            )
        else:
            available_destinations = all_destinations
        
        selected_destination = st.selectbox(
            "Filter by destination",
            options=["All"] + available_destinations,
            index=0,
            key="performance_destination"
        )
    
    # Apply filters
    if selected_departure != "All":
        df_preds = df_preds[df_preds['departure_from'] == selected_departure]
        df_avail = df_avail[df_avail['departure_from'] == selected_departure]
    
    if selected_destination != "All":
        df_preds = df_preds[df_preds['departure_to'] == selected_destination]
        df_avail = df_avail[df_avail['departure_to'] == selected_destination]
    
    if df_preds.empty:
        st.warning("No prediction data for the selected filters.")
        return
    
    # For actuals lookup, we need date-level granularity (availabilities are per-day)
    df_preds['target_date'] = df_preds['availability_start'].dt.date
    df_avail['target_date'] = df_avail['availability_start'].dt.date
    
    # Calculate lag as precise timedelta, then convert to fractional days
    # availability_start is the event datetime, prediction_time is when prediction was made
    df_preds['lag_timedelta'] = df_preds['availability_start'] - df_preds['prediction_time']
    df_preds['lag_days_exact'] = df_preds['lag_timedelta'].dt.total_seconds() / (24 * 3600)
    
    # Integer lag days for grouping metrics
    df_preds['lag_days'] = df_preds['lag_days_exact'].apply(lambda x: int(x))
    
    # Filter out predictions made after the event (lag < 0)
    df_preds = df_preds[df_preds['lag_days_exact'] >= 0]
    
    if df_preds.empty:
        st.warning("No valid predictions found (all predictions were made after their target dates).")
        return
    
    # Create actual outcome lookup: 1 if flight was available, 0 otherwise
    # Create actuals set for fast lookup
    actuals_set = set(
        zip(df_avail['departure_from'], df_avail['departure_to'], df_avail['target_date'])
    )
    
    # Add actual outcome to predictions
    df_preds['actual'] = df_preds.apply(
        lambda row: 1 if (row['departure_from'], row['departure_to'], row['target_date']) in actuals_set else 0,
        axis=1
    )
    
    # Use the binary prediction directly (predicted_available is already 0 or 1)
    df_preds['predicted'] = df_preds['predicted_available'].astype(int)
    
    # Group by lag_days and calculate metrics
    lag_days_range = sorted(df_preds['lag_days'].unique())
    
    metrics_data = []
    for lag in lag_days_range:
        subset = df_preds[df_preds['lag_days'] == lag]
        if len(subset) < 10:  # Skip if too few samples
            continue
            
        y_true = subset['actual']
        y_pred = subset['predicted']
        
        # Calculate metrics (handle edge cases)
        acc = accuracy_score(y_true, y_pred)
        
        # Precision/Recall/F1 need at least one positive prediction or actual
        if y_pred.sum() == 0 and y_true.sum() == 0:
            prec, rec, f1 = 1.0, 1.0, 1.0
        elif y_pred.sum() == 0:
            prec, rec, f1 = 0.0, 0.0, 0.0
        else:
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        
        metrics_data.append({
            'lag_days': lag,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'n_samples': len(subset)
        })
    
    if not metrics_data:
        st.warning("Not enough data to calculate metrics (need at least 10 samples per lag day).")
        return
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Display summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Accuracy", f"{df_metrics['accuracy'].mean():.1%}")
    with col2:
        st.metric("Avg Precision", f"{df_metrics['precision'].mean():.1%}")
    with col3:
        st.metric("Avg Recall", f"{df_metrics['recall'].mean():.1%}")
    with col4:
        st.metric("Avg F1 Score", f"{df_metrics['f1_score'].mean():.1%}")
    
    # All metrics combined chart
    st.subheader("All Metrics by Prediction Lead Time")
    df_melted = df_metrics.melt(
        id_vars=['lag_days', 'n_samples'],
        value_vars=['accuracy', 'precision', 'recall', 'f1_score'],
        var_name='Metric',
        value_name='Value'
    )
    fig_combined = px.line(
        df_melted,
        x='lag_days',
        y='Value',
        color='Metric',
        markers=True,
        labels={'lag_days': 'Days Before Event', 'Value': 'Score'},
    )
    fig_combined.update_layout(
        yaxis_range=[0, 1],
        xaxis_title="Days Before Event",
        yaxis_title="Score",
        legend_title="Metric",
    )
    st.plotly_chart(fig_combined, use_container_width=True)
    
    # Show sample counts
    st.subheader("Sample Count by Lead Time")
    fig_samples = px.bar(
        df_metrics,
        x='lag_days',
        y='n_samples',
        labels={'lag_days': 'Days Before Event', 'n_samples': 'Number of Predictions'},
    )
    fig_samples.update_layout(
        xaxis_title="Days Before Event",
        yaxis_title="Number of Predictions",
    )
    st.plotly_chart(fig_samples, use_container_width=True)
    
    # Show raw data table
    with st.expander("View detailed metrics table"):
        display_df = df_metrics.copy()
        display_df['accuracy'] = display_df['accuracy'].map(lambda x: f"{x:.1%}")
        display_df['precision'] = display_df['precision'].map(lambda x: f"{x:.1%}")
        display_df['recall'] = display_df['recall'].map(lambda x: f"{x:.1%}")
        display_df['f1_score'] = display_df['f1_score'].map(lambda x: f"{x:.1%}")
        display_df.columns = ['Lead Time (Days)', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Samples']
        st.dataframe(display_df, use_container_width=True)
