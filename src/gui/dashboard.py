"""Streamlit dashboard for the Trump Mentions Trading Bot."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
import time

# Page config
st.set_page_config(
    page_title="Trump Mentions Trading Bot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000/api"

# --- Helper functions ---

def api_get(endpoint: str, params: dict = None):
    """Make a GET request to the API."""
    try:
        resp = requests.get(f"{API_BASE}{endpoint}", params=params or {}, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def api_post(endpoint: str, data: dict = None):
    """Make a POST request to the API."""
    try:
        resp = requests.post(f"{API_BASE}{endpoint}", json=data or {}, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def api_put(endpoint: str, data: dict = None):
    """Make a PUT request to the API."""
    try:
        resp = requests.put(f"{API_BASE}{endpoint}", json=data or {}, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


# --- Sidebar ---

with st.sidebar:
    st.title("Control Panel")

    # Kalshi connection
    st.subheader("Kalshi Connection")
    if st.button("Login to Kalshi", use_container_width=True):
        result = api_post("/kalshi/login")
        if result:
            st.success(f"Logged in: {result.get('member_id', '')}")

    st.divider()

    # Data refresh controls
    st.subheader("Data Refresh")

    if st.button("Full Refresh", type="primary", use_container_width=True):
        result = api_post("/system/full-refresh")
        if result:
            st.info("Full refresh started...")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Sync Markets", use_container_width=True):
            result = api_post("/markets/sync")
            if result:
                st.info("Syncing...")

        if st.button("Scrape Speeches", use_container_width=True):
            result = api_post("/speeches/scrape")
            if result:
                st.info("Scraping...")

    with col2:
        if st.button("Update Events", use_container_width=True):
            result = api_post("/events/update")
            if result:
                st.info("Updating...")

        if st.button("Generate Predictions", use_container_width=True):
            result = api_post("/predictions/generate")
            if result:
                st.info("Generating...")

    st.divider()

    # Bot config
    st.subheader("Bot Configuration")
    config = api_get("/trading/config") or {}

    auto_trade = st.toggle("Auto-Trade", value=config.get('auto_trade', False))
    max_exposure = st.number_input("Max Exposure ($)", value=config.get('max_total_exposure', 500.0), step=50.0)
    max_daily_loss = st.number_input("Max Daily Loss ($)", value=config.get('max_daily_loss', 50.0), step=10.0)
    min_edge = st.slider("Min Edge", 0.01, 0.30, config.get('min_edge_threshold', 0.05), 0.01)
    min_conf = st.slider("Min Confidence", 0.0, 1.0, config.get('min_confidence', 0.3), 0.05)

    if st.button("Update Config", use_container_width=True):
        api_put("/trading/config", {
            'auto_trade': auto_trade,
            'max_total_exposure': max_exposure,
            'max_daily_loss': max_daily_loss,
            'min_edge_threshold': min_edge,
            'min_confidence': min_conf,
        })
        st.success("Config updated")

    st.divider()

    # Live monitor
    st.subheader("Live Monitor")
    live = api_get("/live/status") or {}
    if live.get('is_monitoring'):
        st.success("Monitor: ACTIVE")
        detections = live.get('total_detections', 0)
        if detections > 0:
            st.metric("Live Detections", detections)
    else:
        st.text("Monitor: Inactive")

    st.divider()

    # Auto-refresh
    auto_refresh = st.toggle("Auto-Refresh Dashboard", value=True)
    refresh_interval = st.select_slider("Interval (sec)", [10, 30, 60, 120, 300], value=60)

    if auto_refresh:
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=refresh_interval * 1000, key="auto_refresh")
        except ImportError:
            st.warning("Install streamlit-autorefresh for auto-refresh")


# --- Main Dashboard ---

st.title("Trump Mentions Trading Bot")

# Unread alerts badge
unread = api_get("/alerts/count") or {}
unread_count = unread.get('unread', 0)
if unread_count > 0:
    st.warning(f"You have {unread_count} unread alert{'s' if unread_count > 1 else ''}!")

# Live alerts bar
live_events = api_get("/events/live") or []
if live_events:
    for event in live_events:
        st.error(f"🔴 LIVE NOW: {event['title']} (started {event.get('start_time', 'unknown')})")

# --- Tab Layout ---
tab_markets, tab_terms, tab_predictions, tab_trading, tab_events, tab_live, tab_ml, tab_data = st.tabs([
    "Markets", "Terms Database", "Predictions", "Trading",
    "Events Calendar", "Live Monitor", "ML Models", "Data Stats"
])


# --- Markets Tab ---
with tab_markets:
    st.header("Kalshi Trump Mentions Markets")

    markets = api_get("/markets") or []

    if markets:
        # Summary metrics
        active = [m for m in markets if m.get('status') in ('active', 'open')]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Markets", len(markets))
        col2.metric("Active Markets", len(active))
        col3.metric("Total Volume", sum(m.get('volume', 0) or 0 for m in markets))
        col4.metric("Unique Terms", len(set(t for m in markets for t in m.get('terms', []))))

        # Active markets table
        if active:
            st.subheader("Active Markets")
            df = pd.DataFrame(active)
            df = df[['ticker', 'title', 'yes_price', 'no_price', 'volume', 'close_time', 'terms']]
            df['terms'] = df['terms'].apply(lambda x: ', '.join(x) if x else '')

            # Color code by price
            st.dataframe(
                df,
                column_config={
                    'ticker': st.column_config.TextColumn("Ticker", width="medium"),
                    'title': st.column_config.TextColumn("Market", width="large"),
                    'yes_price': st.column_config.NumberColumn("Yes $", format="%.2f"),
                    'no_price': st.column_config.NumberColumn("No $", format="%.2f"),
                    'volume': st.column_config.NumberColumn("Volume"),
                    'close_time': st.column_config.TextColumn("Closes"),
                    'terms': st.column_config.TextColumn("Terms", width="large"),
                },
                use_container_width=True,
                hide_index=True,
            )

        # Price distribution chart
        if active:
            prices = [m['yes_price'] for m in active if m.get('yes_price')]
            if prices:
                fig = px.histogram(x=prices, nbins=20,
                                   title="Yes Price Distribution (Active Markets)",
                                   labels={'x': 'Yes Price', 'y': 'Count'})
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No markets loaded yet. Click 'Sync Markets' in the sidebar.")


# --- Terms Tab ---
with tab_terms:
    st.header("Terms Master Database")

    terms = api_get("/terms") or []

    if terms:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Terms", len(terms))
        compound = sum(1 for t in terms if t.get('is_compound'))
        col2.metric("Compound Terms", compound)
        col3.metric("Total Occurrences", sum(t.get('total_occurrences', 0) for t in terms))

        # Terms table
        df = pd.DataFrame(terms)
        df = df.sort_values('total_occurrences', ascending=False)
        st.dataframe(
            df[['term', 'normalized_term', 'is_compound', 'total_occurrences',
                'trend_score', 'market_count']],
            column_config={
                'term': st.column_config.TextColumn("Term", width="large"),
                'normalized_term': st.column_config.TextColumn("Normalized"),
                'is_compound': st.column_config.CheckboxColumn("Compound?"),
                'total_occurrences': st.column_config.NumberColumn("Occurrences"),
                'trend_score': st.column_config.NumberColumn("Trend", format="%.2f"),
                'market_count': st.column_config.NumberColumn("Markets"),
            },
            use_container_width=True,
            hide_index=True,
        )

        # Top terms chart
        top_terms = df.head(20)
        if not top_terms.empty:
            fig = px.bar(
                top_terms, x='term', y='total_occurrences',
                color='trend_score',
                color_continuous_scale='RdYlGn',
                title="Top 20 Terms by Occurrence",
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        # Term detail view
        st.subheader("Term Detail")
        selected_term = st.selectbox(
            "Select a term", [t['term'] for t in terms]
        )
        if selected_term:
            term_data = next((t for t in terms if t['term'] == selected_term), None)
            if term_data:
                history = api_get(f"/terms/{term_data['id']}/history", {'days': 365}) or []
                if history:
                    hist_df = pd.DataFrame(history)
                    fig = px.line(hist_df, x='date', y='count',
                                  title=f"'{selected_term}' Usage Over Time")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No usage history available yet.")
    else:
        st.info("No terms loaded. Run market sync first.")


# --- Predictions Tab ---
with tab_predictions:
    st.header("Term Predictions")

    predictions = api_get("/predictions") or []

    if predictions:
        # Prediction summary
        high_prob = [p for p in predictions if p.get('probability', 0) > 0.7]
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Predictions", len(predictions))
        col2.metric("High Probability (>70%)", len(high_prob))
        avg_conf = sum(p.get('confidence', 0) for p in predictions) / len(predictions)
        col3.metric("Avg Confidence", f"{avg_conf:.1%}")

        # Predictions table
        df = pd.DataFrame(predictions)
        df = df.sort_values('probability', ascending=False)
        st.dataframe(
            df[['term', 'probability', 'confidence', 'model_name']],
            column_config={
                'term': st.column_config.TextColumn("Term", width="large"),
                'probability': st.column_config.ProgressColumn("Probability", min_value=0, max_value=1),
                'confidence': st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
                'model_name': st.column_config.TextColumn("Model"),
            },
            use_container_width=True,
            hide_index=True,
        )

        # Prediction heatmap
        if len(predictions) > 1:
            fig = px.scatter(
                df, x='probability', y='confidence',
                text='term', size_max=15,
                title="Predictions: Probability vs Confidence",
                labels={'probability': 'Predicted Probability', 'confidence': 'Model Confidence'},
            )
            fig.update_traces(textposition='top center')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No predictions generated yet. Click 'Generate Predictions'.")


# --- Trading Tab ---
with tab_trading:
    st.header("Trading Dashboard")

    # Portfolio summary
    portfolio = api_get("/trading/portfolio") or {}
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Balance", f"${portfolio.get('balance', 0):.2f}")
    col2.metric("Open Orders", portfolio.get('open_orders', 0))
    col3.metric("Total Trades", portfolio.get('total_trades', 0))
    col4.metric("Total P&L", f"${portfolio.get('total_pnl', 0):.2f}",
                delta_color="normal" if portfolio.get('total_pnl', 0) >= 0 else "inverse")

    st.divider()

    # Trading suggestions
    st.subheader("Trading Suggestions")
    suggestions = api_get("/trading/suggestions") or []

    if suggestions:
        for i, s in enumerate(suggestions):
            with st.expander(
                f"{'🟢' if s['edge'] > 0 else '🔴'} {s['term']} | "
                f"Edge: {s['edge']:+.1%} | "
                f"Side: {s['suggested_side'].upper()} | "
                f"Qty: {s.get('suggested_quantity', 0)}",
                expanded=i < 3
            ):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Our Probability", f"{s['our_probability']:.1%}")
                col2.metric("Market Price", f"{s['market_yes_price']:.1%}")
                col3.metric("Edge", f"{s['edge']:+.1%}")
                col4.metric("Expected Value", f"${s.get('expected_value', 0):.2f}")

                st.text(f"Market: {s['market_ticker']}")
                st.text(f"Confidence: {s.get('confidence', 0):.1%}")
                st.text(f"Kelly Fraction: {s.get('kelly_fraction', 0):.3f}")

                if s.get('reasoning'):
                    st.info(f"Reasoning: {s['reasoning']}")

                # Trade button
                if st.button(f"Execute Trade", key=f"trade_{i}"):
                    result = api_post("/trading/execute", {
                        'market_ticker': s['market_ticker'],
                        'side': s['suggested_side'],
                        'quantity': s.get('suggested_quantity', 1),
                    })
                    if result:
                        st.success(f"Trade placed: {result}")
    else:
        st.info("No trading opportunities found. Adjust min edge threshold or wait for new data.")

    # Current positions
    positions = portfolio.get('positions', [])
    if positions:
        st.subheader("Current Positions")
        st.dataframe(pd.DataFrame(positions), use_container_width=True, hide_index=True)


# --- Events Calendar Tab ---
with tab_events:
    st.header("Trump Events Calendar")

    events = api_get("/events", {'days': 60}) or []

    if events:
        # Live events alert
        live = [e for e in events if e.get('is_live')]
        if live:
            for e in live:
                st.error(f"🔴 LIVE NOW: {e['title']}")

        # Calendar view
        confirmed = [e for e in events if e.get('is_confirmed')]
        unconfirmed = [e for e in events if not e.get('is_confirmed')]

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Upcoming", len(events))
        col2.metric("Confirmed", len(confirmed))
        col3.metric("Unconfirmed", len(unconfirmed))

        # Timeline chart
        if events:
            event_df = pd.DataFrame(events)
            event_df['start_time'] = pd.to_datetime(event_df['start_time'])
            event_df = event_df.dropna(subset=['start_time'])

            if not event_df.empty:
                fig = px.timeline(
                    event_df,
                    x_start='start_time',
                    x_end='start_time',  # single point
                    y='event_type',
                    color='is_confirmed',
                    hover_data=['title', 'location'],
                    title="Upcoming Events Timeline",
                )
                st.plotly_chart(fig, use_container_width=True)

        # Events list
        st.subheader("Event Details")
        for e in events:
            status = "🟢 Confirmed" if e.get('is_confirmed') else "🟡 Unconfirmed"
            live_badge = " 🔴 LIVE" if e.get('is_live') else ""
            with st.expander(f"{status}{live_badge} {e['title']} - {e.get('start_time', 'TBD')}"):
                st.text(f"Type: {e.get('event_type', 'unknown')}")
                st.text(f"Location: {e.get('location', 'TBD')}")
                st.text(f"Start: {e.get('start_time', 'TBD')}")
                if e.get('source_url'):
                    st.markdown(f"[Source]({e['source_url']})")
                if e.get('topics'):
                    st.text(f"Expected topics: {', '.join(e['topics'])}")
    else:
        st.info("No upcoming events found. Click 'Update Events'.")


# --- Live Monitor Tab ---
with tab_live:
    st.header("Live Speech Monitor")

    live_status = api_get("/live/status") or {}

    col1, col2 = st.columns(2)
    with col1:
        if live_status.get('is_monitoring'):
            st.success("Monitor is ACTIVE")
            if st.button("Stop Monitoring", type="secondary"):
                api_post("/live/stop")
                st.rerun()
        else:
            st.warning("Monitor is INACTIVE")
            if st.button("Start Monitoring", type="primary"):
                api_post("/live/start")
                st.rerun()

    with col2:
        if live_status.get('current_event'):
            event = live_status['current_event']
            st.error(f"Currently tracking: {event.get('title', 'Unknown')}")
            st.text(f"Source: {event.get('source', '?')}")
            st.text(f"Started: {event.get('started_at', '?')}")

    # Live detected terms
    detected = live_status.get('detected_terms', {})
    if detected:
        st.subheader("Detected Terms (Live)")
        total = live_status.get('total_detections', 0)
        st.metric("Total Detections", total)

        # Bar chart of detected terms
        det_df = pd.DataFrame([
            {'term': k, 'count': v} for k, v in detected.items()
        ]).sort_values('count', ascending=False)

        fig = px.bar(det_df, x='term', y='count',
                     title="Live Term Detections",
                     color='count', color_continuous_scale='Reds')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        # Table
        st.dataframe(det_df, use_container_width=True, hide_index=True)
    else:
        st.info("No terms detected yet. Start monitoring and wait for a live speech.")

    # Real-time feed placeholder
    st.subheader("Live Feed")
    st.info("When Trump speaks live, detected terms will appear here in real-time. "
            "Dashboard auto-refreshes to show latest detections.")


# --- ML Models Tab ---
with tab_ml:
    st.header("ML Model Management")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Train Models", type="primary", use_container_width=True):
            result = api_post("/ml/train")
            if result:
                st.info("Training started... This may take a few minutes.")

    with col2:
        if st.button("Get ML Predictions", use_container_width=True):
            ml_preds = api_get("/ml/predictions") or []
            if ml_preds:
                st.session_state['ml_predictions'] = ml_preds

    # Model info
    model_info = api_get("/ml/info") or {}

    if 'results' in model_info:
        st.subheader("Training Results")
        st.text(f"Last trained: {model_info.get('timestamp', 'unknown')}")

        for model_name, metrics in model_info.get('results', {}).items():
            if 'error' in metrics:
                st.error(f"{model_name}: {metrics['error']}")
                continue

            with st.expander(f"{model_name} (CV AUC: {metrics.get('cv_auc_mean', 0):.3f})", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("CV AUC", f"{metrics.get('cv_auc_mean', 0):.3f}")
                col2.metric("Accuracy", f"{metrics.get('train_accuracy', 0):.1%}")
                col3.metric("F1 Score", f"{metrics.get('train_f1', 0):.3f}")
                col4.metric("Brier Score", f"{metrics.get('train_brier', 0):.4f}")

                st.text(f"Samples: {metrics.get('n_samples', 0)} | Features: {metrics.get('n_features', 0)}")

                top_features = metrics.get('top_features', {})
                if top_features:
                    feat_df = pd.DataFrame([
                        {'feature': k, 'importance': v}
                        for k, v in top_features.items()
                    ]).sort_values('importance', ascending=True)

                    fig = px.bar(feat_df, x='importance', y='feature',
                                 orientation='h', title=f"{model_name} Feature Importance")
                    st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No models trained yet. Click 'Train Models' to start.")

    # ML predictions display
    ml_preds = st.session_state.get('ml_predictions', [])
    if ml_preds:
        st.subheader("ML Model Predictions")
        ml_df = pd.DataFrame(ml_preds)
        ml_df = ml_df.sort_values('probability', ascending=False)

        st.dataframe(
            ml_df[['term', 'probability', 'confidence', 'model_name']],
            column_config={
                'term': st.column_config.TextColumn("Term"),
                'probability': st.column_config.ProgressColumn("Probability", min_value=0, max_value=1),
                'confidence': st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
            },
            use_container_width=True,
            hide_index=True,
        )

        # Model agreement chart
        if 'model_probabilities' in ml_df.columns:
            st.subheader("Model Agreement")
            for _, row in ml_df.head(10).iterrows():
                model_probs = row.get('model_probabilities', {})
                if model_probs:
                    fig = go.Figure(data=[
                        go.Bar(x=list(model_probs.keys()),
                               y=list(model_probs.values()),
                               name=row['term'])
                    ])
                    fig.update_layout(
                        title=f"'{row['term']}' - Model Predictions",
                        yaxis_title="Probability",
                        height=300,
                    )
                    st.plotly_chart(fig, use_container_width=True)


# --- Data Stats Tab ---
with tab_data:
    st.header("Data Collection Statistics")

    speech_stats = api_get("/speeches/stats") or {}

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Speeches", speech_stats.get('total_speeches', 0))
    col2.metric("With Transcripts", speech_stats.get('with_transcripts', 0))
    col3.metric("Processed", speech_stats.get('processed', 0))

    # Term report
    st.subheader("Term Frequency Report")
    report = api_get("/terms/report") or []
    if report:
        for item in report[:20]:
            with st.expander(f"{item['term']} - {item['total_occurrences']} occurrences (trend: {item['trend_score']:+.2f})"):
                if item.get('recent_speeches'):
                    for speech in item['recent_speeches']:
                        st.text(f"  [{speech.get('date', '?')}] {speech.get('speech_title', '?')} ({speech['count']}x)")

    # System health
    st.subheader("System Health")
    health = api_get("/system/health") or {}
    if health.get('status') == 'ok':
        st.success(f"API is healthy - {health.get('timestamp', '')}")
    else:
        st.error("API is not responding")
