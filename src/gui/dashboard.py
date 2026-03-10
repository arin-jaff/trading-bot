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
    page_title="TrumpGPT Trading Bot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000/api"

# --- ASCII art (trimmed) ---
TRUMP_ART = """
                                 ..... ..... ....
                                .%%%%%%%%%%%%%%%%@%.
                             .#%%%%%%%%%%%%%%%%%%%%%%%*.
                            .%%%%%%%%%%%%%%%%%%%%%%%%%%%%.
                           .%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%.
                          .%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%:
                          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%@......
                         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%-
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%+.
                       .%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%.
                      .%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%.
                       .@%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%-.
                        +%%.-%%%%%%%%%%%%%%%%%%%%%%%%%%%%%..
                     .%%%%%%#..+%%%%%%%%%%%%%%%%%%%%%%%%%%..
                  .#%%%%%%%%%%=...@%%%%%%%%%%%%%%%%%%%%%%.
               ..@%%%%%%%%%%%%%%.   +@%%%%%%%%%%%%%%%%%%%.
              .%%%%%%%%%%%%%%%%%%%.  .:%%%%%%%%%%%%%%%%%%.
            .%%%%%%%%%%%%%%%%%%%%%%%   .+%%%%%%%%%%%%%#.
            %%%%%%%%%%%%%%%%%%%%%%%%%.   .%%%%%%@*..
          .@%%%%%%%%%%%%%%%%%%%%%%%%%%..  .@%%: .
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%.   *@
         +%%%%%%%%%%%%%%%%%%%%%%%%%%%%%@  :%%.
        .%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%+.%%%.
       .%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%@...
      .%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%=
     .%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%@.
    .%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%:
    -%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%@.
    .............................................
"""

# --- Helper functions ---

def api_get(endpoint: str, params: dict = None):
    try:
        resp = requests.get(f"{API_BASE}{endpoint}", params=params or {}, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def api_post(endpoint: str, data: dict = None):
    try:
        resp = requests.post(f"{API_BASE}{endpoint}", json=data or {}, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def api_put(endpoint: str, data: dict = None):
    try:
        resp = requests.put(f"{API_BASE}{endpoint}", json=data or {}, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


# --- Session state init ---
if 'trade_queue' not in st.session_state:
    st.session_state.trade_queue = []
if 'suggestion_index' not in st.session_state:
    st.session_state.suggestion_index = 0
if 'trashed_terms' not in st.session_state:
    st.session_state.trashed_terms = set()
if 'confirm_trash' not in st.session_state:
    st.session_state.confirm_trash = None


# --- Sidebar ---

with st.sidebar:
    st.title("Control Panel")

    # Kalshi connection
    st.subheader("Kalshi Connection")
    if st.button("Login to Kalshi", width="stretch"):
        result = api_post("/kalshi/login")
        if result:
            st.success(f"Logged in: {result.get('member_id', '')}")

    st.divider()

    # Data refresh controls
    st.subheader("Data Refresh")

    if st.button("Full Refresh", type="primary", width="stretch"):
        result = api_post("/system/full-refresh")
        if result:
            st.success("Full refresh started. Check status below.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Sync Markets", width="stretch"):
            result = api_post("/markets/sync")
            if result:
                st.success("Market sync started.")

        if st.button("Scrape Speeches", width="stretch"):
            result = api_post("/speeches/scrape")
            if result:
                st.success("Speech scraping started.")

    with col2:
        if st.button("Update Events", width="stretch"):
            result = api_post("/events/update")
            if result:
                st.success("Event update started.")

        if st.button("Generate Predictions", width="stretch"):
            result = api_post("/predictions/generate")
            if result:
                st.success("Prediction generation started.")

    if st.button("Pull Predictions from Drive", width="stretch"):
        with st.spinner("Downloading from Google Drive..."):
            result = api_post("/drive/download-and-import")
        if result:
            st.success("Predictions downloaded and imported!")

    # Job status display
    jobs = api_get("/jobs/status") or {}
    active_jobs = {k: v for k, v in jobs.items() if not v.get('done', True)}
    done_jobs = {k: v for k, v in jobs.items() if v.get('done', True) and not v.get('error')}
    error_jobs = {k: v for k, v in jobs.items() if v.get('error')}

    if active_jobs:
        st.markdown("---")
        st.caption("Running:")
        for name, status in active_jobs.items():
            step = status.get('step', '')
            progress = status.get('progress', 0)
            total = status.get('total', 0)
            label = name.replace('_', ' ').title()
            if total > 0:
                st.progress(progress / total, text=f"{label}: {step}")
            else:
                st.info(f"{label}: {step}")

    if error_jobs:
        for name, status in error_jobs.items():
            st.error(f"{name.replace('_', ' ').title()}: {status['error'][:80]}")

    if done_jobs:
        recent_done = sorted(done_jobs.items(),
                             key=lambda x: x[1].get('updated_at', ''), reverse=True)[:2]
        for name, status in recent_done:
            if status.get('step'):
                st.caption(f"{status['step']}")

    st.divider()

    # Bot config
    st.subheader("Bot Configuration")
    config = api_get("/trading/config") or {}

    auto_trade = st.toggle("Auto-Trade", value=config.get('auto_trade', False))
    max_exposure = st.number_input("Max Exposure ($)", value=config.get('max_total_exposure', 500.0), step=50.0)
    max_daily_loss = st.number_input("Max Daily Loss ($)", value=config.get('max_daily_loss', 50.0), step=10.0)
    min_edge = st.slider("Min Edge", 0.01, 0.30, config.get('min_edge_threshold', 0.05), 0.01)
    min_conf = st.slider("Min Confidence", 0.0, 1.0, config.get('min_confidence', 0.3), 0.05)

    if st.button("Update Config", width="stretch"):
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

# Unread alerts badge
unread = api_get("/alerts/count") or {}
unread_count = unread.get('unread', 0)
if unread_count > 0:
    st.warning(f"You have {unread_count} unread alert{'s' if unread_count > 1 else ''}!")

# Live alerts bar
live_events = api_get("/events/live") or []
if live_events:
    for event in live_events:
        st.error(f"LIVE NOW: {event['title']} (started {event.get('start_time', 'unknown')})")

# --- Tab Layout ---
tab_home, tab_markets, tab_terms, tab_trading, tab_events, tab_live, tab_ml, tab_data = st.tabs([
    "Home", "Markets", "Terms", "Trading",
    "Events", "Live Monitor", "TrumpGPT", "Data"
])


# ═══════════════════════════════════════════════════════════════════
# HOME TAB — Welcome + Trade Suggestions
# ═══════════════════════════════════════════════════════════════════
with tab_home:
    # Header with ASCII art
    col_art, col_title = st.columns([1, 2])
    with col_art:
        st.code(TRUMP_ART, language=None)
    with col_title:
        st.title("TrumpGPT")
        st.caption("LoRA fine-tuned Llama-3.1-8B + Multi-Scenario Monte Carlo")

        # Quick stats
        model_status = api_get("/model/status") or {}
        health = api_get("/system/health") or {}

        col1, col2, col3 = st.columns(3)
        col1.metric("Terms Tracked", model_status.get('total_terms_tracked', 0))
        col2.metric("Predictions", model_status.get('colab_predictions_count', 0))
        last_run = model_status.get('last_run', '')
        col3.metric("Last Run", last_run[:10] if last_run else "Never")

        if health.get('status') == 'ok':
            st.success("System online")
        else:
            st.error("API not responding")

    st.divider()

    # ── Trade Suggestions Card System ──
    st.header("Trade Suggestions")
    st.caption("TrumpGPT's best opportunities ranked by edge. Accept to queue, deny to skip, trash to remove.")

    # Get suggestions, filter out trashed and already-said (99%+)
    all_suggestions = api_get("/predictions/final") or []
    suggestions = [
        s for s in all_suggestions
        if s.get('term', '').lower() not in st.session_state.trashed_terms
        and abs(s.get('edge', 0)) >= 0.01
        and (s.get('market_yes_price') or 0) < 0.99  # 99%+ = already said, skip
    ]
    # Sort by absolute edge descending
    suggestions.sort(key=lambda x: abs(x.get('edge', 0)), reverse=True)

    if suggestions:
        idx = st.session_state.suggestion_index % len(suggestions)
        remaining = len(suggestions) - idx
        st.caption(f"Showing suggestion {idx + 1} of {len(suggestions)}")

        s = suggestions[idx]
        term = s.get('term', '?')
        edge = s.get('edge', 0)
        our_prob = s.get('final_probability', 0)
        market_price = s.get('market_yes_price', 0.5)
        side = 'YES' if edge > 0 else 'NO'
        edge_abs = abs(edge)
        hist = s.get('historical_market_record', {})
        past_yes = hist.get('yes', 0)
        past_no = hist.get('no', 0)

        # Confidence descriptor
        if edge_abs >= 0.20:
            conf_label = "Very High"
            conf_color = "#2ecc71"
        elif edge_abs >= 0.10:
            conf_label = "High"
            conf_color = "#27ae60"
        elif edge_abs >= 0.05:
            conf_label = "Moderate"
            conf_color = "#f39c12"
        else:
            conf_label = "Low"
            conf_color = "#e74c3c"

        # Main card — market title with week range
        close_time = s.get('close_time', '')
        week_label = ''
        if close_time:
            try:
                close_dt = datetime.fromisoformat(close_time)
                week_start = close_dt - timedelta(days=close_dt.weekday())
                week_end = week_start + timedelta(days=6)
                week_label = f" for the week of {week_start.strftime('%b %d')} – {week_end.strftime('%b %d')}"
            except Exception:
                pass

        market_title = s.get('market_title', f'Will Trump say "{term}"?')
        st.markdown(f"### {market_title}")
        if week_label:
            st.caption(week_label)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("TrumpGPT Says", f"{our_prob:.0%}")
        col2.metric("Market Says", f"{market_price:.0%}")
        col3.metric("Edge", f"{edge:+.1%}")
        col4.metric("Suggested Side", side)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Confidence", conf_label)
        col2.metric("Historical Record", f"{past_yes}W / {past_no}L" if past_yes + past_no > 0 else "No history")
        col3.metric("Speeches w/ Term", s.get('speeches_with_term', 0))
        col4.metric("Recency Weight", f"x{s.get('recency_weight', 1.0):.2f}")

        # Market outcome history for this term (yes/no per week)
        weekly_data = api_get("/markets/weekly-payouts", {'weeks': 16}) or []
        term_lower = term.lower().strip()
        term_weeks = []
        for w in weekly_data:
            yes_terms = [t.lower() for t in w.get('yes_terms', [])]
            no_terms = [t.lower() for t in w.get('no_terms', [])]
            if term_lower in yes_terms:
                term_weeks.append({'Week': w['week'], 'Result': 'Said', 'value': 1})
            elif term_lower in no_terms:
                term_weeks.append({'Week': w['week'], 'Result': 'Not Said', 'value': -1})

        if term_weeks:
            tw_df = pd.DataFrame(term_weeks).sort_values('Week')
            colors = {'Said': '#2ecc71', 'Not Said': '#e74c3c'}
            fig = go.Figure()
            for result, color in colors.items():
                mask = tw_df[tw_df['Result'] == result]
                if not mask.empty:
                    fig.add_trace(go.Bar(
                        x=mask['Week'], y=[1] * len(mask),
                        name=result, marker_color=color,
                    ))
            fig.update_layout(
                title=f'"{term}" — past market outcomes',
                height=220,
                margin=dict(t=35, b=20, l=40, r=20),
                yaxis=dict(visible=False),
                barmode='stack',
                showlegend=True,
                legend=dict(orientation='h', y=1.15),
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption(f"No past market history for \"{term}\".")

        # Component breakdown
        with st.expander("Signal breakdown"):
            comp = s.get('component_scores', {})
            if comp:
                for signal, score in comp.items():
                    if score is not None:
                        bar_len = int(score * 30)
                        bar = "█" * bar_len + "░" * (30 - bar_len)
                        st.text(f"  {signal:<22} {score:.1%}  {bar}")

            by_scenario = s.get('by_scenario', {})
            if by_scenario:
                st.markdown("**Per-scenario:**")
                for sc, sc_data in by_scenario.items():
                    prob = sc_data.get('probability', 0) if isinstance(sc_data, dict) else sc_data
                    st.text(f"  {sc.replace('_', ' '):<22} {prob:.0%}")

        # Action buttons
        st.markdown("---")
        col_accept, col_deny, col_trash = st.columns(3)

        with col_accept:
            if st.button("Accept — Add to Queue", type="primary", key=f"accept_{idx}",
                         use_container_width=True):
                # Add to trade queue
                queue_item = {
                    'term': term,
                    'market_ticker': s.get('market_ticker', ''),
                    'market_title': s.get('market_title', ''),
                    'side': side.lower(),
                    'our_probability': our_prob,
                    'market_price': market_price,
                    'edge': edge,
                    'confidence': conf_label,
                    'added_at': datetime.utcnow().isoformat(),
                }
                # Avoid duplicates
                existing_terms = {q['term'] for q in st.session_state.trade_queue}
                if term not in existing_terms:
                    st.session_state.trade_queue.append(queue_item)
                st.session_state.suggestion_index += 1
                st.rerun()

        with col_deny:
            if st.button("Deny — Skip", key=f"deny_{idx}",
                         use_container_width=True):
                st.session_state.suggestion_index += 1
                st.rerun()

        with col_trash:
            if st.session_state.confirm_trash == term:
                st.warning(f"Remove \"{term}\" from all suggestions?")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Yes, trash it", key=f"confirm_yes_{idx}",
                                 type="primary", use_container_width=True):
                        st.session_state.trashed_terms.add(term.lower())
                        st.session_state.confirm_trash = None
                        # Don't increment index since list shrinks
                        st.rerun()
                with c2:
                    if st.button("Cancel", key=f"confirm_no_{idx}",
                                 use_container_width=True):
                        st.session_state.confirm_trash = None
                        st.rerun()
            else:
                if st.button("Trash — Remove", key=f"trash_{idx}",
                             use_container_width=True):
                    st.session_state.confirm_trash = term
                    st.rerun()

        # Queue preview
        if st.session_state.trade_queue:
            st.divider()
            st.subheader(f"Trade Queue ({len(st.session_state.trade_queue)})")
            q_df = pd.DataFrame(st.session_state.trade_queue)
            st.dataframe(
                q_df[['term', 'side', 'our_probability', 'market_price', 'edge', 'confidence']],
                column_config={
                    'term': st.column_config.TextColumn("Term", width="large"),
                    'side': st.column_config.TextColumn("Side"),
                    'our_probability': st.column_config.NumberColumn("Our Prob", format="%.1%%"),
                    'market_price': st.column_config.NumberColumn("Market", format="%.1%%"),
                    'edge': st.column_config.NumberColumn("Edge", format="%+.1%%"),
                    'confidence': st.column_config.TextColumn("Confidence"),
                },
                width="stretch",
                hide_index=True,
            )
            st.caption("Go to the Trading tab to review and execute queued trades.")

    else:
        st.info("No trade suggestions available. Sync markets and generate predictions first.")

    # Trashed terms info
    if st.session_state.trashed_terms:
        with st.expander(f"Trashed terms ({len(st.session_state.trashed_terms)})"):
            st.caption("These terms are hidden from suggestions this session.")
            for t in sorted(st.session_state.trashed_terms):
                st.text(f"  {t}")
            if st.button("Restore all trashed terms"):
                st.session_state.trashed_terms = set()
                st.rerun()


# ═══════════════════════════════════════════════════════════════════
# MARKETS TAB — Simplified
# ═══════════════════════════════════════════════════════════════════
with tab_markets:
    st.header("Kalshi Trump Mentions Markets")

    markets = api_get("/markets") or []

    if markets:
        active = [m for m in markets if m.get('status') in ('active', 'open')]

        # Summary row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Markets", len(markets))
        col2.metric("Active", len(active))
        col3.metric("Volume", f"{sum(m.get('volume', 0) or 0 for m in markets):,}")
        col4.metric("Unique Terms", len(set(t for m in markets for t in m.get('terms', []))))

        # Active markets table
        if active:
            df = pd.DataFrame(active)
            df = df[['ticker', 'title', 'yes_price', 'no_price', 'volume', 'close_time', 'terms']]
            df['terms'] = df['terms'].apply(lambda x: ', '.join(x) if x else '')

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
                width="stretch",
                hide_index=True,
            )

        # Details at the bottom
        with st.expander("Price Distribution"):
            if active:
                prices = [m['yes_price'] for m in active if m.get('yes_price')]
                if prices:
                    fig = px.histogram(x=prices, nbins=20,
                                       title="Yes Price Distribution",
                                       labels={'x': 'Yes Price', 'y': 'Count'})
                    st.plotly_chart(fig, width="stretch")

        with st.expander("Weekly Term Payouts"):
            weekly_data = api_get("/markets/weekly-payouts", {'weeks': 12}) or []
            if weekly_data:
                weeks_df = pd.DataFrame([
                    {
                        'Week': w['week'],
                        'Said (Yes)': w['yes_count'],
                        'Not Said (No)': w['no_count'],
                    }
                    for w in weekly_data
                ]).sort_values('Week')

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=weeks_df['Week'], y=weeks_df['Said (Yes)'],
                    name='Said (Yes)', marker_color='#2ecc71',
                ))
                fig.add_trace(go.Bar(
                    x=weeks_df['Week'], y=weeks_df['Not Said (No)'],
                    name='Not Said (No)', marker_color='#e74c3c',
                ))
                fig.update_layout(barmode='stack', height=400,
                                  xaxis_title='Week', yaxis_title='Markets')
                st.plotly_chart(fig, width="stretch")

                for w in weekly_data:
                    with st.expander(
                        f"Week of {w['week']} — {w['yes_count']} Yes, {w['no_count']} No"
                    ):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Said (Yes):**")
                            for m in w.get('yes_markets', []):
                                terms_str = ', '.join(m['terms']) if m['terms'] else m['title']
                                st.text(f"  {terms_str}")
                        with col2:
                            st.markdown("**Not Said (No):**")
                            for m in w.get('no_markets', []):
                                terms_str = ', '.join(m['terms']) if m['terms'] else m['title']
                                st.text(f"  {terms_str}")
            else:
                st.info("No settled markets yet.")

    else:
        st.info("No markets loaded yet. Click 'Sync Markets' in the sidebar.")


# ═══════════════════════════════════════════════════════════════════
# TERMS TAB
# ═══════════════════════════════════════════════════════════════════
with tab_terms:
    st.header("Terms Database")

    terms = api_get("/terms") or []

    if terms:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Terms", len(terms))
        compound = sum(1 for t in terms if t.get('is_compound'))
        col2.metric("Compound Terms", compound)
        col3.metric("Total Occurrences", sum(t.get('total_occurrences', 0) for t in terms))

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
            width="stretch",
            hide_index=True,
        )

        with st.expander("Top 20 Terms Chart"):
            top_terms = df.head(20)
            if not top_terms.empty:
                fig = px.bar(top_terms, x='term', y='total_occurrences',
                             color='trend_score', color_continuous_scale='RdYlGn',
                             title="Top 20 Terms by Occurrence")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, width="stretch")

        st.subheader("Term Detail")
        selected_term = st.selectbox("Select a term", [t['term'] for t in terms])
        if selected_term:
            term_data = next((t for t in terms if t['term'] == selected_term), None)
            if term_data:
                history = api_get(f"/terms/{term_data['id']}/history", {'days': 365}) or []
                if history:
                    hist_df = pd.DataFrame(history)
                    fig = px.line(hist_df, x='date', y='count',
                                  title=f"'{selected_term}' Usage Over Time")
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("No usage history available yet.")
    else:
        st.info("No terms loaded. Run market sync first.")


# ═══════════════════════════════════════════════════════════════════
# TRADING TAB — Queue + Execution
# ═══════════════════════════════════════════════════════════════════
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

    # Trade queue from Home tab
    if st.session_state.trade_queue:
        st.subheader(f"Queued Trades ({len(st.session_state.trade_queue)})")
        st.caption("Trades accepted from the Home tab. Review and execute below.")

        for i, trade in enumerate(st.session_state.trade_queue):
            with st.expander(
                f"{'BUY' if trade['side'] == 'yes' else 'SELL'} {trade['side'].upper()} — "
                f"{trade['term']} | Edge: {trade['edge']:+.1%}",
                expanded=True,
            ):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Our Probability", f"{trade['our_probability']:.1%}")
                col2.metric("Market Price", f"{trade['market_price']:.1%}")
                col3.metric("Edge", f"{trade['edge']:+.1%}")
                col4.metric("Confidence", trade['confidence'])

                st.text(f"Market: {trade['market_ticker']}")

                col_exec, col_remove = st.columns(2)
                with col_exec:
                    if st.button("Execute Trade", key=f"exec_{i}", type="primary",
                                 use_container_width=True):
                        result = api_post("/trading/execute", {
                            'market_ticker': trade['market_ticker'],
                            'side': trade['side'],
                            'quantity': 1,
                        })
                        if result:
                            st.success(f"Trade executed: {result}")
                            st.session_state.trade_queue.pop(i)
                            st.rerun()
                with col_remove:
                    if st.button("Remove from Queue", key=f"remove_{i}",
                                 use_container_width=True):
                        st.session_state.trade_queue.pop(i)
                        st.rerun()

        if st.button("Clear All Queued Trades"):
            st.session_state.trade_queue = []
            st.rerun()

        st.divider()

    # API-generated suggestions (existing)
    st.subheader("Auto-Generated Suggestions")
    suggestions = api_get("/trading/suggestions") or []

    if suggestions:
        for i, s in enumerate(suggestions):
            with st.expander(
                f"{'BUY YES' if s['edge'] > 0 else 'BUY NO'} {s['term']} | "
                f"Edge: {s['edge']:+.1%} | Qty: {s.get('suggested_quantity', 0)}",
                expanded=i < 3,
            ):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Our Probability", f"{s['our_probability']:.1%}")
                col2.metric("Market Price", f"{s['market_yes_price']:.1%}")
                col3.metric("Edge", f"{s['edge']:+.1%}")
                col4.metric("Kelly Fraction", f"{s.get('kelly_fraction', 0):.3f}")

                st.text(f"Market: {s['market_ticker']}")
                if s.get('reasoning'):
                    st.info(f"Reasoning: {s['reasoning']}")

                if st.button("Execute Trade", key=f"trade_{i}"):
                    result = api_post("/trading/execute", {
                        'market_ticker': s['market_ticker'],
                        'side': s['suggested_side'],
                        'quantity': s.get('suggested_quantity', 1),
                    })
                    if result:
                        st.success(f"Trade placed: {result}")
    else:
        st.info("No auto-generated suggestions. Adjust min edge or wait for new data.")

    # Current positions
    positions = portfolio.get('positions', [])
    if positions:
        st.subheader("Current Positions")
        st.dataframe(pd.DataFrame(positions), width="stretch", hide_index=True)


# ═══════════════════════════════════════════════════════════════════
# EVENTS TAB
# ═══════════════════════════════════════════════════════════════════
with tab_events:
    st.header("Trump Events Calendar")

    events = api_get("/events", {'days': 60}) or []

    if events:
        live = [e for e in events if e.get('is_live')]
        if live:
            for e in live:
                st.error(f"LIVE NOW: {e['title']}")

        confirmed = [e for e in events if e.get('is_confirmed')]
        unconfirmed = [e for e in events if not e.get('is_confirmed')]

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Upcoming", len(events))
        col2.metric("Confirmed", len(confirmed))
        col3.metric("Unconfirmed", len(unconfirmed))

        if events:
            event_df = pd.DataFrame(events)
            event_df['start_time'] = pd.to_datetime(event_df['start_time'])
            event_df = event_df.dropna(subset=['start_time'])

            if not event_df.empty:
                fig = px.timeline(
                    event_df, x_start='start_time', x_end='start_time',
                    y='event_type', color='is_confirmed',
                    hover_data=['title', 'location'],
                    title="Upcoming Events Timeline",
                )
                st.plotly_chart(fig, width="stretch")

        for e in events:
            status_icon = "Confirmed" if e.get('is_confirmed') else "Unconfirmed"
            live_badge = " LIVE" if e.get('is_live') else ""
            with st.expander(f"{status_icon}{live_badge} {e['title']} - {e.get('start_time', 'TBD')}"):
                st.text(f"Type: {e.get('event_type', 'unknown')}")
                st.text(f"Location: {e.get('location', 'TBD')}")
                st.text(f"Start: {e.get('start_time', 'TBD')}")
                if e.get('source_url'):
                    st.markdown(f"[Source]({e['source_url']})")
                if e.get('topics'):
                    st.text(f"Expected topics: {', '.join(e['topics'])}")
    else:
        st.info("No upcoming events found. Click 'Update Events'.")


# ═══════════════════════════════════════════════════════════════════
# LIVE MONITOR TAB
# ═══════════════════════════════════════════════════════════════════
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

    detected = live_status.get('detected_terms', {})
    if detected:
        st.subheader("Detected Terms (Live)")
        total = live_status.get('total_detections', 0)
        st.metric("Total Detections", total)

        det_df = pd.DataFrame([
            {'term': k, 'count': v} for k, v in detected.items()
        ]).sort_values('count', ascending=False)

        fig = px.bar(det_df, x='term', y='count', title="Live Term Detections",
                     color='count', color_continuous_scale='Reds')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, width="stretch")

        st.dataframe(det_df, width="stretch", hide_index=True)
    else:
        st.info("No terms detected yet. Start monitoring and wait for a live speech.")

    st.subheader("Live Feed")
    st.info("When Trump speaks live, detected terms will appear here in real-time.")


# ═══════════════════════════════════════════════════════════════════
# TRUMPGPT TAB — Model status, no ASCII art (moved to Home)
# ═══════════════════════════════════════════════════════════════════
with tab_ml:
    st.header("TrumpGPT")
    st.caption("LoRA fine-tuned Llama-3.1-8B + Multi-Scenario Monte Carlo Simulation Engine")

    model_status = api_get("/model/status") or {}

    if model_status.get('model_name'):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Model", model_status.get('base_model', '').split('/')[-1])
        col2.metric("Terms Tracked", model_status.get('total_terms_tracked', 0))
        col3.metric("Colab Predictions", model_status.get('colab_predictions_count', 0))
        col4.metric("Discovered Phrases", model_status.get('discovered_phrases_count', 0))

        last_run = model_status.get('last_run', '')
        if last_run:
            st.success(f"Last TrumpGPT run: {last_run}")
        else:
            st.warning("TrumpGPT has not run yet. Run the Colab notebooks.")

        new_info = model_status.get('new_iteration_info', {})
        if new_info:
            if new_info.get('would_retrain'):
                st.info(f"Ready for retraining: {new_info['new_speeches_last_24h']} new speeches available.")
            else:
                st.caption(new_info.get('description', ''))

        # Ensemble weights
        st.subheader("Ensemble Signal Weights")
        weights = model_status.get('ensemble_weights', {})
        if weights:
            w_df = pd.DataFrame([
                {'Signal': k.replace('_', ' ').title(), 'Weight': v}
                for k, v in weights.items()
            ]).sort_values('Weight', ascending=True)
            fig = px.bar(w_df, x='Weight', y='Signal', orientation='h',
                         color='Weight', color_continuous_scale='Blues',
                         title='How TrumpGPT Blends Prediction Signals')
            fig.update_layout(height=300)
            st.plotly_chart(fig, width="stretch")

        # Scenario weights
        sc_weights = model_status.get('scenario_weights', {})
        if sc_weights:
            st.subheader("Monte Carlo Scenario Allocation")
            sc_counts = model_status.get('scenario_counts', {})
            sc_df = pd.DataFrame([
                {
                    'Scenario': k.replace('_', ' ').title(),
                    'Weight': v,
                    'Simulations': sc_counts.get(k, 0),
                }
                for k, v in sc_weights.items()
            ])
            fig = px.pie(sc_df, values='Weight', names='Scenario',
                         title='Simulation Budget by Scenario Type',
                         color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, width="stretch")

        # Gemini enrichment
        gemini = model_status.get('gemini_enrichment', {})
        topics = gemini.get('enriched_topics', [])
        hot_words = gemini.get('scenario_hot_words', {})
        if topics or hot_words:
            st.subheader("Gemini Current Events Enrichment")
            if topics:
                st.markdown("**Enriched Topics:**")
                for t in topics:
                    st.markdown(f"- {t}")
            if hot_words:
                st.markdown("**Per-Scenario Hot Words:**")
                for scenario, words in hot_words.items():
                    st.markdown(f"- **{scenario.replace('_', ' ').title()}**: {', '.join(words)}")

        # Top predictions
        top_preds = model_status.get('top_predictions', [])
        if top_preds:
            st.subheader("Top TrumpGPT Predictions")
            tp_df = pd.DataFrame(top_preds)
            st.dataframe(
                tp_df,
                column_config={
                    'term': st.column_config.TextColumn("Term", width="large"),
                    'probability': st.column_config.ProgressColumn("Probability", min_value=0, max_value=1),
                    'recency_weight': st.column_config.NumberColumn("Recency Wt", format="%.2f"),
                },
                width="stretch",
                hide_index=True,
            )

    st.divider()

    # Local ML models
    st.subheader("Local ML Models")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Train Local Models", type="primary", width="stretch"):
            result = api_post("/ml/train")
            if result:
                st.success("Training started.")

    with col2:
        if st.button("Get ML Predictions", width="stretch"):
            ml_preds = api_get("/ml/predictions") or []
            if ml_preds:
                st.session_state['ml_predictions'] = ml_preds

    model_info = api_get("/ml/info") or {}

    if 'results' in model_info:
        for model_name, metrics in model_info.get('results', {}).items():
            if 'error' in metrics:
                st.error(f"{model_name}: {metrics['error']}")
                continue

            with st.expander(f"{model_name} (CV AUC: {metrics.get('cv_auc_mean', 0):.3f})"):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("CV AUC", f"{metrics.get('cv_auc_mean', 0):.3f}")
                col2.metric("Accuracy", f"{metrics.get('train_accuracy', 0):.1%}")
                col3.metric("F1 Score", f"{metrics.get('train_f1', 0):.3f}")
                col4.metric("Brier Score", f"{metrics.get('train_brier', 0):.4f}")

                top_features = metrics.get('top_features', {})
                if top_features:
                    feat_df = pd.DataFrame([
                        {'feature': k, 'importance': v}
                        for k, v in top_features.items()
                    ]).sort_values('importance', ascending=True)
                    fig = px.bar(feat_df, x='importance', y='feature',
                                 orientation='h', title=f"{model_name} Feature Importance")
                    st.plotly_chart(fig, width="stretch")

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
            width="stretch",
            hide_index=True,
        )


# ═══════════════════════════════════════════════════════════════════
# DATA STATS TAB
# ═══════════════════════════════════════════════════════════════════
with tab_data:
    st.header("Data Collection Statistics")

    speech_stats = api_get("/speeches/stats") or {}

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Speeches", speech_stats.get('total_speeches', 0))
    col2.metric("With Transcripts", speech_stats.get('with_transcripts', 0))
    col3.metric("Processed", speech_stats.get('processed', 0))

    st.subheader("Term Frequency Report")
    report = api_get("/terms/report") or []
    if report:
        for item in report[:20]:
            with st.expander(f"{item['term']} - {item['total_occurrences']} occurrences (trend: {item['trend_score']:+.2f})"):
                if item.get('recent_speeches'):
                    for speech in item['recent_speeches']:
                        st.text(f"  [{speech.get('date', '?')}] {speech.get('speech_title', '?')} ({speech['count']}x)")

    st.subheader("System Health")
    health = api_get("/system/health") or {}
    if health.get('status') == 'ok':
        st.success(f"API is healthy - {health.get('timestamp', '')}")
    else:
        st.error("API is not responding")
