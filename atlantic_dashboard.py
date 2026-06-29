"""
United States Top 50 Playlist Performance and Song Popularity Trend Analysis
Unified Mentor Internship Project — Data Analyst Batch Jan 2026
Atlantic Recording Corporation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(
    page_title="Atlantic US Top 50 Analytics",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    df = pd.read_csv("Atlantic_United_States.csv")

    # Force all columns to standard types
    df['date']        = pd.to_datetime(df['date'], dayfirst=True)
    df['position']    = df['position'].astype(int)
    df['song']        = df['song'].astype(str)
    df['artist']      = df['artist'].astype(str)
    df['popularity']  = df['popularity'].astype(int)
    df['duration_ms'] = df['duration_ms'].astype(int)
    df['album_type']  = df['album_type'].astype(str)
    df['total_tracks']= df['total_tracks'].astype(int)
    df['is_explicit'] = df['is_explicit'].astype(bool)

    df = df.sort_values('date').reset_index(drop=True)

    # ── Feature Engineering ───────────────────────────────────────────────────
    # Duration in minutes
    df['duration_min'] = (df['duration_ms'] / 60000).round(2)

    # Days on Chart per song
    days_map = df.groupby('song')['date'].nunique()
    df['days_on_chart'] = df['song'].map(days_map)

    # Average Rank per song
    avg_rank_map = df.groupby('song')['position'].mean()
    df['avg_rank'] = df['song'].map(avg_rank_map).round(2)

    # Best Rank Achieved per song
    best_rank_map = df.groupby('song')['position'].min()
    df['best_rank'] = df['song'].map(best_rank_map)

    # Rank Volatility Index (std dev of rank per song)
    vol_map = df.groupby('song')['position'].std().fillna(0)
    df['rank_volatility'] = df['song'].map(vol_map).round(2)

    # Popularity Trend Score (7-day rolling average per song)
    df = df.sort_values(['song', 'date'])
    df['popularity_trend'] = (
        df.groupby('song')['popularity']
        .transform(lambda x: x.rolling(7, min_periods=1).mean().round(2))
    )
    df = df.sort_values('date').reset_index(drop=True)

    # Time columns
    df['year_month'] = df['date'].dt.to_period('M').astype(str)
    df['week']       = df['date'].dt.to_period('W').astype(str)

    # Content type label
    df['content_type'] = df['is_explicit'].map({True: 'Explicit', False: 'Non-Explicit'})

    return df


df = load_data()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎵 Atlantic US Top 50")
    st.markdown("**Atlantic Recording Corporation**  \nUnified Mentor · Data Analyst Intern · Jan 2026")
    st.divider()
    st.subheader("Filters")

    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    date_range = st.date_input(
        "Date range selector",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    all_artists = sorted(df['artist'].unique().tolist())
    selected_artists = st.multiselect("Artist filter", all_artists, default=[])

    all_songs = sorted(df['song'].unique().tolist())
    selected_songs = st.multiselect("Song filter", all_songs, default=[])

    rank_range = st.slider("Rank range", min_value=1, max_value=50, value=(1, 50))

    album_types = st.multiselect(
        "Album type toggle",
        ["album", "single", "compilation"],
        default=["album", "single", "compilation"]
    )

    st.divider()
    st.caption(f"Data: May 2024 – Nov 2025 · {df['date'].nunique()} snapshot days · {df['song'].nunique()} unique songs")


# ─── Apply Filters ────────────────────────────────────────────────────────────
dff = df.copy()

if len(date_range) == 2:
    s, e = date_range
    dff = dff[(dff['date'].dt.date >= s) & (dff['date'].dt.date <= e)]

if selected_artists:
    dff = dff[dff['artist'].isin(selected_artists)]

if selected_songs:
    dff = dff[dff['song'].isin(selected_songs)]

dff = dff[(dff['position'] >= rank_range[0]) & (dff['position'] <= rank_range[1])]

if album_types:
    dff = dff[dff['album_type'].isin(album_types)]

dff = dff.reset_index(drop=True)

# ─── KPIs ─────────────────────────────────────────────────────────────────────
avg_days       = round(dff.groupby('song')['days_on_chart'].first().mean(), 1) if len(dff) else 0
overall_rank   = round(dff['position'].mean(), 1) if len(dff) else 0
avg_vol        = round(dff.groupby('song')['rank_volatility'].first().mean(), 1) if len(dff) else 0
latest_pop     = round(dff[dff['date'] == dff['date'].max()]['popularity'].mean(), 1) if len(dff) else 0
artist_days    = dff.groupby('artist')['date'].nunique()
top_artist     = artist_days.idxmax() if len(artist_days) else "N/A"
explicit_share = round(dff['is_explicit'].mean() * 100, 1) if len(dff) else 0

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("# 🎵 US Top 50 Playlist Performance & Song Popularity Trend Analysis")
st.markdown(
    f"**Atlantic Recording Corporation** · "
    f"**{dff['date'].min().strftime('%b %Y') if len(dff) else 'N/A'}"
    f" – {dff['date'].max().strftime('%b %Y') if len(dff) else 'N/A'}**"
)
st.divider()

# ─── KPI Cards ───────────────────────────────────────────────────────────────
st.subheader("📊 KPI Summary Cards")
k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.metric("Days on Chart",          f"{avg_days}",       help="Longevity indicator")
with k2:
    st.metric("Average Rank",           f"{overall_rank}",   help="Overall performance")
with k3:
    st.metric("Rank Volatility Index",  f"{avg_vol}",        help="Stability metric")
with k4:
    st.metric("Popularity Score Trend", f"{latest_pop}",     help="Listener engagement")
with k5:
    st.metric("Artist Dominance",       f"{top_artist}",     help="Most dominant artist")
with k6:
    st.metric("Explicit Content Share", f"{explicit_share}%",help="Content strategy insight")

st.divider()

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📅 Playlist Timeline",
    "📈 Song Ranking Trends",
    "🏆 Artist Dominance",
    "⭐ Popularity vs Rank",
    "🔞 Explicit vs Non-Explicit"
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — PLAYLIST TIMELINE EXPLORER
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Playlist Timeline Explorer")

    # Daily rank distribution
    st.markdown("**Daily rank distribution**")
    daily = dff.groupby('date')['position'].mean().reset_index()
    daily.columns = ['date', 'avg_position']
    fig1 = go.Figure(go.Scatter(
        x=daily['date'], y=daily['avg_position'],
        mode='lines', line=dict(color='#1DB954', width=2)
    ))
    fig1.update_layout(
        height=300, template='plotly_white', hovermode='x unified',
        xaxis_title='Date', yaxis_title='Avg Rank',
        yaxis=dict(autorange='reversed'),
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig1, width='stretch')

    # Entry vs exit behavior
    st.markdown("**Entry vs exit — new songs entering chart each month**")
    monthly_new = (
        dff.sort_values('date')
        .groupby(['year_month', 'song'])['date'].min()
        .reset_index()
        .groupby('year_month')['song'].count()
        .reset_index()
    )
    monthly_new.columns = ['year_month', 'new_entries']
    fig2 = go.Figure(go.Bar(
        x=monthly_new['year_month'], y=monthly_new['new_entries'],
        marker_color='#1DB954'
    ))
    fig2.update_layout(
        height=280, template='plotly_white',
        xaxis_title='Month', yaxis_title='New songs entering chart',
        margin=dict(l=10, r=10, t=10, b=10)
    )
    fig2.update_xaxes(tickangle=45, tickfont=dict(size=9))
    st.plotly_chart(fig2, width='stretch')

    # Fast risers
    st.markdown("**Fast risers — songs with biggest rank improvement**")
    song_stats = dff.groupby('song').agg(
        avg_rank=('position', 'mean'),
        best_rank=('position', 'min')
    ).reset_index()
    song_stats['rank_improvement'] = (song_stats['avg_rank'] - song_stats['best_rank']).round(1)
    fast_risers = song_stats.sort_values('rank_improvement', ascending=False).head(15)
    fig3 = go.Figure(go.Bar(
        x=fast_risers['rank_improvement'],
        y=fast_risers['song'],
        orientation='h', marker_color='#1DB954'
    ))
    fig3.update_layout(
        height=420, template='plotly_white',
        xaxis_title='Rank improvement (avg − best rank)',
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig3, width='stretch')


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — SONG RANKING TREND CHARTS
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Song Ranking Trend Charts")

    # Longest playlist presence
    st.markdown("**Songs with longest playlist presence**")
    longevity = dff.groupby('song')['days_on_chart'].first().reset_index()
    longevity = longevity.sort_values('days_on_chart', ascending=False).head(20)
    fig4 = go.Figure(go.Bar(
        x=longevity['days_on_chart'], y=longevity['song'],
        orientation='h', marker_color='#378ADD'
    ))
    fig4.update_layout(
        height=500, template='plotly_white',
        xaxis_title='Days on Chart', margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig4, width='stretch')

    # Highest average popularity
    st.markdown("**Songs with highest average popularity**")
    top_pop = dff.groupby('song')['popularity'].mean().reset_index()
    top_pop.columns = ['song', 'avg_popularity']
    top_pop = top_pop.sort_values('avg_popularity', ascending=False).head(20)
    fig5 = go.Figure(go.Bar(
        x=top_pop['avg_popularity'].round(1), y=top_pop['song'],
        orientation='h', marker_color='#D85A30'
    ))
    fig5.update_layout(
        height=500, template='plotly_white',
        xaxis_title='Average Popularity Score', margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig5, width='stretch')

    # Peak rank vs longevity
    st.markdown("**Peak rank vs longevity comparison**")
    pr_lon = dff.groupby('song').agg(
        best_rank=('position', 'min'),
        days_on_chart=('date', 'nunique'),
        avg_popularity=('popularity', 'mean')
    ).reset_index()
    fig6 = px.scatter(
        pr_lon, x='best_rank', y='days_on_chart',
        size='avg_popularity', color='avg_popularity',
        hover_name='song',
        color_continuous_scale='Viridis',
        labels={'best_rank': 'Best Rank Achieved', 'days_on_chart': 'Days on Chart'}
    )
    fig6.update_layout(height=420, template='plotly_white', margin=dict(l=10, r=10, t=10, b=10))
    fig6.update_xaxes(autorange='reversed')
    st.plotly_chart(fig6, width='stretch')

    # Rank Volatility Index
    st.markdown("**Rank Volatility Index — most volatile songs**")
    vol = dff.groupby('song')['rank_volatility'].first().reset_index()
    vol = vol.sort_values('rank_volatility', ascending=False).head(20)
    fig7 = go.Figure(go.Bar(
        x=vol['rank_volatility'], y=vol['song'],
        orientation='h', marker_color='#E24B4A'
    ))
    fig7.update_layout(
        height=500, template='plotly_white',
        xaxis_title='Rank Volatility Index (Std Dev)', margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig7, width='stretch')


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — ARTIST DOMINANCE LEADERBOARD
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Artist Dominance Leaderboard")

    # Unique songs per artist
    st.markdown("**Number of unique songs per artist (Top 20)**")
    art_songs = dff.groupby('artist')['song'].nunique().reset_index()
    art_songs.columns = ['artist', 'unique_songs']
    art_songs = art_songs.sort_values('unique_songs', ascending=False).head(20)
    fig8 = go.Figure(go.Bar(
        x=art_songs['unique_songs'], y=art_songs['artist'],
        orientation='h', marker_color='#8B5CF6'
    ))
    fig8.update_layout(
        height=500, template='plotly_white',
        xaxis_title='Unique Songs on Playlist', margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig8, width='stretch')

    # Total days on playlist
    st.markdown("**Total days artist appears on playlist (Top 20)**")
    art_days = dff.groupby('artist')['date'].nunique().reset_index()
    art_days.columns = ['artist', 'total_days']
    art_days = art_days.sort_values('total_days', ascending=False).head(20)
    fig9 = go.Figure(go.Bar(
        x=art_days['total_days'], y=art_days['artist'],
        orientation='h', marker_color='#1D9E75'
    ))
    fig9.update_layout(
        height=500, template='plotly_white',
        xaxis_title='Total Days on Playlist', margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig9, width='stretch')

    # Artist dominance over time
    st.markdown("**Artist dominance over time — top 5 artists monthly presence**")
    top5 = art_days.head(5)['artist'].tolist()
    art_time = (
        dff[dff['artist'].isin(top5)]
        .groupby(['year_month', 'artist'])['date']
        .nunique().reset_index()
    )
    art_time.columns = ['year_month', 'artist', 'days_present']
    fig10 = px.line(
        art_time, x='year_month', y='days_present', color='artist',
        labels={'year_month': 'Month', 'days_present': 'Days on Chart'}
    )
    fig10.update_layout(
        height=380, template='plotly_white', hovermode='x unified',
        margin=dict(l=10, r=10, t=10, b=10)
    )
    fig10.update_xaxes(tickangle=45, tickfont=dict(size=9))
    st.plotly_chart(fig10, width='stretch')


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — POPULARITY VS RANK SCATTER PLOTS
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Popularity vs Rank Scatter Plots")

    # Popularity vs rank correlation
    st.markdown("**Popularity vs rank — correlation**")
    fig11 = px.scatter(
        dff, x='position', y='popularity',
        color='album_type', hover_name='song', opacity=0.5,
        labels={'position': 'Playlist Rank', 'popularity': 'Popularity Score', 'album_type': 'Album Type'}
    )
    fig11.update_layout(height=400, template='plotly_white', margin=dict(l=10, r=10, t=10, b=10))
    fig11.update_xaxes(autorange='reversed')
    st.plotly_chart(fig11, width='stretch')

    # Popularity distribution across Top 10, Top 20, Top 50
    st.markdown("**Popularity distribution across Top 10, Top 20, Top 50**")
    dff2 = dff.copy()
    dff2['rank_tier'] = pd.cut(
        dff2['position'], bins=[0, 10, 20, 50],
        labels=['Top 10', 'Top 11–20', 'Top 21–50']
    )
    fig12 = px.box(
        dff2, x='rank_tier', y='popularity', color='rank_tier',
        labels={'rank_tier': 'Rank Tier', 'popularity': 'Popularity Score'}
    )
    fig12.update_layout(
        height=380, template='plotly_white',
        margin=dict(l=10, r=10, t=10, b=10), showlegend=False
    )
    st.plotly_chart(fig12, width='stretch')

    # Popularity stability vs chart volatility
    st.markdown("**Popularity stability vs chart volatility — per song**")
    stab = dff.groupby('song').agg(
        pop_std=('popularity', 'std'),
        rank_vol=('position', 'std'),
        avg_pop=('popularity', 'mean')
    ).reset_index().fillna(0)
    fig13 = px.scatter(
        stab, x='rank_vol', y='pop_std',
        size='avg_pop', hover_name='song', color='avg_pop',
        color_continuous_scale='Blues',
        labels={'rank_vol': 'Rank Volatility', 'pop_std': 'Popularity Std Dev', 'avg_pop': 'Avg Popularity'}
    )
    fig13.update_layout(height=400, template='plotly_white', margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig13, width='stretch')


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — EXPLICIT VS NON-EXPLICIT
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Explicit vs Non-Explicit Performance Panel")

    # Avg popularity
    st.markdown("**Explicit vs non-explicit — average popularity score**")
    exp_pop = dff.groupby('content_type')['popularity'].mean().reset_index()
    fig14 = go.Figure(go.Bar(
        x=exp_pop['content_type'], y=exp_pop['popularity'].round(1),
        marker_color=['#E24B4A', '#1D9E75']
    ))
    fig14.update_layout(
        height=300, template='plotly_white',
        yaxis_title='Average Popularity Score', margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig14, width='stretch')

    # Avg rank
    st.markdown("**Explicit vs non-explicit — average rank**")
    exp_rank = dff.groupby('content_type')['position'].mean().reset_index()
    fig15 = go.Figure(go.Bar(
        x=exp_rank['content_type'], y=exp_rank['position'].round(1),
        marker_color=['#E24B4A', '#1D9E75']
    ))
    fig15.update_layout(
        height=300, template='plotly_white',
        yaxis_title='Average Rank (lower = better)', margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig15, width='stretch')

    # Single vs album track comparison
    st.markdown("**Single vs album track — popularity and rank comparison**")
    alb_stats = dff.groupby('album_type').agg(
        avg_popularity=('popularity', 'mean'),
        avg_rank=('position', 'mean'),
        song_count=('song', 'nunique')
    ).reset_index().round(1)
    fig16 = px.bar(
        alb_stats, x='album_type', y='avg_popularity', color='album_type',
        labels={'album_type': 'Album Type', 'avg_popularity': 'Avg Popularity'}
    )
    fig16.update_layout(
        height=300, template='plotly_white',
        margin=dict(l=10, r=10, t=10, b=10), showlegend=False
    )
    st.plotly_chart(fig16, width='stretch')

    # Song duration impact on popularity and rank
    st.markdown("**Song duration impact on popularity and rank**")
    fig17 = px.scatter(
        dff, x='duration_min', y='popularity', color='content_type',
        opacity=0.4, hover_name='song',
        labels={'duration_min': 'Duration (minutes)', 'popularity': 'Popularity Score', 'content_type': 'Content Type'}
    )
    fig17.update_layout(height=380, template='plotly_white', margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig17, width='stretch')

    # Album size vs song success
    st.markdown("**Album size (total tracks) vs song success**")
    fig18 = px.scatter(
        dff, x='total_tracks', y='popularity', color='album_type',
        opacity=0.4, hover_name='song',
        labels={'total_tracks': 'Total Tracks in Album', 'popularity': 'Popularity Score', 'album_type': 'Album Type'}
    )
    fig18.update_layout(height=380, template='plotly_white', margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig18, width='stretch')


# ─── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='font-size:12px;color:#999;text-align:center;'>"
    "US Top 50 Playlist Analytics · Atlantic Recording Corporation · "
    "Unified Mentor Internship Jan 2026 Batch · Built with Streamlit & Plotly"
    "</p>",
    unsafe_allow_html=True
)