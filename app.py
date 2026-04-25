import math
import socket
import dash
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd


class CallableDash(dash.Dash):
    """Dash app that also acts as a WSGI callable for Gunicorn."""

    def __call__(self, environ, start_response):
        return self.server(environ, start_response)

# ==========================================
# 1. Data Processing & Performance Setup
# ==========================================

file_path = "student_mental_health_burnout_5K.csv"
try:
    df_full = pd.read_csv(file_path)
except FileNotFoundError:
    # Fallback to the file in the workspace just in case
    df_full = pd.read_csv("student_mental_health_burnout_100K.csv")

# Keep only the 10 requested features directly in program memory.
REQUESTED_COLUMNS = [
    'burnout_score',
    'age',
    'sleep_hours',
    'physical_activity',
    'academic_year',
    'stress_level',
    'gender',
    'screen_time',
    'anxiety_score',
    'depression_score'
]
missing_columns = [col for col in REQUESTED_COLUMNS if col not in df_full.columns]
if missing_columns:
    raise ValueError(f"Missing required columns in dataset: {missing_columns}")

df_full = df_full[REQUESTED_COLUMNS].copy()

# Keep tail behavior visible by stratifying the sample on burnout quantiles.
def stratified_sample_by_burnout(data, n=5000, quantiles=20, random_state=1):
    if len(data) <= n:
        return data.copy()

    data = data.copy()
    q = max(2, min(quantiles, data['burnout_score'].nunique()))
    data['_burnout_bin'] = pd.qcut(data['burnout_score'], q=q, duplicates='drop')

    sampled_chunks = []
    for _, group in data.groupby('_burnout_bin', observed=False):
        take_n = max(1, int(np.floor(n * len(group) / len(data))))
        take_n = min(take_n, len(group))
        sampled_chunks.append(group.sample(n=take_n, random_state=random_state))

    sampled = pd.concat(sampled_chunks, axis=0).drop(columns=['_burnout_bin'])
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=random_state)
    elif len(sampled) < n:
        deficit = n - len(sampled)
        remainder = data.drop(index=sampled.index).drop(columns=['_burnout_bin'])
        sampled = pd.concat([sampled, remainder.sample(n=deficit, random_state=random_state)])

    return sampled.copy()


df = stratified_sample_by_burnout(df_full, n=5000, random_state=1)
df = df.reset_index(drop=True)
df['_row_id'] = np.arange(len(df))

# Ensure academic_year is treated as a categorical/string variable
df['academic_year'] = df['academic_year'].astype(str)

NUMERIC_FEATURES = [
    col for col in df.select_dtypes(include='number').columns
    if col not in {'burnout_score', '_row_id'}
]
DEFAULT_MASTER_X = 'stress_level' if 'stress_level' in NUMERIC_FEATURES else NUMERIC_FEATURES[0]
DEFAULT_HEATMAP_Y = 'stress_level' if 'stress_level' in NUMERIC_FEATURES else NUMERIC_FEATURES[0]


def format_feature_name(feature_name):
    return feature_name.replace('_', ' ').title()


def format_numeric_label(value):
    return f"{float(value):.2f}".rstrip('0').rstrip('.')


def spearman_corr_no_scipy(x_values, y_values):
    """Compute Spearman correlation via ranked Pearson correlation (no SciPy dependency)."""
    pair_df = pd.DataFrame({'x': x_values, 'y': y_values}).dropna()
    if len(pair_df) < 2:
        return np.nan
    if pair_df['x'].nunique() <= 1 or pair_df['y'].nunique() <= 1:
        return np.nan
    return pair_df['x'].rank(method='average').corr(pair_df['y'].rank(method='average'))


def get_filtered_df(selected_data):
    if selected_data and selected_data.get('points'):
        selected_row_ids = []
        for point in selected_data['points']:
            custom_data = point.get('customdata')
            if custom_data and len(custom_data) > 0:
                selected_row_ids.append(int(custom_data[0]))

        if selected_row_ids:
            return df[df['_row_id'].isin(selected_row_ids)].copy()

        selected_indices = [point['pointIndex'] for point in selected_data['points'] if 'pointIndex' in point]
        if selected_indices:
            safe_indices = [idx for idx in selected_indices if 0 <= idx < len(df)]
            return df.iloc[safe_indices].copy()

    return df.copy()


def compute_feature_impact(input_df):
    rows = []
    for feature in NUMERIC_FEATURES:
        pair_df = input_df[[feature, 'burnout_score']].dropna()
        if len(pair_df) < 30 or pair_df[feature].nunique() < 4:
            continue

        spearman_corr = spearman_corr_no_scipy(pair_df[feature], pair_df['burnout_score'])
        low_threshold = pair_df[feature].quantile(0.10)
        high_threshold = pair_df[feature].quantile(0.90)

        low_burnout_mean = pair_df.loc[pair_df[feature] <= low_threshold, 'burnout_score'].mean()
        high_burnout_mean = pair_df.loc[pair_df[feature] >= high_threshold, 'burnout_score'].mean()
        burnout_lift = high_burnout_mean - low_burnout_mean

        rows.append({
            'feature': feature,
            'feature_label': format_feature_name(feature),
            'spearman_corr': spearman_corr,
            'burnout_lift': burnout_lift,
            'direction': 'Increases burnout' if burnout_lift >= 0 else 'Decreases burnout'
        })

    if not rows:
        return pd.DataFrame(columns=['feature', 'feature_label', 'spearman_corr', 'burnout_lift', 'direction'])

    impact_df = pd.DataFrame(rows)
    impact_df = impact_df.sort_values(['burnout_lift', 'spearman_corr'], ascending=False)
    return impact_df



# Burnout level bins and color map for Master View
def burnout_level(score):
    if score >= 6:
        return "High"
    elif score >= 2:
        return "Medium"
    else:
        return "Low"

BURNOUT_LEVEL_COLORS = {"Low": "#029820", "Medium": "#f59e0b", "High": "#d62728"}
STRESS_BIN_SIZE = 1.0
STRESS_MIN = 0.0
STRESS_MAX = 10.0
HEATMAP_GENERAL_BIN_COUNT = 8
BURNOUT_COLOR_MIN = 1.0
BURNOUT_COLOR_MAX = 8.0


def style_figure(fig, dark_mode=False):
    theme = {
        'font': '#12212f',
        'grid': 'rgba(18, 33, 47, 0.12)',
        'zero': 'rgba(18, 33, 47, 0.16)',
        'paper': 'rgba(0,0,0,0)'
    }
    if dark_mode:
        theme = {
            'font': '#d9e4ee',
            'grid': 'rgba(217, 228, 238, 0.16)',
            'zero': 'rgba(217, 228, 238, 0.24)',
            'paper': 'rgba(0,0,0,0)'
        }

    fig.update_layout(
        template='plotly_white',
        font=dict(family='Manrope, IBM Plex Sans, Avenir Next, sans-serif', color=theme['font']),
        title_font=dict(size=18),
        paper_bgcolor=theme['paper'],
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=42, r=18, t=52, b=34),
        transition={'duration': 650, 'easing': 'cubic-in-out'}
    )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=theme['grid'], zeroline=True, zerolinecolor=theme['zero'])
    fig.update_xaxes(showgrid=False, zeroline=False)
    return fig


def graph_card(title, subtitle, graph_id, controls=None, animate_graph=True):
    controls = controls or []
    return html.Div([
        html.Div([
            html.H5(title, className='card-title'),
            html.P(subtitle, className='card-subtitle'),
            *controls
        ], className='card-header-lite'),
        dcc.Graph(
            id=graph_id,
            className='graph-shell',
            config={'displayModeBar': False},
            animate=animate_graph,
            animation_options={
                'transition': {'duration': 650, 'easing': 'cubic-in-out'},
                'frame': {'duration': 450}
            }
        )
    ], className='viz-card')


def find_available_port(start_port=8050, max_tries=50):
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(('127.0.0.1', port)) != 0:
                return port
    return start_port

# ==========================================
# 2. UI Layout & Design Guidelines
# ==========================================

# Initialize Dash app with a clean, professional Bootstrap theme
app = CallableDash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server
app.title = "Student Burnout Analysis"

app.layout = html.Div([
    dcc.Store(id='kpi-target-store', data={'sample': len(df), 'selected': len(df), 'avg_burnout': float(df['burnout_score'].mean())}),
    dcc.Store(id='kpi-current-store', data={'sample': 0, 'selected': 0, 'avg_burnout': 0.0}),
    dcc.Interval(id='kpi-interval', interval=30, n_intervals=0, disabled=False),

    html.Div([
        html.Aside([
            html.Div([
                html.Div('BI', className='brand-mark'),
                html.Div([
                    html.H4('BurnoutIQ', className='brand-title'),
                    html.P('Decision Console', className='brand-subtitle')
                ])
            ], className='brand-shell'),

            html.Nav([
                html.A('Overview', href='#section-overview', className='nav-link'),
                html.A('Primary Insights', href='#section-insights', className='nav-link'),
                html.A('Diagnostics', href='#section-diagnostics', className='nav-link')
            ], className='side-nav'),

            html.Div([
                html.Div('Theme', className='theme-title'),
                dbc.Switch(id='theme-toggle', value=False, className='theme-switch'),
                html.Div('Light Mode', id='theme-label', className='theme-label')
            ], className='theme-panel')
        ], className='sidebar-shell'),

        html.Main([
            dbc.Container([
                html.Div([
                    html.Div([
                        html.Span('Burnout Intelligence Dashboard', className='eyebrow'),
                        html.H1('Student Burnout Drivers', className='hero-title'),
                        html.P(
                            'What actually causes burnout among students? Explore the data!',
                            className='hero-subtitle'
                        )
                    ], className='hero-main'),
                    html.Div([
                        html.Div([
                            html.P('Sample Size', className='kpi-label'),
                            html.H3('0', id='kpi-sample-size', className='kpi-value')
                        ], className='kpi-box'),
                        html.Div([
                            html.P('Selected Students', className='kpi-label'),
                            html.H3('0', id='kpi-selected-size', className='kpi-value')
                        ], className='kpi-box'),
                        html.Div([
                            html.P('Avg Burnout (Selection)', className='kpi-label'),
                            html.H3('0.00', id='kpi-avg-burnout', className='kpi-value')
                        ], className='kpi-box')
                    ], className='kpi-row')
                ], id='section-overview', className='hero-shell'),

                html.Div([
                    html.Div([
                        html.H4('Primary Insights', className='section-title'),
                        html.P(
                            'Use lasso or box selection in the scatter to recompute drivers for a specific subgroup.',
                            className='section-subtitle'
                        )
                    ], className='section-head'),
                    dbc.Row([
                        dbc.Col(
                            graph_card(
                                'Feature Impact on Burnout',
                                'How each feature shifts burnout: average in top 10% minus bottom 10% of that feature',
                                'impact-plot'
                            ),
                            xs=12,
                            lg=6
                        ),
                        dbc.Col(
                            graph_card(
                                'Master View',
                                'Choose x-axis feature to find out correlation with burnout',
                                'scatter-plot',
                                controls=[
                                    html.Div(
                                        [
                                            dcc.Dropdown(
                                                id='master-x-feature',
                                                options=[
                                                    {'label': format_feature_name(col), 'value': col}
                                                    for col in NUMERIC_FEATURES
                                                ],
                                                value=DEFAULT_MASTER_X,
                                                clearable=False,
                                                className='feature-dropdown',
                                                style={'flex': '1 1 240px'}
                                            ),
                                            dbc.Button(
                                                'Reset',
                                                id='master-x-reset',
                                                color='secondary',
                                                outline=True,
                                                size='sm',
                                                style={'whiteSpace': 'nowrap'}
                                            )
                                        ],
                                        style={
                                            'display': 'flex',
                                            'gap': '8px',
                                            'alignItems': 'center',
                                            'marginTop': '10px',
                                            'flexWrap': 'wrap'
                                        }
                                    ),
                                    html.Div(
                                        'Correlation Trend: Near Flat',
                                        id='master-trend-banner',
                                        className='master-trend-banner trend-neutral'
                                    )
                                ],
                                animate_graph=False
                            ),
                            xs=12,
                            lg=6
                        )
                    ], className='g-3')
                ], id='section-insights', className='section-shell'),

                html.Div([
                    html.Div([
                        html.H4('Diagnostic Views', className='section-title'),
                        html.P('Context charts to explain why selected groups behave differently.', className='section-subtitle')
                    ], className='section-head'),
                    dbc.Row([
                        dbc.Col(
                            graph_card(
                                'Burnout Heatmap',
                                'Average burnout by selected y-axis feature and academic year',
                                'stress-heatmap-plot',
                                controls=[
                                    html.Div(
                                        [
                                            dcc.Dropdown(
                                                id='heatmap-y-feature',
                                                options=[
                                                    {'label': format_feature_name(col), 'value': col}
                                                    for col in NUMERIC_FEATURES
                                                ],
                                                value=DEFAULT_HEATMAP_Y,
                                                clearable=False,
                                                className='feature-dropdown',
                                                style={'flex': '1 1 240px'}
                                            ),
                                            dbc.Button(
                                                'Reset',
                                                id='heatmap-y-reset',
                                                color='secondary',
                                                outline=True,
                                                size='sm',
                                                style={'whiteSpace': 'nowrap'}
                                            )
                                        ],
                                        style={
                                            'display': 'flex',
                                            'gap': '8px',
                                            'alignItems': 'center',
                                            'marginTop': '10px',
                                            'flexWrap': 'wrap'
                                        }
                                    )
                                ]
                            ),
                            xs=12,
                            lg=6
                        ),
                        dbc.Col(graph_card('Sleep Distribution', 'Selected subgroup compared to all students', 'histogram-plot'), xs=12, lg=6),
                        dbc.Col(graph_card('Average Burnout by Year', 'Mean burnout score for selected population', 'bar-plot'), xs=12, lg=6),
                        dbc.Col(graph_card('Mental Health Distribution', 'Anxiety and depression spread in the selection', 'box-plot'), xs=12, lg=6)
                    ], className='g-3')
                ], id='section-diagnostics', className='section-shell mb-4')
            ], fluid=True, className='app-shell')
        ], className='main-shell')
    ], id='theme-root', className='theme-root theme-light dashboard-shell')
], className='page-shell')

# ==========================================
# 3 & 4. Callback Logic & Chart Generation
# ==========================================

@app.callback(
    [Output('histogram-plot', 'figure'),
     Output('bar-plot', 'figure'),
     Output('box-plot', 'figure'),
     Output('impact-plot', 'figure'),
     Output('stress-heatmap-plot', 'figure'),
    Output('kpi-target-store', 'data')],
    [Input('scatter-plot', 'selectedData'),
    Input('heatmap-y-feature', 'value'),
    Input('theme-toggle', 'value')]
)
def update_linked_charts(selectedData, heatmap_y_feature, dark_mode):
    """
    Brushing and Linking Logic:
    This callback triggers whenever a user selects points on the master scatter plot.
    If points are selected, it filters the 5,000-row DataFrame to only include the brushed students.
    If nothing is selected (default state), it uses the full 5,000-row DataFrame.
    """
    
    filtered_df = get_filtered_df(selectedData)

    # --------------------------------------------------------- 
    # Chart 2: Sleep Histogram (Linked, 1-hour bins, x=sleep_hours, y=count)
    # ---------------------------------------------------------
    min_sleep = int(np.floor(min(df['sleep_hours'].min(), filtered_df['sleep_hours'].min())))
    max_sleep = int(np.ceil(max(df['sleep_hours'].max(), filtered_df['sleep_hours'].max())))
    bins = list(range(min_sleep, max_sleep + 2))  # +2 to include last edge

    fig_hist = go.Figure()
    # Global population distribution (static, light gray)
    fig_hist.add_trace(go.Histogram(
        x=df['sleep_hours'],
        name='All Students',
        marker_color='lightgray',
        opacity=0.3,
        xbins=dict(start=min_sleep, end=max_sleep+1, size=1)
    ))
    # Selected population distribution (updates based on brush, prominent color)
    fig_hist.add_trace(go.Histogram(
        x=filtered_df['sleep_hours'],
        name='Selected Students',
        marker_color='steelblue',
        xbins=dict(start=min_sleep, end=max_sleep+1, size=1)
    ))
    # barmode='overlay' puts the selected active histogram over the static global one
    fig_hist.update_layout(
        title="Distribution of Sleep Hours",
        xaxis_title="Sleep Hours",
        yaxis_title="Count",
        barmode='overlay',
        template="plotly_white"
    )
    fig_hist = style_figure(fig_hist, dark_mode)

    # ---------------------------------------------------------
    # Chart 3: Average Burnout by Year (Linked)
    # ---------------------------------------------------------
    burnout_by_year_df = filtered_df.groupby('academic_year', as_index=False)['burnout_score'].mean()
    burnout_by_year_df['academic_year_num'] = pd.to_numeric(burnout_by_year_df['academic_year'], errors='coerce')
    burnout_by_year_df = burnout_by_year_df.sort_values(['academic_year_num', 'academic_year'])
    
    fig_bar = px.bar(
        burnout_by_year_df,
        x='academic_year', 
        y='burnout_score',
        title="Average Burnout by Academic Year",
        labels={'academic_year': 'Academic Year', 'burnout_score': 'Avg Burnout Score'},
        template="plotly_white",
        color_discrete_sequence=['#17becf']
    )
    fig_bar.update_xaxes(showgrid=False, type='category')
    fig_bar = style_figure(fig_bar, dark_mode)

    # ---------------------------------------------------------
    # Chart 4: Mental Health Box Plots (Linked)
    # ---------------------------------------------------------
    # We need to present Anxiety and Depression side-by-side. 
    # Best way is to melt the dataframe so we have a 'Metric' column and a 'Score' column
    melted_df = filtered_df.melt(
        id_vars=['academic_year'], 
        value_vars=['anxiety_score', 'depression_score'], 
        var_name='Mental Health Metric', 
        value_name='Score'
    )
    
    # Rename for cleaner UI
    melted_df['Mental Health Metric'] = melted_df['Mental Health Metric'].str.replace('_score', '').str.title()
    
    fig_box = px.box(
        melted_df, 
        x='Mental Health Metric', 
        y='Score', 
        color='Mental Health Metric',
        title="Mental Health Scores Distribution",
        template="plotly_white"
    )
    fig_box.update_layout(showlegend=False)
    fig_box = style_figure(fig_box, dark_mode)

    # ---------------------------------------------------------
    # Chart 5: Burnout Driver Ranking (Linked)
    # ---------------------------------------------------------
    impact_df = compute_feature_impact(filtered_df)
    impact_display_df = impact_df.head(10)

    fig_impact = px.bar(
        impact_display_df.sort_values('burnout_lift'),
        x='burnout_lift',
        y='feature_label',
        orientation='h',
        color='direction',
        color_discrete_map={
            'Increases burnout': '#d62728',
            'Decreases burnout': '#1f77b4'
        },
        title='Feature Impact on Burnout (Top 10% - Bottom 10%)',
        labels={
            'burnout_lift': 'Burnout Difference (Top 10% - Bottom 10%)',
            'feature_label': 'Feature',
            'direction': 'Effect'
        },
        template='plotly_white'
    )
    fig_impact.add_vline(x=0, line_dash='dash', line_color='gray')
    fig_impact.update_layout(legend_title='')
    fig_impact.update_yaxes(showgrid=False)
    fig_impact = style_figure(fig_impact, dark_mode)

    # ---------------------------------------------------------
    # Chart 6: Burnout Heatmap by selected y-axis feature and Academic Year
    # ---------------------------------------------------------
    selected_heatmap_y = heatmap_y_feature if heatmap_y_feature in NUMERIC_FEATURES else DEFAULT_HEATMAP_Y
    heatmap_y_label = format_feature_name(selected_heatmap_y)

    heatmap_source = filtered_df[[selected_heatmap_y, 'academic_year', 'burnout_score']].copy()
    heatmap_source[selected_heatmap_y] = pd.to_numeric(heatmap_source[selected_heatmap_y], errors='coerce')
    heatmap_source = heatmap_source.dropna(subset=[selected_heatmap_y, 'academic_year', 'burnout_score'])

    heatmap_bin_labels = []
    if selected_heatmap_y == 'stress_level':
        heatmap_source[selected_heatmap_y] = heatmap_source[selected_heatmap_y].clip(lower=STRESS_MIN, upper=STRESS_MAX)
        stress_bin_edges = np.arange(STRESS_MIN, STRESS_MAX + STRESS_BIN_SIZE, STRESS_BIN_SIZE)
        if stress_bin_edges[-1] < STRESS_MAX:
            stress_bin_edges = np.append(stress_bin_edges, STRESS_MAX)
        heatmap_bin_labels = [
            f"{int(left) if float(left).is_integer() else left:g}-{int(right) if float(right).is_integer() else right:g}"
            for left, right in zip(stress_bin_edges[:-1], stress_bin_edges[1:])
        ]
        heatmap_source['heatmap_y_bin'] = pd.cut(
            heatmap_source[selected_heatmap_y],
            bins=stress_bin_edges,
            labels=heatmap_bin_labels,
            include_lowest=True,
            right=True
        )
    else:
        unique_values = np.sort(heatmap_source[selected_heatmap_y].dropna().unique())
        if len(unique_values) <= 12:
            heatmap_source['heatmap_y_bin'] = heatmap_source[selected_heatmap_y].apply(
                lambda value: format_numeric_label(value) if pd.notna(value) else np.nan
            )
            heatmap_bin_labels = sorted(
                list({format_numeric_label(value) for value in unique_values}),
                key=lambda value: float(value)
            )
        else:
            min_value = float(heatmap_source[selected_heatmap_y].min())
            max_value = float(heatmap_source[selected_heatmap_y].max())
            if math.isclose(min_value, max_value):
                min_value -= 0.5
                max_value += 0.5
            generic_bin_edges = np.linspace(min_value, max_value, HEATMAP_GENERAL_BIN_COUNT + 1)
            generic_bin_edges = np.unique(generic_bin_edges)
            if len(generic_bin_edges) < 2:
                generic_bin_edges = np.array([min_value - 0.5, max_value + 0.5])
            heatmap_bin_labels = [
                f"{format_numeric_label(left)}-{format_numeric_label(right)}"
                for left, right in zip(generic_bin_edges[:-1], generic_bin_edges[1:])
            ]
            heatmap_source['heatmap_y_bin'] = pd.cut(
                heatmap_source[selected_heatmap_y],
                bins=generic_bin_edges,
                labels=heatmap_bin_labels,
                include_lowest=True,
                right=True
            )

    heatmap_df = heatmap_source.pivot_table(
        index='heatmap_y_bin',
        columns='academic_year',
        values='burnout_score',
        aggfunc='mean',
        observed=False
    )

    if not heatmap_df.empty:
        heatmap_df = heatmap_df.reindex(heatmap_bin_labels)
        heatmap_df = heatmap_df.reindex(sorted(heatmap_df.columns, key=lambda x: (pd.to_numeric(x, errors='coerce'), x)), axis=1)
        fig_heatmap = px.imshow(
            heatmap_df,
            labels={'x': 'Academic Year', 'y': heatmap_y_label, 'color': 'Avg Burnout'},
            color_continuous_scale='YlOrRd',
            aspect='auto',
            title=f'Average Burnout: {heatmap_y_label} x Academic Year',
            zmin=BURNOUT_COLOR_MIN,
            zmax=BURNOUT_COLOR_MAX,
            origin='lower'
        )
        fig_heatmap.update_coloraxes(
            cmin=BURNOUT_COLOR_MIN,
            cmax=BURNOUT_COLOR_MAX,
            colorbar=dict(tickmode='array', tickvals=[1, 2, 3, 4, 5, 6, 7, 8])
        )
    else:
        fig_heatmap = go.Figure()
        fig_heatmap.update_layout(
            title=f'Average Burnout: {heatmap_y_label} x Academic Year',
            xaxis_title='Academic Year',
            yaxis_title=heatmap_y_label,
            annotations=[
                dict(
                    text='Not enough data for heatmap',
                    x=0.5,
                    y=0.5,
                    xref='paper',
                    yref='paper',
                    showarrow=False,
                    font=dict(size=15)
                )
            ]
        )
    fig_heatmap = style_figure(fig_heatmap, dark_mode)

    kpi_target = {
        'sample': int(len(df)),
        'selected': int(len(filtered_df)),
        'avg_burnout': float(filtered_df['burnout_score'].mean())
    }

    return fig_hist, fig_bar, fig_box, fig_impact, fig_heatmap, kpi_target


@app.callback(
    Output('master-x-feature', 'value'),
    Input('master-x-reset', 'n_clicks'),
    prevent_initial_call=True
)
def reset_master_x_axis(_):
    return DEFAULT_MASTER_X


@app.callback(
    Output('heatmap-y-feature', 'value'),
    Input('heatmap-y-reset', 'n_clicks'),
    prevent_initial_call=True
)
def reset_heatmap_y_axis(_):
    return DEFAULT_HEATMAP_Y


@app.callback(
    [Output('scatter-plot', 'figure'),
     Output('master-trend-banner', 'children'),
     Output('master-trend-banner', 'className')],
    [Input('scatter-plot', 'id'),
     Input('master-x-feature', 'value'),
     Input('theme-toggle', 'value')]
)
def generate_master_scatter(_, selected_x_feature, dark_mode):
    # ---------------------------------------------------------
    # Chart 1: The Master Scatter Plot (The Source)
    # ---------------------------------------------------------
    x_feature = selected_x_feature if selected_x_feature in NUMERIC_FEATURES else DEFAULT_MASTER_X

    hover_cols = [col for col in ['sleep_hours', 'screen_time', 'anxiety_score', 'depression_score'] if col != x_feature]

    plot_cols = [x_feature, 'burnout_score', '_row_id', *hover_cols]
    plot_df = df[plot_cols].copy()
    plot_df[x_feature] = pd.to_numeric(plot_df[x_feature], errors='coerce')
    plot_df['burnout_score'] = pd.to_numeric(plot_df['burnout_score'], errors='coerce')
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_feature, 'burnout_score'])
    plot_df['burnout_level'] = plot_df['burnout_score'].apply(burnout_level)
    plot_df['burnout_level'] = pd.Categorical(
        plot_df['burnout_level'],
        categories=['High', 'Medium', 'Low'],
        ordered=True
    )

    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"{format_feature_name(x_feature)} vs Burnout Score",
            xaxis_title=format_feature_name(x_feature),
            yaxis_title='Burnout Score',
            annotations=[
                dict(
                    text='No valid data for selected feature',
                    x=0.5,
                    y=0.5,
                    xref='paper',
                    yref='paper',
                    showarrow=False,
                    font=dict(size=15)
                )
            ]
        )
        return style_figure(fig, dark_mode), 'Correlation Trend: Near Flat', 'master-trend-banner trend-neutral'

    fig = px.scatter(
        plot_df,
        x=x_feature,
        y='burnout_score',
        color='burnout_level',
        custom_data=['_row_id'],
        color_discrete_map=BURNOUT_LEVEL_COLORS,
        hover_data=hover_cols,
        title=f"{format_feature_name(x_feature)} vs Burnout Score (Master View)",
        labels={
            x_feature: format_feature_name(x_feature),
            'burnout_score': 'Burnout Score',
            'burnout_level': 'Burnout Level'
        },
        category_orders={'burnout_level': ['High', 'Medium', 'Low']},
        template="plotly_white",
        opacity=0.65,
        render_mode='auto'
    )

    x_series = plot_df[x_feature]
    y_series = plot_df['burnout_score']
    trend_label = 'Near Flat'
    trend_class = 'master-trend-banner trend-neutral'

    if x_series.nunique() > 1 and len(plot_df) >= 10:
        slope, intercept = np.polyfit(x_series.to_numpy(), y_series.to_numpy(), deg=1)
        x_min = float(x_series.min())
        x_max = float(x_series.max())
        trend_x = [x_min, x_max]
        trend_y = [slope * x_min + intercept, slope * x_max + intercept]
        fig.add_trace(
            go.Scatter(
                x=trend_x,
                y=trend_y,
                mode='lines',
                name='Correlation Trend',
                line=dict(color='#111827', width=3, dash='dash')
            )
        )

        spearman_rho = spearman_corr_no_scipy(x_series, y_series)
        if pd.isna(spearman_rho):
            trend_label = 'Near Flat'
            trend_class = 'master-trend-banner trend-neutral'
        elif spearman_rho >= 0.35:
            trend_label = f'Increasing ({spearman_rho:.2f})'
            trend_class = 'master-trend-banner trend-increasing'
        elif spearman_rho >= 0.10:
            trend_label = f'Slightly Increasing ({spearman_rho:.2f})'
            trend_class = 'master-trend-banner trend-slightly-increasing'
        elif spearman_rho <= -0.35:
            trend_label = f'Decreasing ({spearman_rho:.2f})'
            trend_class = 'master-trend-banner trend-decreasing'
        elif spearman_rho <= -0.05:
            trend_label = f'Slightly Decreasing ({spearman_rho:.2f})'
            trend_class = 'master-trend-banner trend-slightly-decreasing'
        else:
            trend_label = f'Near Flat ({spearman_rho:.2f})'
            trend_class = 'master-trend-banner trend-neutral'

    fig.update_layout(
        dragmode='select',
        title=f"{format_feature_name(x_feature)} vs Burnout Score",
        legend_title='Burnout Level'
    )

    return style_figure(fig, dark_mode), f'Correlation Trend: {trend_label}', trend_class


@app.callback(
    [Output('theme-root', 'className'),
     Output('theme-label', 'children')],
    [Input('theme-toggle', 'value')]
)
def update_theme(dark_mode):
    if dark_mode:
        return 'theme-root theme-dark', 'Dark Mode'
    return 'theme-root theme-light', 'Light Mode'


@app.callback(
    [Output('kpi-sample-size', 'children'),
     Output('kpi-selected-size', 'children'),
     Output('kpi-avg-burnout', 'children'),
     Output('kpi-current-store', 'data'),
     Output('kpi-interval', 'disabled')],
    [Input('kpi-interval', 'n_intervals'),
     Input('kpi-target-store', 'data')],
    [State('kpi-current-store', 'data')]
)
def animate_kpis(_, target, current):
    if not target:
        return no_update, no_update, no_update, no_update, True

    if not current:
        current = {'sample': 0, 'selected': 0, 'avg_burnout': 0.0}

    def step_int(now_value, target_value):
        if now_value == target_value:
            return now_value
        distance = abs(target_value - now_value)
        step = max(1, int(math.ceil(distance * 0.18)))
        if now_value < target_value:
            return min(target_value, now_value + step)
        return max(target_value, now_value - step)

    def step_float(now_value, target_value):
        if abs(now_value - target_value) < 0.01:
            return target_value
        distance = abs(target_value - now_value)
        step = max(0.01, distance * 0.22)
        if now_value < target_value:
            return min(target_value, now_value + step)
        return max(target_value, now_value - step)

    next_sample = step_int(int(current['sample']), int(target['sample']))
    next_selected = step_int(int(current['selected']), int(target['selected']))
    next_avg = step_float(float(current['avg_burnout']), float(target['avg_burnout']))

    next_state = {
        'sample': int(next_sample),
        'selected': int(next_selected),
        'avg_burnout': float(next_avg)
    }

    done = (
        next_state['sample'] == int(target['sample'])
        and next_state['selected'] == int(target['selected'])
        and abs(next_state['avg_burnout'] - float(target['avg_burnout'])) < 0.01
    )

    return (
        f"{next_state['sample']:,}",
        f"{next_state['selected']:,}",
        f"{next_state['avg_burnout']:.2f}",
        next_state,
        done
    )


if __name__ == '__main__':
    # Run the Dash app
    app.run(debug=True, port=find_available_port(8050))