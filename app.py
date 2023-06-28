from dash import Dash, dcc, html, Output, Input, State, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash_bootstrap_templates import ThemeSwitchAIO

import pandas as pd
import numpy as np

# Dataset
df = pd.read_csv('dataset/data_gas.csv')
df = df[df['PRODUTO']=='GASOLINA COMUM']
df['DATA'] = df[['DATA INICIAL','DATA FINAL']].astype('datetime64[ms]').apply(np.mean, axis=1)
df = df[['DATA','REGIÃO','ESTADO','PREÇO MÉDIO REVENDA' ]]
df.sort_values(by=['DATA','REGIÃO','ESTADO'], ignore_index=True, inplace=True)

# style
tab_card = {'height': '100%'}
theme_1 = 'flatly'
theme_2 = 'vapor'

graph_main_config = {
    "hovermode": "x unified",
    "legend": {"yanchor":"top", 
                "y":0.9, 
                "xanchor":"left",
                "x":0.1,
                "title": {"text": None},
                "font" :{"color":"white"},
                "bgcolor": "rgba(0,0,0,0.5)"},
    "margin": {"l":0, "r":0, "t":10, "b":0}
}

font_awesome  = "https://use.fontawesome.com/releases/v5.10.2/css/all.css"
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.4/dbc.min.css"

# App
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY, font_awesome, dbc_css])

app.layout = dbc.Container([
    dcc.Store(id='dataset-filtered'),
    dcc.Store(id='controler', data={'play':False}),
    dcc.Interval(id='interval', interval=2000, disabled=True),
    dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(html.Legend('Gas Prices Analysis'), sm=8),
                        dbc.Col(html.I(className='fa fa-filter', style={'font-size': '300%'}), sm=4, align='center')  
                    ]),
                    dbc.Row([
                        ThemeSwitchAIO(aio_id='theme', themes=[dbc.themes.FLATLY, dbc.themes.VAPOR]),
                        html.Legend('Dash Allan'),
                        dbc.Button('Visite nosso site', href='https://www.google.com', target='_blank')
                    ], style={'margin-top': '10px'})
                    
                ]), style=tab_card
            ), sm=4, lg=2,
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H3('Maximos e Minimos'),
                    dcc.Graph(id='graph-max-min', config={'displayModeBar': False, 'showTips':False} )
                ]), style=tab_card
            ), sm=8, lg=3
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6('Ano de analise:'),
                            dcc.Dropdown(
                                id='select-year',
                                value=df['DATA'].dt.year.max(),
                                options=[{'label':x, 'value': x} for x in df['DATA'].dt.year.unique()],
                                clearable=False,
                                className='dbc'
                            ),
                            dcc.Graph(id='graph-region-bar', config={'displayModeBar':False, 'showTips':False})
                        ],sm=6 ),
                        dbc.Col([
                            html.H6('Região de analise:'),
                            dcc.Dropdown(
                                id='select-region',
                                value=df['REGIÃO'].unique()[1],
                                options=[{'label': x, 'value': x} for x in df['REGIÃO'].unique()],
                                className='dbc',
                                clearable=False
                            ),
                            dcc.Graph(id='graph-estado-bar', config={'displayModeBar':False, 'showTips':False})
                        ], sm=6)
                    ])
                ]), style=tab_card
            ), sm=12, lg=7
        )
    ], className='main_row g-2 my-auto'),
    dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H3('Preço x Estado'),
                    html.H6('Comparação temporal entre estados'),
                    dcc.Dropdown(
                        id='select-states',
                        value=['ACRE', 'SAO PAULO'],
                        options=[{'label':x, 'value': x} for x in df['ESTADO'].unique()],
                        multi=True,
                        clearable=False,
                        className='dbc'
                    ),
                    dcc.Graph(id='graph-comparison-states', config={'displayModeBar':False, 'showTips':False})
                ]), style=tab_card
            ), sm=12, md=6, lg=5
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H3('Comparação direta'),
                    html.H6('Qual preço menor em um dado periodo de tempo'),
                    dbc.Row([
                        dbc.Col(
                            dcc.Dropdown(
                                id='select-direct-state-1',
                                value='CEARA',
                                options=[{'label':x, 'value': x} for x in df['ESTADO'].unique()],
                                clearable=False,
                                className='dbc'
                                )
                            ),
                        dbc.Col(
                            dcc.Dropdown(
                                id='select-direct-state-2',
                                value='RIO GRANDE DO SUL',
                                options=[{'label':x, 'value': x} for x in df['ESTADO'].unique()],
                                clearable=False,
                                className='dbc'
                                )
                        )
                    ]),
                    dcc.Graph(id='graph-comparison-direct', config={'displayModeBar':False, 'showTips':False}),
                    html.P(id='text-comparison-direct', style={'font-size': '80%', 'color': 'gray'})
                ]), style=tab_card
            ), sm=12, md=6, lg=4
        ),
        dbc.Col([
            dbc.Row(
                dbc.Col(
                    dbc.Card(  
                        dbc.CardBody([
                            dcc.Graph(id='graph-indicator-1', config={'displayModeBar': False, 'showTips':False})
                        ])
                    )
                ), className='g-2'
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Card(  
                        dbc.CardBody([
                            dcc.Graph(id='graph-indicator-2', config={'displayModeBar':False, 'showTips': False})
                        ])
                    )
                ), className='g-2 my-auto'
            )
        ], sm=12, lg=3)
    ], className='main_row g-2 my-auto'),
    dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    dbc.Row([
                        dbc.Col([
                            html.I(id='play-button', className='fa fa-play', style={'margin-right':'10px'}),
                            html.I(id='stop-button', className='fa fa-stop')
                        ], sm=12, md=1, style={'text-align':'center', 'margin-top':'10px'}),
                        dbc.Col(
                            dcc.RangeSlider(
                                id='select-range-years',
                                marks={str(x) :str(x) for x in df['DATA'].dt.year.unique()},
                                step=3,
                                min=df['DATA'].dt.year.unique().min(),
                                max=df['DATA'].dt.year.unique().max(),
                                className='dbc',
                                value=[df['DATA'].dt.year.unique().min(),df['DATA'].dt.year.unique().max()],
                                dots=True,
                                pushable=3
                            ), sm=12, md=11, style={'margin-top':'15px'}
                        )
                    ], style={'height':'20%'})
                )
            )
        )   
    ], className='main_row g-2 my-auto')
], fluid=True)


@app.callback(
        Output('dataset-filtered', 'data'),
        Input('select-range-years', 'value')
)
def filtered_dataset(years):
    df_filtered = df[(df['DATA'].dt.year >= years[0]) & (df['DATA'].dt.year <= years[1])].to_dict()
    return df_filtered

@app.callback(
    [
        Output('select-range-years', 'value'),
        Output('controler', 'data'),
        Output('interval', 'disabled')

    ],
    [
        Input('interval', 'n_intervals'),
        Input('play-button', 'n_clicks'),
        Input('stop-button','n_clicks')
    ],
    [
        State('select-range-years', 'value'),
        State('controler', 'data')
    ], prevent_initial_call=True
)
def controler(n_intervals, play, stop, range_year, controler):
    trigg = callback_context.triggered[0]['prop_id']
    interval_disabled=False

    if 'play-button' in trigg and not controler['play']:
        controler['play'] = True
        range_year[1]=2004
        interval_disabled = False

    elif 'stop-button' in trigg and controler['play']:
        controler['play'] = False

    if controler['play']:
        if range_year[1]==2021:
            controler['play']=False
            interval_disabled=True
        range_year[1]+=1 if range_year[1]<2021 else 0
        
    return range_year, controler, interval_disabled

@app.callback(
    Output('graph-max-min', 'figure'),
    [
        Input('dataset-filtered', 'data'),
        Input(ThemeSwitchAIO.ids.switch("theme"), 'value')
     ]
    )
def graph_max_min(data, toggle):
    theme = theme_1 if toggle else theme_2
    df_fig = pd.DataFrame(data)
    df_fig['DATA'] = pd.to_datetime(df_fig['DATA'])
    df_fig = df_fig.groupby(df_fig['DATA'].dt.year)['PREÇO MÉDIO REVENDA'].agg({'max', 'min'}).reset_index()
    
    fig = px.line(df_fig, x='DATA', y=['max','min'], template=theme)
    fig.update_layout(graph_main_config, height=150, xaxis_title=None, yaxis_title=None)
    return fig

@app.callback(
    [
        Output('graph-region-bar', 'figure'),
        Output('graph-estado-bar','figure')
    ],
    [
        Input('select-year', 'value'),
        Input('select-region', 'value'),
        Input(ThemeSwitchAIO.ids.switch('theme'), 'value')
    ]
    )
def graph_bar_flat(year, region, toggle):
    theme = theme_1 if toggle else theme_2
    df_region = df[df['DATA'].dt.year==year]
    df_region = (df_region.groupby('REGIÃO')['PREÇO MÉDIO REVENDA']
                 .mean().sort_values(ascending=True))
    fig_region = go.Figure(go.Bar(
        x=df_region,
        y=df_region.index,
        orientation='h',
        text=[f'{i[0]}, R${i[1]:.2f}' for i in df_region.items()],
        textposition='auto',
        insidetextanchor='end',
        insidetextfont={'family': 'Times', 'size': 12}
    ))
    fig_region.update_layout(graph_main_config, height=150, template=theme,
                             yaxis={'showticklabels':False},
                             xaxis_range=[df_region.max(),df_region.min()-0.15])
    

    df_states = df[(df['DATA'].dt.year==year) & (df['REGIÃO']==region)]
    df_states = (df_states.groupby('ESTADO')['PREÇO MÉDIO REVENDA']
                 .mean().sort_values(ascending=True))
    
    fig_states = go.Figure(go.Bar(
        x=df_states,
        y=df_states.index,
        orientation='h',
        text=[f'{i[0]}, R${i[1]:.2f}' for i in df_states.items()],
        textposition='auto',
        insidetextanchor='end',
        insidetextfont={'family':'Times', 'size': 12}
    ))
    fig_states.update_layout(graph_main_config, height=150, template=theme,
                             yaxis={'showticklabels':False},
                             xaxis_range=[df_states.min()-0.15, df_states.max()])

    return fig_region, fig_states

@app.callback(
    Output('graph-comparison-states', 'figure'),
    [
        Input('dataset-filtered', 'data'),
        Input('select-states', 'value'),
        Input(ThemeSwitchAIO.ids.switch('theme'), 'value')
    ]
)
def graph_comparison_states(data, states, toggle):
    theme = theme_1 if toggle else theme_2
    df_fig = pd.DataFrame(data)
    df_fig['DATA'] = pd.to_datetime(df_fig['DATA'])
    df_fig = df_fig[df_fig['ESTADO'].isin(states)]
    fig = px.line(df_fig, x='DATA', y='PREÇO MÉDIO REVENDA', color='ESTADO')
    fig.update_layout(graph_main_config, height=300,
                      xaxis_title=None, yaxis_title=None,
                      template=theme)
    return fig

@app.callback(
    [
        Output('graph-comparison-direct','figure'),
        Output('text-comparison-direct','children'),
        Output('graph-indicator-1','figure'),
        Output('graph-indicator-2','figure')
    ],    
    [
        Input('dataset-filtered', 'data'),
        Input(ThemeSwitchAIO.ids.switch('theme'), 'value'),
        Input('select-direct-state-1','value'),
        Input('select-direct-state-2','value')
    
    ]
    )
def graph_comparison_direct(data, toggle ,state_1, state_2):
    theme = theme_1 if toggle else theme_2
    df_filtered = pd.DataFrame(data=data)
    df_filtered['DATA'] = pd.to_datetime(df_filtered['DATA'])

    serie_state_1 = (df_filtered[df_filtered['ESTADO']==state_1]
               .groupby(df_filtered['DATA'].dt.strftime('%Y-%m'))['PREÇO MÉDIO REVENDA']
               .mean())
    serie_state_2 = (df_filtered[df_filtered['ESTADO']==state_2]
               .groupby(df_filtered['DATA'].dt.strftime('%Y-%m'))['PREÇO MÉDIO REVENDA']
               .mean())

    df_final = pd.DataFrame(serie_state_1 - serie_state_2)

    fig_graph = go.Figure()
    fig_graph.add_trace(
        go.Scatter(x=df_final.index, y=df_final['PREÇO MÉDIO REVENDA'], fill='tonexty')
    )
    fig_graph.update_layout(graph_main_config, height=300, template=theme)

    fig_card_1 = go.Figure()
    fig_card_1.add_trace(go.Indicator(
        mode='number+delta',
        title={"text": f"{state_1}<br><span style='font-size:0.8em;color:gray'>{df_filtered[df_filtered['ESTADO']==state_1].iloc[0]['DATA'].year} - {df_filtered[df_filtered['ESTADO']==state_1].iloc[-1]['DATA'].year}</span>"},
        number={'prefix': 'R$', 'valueformat':'.2f'},
        value=df_filtered[df_filtered['ESTADO']==state_1].iloc[-1]['PREÇO MÉDIO REVENDA'],
        delta={'reference': df_filtered[df_filtered['ESTADO']=='ACRE'].iloc[0]['PREÇO MÉDIO REVENDA'],
               'relative': True, 'valueformat': '.1%'},
        domain = {'y': [0, 0.75], 'x': [0, 1]}
        ))
    fig_card_1.update_layout(graph_main_config, height=210, template=theme)

    text = f"Comparando {state_1} e {state_2}. Se a linha estiver acima do eixo X, {state_2} tinha menor preço, do contrário, {state_1} tinha um valor inferior."

    fig_card_2 = go.Figure()
    fig_card_2.add_trace(go.Indicator(
        mode='number+delta',
        number={'prefix':'R$','valueformat':'.2f'},
        title={'text': f"{state_2}<br><span style='font-size:0.8em;color:gray'>{df_filtered[df_filtered['ESTADO']==state_2].iloc[0]['DATA'].year} - {df_filtered[df_filtered['ESTADO']==state_2].iloc[-1]['DATA'].year}</span>"},
        value=df_filtered[df_filtered['ESTADO']==state_2].iloc[-1]['PREÇO MÉDIO REVENDA'],
        delta={'reference': df_filtered[df_filtered['ESTADO']==state_2].iloc[0]['PREÇO MÉDIO REVENDA'],
               'valueformat':'.1%', 'relative': True},
        domain = {'y': [0, 0.75], 'x': [0, 1]}
        ))
    fig_card_2.update_layout(graph_main_config, height=210, template=theme)

    return fig_graph, text, fig_card_1, fig_card_2

if __name__ =='__main__':
    app.run_server(debug=True, port=8051)