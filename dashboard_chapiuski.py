import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter
from itertools import combinations
import plotly.express as px
import plotly.graph_objects as go
import os

# Inicializar o app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# URL dos dados
dados_chap = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQqKawlrhvZxCUepOzcl4jG9ejActoqNd11Hs6hDverwxV0gv9PRYjwVxs6coMWsoopfH41EuSLRN-v/pub?output=csv"

# --- FUNÇÕES AUXILIARES ---
def carregar_dados():
    try:
        df = pd.read_csv(dados_chap, decimal=',')
        df['Gol'] = pd.to_numeric(df['Gol'], errors='coerce').fillna(0).astype(int)
        df['Assistência'] = pd.to_numeric(df['Assistência'], errors='coerce').fillna(0).astype(int)
        df['Data'] = pd.to_datetime(df['Data'], dayfirst=True)
        df = df.sort_values('Data')
        primeira_data = df['Data'].min()
        df['Semana'] = ((df['Data'] - primeira_data).dt.days // 7) + 1
        df['Jogador'] = df['Jogador'].str.strip()
        return df
    except Exception as e:
        print(f"Erro ao carregar os dados: {str(e)}")
        return None

def criar_grafico_evolucao(df, jogadores, coluna, data_inicio, data_fim):
    data_inicio_dt = pd.to_datetime(data_inicio)
    data_fim_dt = pd.to_datetime(data_fim)
    
    while data_inicio_dt.weekday() != 5:
        data_inicio_dt += pd.Timedelta(days=1)
    
    todos_sabados = pd.date_range(start=data_inicio_dt, end=data_fim_dt, freq='W-SAT')
    
    dados_acumulados = pd.DataFrame()
    for jogador in jogadores:
        dados_jogador = pd.DataFrame({'Data': todos_sabados})
        dados_jogador['Jogador'] = jogador
        
        dados_reais = df[
            (df['Jogador'] == jogador) &
            (df['Data'] >= data_inicio_dt) &
            (df['Data'] <= data_fim_dt)
        ].groupby('Data')[coluna].sum().reset_index()
        
        dados_jogador = dados_jogador.merge(dados_reais, on='Data', how='left')
        dados_jogador[coluna] = dados_jogador[coluna].fillna(0)
        dados_jogador = dados_jogador.sort_values('Data')
        dados_jogador[f'{coluna}_Acumulado'] = dados_jogador[coluna].cumsum()
        
        dados_acumulados = pd.concat([dados_acumulados, dados_jogador])
    
    fig = px.line(dados_acumulados, 
                  x='Data', 
                  y=f'{coluna}_Acumulado',
                  color='Jogador',
                  title=f'Evolução de {coluna}s Acumulados')
    
    fig.update_layout(
        xaxis_title='Data',
        yaxis_title=f'Total de {coluna}s',
        showlegend=True
    )
    
    return fig

def criar_ranking(df, coluna, titulo):
    rankings = df.groupby('Jogador')[coluna].sum().sort_values(ascending=True).tail(10)
    
    fig = go.Figure(go.Bar(
        x=rankings.values,
        y=rankings.index,
        orientation='h',
        text=rankings.values,
        textposition='auto',
    ))
    
    fig.update_layout(
        title=titulo,
        xaxis_title=coluna,
        yaxis_title='Jogador',
        height=400
    )
    
    return fig

def criar_heatmap_participacao(df):
    top_10_frequentes = df['Jogador'].value_counts().head(10).index.tolist()
    participacao = (
        df[df['Jogador'].isin(top_10_frequentes)]
        .groupby(['Jogador', 'Semana'])
        .size()
        .unstack(fill_value=0)
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=participacao.values,
        x=participacao.columns,
        y=participacao.index,
        colorscale='Greens'
    ))
    
    fig.update_layout(
        title='Participação por Semana',
        xaxis_title='Semana',
        yaxis_title='Jogador',
        height=400
    )
    
    return fig

def analisar_momento(df, jogador, coluna):
    datas_com_jogos = pd.Series(df['Data'].unique()).sort_values()
    dados_jogador = pd.DataFrame({'Data': datas_com_jogos})
    jogos_jogador = df[df['Jogador'] == jogador][['Data', coluna]].copy()
    
    dados_jogador = dados_jogador.merge(jogos_jogador, on='Data', how='left')
    dados_jogador[coluna] = dados_jogador[coluna].fillna(0)
    dados_jogador['Media_Movel'] = dados_jogador[coluna].rolling(window=4, min_periods=1).mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dados_jogador['Data'],
        y=dados_jogador[coluna],
        name='Por Jogo',
        mode='lines+markers'
    ))
    
    fig.add_trace(go.Scatter(
        x=dados_jogador['Data'],
        y=dados_jogador['Media_Movel'],
        name='Média Móvel (4 jogos)',
        line=dict(dash='dash')
    ))
    
    fig.update_layout(
        title=f'Evolução de {coluna}s - {jogador}',
        xaxis_title='Data',
        yaxis_title='Quantidade',
        height=400
    )
    
    return fig

def criar_radar(df, jogadores):
    metricas = []
    total_semanas = df['Semana'].nunique()
    
    for jogador in jogadores:
        dados_jogador = df[df['Jogador'] == jogador]
        jogos = len(dados_jogador)
        if jogos > 0:
            semanas_jogadas = dados_jogador['Semana'].nunique()
            vitorias = (dados_jogador['Situação'] == 'Vitória').sum()
            taxa_vitoria = (vitorias / jogos * 100)
            
            metricas.append({
                'Jogador': jogador,
                'Gols/Jogo': dados_jogador['Gol'].mean(),
                'Assistências/Jogo': dados_jogador['Assistência'].mean(),
                'Frequência (%)': (semanas_jogadas / total_semanas * 100),
                'Taxa de Vitória (%)': taxa_vitoria,
                'Participações/Jogo': (dados_jogador['Gol'].sum() + dados_jogador['Assistência'].sum()) / jogos
            })
    
    if not metricas:
        return go.Figure()
    
    df_metricas = pd.DataFrame(metricas)
    
    fig = go.Figure()
    
    categorias = ['Gols/Jogo', 'Assistências/Jogo', 'Frequência (%)', 'Taxa de Vitória (%)', 'Participações/Jogo']
    
    for jogador in df_metricas['Jogador']:
        valores = df_metricas[df_metricas['Jogador'] == jogador][categorias].values[0]
        valores = np.append(valores, valores[0])  # Fecha o polígono
        
        fig.add_trace(go.Scatterpolar(
            r=valores,
            theta=categorias + [categorias[0]],
            name=jogador,
            fill='toself'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max([
                    df_metricas['Gols/Jogo'].max(),
                    df_metricas['Assistências/Jogo'].max(),
                    df_metricas['Frequência (%)'].max() / 100,
                    df_metricas['Taxa de Vitória (%)'].max() / 100,
                    df_metricas['Participações/Jogo'].max()
                ])]
            )
        ),
        title='Comparação entre Jogadores',
        showlegend=True
    )
    
    return fig

def criar_rede_entrosamento(df, coluna, titulo):
    top_7_frequentes = df['Jogador'].value_counts().head(7).index.tolist()
    
    soma = Counter()
    semanas = Counter()
    for semana, grupo in df.groupby('Semana'):
        jogs = grupo['Jogador'].unique()
        for par in combinations(sorted(jogs), 2):
            if par[0] in top_7_frequentes and par[1] in top_7_frequentes:
                soma[par] += grupo[grupo['Jogador'].isin(par)][coluna].sum()
                semanas[par] += 1
    
    media_dict = {par: soma[par]/semanas[par] if semanas[par]>0 else 0 for par in soma}
    
    nodes = list(set([j for par in media_dict.keys() for j in par]))
    node_indices = {node: i for i, node in enumerate(nodes)}
    
    edge_x = []
    edge_y = []
    edge_weights = []
    
    for (j1, j2), peso in media_dict.items():
        if peso > 0:
            x0 = np.cos(2*np.pi*node_indices[j1]/len(nodes))
            y0 = np.sin(2*np.pi*node_indices[j1]/len(nodes))
            x1 = np.cos(2*np.pi*node_indices[j2]/len(nodes))
            y1 = np.sin(2*np.pi*node_indices[j2]/len(nodes))
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(peso)
    
    edges_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=np.array(edge_weights)*2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    node_x = [np.cos(2*np.pi*i/len(nodes)) for i in range(len(nodes))]
    node_y = [np.sin(2*np.pi*i/len(nodes)) for i in range(len(nodes))]
    
    nodes_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=nodes,
        textposition="top center",
        marker=dict(
            size=20,
            color='#1f77b4',
            line_width=2
        )
    )
    
    fig = go.Figure(data=[edges_trace, nodes_trace],
                   layout=go.Layout(
                       title=titulo,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                   ))
    
    return fig

def analisar_tendencia(df, jogador, coluna):
    ultimos_4 = df[df['Jogador'] == jogador][coluna].tail(4).mean()
    media_geral = df[df['Jogador'] == jogador][coluna].mean()
    
    if media_geral == 0:
        return "❌ Sem dados suficientes", 0
    else:
        momento = "🔥 Em Alta" if ultimos_4 > media_geral * 1.2 else "❄️ Em Baixa" if ultimos_4 < media_geral * 0.8 else "➡️ Estável"
        diferenca = ((ultimos_4/media_geral - 1)*100) if media_geral > 0 else 0
        return momento, diferenca

# Layout do Dashboard
app.layout = dbc.Container([
    html.H1('Chapiuski FC', className='text-center my-4'),
    
    # Filtros
    dbc.Card([
        dbc.CardBody([
            html.H3('⚙️ Filtros de Análise', className='card-title'),
            dbc.Row([
                dbc.Col([
                    html.Label('Data Inicial:'),
                    dcc.DatePickerSingle(
                        id='data-inicio',
                        min_date_allowed=datetime(2023, 1, 1),
                        max_date_allowed=datetime(2024, 12, 31),
                        initial_visible_month=datetime(2024, 1, 1),
                        date=datetime(2024, 1, 1)
                    )
                ]),
                dbc.Col([
                    html.Label('Data Final:'),
                    dcc.DatePickerSingle(
                        id='data-fim',
                        min_date_allowed=datetime(2023, 1, 1),
                        max_date_allowed=datetime(2024, 12, 31),
                        initial_visible_month=datetime(2024, 1, 1),
                        date=datetime(2024, 12, 31)
                    )
                ])
            ])
        ])
    ], className='mb-4'),
    
    # Números do Período
    dbc.Card([
        dbc.CardBody([
            html.H3('📊 Números do Período', className='card-title'),
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H5('Total de Jogos', className='card-title'),
                        html.H3(id='total-jogos')
                    ])
                ])),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H5('Total de Gols', className='card-title'),
                        html.H3(id='total-gols')
                    ])
                ])),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H5('Total de Assistências', className='card-title'),
                        html.H3(id='total-assists')
                    ])
                ]))
            ])
        ])
    ], className='mb-4'),
    
    # Rankings
    dbc.Card([
        dbc.CardBody([
            html.H3('🏆 Rankings do Período', className='card-title'),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='ranking-gols')
                ]),
                dbc.Col([
                    dcc.Graph(id='ranking-assists')
                ])
            ])
        ])
    ], className='mb-4'),
    
    # Evolução
    dbc.Card([
        dbc.CardBody([
            html.H3('📈 Evolução', className='card-title'),
            dcc.Graph(id='evolucao-gols'),
            dcc.Graph(id='evolucao-assists')
        ])
    ], className='mb-4'),
    
    # Heatmap de Participação
    dbc.Card([
        dbc.CardBody([
            html.H3('🎯 Participação por Semana', className='card-title'),
            dcc.Graph(id='heatmap-participacao')
        ])
    ], className='mb-4'),
    
    # Comparação entre Jogadores
    dbc.Card([
        dbc.CardBody([
            html.H3('📊 Comparação entre Jogadores', className='card-title'),
            dcc.Dropdown(
                id='jogadores-comparacao',
                multi=True,
                placeholder='Selecione os jogadores para comparar (máximo 5)',
                style={'marginBottom': '20px'}
            ),
            dcc.Graph(id='radar-plot')
        ])
    ], className='mb-4'),
    
    # Redes de Entrosamento
    dbc.Card([
        dbc.CardBody([
            html.H3('🤝 Redes de Entrosamento', className='card-title'),
            dbc.Row([
                dbc.Col([
                    html.H5('⚽ Rede de Gols', className='text-center'),
                    dcc.Graph(id='rede-gols')
                ]),
                dbc.Col([
                    html.H5('👟 Rede de Assistências', className='text-center'),
                    dcc.Graph(id='rede-assists')
                ])
            ])
        ])
    ], className='mb-4'),
    
    # Análise Individual
    dbc.Card([
        dbc.CardBody([
            html.H3('👤 Análise Individual', className='card-title'),
            dcc.Dropdown(
                id='jogador-select',
                placeholder='Selecione um jogador'
            ),
            html.Div(id='metricas-jogador'),
            dcc.Graph(id='momento-gols'),
            dcc.Graph(id='momento-assists')
        ])
    ])
], fluid=True)

# Callbacks
@app.callback(
    [Output('total-jogos', 'children'),
     Output('total-gols', 'children'),
     Output('total-assists', 'children'),
     Output('ranking-gols', 'figure'),
     Output('ranking-assists', 'figure'),
     Output('evolucao-gols', 'figure'),
     Output('evolucao-assists', 'figure'),
     Output('heatmap-participacao', 'figure'),
     Output('jogadores-comparacao', 'options'),
     Output('radar-plot', 'figure'),
     Output('rede-gols', 'figure'),
     Output('rede-assists', 'figure')],
    [Input('data-inicio', 'date'),
     Input('data-fim', 'date'),
     Input('jogadores-comparacao', 'value')]
)
def update_dashboard(data_inicio, data_fim, jogadores_selecionados):
    df = carregar_dados()
    if df is None:
        return dash.no_update
    
    df_filt = df[
        (df['Data'].dt.date >= pd.to_datetime(data_inicio).date()) &
        (df['Data'].dt.date <= pd.to_datetime(data_fim).date())
    ]
    
    total_jogos = df_filt['Semana'].nunique()
    total_gols = int(df_filt['Gol'].sum())
    total_assists = int(df_filt['Assistência'].sum())
    
    ranking_gols = criar_ranking(df_filt, 'Gol', 'Ranking de Goleadores')
    ranking_assists = criar_ranking(df_filt, 'Assistência', 'Ranking de Assistentes')
    
    rankings = df_filt.groupby('Jogador').agg({
        'Gol': 'sum',
        'Assistência': 'sum'
    }).reset_index()
    
    evolucao_gols = criar_grafico_evolucao(
        df_filt, 
        rankings.nlargest(5, 'Gol')['Jogador'].tolist(),
        'Gol',
        data_inicio,
        data_fim
    )
    
    evolucao_assists = criar_grafico_evolucao(
        df_filt,
        rankings.nlargest(5, 'Assistência')['Jogador'].tolist(),
        'Assistência',
        data_inicio,
        data_fim
    )
    
    heatmap = criar_heatmap_participacao(df_filt)
    
    jogadores_options = [{'label': j, 'value': j} for j in sorted(df_filt['Jogador'].unique())]
    
    # Radar Plot
    radar_fig = criar_radar(df_filt, jogadores_selecionados if jogadores_selecionados else [])
    
    # Redes de Entrosamento
    rede_gols = criar_rede_entrosamento(df_filt, 'Gol', 'Rede de Entrosamento - Gols')
    rede_assists = criar_rede_entrosamento(df_filt, 'Assistência', 'Rede de Entrosamento - Assistências')
    
    return (
        total_jogos,
        total_gols,
        total_assists,
        ranking_gols,
        ranking_assists,
        evolucao_gols,
        evolucao_assists,
        heatmap,
        jogadores_options,
        radar_fig,
        rede_gols,
        rede_assists
    )

@app.callback(
    [Output('metricas-jogador', 'children'),
     Output('momento-gols', 'figure'),
     Output('momento-assists', 'figure')],
    [Input('jogador-select', 'value'),
     Input('data-inicio', 'date'),
     Input('data-fim', 'date')]
)
def update_analise_individual(jogador, data_inicio, data_fim):
    if not jogador:
        return dash.no_update
    
    df = carregar_dados()
    if df is None:
        return dash.no_update
    
    df_filt = df[
        (df['Data'].dt.date >= pd.to_datetime(data_inicio).date()) &
        (df['Data'].dt.date <= pd.to_datetime(data_fim).date())
    ]
    
    dados_jogador = df_filt[df_filt['Jogador'] == jogador]
    total_jogos = len(dados_jogador)
    
    if total_jogos == 0:
        return html.Div('Não há dados para este jogador no período selecionado.'), {}, {}
    
    gols = dados_jogador['Gol'].sum()
    assists = dados_jogador['Assistência'].sum()
    vitorias = (dados_jogador['Situação'] == 'Vitória').sum()
    
    # Análise de tendência
    momento_gols, diferenca_gols = analisar_tendencia(df_filt, jogador, 'Gol')
    momento_assists, diferenca_assists = analisar_tendencia(df_filt, jogador, 'Assistência')
    
    metricas = dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5('Jogos'),
                html.H3(total_jogos)
            ])
        ])),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5('Gols'),
                html.H3(f"{gols} ({momento_gols})")
            ])
        ])),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5('Assistências'),
                html.H3(f"{assists} ({momento_assists})")
            ])
        ])),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5('Taxa de Vitória'),
                html.H3(f'{(vitorias/total_jogos*100):.1f}%')
            ])
        ]))
    ])
    
    momento_gols = analisar_momento(df_filt, jogador, 'Gol')
    momento_assists = analisar_momento(df_filt, jogador, 'Assistência')
    
    return metricas, momento_gols, momento_assists

if __name__ == '__main__':
    app.run_server()