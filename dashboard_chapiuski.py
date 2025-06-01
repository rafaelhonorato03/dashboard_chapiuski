import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from collections import Counter
from itertools import combinations

dados_chap = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQqKawlrhvZxCUepOzcl4jG9ejActoqNd11Hs6hDverwxV0gv9PRYjwVxs6coMWsoopfH41EuSLRN-v/pub?output=csv"

# --- FUNÇÕES AUXILIARES ---
def criar_grafico_evolucao(df, jogadores, coluna):
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
    
    # Usar st.line_chart do Streamlit
    chart_data = dados_acumulados.pivot(index='Data', 
                                      columns='Jogador', 
                                      values=f'{coluna}_Acumulado')
    st.line_chart(chart_data)

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
    
    # Criar DataFrame para visualização
    conexoes_df = pd.DataFrame([
        {'Jogador 1': j1, 'Jogador 2': j2, 'Média': val}
        for (j1, j2), val in media_dict.items()
        if val > 0
    ]).sort_values('Média', ascending=False)
    
    if len(conexoes_df) == 0:
        st.info(f"Nenhuma conexão encontrada para {titulo}")
        return
    
    st.dataframe(
        conexoes_df,
        column_config={
            'Média': st.column_config.ProgressColumn(
                'Média',
                format='%.2f',
                min_value=0,
                max_value=conexoes_df['Média'].max()
            )
        }
    )

def criar_radar(df, jogadores, titulo):
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
                'Gols': dados_jogador['Gol'].sum(),
                'Gols/Jogo': dados_jogador['Gol'].mean(),
                'Assistências': dados_jogador['Assistência'].sum(),
                'Assistências/Jogo': dados_jogador['Assistência'].mean(),
                'Frequência (%)': (semanas_jogadas / total_semanas * 100),
                'Taxa de Vitória (%)': taxa_vitoria
            })
    
    if not metricas:
        return
        
    df_metricas = pd.DataFrame(metricas)
    st.dataframe(
        df_metricas.set_index('Jogador'),
        column_config={
            'Gols': st.column_config.NumberColumn('Gols', format='%d'),
            'Gols/Jogo': st.column_config.NumberColumn('Gols/Jogo', format='%.2f'),
            'Assistências': st.column_config.NumberColumn('Assistências', format='%d'),
            'Assistências/Jogo': st.column_config.NumberColumn('Assistências/Jogo', format='%.2f'),
            'Frequência (%)': st.column_config.ProgressColumn('Frequência (%)', format='%.1f%%'),
            'Taxa de Vitória (%)': st.column_config.ProgressColumn('Taxa de Vitória (%)', format='%.1f%%')
        }
    )

def analisar_momento(df, jogador, coluna, titulo, cor):
    datas_com_jogos = pd.Series(df['Data'].unique()).sort_values()
    dados_jogador = pd.DataFrame({'Data': datas_com_jogos})
    jogos_jogador = df[df['Jogador'] == jogador][['Data', coluna]].copy()
    
    dados_jogador = dados_jogador.merge(jogos_jogador, on='Data', how='left')
    dados_jogador[coluna] = dados_jogador[coluna].fillna(0)
    dados_jogador['Media_Movel'] = dados_jogador[coluna].rolling(window=4, min_periods=1).mean()
    
    # Usar st.line_chart do Streamlit
    chart_data = pd.DataFrame({
        'Por Jogo': dados_jogador[coluna],
        'Média Móvel (4 jogos)': dados_jogador['Media_Movel']
    }, index=dados_jogador['Data'])
    
    st.line_chart(chart_data)
    
    ultimos_4 = dados_jogador[coluna].tail(4).mean()
    media_geral = dados_jogador[coluna].mean()
    
    if media_geral == 0:
        momento = "❌ Sem dados suficientes"
        diferenca = 0
    else:
        momento = "🔥 Em Alta" if ultimos_4 > media_geral * 1.2 else "❄️ Em Baixa" if ultimos_4 < media_geral * 0.8 else "➡️ Estável"
        diferenca = ((ultimos_4/media_geral - 1)*100) if media_geral > 0 else 0
    
    ultimos_4_valores = dados_jogador[coluna].tail(4).tolist()
    ultimos_4_datas = dados_jogador['Data'].tail(4).dt.strftime('%d/%m').tolist()
    
    metrica = "gols" if coluna == 'Gol' else "assistências" if coluna == 'Assistência' else "vitórias"
    
    st.markdown(f"""
    **Análise de {metrica.title()} - {jogador}**
    - Status: {momento}
    - Média de {metrica} nos últimos 4 jogos: {ultimos_4:.2f}
    - Média geral de {metrica}: {media_geral:.2f}
    - Diferença: {diferenca:.1f}% em relação à média
    
    **Últimos 4 jogos ({metrica}):**
    - {ultimos_4_datas[0]}: {ultimos_4_valores[0]:.0f}
    - {ultimos_4_datas[1]}: {ultimos_4_valores[1]:.0f}
    - {ultimos_4_datas[2]}: {ultimos_4_valores[2]:.0f}
    - {ultimos_4_datas[3]}: {ultimos_4_valores[3]:.0f}
    """)

def calcular_vitorias(df):
    df = df.copy()
    df['vitoria'] = (df['Situação'] == 'Vitória').astype(int)
    return df

def analisar_entrosamento(df, jogador_central):
    conexoes = {}
    for semana, grupo in df.groupby('Semana'):
        if jogador_central in grupo['Jogador'].values:
            jogadores_semana = grupo['Jogador'].unique()
            for outro_jogador in jogadores_semana:
                if outro_jogador != jogador_central:
                    dados_dupla = grupo[grupo['Jogador'].isin([jogador_central, outro_jogador])]
                    
                    if len(dados_dupla) == 2:
                        gols = dados_dupla['Gol'].sum()
                        assists = dados_dupla['Assistência'].sum()
                        vitoria_na_semana = (dados_dupla['Situação'] == 'Vitória').all()
                        
                        if outro_jogador not in conexoes:
                            conexoes[outro_jogador] = {
                                'gols': 0, 
                                'assists': 0, 
                                'vitorias': 0, 
                                'jogos': 0,
                                'semanas_juntos': set()
                            }
                        
                        conexoes[outro_jogador]['gols'] += gols
                        conexoes[outro_jogador]['assists'] += assists
                        conexoes[outro_jogador]['vitorias'] += 1 if vitoria_na_semana else 0
                        conexoes[outro_jogador]['jogos'] += 1
                        conexoes[outro_jogador]['semanas_juntos'].add(semana)
    
    if conexoes:
        df_conexoes = pd.DataFrame.from_dict(conexoes, orient='index')
        df_conexoes['total_participacoes'] = df_conexoes['gols'] + df_conexoes['assists']
        df_conexoes['taxa_vitoria'] = (df_conexoes['vitorias'] / df_conexoes['jogos'] * 100).round(1)
        df_conexoes['semanas_juntos'] = df_conexoes['semanas_juntos'].apply(len)
        return df_conexoes
    return None

# --- CARREGAMENTO E PREPARAÇÃO DOS DADOS ---
@st.cache_data
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
        st.error(f"Erro ao carregar os dados: {str(e)}")
        return None

# 1 - TÍTULO DA PÁGINA
st.title('Chapiuski FC')

df = carregar_dados()
if df is None:
    st.stop()

# 2 - FILTRO DE DATAS
st.markdown("### ⚙️ Filtros de Análise")

col1, col2 = st.columns(2)
with col1:
    data_inicio = st.date_input(
        'Data Inicial:',
        value=min(df['Data'].dt.date),
        min_value=min(df['Data'].dt.date),
        max_value=max(df['Data'].dt.date)
    )

with col2:
    data_fim = st.date_input(
        'Data Final:',
        value=max(df['Data'].dt.date),
        min_value=min(df['Data'].dt.date),
        max_value=max(df['Data'].dt.date)
    )

# Filtrar dados por data
df_filt = df[
    (df['Data'].dt.date >= data_inicio) &
    (df['Data'].dt.date <= data_fim)
]

# Lista de todos os jogadores ordenada alfabeticamente
todos_jogadores = sorted(df_filt['Jogador'].unique())

# 3 - NÚMEROS TOTAIS
st.markdown("### 📊 Números do Período")
total_jogos = df_filt['Semana'].nunique()
total_gols = df_filt['Gol'].sum()
total_assist = df_filt['Assistência'].sum()

col1, col2, col3 = st.columns(3)
col1.metric("Total de Jogos", total_jogos)
col2.metric("Total de Gols", int(total_gols))
col3.metric("Total de Assistências", int(total_assist))

# 4 - RANKINGS DO PERÍODO
st.markdown("### 🏆 Rankings do Período")

# Preparar dados para os rankings
rankings = df_filt.groupby('Jogador').agg({
    'Gol': 'sum',
    'Assistência': 'sum',
    'Semana': 'nunique'
}).reset_index()

# Ranking de Gols
st.subheader("⚽ Ranking de Goleadores")
gols_plot = rankings.nlargest(10, 'Gol').sort_values('Gol', ascending=True)
st.bar_chart(gols_plot.set_index('Jogador')['Gol'])

# Ranking de Assistências
st.subheader("👟 Ranking de Assistentes")
assist_plot = rankings.nlargest(10, 'Assistência').sort_values('Assistência', ascending=True)
st.bar_chart(assist_plot.set_index('Jogador')['Assistência'])

# 5 - EVOLUÇÃO
st.markdown("### 📈 Evolução")

# Gols Acumulados (Top 5)
st.subheader('Gols Acumulados por Semana (Top 5)')
top_5_goleadores = rankings.nlargest(5, 'Gol')['Jogador'].tolist()
gols_acumulados = criar_grafico_evolucao(df_filt, top_5_goleadores, 'Gol')

# Assistências Acumuladas (Top 5)
st.subheader('Assistências Acumuladas por Semana (Top 5)')
top_5_assistentes = rankings.nlargest(5, 'Assistência')['Jogador'].tolist()
assists_acumulados = criar_grafico_evolucao(df_filt, top_5_assistentes, 'Assistência')

# 6 - HEATMAP DE PARTICIPAÇÃO
st.markdown("### 🎯 Participação por Semana")
st.markdown("Visualize a frequência de participação dos jogadores mais ativos")

top_10_frequentes = df_filt['Jogador'].value_counts().head(10).index.tolist()
participacao = (
    df_filt[df_filt['Jogador'].isin(top_10_frequentes)]
    .groupby(['Jogador', 'Semana'])
    .size()
    .unstack(fill_value=0)
)

# Usar st.dataframe com formatação condicional
st.dataframe(
    participacao,
    use_container_width=True,
    hide_index=False,
    column_config={col: st.column_config.NumberColumn(
        str(col),
        format="%d",
        background="rgb(0, 200, 0, {value/max_value})"
    ) for col in participacao.columns}
)

# 7 - REDES DE ENTROSAMENTO
st.markdown("### 🤝 Redes de Entrosamento")
st.markdown("Visualize as conexões entre os jogadores")

criar_rede_entrosamento(df_filt, 'Gol', '⚽ Rede de Gols')
criar_rede_entrosamento(df_filt, 'Assistência', '👟 Rede de Assistências')

# 8 - COMPARAÇÃO ENTRE JOGADORES
st.markdown("### 📊 Comparação entre Jogadores")
st.markdown("Compare as métricas de diferentes jogadores")

jogadores_selecionados = st.multiselect(
    "Selecione os jogadores para comparar (máximo 5):",
    todos_jogadores,
    default=df_filt['Jogador'].value_counts().head(3).index.tolist(),
    max_selections=5
)

if jogadores_selecionados:
    criar_radar(df_filt, jogadores_selecionados, "Comparação entre Jogadores Selecionados")

# 9 - ANÁLISE INDIVIDUAL
st.markdown("### 👤 Análise Individual")
st.markdown("Análise detalhada por jogador")

# Seletor de jogador
jogador_mais_frequente = df_filt['Jogador'].value_counts().index[0]
jogador_selecionado = st.selectbox(
    "Selecione um jogador:",
    todos_jogadores,
    index=todos_jogadores.index(jogador_mais_frequente)
)

# Métricas individuais
dados_jogador = df_filt[df_filt['Jogador'] == jogador_selecionado]
total_jogos_jogador = len(dados_jogador)
if total_jogos_jogador > 0:
    gols_jogador = dados_jogador['Gol'].sum()
    assists_jogador = dados_jogador['Assistência'].sum()
    vitorias_jogador = (dados_jogador['Situação'] == 'Vitória').sum()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Jogos", total_jogos_jogador)
    col2.metric("Total de Gols", int(gols_jogador))
    col3.metric("Total de Assistências", int(assists_jogador))
    
    col4, col5, col6 = st.columns(3)
    col4.metric("Gols por Jogo", f"{gols_jogador/total_jogos_jogador:.2f}")
    col5.metric("Assistências por Jogo", f"{assists_jogador/total_jogos_jogador:.2f}")
    col6.metric("Taxa de Vitória", f"{(vitorias_jogador/total_jogos_jogador*100):.1f}%")
    
    st.markdown("#### 📈 Momento do Jogador")
    st.markdown("Análise da evolução de gols, assistências e vitórias nas últimas semanas")
    analisar_momento(df_filt, jogador_selecionado, 'Gol', 'Evolução de Gols por Jogo', 'green')
    analisar_momento(df_filt, jogador_selecionado, 'Assistência', 'Evolução de Assistências por Jogo', 'purple')
    df_vitorias = calcular_vitorias(df_filt)
    analisar_momento(df_vitorias, jogador_selecionado, 'vitoria', 'Evolução de Vitórias por Jogo', 'gold')
    
    st.markdown("#### 🤝 Parcerias do Jogador")
    st.markdown("Análise das principais conexões em campo")
    conexoes = analisar_entrosamento(df_filt, jogador_selecionado)
    
    if not conexoes is None and not conexoes.empty:
        top_conexoes = conexoes.nlargest(5, 'total_participacoes')
        
        # Parcerias
        st.subheader("🎯 Top 5 Parcerias")
        st.dataframe(
            top_conexoes,
            column_config={
                'gols': st.column_config.NumberColumn('Gols', format='%d'),
                'assists': st.column_config.NumberColumn('Assistências', format='%d'),
                'total_participacoes': st.column_config.NumberColumn('Total', format='%d'),
                'taxa_vitoria': st.column_config.ProgressColumn('Taxa de Vitória', format='%.1f%%'),
                'semanas_juntos': st.column_config.NumberColumn('Semanas Juntos', format='%d')
            }
        )
else:
    st.info(f"Não há dados para {jogador_selecionado} no período selecionado.")