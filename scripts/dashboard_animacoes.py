# -*- coding: utf-8 -*-
"""
Dashboard de Animações e Visualizações de Dados Esportivos.

Este script carrega dados de um arquivo Excel, processa-os e gera 
múltiplas visualizações, incluindo:
1. Animações da evolução de Gols, Assistências e Frequência.
2. Gráficos de ranking (Bump Chart).
3. Mapas de calor (Heatmap) de performance.
4. Gráficos de barras animados.
5. Gráficos de linhas de valores acumulados.
"""

# ==============================================================================
# 1. IMPORTAÇÕES E CONFIGURAÇÕES GLOBAIS
# ==============================================================================
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import plotly.express as px

# Ignorar avisos para uma saída mais limpa
warnings.filterwarnings('ignore')

# Configurações de estilo para os gráficos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# --- Constantes de Configuração ---
# ATENÇÃO: Altere o caminho do arquivo para o local em seu computador.
FILE_PATH = r"C:\Users\tabat\Documents\GitHub\dashboard_chapiuski\dados\Chapiuski Gols.xlsx"

# Cores para os gráficos (suficiente para o maior top_n)
PLAYER_COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
    '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'
]

# ==============================================================================
# 2. FUNÇÕES DE PREPARAÇÃO DE DADOS
# ==============================================================================

def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """
    Carrega os dados de um arquivo Excel e realiza a limpeza e preparação inicial.

    Args:
        filepath (str): O caminho para o arquivo Excel.

    Returns:
        pd.DataFrame: DataFrame processado e pronto para análise.
    """
    print("📊 Carregando e preparando dados...")
    try:
        df = pd.read_excel(filepath)
    except FileNotFoundError:
        print(f"❌ ERRO: O arquivo não foi encontrado em '{filepath}'")
        return pd.DataFrame()

    # Limpeza e conversão de tipos
    df['Gol'] = pd.to_numeric(df['Gol'], errors='coerce').fillna(0)
    df['Assistência'] = pd.to_numeric(df['Assistência'], errors='coerce').fillna(0)
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
    df.dropna(subset=['Data'], inplace=True)
    df.sort_values('Data', inplace=True)

    # Cálculo de métricas acumuladas para o período total
    df['Gols Acumulados'] = df.groupby('Jogador')['Gol'].cumsum()
    df['Assistências Acumuladas'] = df.groupby('Jogador')['Assistência'].cumsum()
    df['Frequência'] = df.groupby('Jogador').cumcount() + 1
    
    # Coluna auxiliar para agrupamentos mensais
    df['AnoMes'] = df['Data'].dt.to_period('M').astype(str)

    print(f"✓ Dados carregados: {len(df)} registros de {df['Data'].min():%d/%m/%Y} a {df['Data'].max():%d/%m/%Y}")
    return df

# ==============================================================================
# 3. FUNÇÕES DE VISUALIZAÇÃO
# ==============================================================================

def create_evolution_animation(df_period: pd.DataFrame, value_col: str, title: str, top_n: int = 8):
    """
    Cria e salva uma animação da evolução de uma métrica ao longo do tempo.

    Args:
        df_period (pd.DataFrame): DataFrame do período a ser animado.
        value_col (str): Nome da coluna com os valores a serem plotados.
        title (str): Título principal do gráfico.
        top_n (int): Número de jogadores a serem exibidos no ranking.
    """
    print(f"\n🎬 Criando animação: {title}")
    
    if df_period.empty:
        print(f"⚠️ Aviso: Não há dados para o período de '{title}'. Animação pulada.")
        return

    # Filtra os top N jogadores baseado no valor máximo do período
    top_players = df_period.groupby('Jogador')[value_col].max().nlargest(top_n).index
    df_top = df_period[df_period['Jogador'].isin(top_players)]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#ffffff')

    unique_dates = df_top['Data'].unique()

    def animate(frame):
        ax.clear()
        current_date = unique_dates[frame]
        data_so_far = df_top[df_top['Data'] <= current_date]
        
        # Plotar linhas para cada jogador
        for i, player in enumerate(top_players):
            player_data = data_so_far[data_so_far['Jogador'] == player]
            if not player_data.empty:
                ax.plot(player_data['Data'], player_data[value_col], 
                        marker='o', label=player, linewidth=3, markersize=8, 
                        color=PLAYER_COLORS[i], alpha=0.8, markerfacecolor='white', 
                        markeredgewidth=2, markeredgecolor=PLAYER_COLORS[i])
                
                # Anotar o último ponto
                last_entry = player_data.iloc[-1]
                sufix = 'g' if 'Gol' in value_col else 'a' if 'Assist' in value_col else 'j'
                ax.text(last_entry['Data'], last_entry[value_col] + 0.1, f" {last_entry[value_col]:.0f}{sufix}",
                        fontsize=12, fontweight='bold', color=PLAYER_COLORS[i])

        # Configurações do eixo e títulos
        ax.set_xlim(df_top['Data'].min(), df_top['Data'].max())
        ax.set_ylim(0, df_top[value_col].max() * 1.15)
        ax.set_title(f'{title}\n{pd.to_datetime(current_date).strftime("%d/%m/%Y")}', 
                     fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel('Data', fontsize=14)
        ax.set_ylabel(value_col.replace('_', ' '), fontsize=14)
        ax.legend(title='Jogador', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
        ax.grid(True, which='major', linestyle='--', linewidth=0.5)
        plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Criar e salvar a animação
    ani = animation.FuncAnimation(fig, animate, frames=len(unique_dates), interval=200, repeat=False)
    
    # Salvar em vez de mostrar para evitar problemas de múltiplos plt.show()
    output_filename = f"{title.replace(' ', '_').lower()}.gif"
    ani.save(output_filename, writer='pillow', fps=5)
    print(f"✓ Animação salva como '{output_filename}'")
    plt.close(fig) # Fecha a figura para liberar memória


def plot_bump_chart(df: pd.DataFrame, value_col: str = 'Gols Acumulados', title: str = '🏆 Ranking de Artilheiros', top_n: int = 8):
    """Cria e exibe um bump chart para mostrar a evolução do ranking."""
    print(f"\n📈 Gerando Bump Chart: {title}")
    
    # Prepara o ranking por data
    datas = df['Data'].sort_values().unique()
    rank_frames = []
    for data in datas:
        temp = df[df['Data'] <= data].copy()
        # Pega o último registro de cada jogador até a data atual
        latest = temp.loc[temp.groupby('Jogador')['Data'].idxmax()]
        latest['Rank'] = latest[value_col].rank(method='first', ascending=False)
        latest = latest[latest['Rank'] <= top_n]
        latest['Data'] = pd.to_datetime(data)
        rank_frames.append(latest[['Data', 'Jogador', 'Rank']])

    df_ranking = pd.concat(rank_frames).drop_duplicates(subset=['Data', 'Jogador'])
    
    plt.figure(figsize=(14, 8))
    for i, player in enumerate(df_ranking['Jogador'].unique()):
        player_data = df_ranking[df_ranking['Jogador'] == player]
        plt.plot(player_data['Data'], player_data['Rank'], marker='o', label=player, color=PLAYER_COLORS[i])

    plt.gca().invert_yaxis() # 1° lugar no topo
    plt.title(title, fontsize=18, weight='bold')
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Posição no Ranking', fontsize=12)
    plt.legend(title='Jogador', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()


def plot_heatmap(df: pd.DataFrame, title: str = "🔥 Heatmap de Gols por Jogador e Mês"):
    """Cria e exibe um heatmap da performance mensal dos jogadores."""
    print(f"\n heatmap: {title}")
    heatmap_df = df.groupby(['Jogador', 'AnoMes'])['Gol'].sum().unstack().fillna(0)
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(heatmap_df, cmap="YlGnBu", annot=True, fmt=".0f", linewidths=.5)
    plt.title(title, fontsize=18, weight='bold')
    plt.xlabel("Mês", fontsize=12)
    plt.ylabel("Jogador", fontsize=12)
    plt.tight_layout()


def plot_animated_bar_chart(df: pd.DataFrame, title: str = '📊 Gols por Jogador (Evolução Mensal)'):
    """Cria e exibe um gráfico de barras animado com Plotly."""
    print(f"\n📊 Gerando Bar Chart animado: {title}")
    df_grouped = df.groupby(['AnoMes', 'Jogador'])['Gol'].sum().reset_index()
    
    fig = px.bar(df_grouped,
                 x='Jogador',
                 y='Gol',
                 color='Jogador',
                 animation_frame='AnoMes',
                 range_y=[0, df_grouped['Gol'].max() + 1],
                 title=title,
                 labels={'Gol': 'Gols no Mês', 'AnoMes': 'Mês'})

    fig.update_layout(xaxis={'categoryorder':'total descending'},
                      font=dict(size=14))
    fig.show()

# ==============================================================================
# 4. FUNÇÃO PRINCIPAL E EXECUÇÃO
# ==============================================================================

def main():
    """
    Função principal que orquestra o carregamento, processamento e visualização dos dados.
    """
    print("🎯 DASHBOARD DE ANIMAÇÕES - CHAPIUSKI GOLS")
    print("="*50)
    
    # Carrega os dados
    df_main = load_and_prepare_data(FILE_PATH)
    if df_main.empty:
        return # Termina a execução se os dados não puderem ser carregados

    # --- Definição dos períodos para as animações ---
    periods = [
        {'name': 'Período Total', 'year': None, 'month': None, 'top_n': 8},
        {'name': 'Ano 2025', 'year': 2025, 'month': None, 'top_n': 6},
        {'name': 'Julho 2025', 'year': 2025, 'month': 7, 'top_n': 5}
    ]
    
    metrics = [
        {'col': 'Gols Acumulados', 'title': '⚽ Gols Acumulados'},
        {'col': 'Assistências Acumuladas', 'title': '🎯 Assistências Acumuladas'},
        {'col': 'Frequência', 'title': '📊 Frequência de Jogos'}
    ]

    # --- Geração das Animações ---
    for period in periods:
        df_filtered = df_main.copy()
        
        # Filtra por ano e mês, se especificado
        if period['year']:
            df_filtered = df_filtered[df_filtered['Data'].dt.year == period['year']]
        if period['month']:
            df_filtered = df_filtered[df_filtered['Data'].dt.month == period['month']]
        
        if df_filtered.empty:
            print(f"\n⚠️ Sem dados para o período: {period['name']}. Gráficos não serão gerados.")
            continue
            
        # Recalcula as métricas acumuladas apenas para o período filtrado
        df_filtered['Gols Acumulados'] = df_filtered.groupby('Jogador')['Gol'].cumsum()
        df_filtered['Assistências Acumuladas'] = df_filtered.groupby('Jogador')['Assistência'].cumsum()
        df_filtered['Frequência'] = df_filtered.groupby('Jogador').cumcount() + 1
        
        for metric in metrics:
            create_evolution_animation(
                df_period=df_filtered,
                value_col=metric['col'],
                title=f"{metric['title']} - {period['name']}",
                top_n=period['top_n']
            )

    # --- Geração de Outros Gráficos ---
    print("\n" + "="*50)
    print("🎨 Gerando visualizações adicionais...")

    # Gráfico de Barras Animado (Plotly)
    plot_animated_bar_chart(df_main)

    # Gráfico de Ranking (Bump Chart) e Heatmap (Matplotlib)
    plot_bump_chart(df_main)
    plot_heatmap(df_main)

    print("\n🚀 Processo concluído! Exibindo gráficos estáticos.")
    print("="*50)
    
    # Mostra todos os gráficos gerados com Matplotlib
    plt.show()


if __name__ == "__main__":
    main()