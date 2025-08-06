# Dashboard Chapiuski

Um dashboard interativo construído com Dash e Python para análise de estatísticas do Chapiuski FC.

## 📁 Estrutura do Projeto

```
dashboard_chapiuski/
├── dados/
│   └── Chapiuski Gols.xlsx          # Dados dos jogos
├── scripts/
│   ├── dashboard_chapiuski.py        # Dashboard interativo
│   └── dashboard_animacoes.py        # Animações interativas
├── config/
│   └── gunicorn_config.py           # Configuração do servidor
├── main.py                           # Script principal
├── requirements.txt                  # Dependências
├── Procfile                         # Configuração para deploy
├── render.yaml                      # Configuração do Render
├── runtime.txt                      # Versão do Python
└── README.md                        # Documentação
```

## 🚀 Configuração do Ambiente

1. **Crie um ambiente virtual Python:**
```bash
python -m venv .venv
```

2. **Ative o ambiente virtual:**
- Windows:
```bash
.venv\Scripts\activate
```
- Linux/Mac:
```bash
source .venv/bin/activate
```

3. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

## 🎯 Executando o Projeto

### Opção 1: Script Principal (Recomendado)
Execute o script principal que oferece um menu interativo:

```bash
python main.py
```

O script oferece as seguintes opções:
- **1. Executar Dashboard Interativo**: Abre o dashboard no navegador
- **2. Executar Animações**: Mostra as animações em janelas separadas
- **3. Sair**: Encerra o programa

### Opção 2: Executar Individualmente

**Dashboard Interativo:**
```bash
python scripts/dashboard_chapiuski.py
```
Acesse: `http://localhost:8050`

**Animações:**
```bash
python scripts/dashboard_animacoes.py
```

## 📊 Funcionalidades

### Dashboard Interativo
- 📈 Gráficos de evolução de gols e assistências
- 🏆 Rankings de jogadores
- 📅 Filtros por período
- 🎯 Análise individual de jogadores
- 📊 Heatmap de participação
- 🎨 Gráfico radar de métricas

### Animações Interativas
- ⚽ Evolução de gols acumulados
- 🎯 Evolução de assistências acumuladas
- 📊 Evolução da frequência de jogos
- 📅 Animações por período (total, 2025, julho 2025)

## 🛠️ Tecnologias Utilizadas

- **Dash**: Framework para dashboards interativos
- **Plotly**: Biblioteca para gráficos interativos
- **Pandas**: Manipulação e análise de dados
- **Matplotlib**: Criação de animações
- **Seaborn**: Estilização de gráficos
- **Bootstrap**: Interface responsiva

## 🌐 Deploy

### Render (Recomendado)
1. Crie uma conta em [render.com](https://render.com)
2. Conecte seu repositório GitHub
3. Crie um novo Web Service
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn scripts.dashboard_chapiuski:server`

### Heroku
1. Crie uma conta no [Heroku](https://heroku.com)
2. Instale o Heroku CLI
3. Execute:
```bash
heroku create seu-app-name
git push heroku main
```

### Railway
1. Crie uma conta em [railway.app](https://railway.app)
2. Conecte seu repositório GitHub
3. O deploy será automático

## 📝 Notas Importantes

- ✅ **Organização**: Projeto reorganizado em pastas específicas
- ✅ **Limpeza**: Removidos todos os arquivos de imagem e GIF desnecessários
- ✅ **Interatividade**: Animações agora são exibidas ao invés de salvas
- ✅ **Simplicidade**: Script principal com menu interativo
- ✅ **Manutenibilidade**: Código mais limpo e organizado

## 🤝 Contribuição

Para contribuir com o projeto:
1. Faça um fork do repositório
2. Crie uma branch para sua feature
3. Faça commit das mudanças
4. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT.

