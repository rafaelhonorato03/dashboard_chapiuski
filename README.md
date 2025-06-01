# Dashboard Chapiuski

Um dashboard interativo construído com Dash e Python para análise de estatísticas do Chapiuski FC.

## Configuração do Ambiente

1. Crie um ambiente virtual Python:
```bash
python -m venv .venv
```

2. Ative o ambiente virtual:
- Windows:
```bash
.venv\Scripts\activate
```
- Linux/Mac:
```bash
source .venv/bin/activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Executando o Projeto

1. Ative o ambiente virtual (se ainda não estiver ativo)
2. Execute o dashboard:
```bash
python dashboard_chapiuski.py
```
3. Abra o navegador em `http://localhost:8050`

## Deploy

Para fazer o deploy do dashboard, você pode usar serviços como:

1. **Render**:
   - Crie uma conta em render.com
   - Conecte seu repositório GitHub
   - Crie um novo Web Service
   - Configure o comando de build: `pip install -r requirements.txt`
   - Configure o comando de start: `gunicorn dashboard_chapiuski:server`

2. **Heroku**:
   - Crie uma conta no Heroku
   - Instale o Heroku CLI
   - Faça login no Heroku CLI
   - Crie um novo app: `heroku create seu-app-name`
   - Faça deploy: `git push heroku main`

3. **Railway**:
   - Crie uma conta em railway.app
   - Conecte seu repositório GitHub
   - O deploy será automático

## Estrutura do Projeto

- `dashboard_chapiuski.py`: Arquivo principal do dashboard
- `requirements.txt`: Lista de dependências do projeto
- `README.md`: Documentação do projeto

