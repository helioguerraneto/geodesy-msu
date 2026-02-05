# GNSS Great Lakes Viewer - Deploy Guide

## 🎯 Sobre o Projeto

Visualizador interativo de estações GNSS da região dos Grandes Lagos com dados hospedados no Google Drive.

**Links dos dados:**
- ✅ Linear: `1UQdyJahGg1gxb2WSUFs1ge9wUYUPXq93`
- ✅ CM: `1Xdcp8j4m4VjOwDe3ryOdNm3ge2aVqmLf`  
- ✅ CF: `1Sc8g2GdEodO7NAeDZWkbM1MlbFNA_T5y`
- ✅ glstations.txt: `17mi5FA44LvnWr-50-bLgrdbrBsuMU-bK`

---

## 📦 Arquivos do Projeto

- `app.py` - Código principal do Dash
- `requirements.txt` - Dependências Python
- `Procfile` - Configuração para Render.com
- `README.md` - Este arquivo

---

## 🚀 Deploy no Render.com - PASSO A PASSO

### PASSO 1: Criar Repositório no GitHub

1. Acesse https://github.com e faça login
2. Clique em **"New repository"** (botão verde)
3. Configure:
   - **Repository name:** `great-lakes-gnss`
   - **Description:** "GNSS Great Lakes Viewer"
   - **Visibilidade:** ✅ **Public** (obrigatório)
   - **Initialize:** Não marque nada
4. Clique em **"Create repository"**

### PASSO 2: Upload dos Arquivos no GitHub

**Opção A: Via Web (Mais Fácil)**

1. Na página do repositório, clique em **"uploading an existing file"**
2. Arraste os 4 arquivos para a área de upload:
   - `app.py`
   - `requirements.txt`
   - `Procfile`
   - `README.md`
3. Role até o fim e clique em **"Commit changes"**

**Opção B: Via Git (Se você usa linha de comando)**

```bash
# No terminal, na pasta onde estão os arquivos:
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/great-lakes-gnss.git
git push -u origin main
```

### PASSO 3: Conectar GitHub ao Render

1. Faça login em https://render.com
2. No painel, clique em **"New +"** → **"Web Service"**
3. Clique em **"Connect a repository"**
4. Se for a primeira vez:
   - Clique em **"Connect GitHub"**
   - Autorize o Render a acessar seus repositórios
5. Selecione o repositório **`great-lakes-gnss`**

### PASSO 4: Configurar o Web Service

Preencha os campos:

- **Name:** `great-lakes-gnss` (ou o nome que preferir)
- **Region:** `Oregon (US West)` ou o mais próximo
- **Branch:** `main`
- **Root Directory:** (deixe em branco)
- **Environment:** **Python 3**
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn app:server`

**Plan:** Selecione **Free**

### PASSO 5: Variáveis de Ambiente (OPCIONAL)

Se quiser, pode adicionar:
- `PORT` = `8050` (mas o Render já define isso automaticamente)

### PASSO 6: Deploy!

1. Clique em **"Create Web Service"**
2. Aguarde 2-5 minutos enquanto o Render:
   - ✅ Clona o repositório
   - ✅ Instala as dependências
   - ✅ Inicia o servidor
3. Quando aparecer **"Your service is live 🎉"**, clique no link (ex: `https://great-lakes-gnss.onrender.com`)

---

## ⚙️ Como o App Funciona

1. **Carrega lista de estações** do Google Drive (`glstations.txt`)
2. **Mostra mapa** com todas as estações
3. **Ao clicar em uma estação:**
   - Descobre o file ID do arquivo `.pfiles` via web scraping
   - Baixa APENAS esse arquivo do Google Drive
   - Processa e mostra os gráficos

**Vantagens:**
- ✅ Não precisa baixar os 9GB de dados
- ✅ Rápido (só baixa o que precisa)
- ✅ Funciona sem API key do Google

---

## 🧪 Testar Localmente (OPCIONAL)

Se quiser testar antes de fazer deploy:

```bash
# Instalar dependências
pip install -r requirements.txt

# Rodar o app
python app.py

# Abrir no navegador
# http://localhost:8050
```

---

## 🐛 Troubleshooting

### Erro: "Could not load data for station X"

**Causa:** O Google Drive pode bloquear web scraping temporariamente

**Soluções:**
1. Tente outra estação
2. Aguarde alguns minutos
3. Se persistir, pode ser necessário criar um `file_index.json` manualmente

### Erro: "Application Error" no Render

**Verifique os logs:**
1. No painel do Render, clique em **"Logs"**
2. Procure por mensagens de erro
3. Erros comuns:
   - Dependência faltando → Adicione no `requirements.txt`
   - Porta incorreta → Já está configurada corretamente
   - Erro de sintaxe → Verifique o `app.py`

### App carrega mas não mostra estações

**Causa:** Problema ao baixar `glstations.txt`

**Solução:**
1. Verifique se o link do Google Drive está público
2. Tente acessar: `https://drive.google.com/uc?export=download&id=17mi5FA44LvnWr-50-bLgrdbrBsuMU-bK`
3. Se não funcionar, re-compartilhe o arquivo

---

## 🔄 Atualizar o App

Sempre que você modificar algum arquivo:

1. Faça commit no GitHub (via web ou git)
2. O Render detecta automaticamente
3. Faz redeploy em ~2 minutos

---

## 📊 Limitações do Plano Gratuito do Render

- ⏱️ App "dorme" após 15 minutos sem uso
- 🐌 Primeiro acesso depois de dormir é lento (~30 segundos)
- 💾 750 horas/mês de uso gratuito (mais que suficiente)

---

## ✅ Checklist Final

Antes de fazer deploy, confirme:

- [ ] Os 4 arquivos foram criados
- [ ] Upload no GitHub foi feito
- [ ] Repositório é público
- [ ] Conectei GitHub ao Render
- [ ] Selecionei Python 3 como environment
- [ ] Start command é `gunicorn app:server`

**Boa sorte! 🚀**

---

## 📞 Próximos Passos

Depois do deploy funcionar:

1. **Teste clicando em várias estações** para verificar se os dados carregam
2. **Se algumas estações não funcionarem**, é normal (pode ser limitação de web scraping)
3. **Compartilhe o link** com seu orientador/equipe

---

## 🎓 Créditos

**MSU Geodesy Lab - The Great Lakes GNSS Stations**

Desenvolvido para visualização de dados de deformação crustal na região dos Grandes Lagos.

