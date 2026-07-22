# Deploying Adaptyv Copilot to a public URL

Goal: a link like `https://adaptyv-copilot.streamlit.app` you can send to Adaptyv.
Hosting is **free** via Streamlit Community Cloud, and it installs everything on
Streamlit's servers — so your own machine's disk space doesn't matter.

## Before you start
- A **GitHub** account — https://github.com/signup
- A **Streamlit Community Cloud** account (sign in with GitHub) — https://share.streamlit.io
- An **Anthropic API key** — https://console.anthropic.com
  → In the console, set a **monthly spend limit** (Billing → Limits). A public
    demo should be capped; the app only spends on Claude tokens, not lab assays.

## Step 1 — put the code on GitHub
From this folder:

```bash
git init
git add .
git commit -m "Adaptyv Copilot"
```

Create an **empty** repo on GitHub (no README), then:

```bash
git remote add origin https://github.com/<you>/adaptyv-copilot.git
git branch -M main
git push -u origin main
```

> `.gitignore` already excludes `.env` and `secrets.toml`, so your keys never
> get uploaded. Good.

## Step 2 — deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io → **Create app** → pick your GitHub repo.
2. Set **Main file path** to `app.py`.
3. Click **Advanced settings → Secrets** and paste (with your real values):

   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ADAPTYV_MODE = "simulated"
   APP_PASSWORD = "a-password-for-adaptyv"
   ```

4. **Deploy.** First build takes a couple of minutes while it installs
   `requirements.txt`. You'll get a public `*.streamlit.app` URL.

## Step 3 — share it
Send Adaptyv the URL and the password. They open it, type a goal like
*"Find me a strong EGFR binder"*, and watch the agent work.

## Safety recap
- **Password gate** (`APP_PASSWORD`) keeps random visitors out.
- **`ADAPTYV_MODE=simulated`** means no real (paid) lab orders are ever placed.
- **Anthropic spend limit** caps the only real cost (Claude tokens).
- To rotate/kill access later: change `APP_PASSWORD` in Secrets, or delete the
  app from the Streamlit dashboard.
