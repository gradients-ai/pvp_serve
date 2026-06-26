# Deploying pvp_serve on a GPU node

Goal: a stable HTTPS endpoint serving the champion model + the play API, which
`gradients-web` points `PUBLIC_PLAY_API_URL` at.

## Sizing

The champion is a ~7B (Qwen2.5-7B lineage) model served in fp16:

- Weights ≈ 14 GB + KV cache + SGLang overhead.
- **24 GB GPU (RTX 4090 / L4)**: works, tighter KV cache.
- **40–48 GB GPU (A100-40G / L40S)**: comfortable, headroom for batching. Recommended.

Single GPU, `--tensor-parallel-size 1`. CPU/RAM modest. The play API itself is
CPU-only and light.

## Layout on the box

```
/opt/pvp_serve            # this repo (+ ./god submodule)
  .venv                   # python env: pip install -e . AND pyspiel/open_spiel
SGLang                    # serves the model on :30000 (loopback)
pvp_serve API             # :8000 (loopback)
Caddy/nginx               # :443 TLS -> 127.0.0.1:8000, public
```

Keep SGLang and the API on loopback; expose only the TLS reverse proxy.

## Steps

```bash
# 0. prereqs: NVIDIA driver + CUDA, python3.10+, git, caddy (or nginx+certbot)

# 1. code
git clone git@github.com:gradients-ai/pvp_serve.git /opt/pvp_serve
cd /opt/pvp_serve
git submodule update --init --recursive    # G.O.D pinned to main

# 2. env
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install open_spiel          # or build from source / vendor a wheel
pip install sglang[all]         # model server

# 3. secrets
export HF_TOKEN=hf_...           # access to gradients-io-tournaments (gated)
# from grads_vps:/root/G.O.D/.vali.env -> HUGGINGFACE_TOKEN

# 4. serve the model (pm2/systemd in prod)
HF_TOKEN=$HF_TOKEN ./scripts/serve_sglang.sh    # downloads + serves on :30000

# 5. run the API
PVP_AGENT_KIND=llm \
PVP_SGLANG_BASE_URL=http://127.0.0.1:30000/v1 \
PVP_INFERENCE_MODEL=champion \
PVP_CORS_ORIGINS=https://gradients.io \
./scripts/run.sh                                 # :8000
```

### Process management (pm2 example)

```bash
pm2 start scripts/serve_sglang.sh --name champion-sglang --update-env
pm2 start scripts/run.sh           --name pvp-api        --update-env
pm2 save
```

### TLS (Caddy)

`gradients-web` is served over HTTPS, so the play API **must** be HTTPS too
(browsers block mixed content). Minimal Caddyfile:

```
play-api.gradients.io {
    reverse_proxy 127.0.0.1:8000
}
```

Caddy auto-provisions the cert. Point a DNS A record for
`play-api.gradients.io` at the box first.

## Wire up the frontend

Set in the `gradients-web` deploy environment:

```
PUBLIC_PLAY_API_URL=https://play-api.gradients.io
```

When set, the app uses `HttpConnector` (live model); when unset it falls back to
the offline `MockConnector`. No rebuild logic change needed.

## Smoke test the live box

```bash
curl https://play-api.gradients.io/health
# create a session and watch the champion move:
curl -s -X POST https://play-api.gradients.io/session \
  -H 'content-type: application/json' -d '{"game":"othello","humanSeat":0}' | jq .view.toMove
```

## Refreshing the champion after a new tournament

1. Query the new winner repo (see README).
2. `PVP_MODEL_REPO=<new repo> pm2 restart champion-sglang --update-env`.
3. No API change needed.
