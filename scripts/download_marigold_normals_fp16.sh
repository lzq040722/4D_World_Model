#!/usr/bin/env bash
set -euo pipefail

cache_root=/root/autodl-tmp/huggingface/hub/models--prs-eth--marigold-normals-v0-1
snapshot="$cache_root/snapshots/b105f7cfa1972b57d8e9ed72a896b2768a52f8e9"
base_url=https://alpha.hf-mirror.com/prs-eth/marigold-normals-v0-1/resolve/main

mkdir -p "$cache_root/blobs" "$cache_root/refs" "$snapshot" "$cache_root/downloads"
printf '%s' b105f7cfa1972b57d8e9ed72a896b2768a52f8e9 > "$cache_root/refs/main"

files=(
  model_index.json
  scheduler/scheduler_config.json
  text_encoder/config.json
  text_encoder/model.fp16.safetensors
  tokenizer/merges.txt
  tokenizer/special_tokens_map.json
  tokenizer/tokenizer_config.json
  tokenizer/vocab.json
  unet/config.json
  unet/diffusion_pytorch_model.fp16.safetensors
  vae/config.json
  vae/diffusion_pytorch_model.fp16.safetensors
)

for source_file in "${files[@]}"; do
  if [ -e "$snapshot/$source_file" ]; then
    printf 'cached %s\n' "$source_file"
    continue
  fi

  staged="$cache_root/downloads/${source_file//\//__}.incomplete"
  curl -fL -C - --retry 20 --retry-all-errors --retry-delay 3 --connect-timeout 20 --max-time 0 \
    "$base_url/$source_file" -o "$staged"
  digest="$(sha256sum "$staged" | awk '{print $1}')"
  mv "$staged" "$cache_root/blobs/$digest"
  mkdir -p "$(dirname "$snapshot/$source_file")"
  ln -s "$cache_root/blobs/$digest" "$snapshot/$source_file"
  printf 'cached %s %s\n' "$source_file" "$digest"
done

printf 'MARIGOLD_NORMALS_CACHE_COMPLETE\n'
