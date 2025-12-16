#!/usr/bin/env bash
set -euo pipefail

print_usage() {
  cat <<'USAGE'
Usage: bash make_alpha.sh <input-prores-mov>

このスクリプトは「DaVinci Resolveでアルファ付きProRes4444として書き出した動画」
から以下の2形式を自動生成します:

  1. WebM (VP9 + Alpha)   → Chrome/Android用
  2. HEVC (H.265 + Alpha) → Safari/iOS用

USAGE
}

if [[ $# -ne 1 ]]; then
  print_usage >&2
  exit 1
fi

INPUT_PATH="$1"

if [[ ! -f "$INPUT_PATH" ]]; then
  echo "入力ファイルが存在しません: $INPUT_PATH" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg が見つかりません。" >&2
  exit 1
fi

INPUT_DIR="$(cd "$(dirname "$INPUT_PATH")" && pwd)"
INPUT_FILE="$(basename "$INPUT_PATH")"
INPUT_BASE="${INPUT_FILE%.*}"

WEBM_PATH="$INPUT_DIR/${INPUT_BASE}.webm"
HEVC_PATH="$INPUT_DIR/${INPUT_BASE}.mp4"
IOS_FRAME_PATTERN="$INPUT_DIR/ios_frame_%03d.png"

echo "入力:         $INPUT_PATH"
echo "出力(WebM):   $WEBM_PATH"
echo "出力(HEVC):   $HEVC_PATH"
echo "iOS PNG:      ${IOS_FRAME_PATTERN//%03d/*}"
echo

#################################
# 1. WebM (VP9 + Alpha)
#################################
echo "==> WebM (VP9 + Alpha) を生成中..."

ffmpeg -hide_banner -y \
  -i "$INPUT_PATH" \
  -c:v libvpx-vp9 \
  -pix_fmt yuva420p \
  -b:v 0 \
  -crf 28 \
  -auto-alt-ref 0 \
  "$WEBM_PATH"

echo "WebM 完了: $WEBM_PATH"
echo

#################################
# 2. HEVC (H.265 + Alpha)
#################################
echo "==> HEVC with Alpha (hvc1) を生成中..."

ffmpeg -hide_banner -y \
  -i "$INPUT_PATH" \
  -c:v hevc_videotoolbox \
  -tag:v hvc1 \
  -alpha_quality 1 \
  "$HEVC_PATH"

echo "HEVC 完了: $HEVC_PATH"
echo

#################################
# 3. Safari用 PNG 連番
#################################
echo "==> iOS Safari 用 PNG連番を生成中..."
ffmpeg -hide_banner -y \
  -i "$INPUT_PATH" \
  -vf "fps=10,scale=720:1280:flags=lanczos" \
  "$IOS_FRAME_PATTERN"

if command -v pngquant >/dev/null 2>&1; then
  echo "==> PNG を pngquant で圧縮中..."
  pngquant --quality=70-95 --speed=1 "$INPUT_DIR"/ios_frame_*.png --ext .png --force
else
  echo "pngquant が見つからないため圧縮をスキップしました。" >&2
fi

echo
echo "=== 全て完了しました ==="
