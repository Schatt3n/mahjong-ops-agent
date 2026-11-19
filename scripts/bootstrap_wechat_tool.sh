#!/bin/bash
# 按固定发布版本安装本机工具，仅向调用方输出可执行文件路径。
set -euo pipefail
umask 077
root="$(cd -- "$(dirname -- "$0")/.." && pwd)"
tool="${1:?需要工具名称 python、node 或 wx}"
platform="$(uname -s)-$(uname -m)"
if [[ "$platform" == Darwin-x86_64 && "$(sysctl -n hw.optional.arm64 2>/dev/null || true)" == 1 ]]; then
    platform=Darwin-arm64
fi

if ! IFS=$'\t' read -r version url expected executable < <(
    awk -F '\t' -v tool="$tool" -v platform="$platform" \
        '$1 == tool && $2 == platform {print $3 "\t" $4 "\t" $5 "\t" $6; exit}' \
        "$root/scripts/wechat_tool_downloads.tsv"
); then
    printf '当前平台 %s 没有可自动安装的 %s，请提供本机可执行文件。\n' "$platform" "$tool" >&2
    exit 1
fi

cache="$root/runtime_data/wechat-stack-tools"
install="$cache/$tool-$version-$platform"
if [[ -x "$install" ]]; then
    printf '%s\n' "$install"
    exit 0
fi

mkdir -p -- "$cache"
staging="$(mktemp -d "$cache/.${tool}.XXXXXXXX")"
trap 'if [[ -n "$staging" ]]; then rm -rf -- "$staging"; fi' EXIT
archive="$staging/download"

printf '首次启动，正在下载 %s %s…\n' "$tool" "$version" >&2
curl_options=(-fsSL)
if [[ -t 2 ]]; then
    curl_options=(-fL --progress-bar)
fi
curl "${curl_options[@]}" --proto '=https' --proto-redir '=https' --retry 2 --connect-timeout 15 --max-time 300 \
    -o "$archive" "$url"
if command -v shasum >/dev/null 2>&1; then
    checksum="$(shasum -a 256 < "$archive")"
else
    checksum="$(sha256sum < "$archive")"
fi
if [[ "${checksum%% *}" != "$expected" ]]; then
    printf '%s 的下载校验失败，请稍后重试。\n' "$tool" >&2
    exit 1
fi

if [[ "$tool" == wx ]]; then
    mv -- "$archive" "$staging/$executable"
    chmod 700 "$staging/$executable"
else
    tar -xzf "$archive" -C "$staging"
    rm -- "$archive"
fi
if [[ ! -f "$staging/$executable" || ! -x "$staging/$executable" ]]; then
    printf '已校验的 %s 安装包缺少可执行文件。\n' "$tool" >&2
    exit 1
fi

# 完整解压后再创建入口；若另一个启动同时安装同一版本，只保留先完成的一份。
if ln -s "${staging##*/}/$executable" "$install" 2>/dev/null; then
    staging=''
elif [[ ! -x "$install" ]]; then
    printf '无法启用 %s，请检查 %s 是否可写。\n' "$tool" "$cache" >&2
    exit 1
fi
printf '%s\n' "$install"
