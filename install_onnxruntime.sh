#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_GPU_DIR="${SCRIPT_DIR}/3rdparty/onnxruntime-gpu"
INSTALL_CPU_DIR="${SCRIPT_DIR}/3rdparty/onnxruntime-cpu"
ORT_VERSION="1.17.3"

mkdir -p "${SCRIPT_DIR}/3rdparty"

# 检测 CUDA 版本
CUDA_MAJOR=12
if command -v nvcc &> /dev/null; then
    CUDA_VER=$(nvcc --version | grep "release" | sed -n -e 's/^.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d'.' -f1)
    echo "检测到系统 CUDA 版本: $CUDA_VER (主版本: $CUDA_MAJOR)"
fi

if [ "$CUDA_MAJOR" -eq 11 ]; then
    GPU_TGZ="onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz"
else
    GPU_TGZ="onnxruntime-linux-x64-gpu-cuda12-${ORT_VERSION}.tgz"
fi
CPU_TGZ="onnxruntime-linux-x64-${ORT_VERSION}.tgz"

CPU_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${CPU_TGZ}"
GPU_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${GPU_TGZ}"

echo "=========================================="
echo "安装 ONNX Runtime ${ORT_VERSION}"
echo "  CPU  -> ${INSTALL_CPU_DIR}"
echo "  GPU  -> ${INSTALL_GPU_DIR}"
echo "=========================================="

download_and_extract() {
    local url="$1"
    local tgz="$2"
    local dest="$3"

    echo "下载 $tgz ..."
    wget -c "$url" -O "$tgz"
    
    echo "解压 $tgz 到 $dest ..."
    local tmp_dir
    tmp_dir=$(mktemp -d "${SCRIPT_DIR}/.ort_tmp_XXXXXX")
    tar -xzf "$tgz" -C "$tmp_dir"
    
    local extracted_dir
    extracted_dir=$(find "$tmp_dir" -mindepth 1 -maxdepth 1 -type d | head -n 1)
    
    mkdir -p "$dest"
    cp -r "$extracted_dir"/* "$dest/"
    
    rm -rf "$tmp_dir" "$tgz"
}

download_and_extract "$CPU_URL" "$CPU_TGZ" "$INSTALL_CPU_DIR"
download_and_extract "$GPU_URL" "$GPU_TGZ" "$INSTALL_GPU_DIR"

echo "ONNX Runtime ${ORT_VERSION} 安装完成！"
