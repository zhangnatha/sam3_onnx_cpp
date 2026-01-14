# SAM3 模型导出与资源说明

本目录包含将 SAM3 PyTorch 模型 (`.pt`) 转换为 ONNX 格式所需的工具，以及 C++ 推理所需的文本分词资源。

👈 **[返回 C++ 推理引擎主文档 (../README.md)](../README.md)**

## 导出工具说明

### 1. 模型导出脚本 (`export_sam3_onnx.py`)
该脚本负责将神经网络权重从 `.pt` 文件导出为 5 个 ONNX 格式的文件。它包含针对 ONNX 兼容性的关键补丁（如 RoPE 旋转位置编码和输入掩码处理）。

**使用方法**:
```bash
python export_sam3_onnx.py
```

### 2. 分词器导出脚本 (`export_tokenizer.py`)
SAM3 的文本分割功能依赖 CLIP 分词器。该脚本将分词器所需的词表和合并规则导出为 C++ 可读的文本格式。

**使用方法**:
```bash
python export_tokenizer.py
```
*注意：需要安装 `transformers` 库。*

---

## 文件列表及作用

### 神经网络模型 (ONNX)
这些文件由 `export_sam3_onnx.py` 生成：

1.  **`sam3_encoder.onnx`** (交互式编码器):
    *   **作用**: 处理点/框分割模式下的输入图像。
    *   **细节**: 生成多尺度图像特征，供 `sam3_decoder.onnx` 使用。
2.  **`sam3_decoder.onnx`** (交互式解码器):
    *   **作用**: 处理点、框提示词并输出分割掩码。
    *   **细节**: 接受编码器特征和坐标提示，输出 3 个候选掩码及 IOU 置信度。
3.  **`sam3_language_encoder.onnx`** (语言编码器):
    *   **作用**: 将输入的文本 ID 序列转换为语义向量。
    *   **细节**: 它是 Grounding (文本分割) 管道的第一步。
4.  **`sam3_grounding_encoder.onnx`** (文本编码器/图像支柱):
    *   **作用**: 专门为文本分割任务提取图像特征。
    *   **细节**: 针对 CLIP 语义空间进行了对齐。
5.  **`sam3_grounding_decoder.onnx`** (文本解码器):
    *   **作用**: 融合图像特征和文本语义，生成物体检测框和掩码。
    *   **细节**: 支持开放词汇检测（Open-Vocabulary Detection）。

### 文本分词资源 (TXT)
这些文件由 `export_tokenizer.py` 生成（或手动放置）：

6.  **`vocab.txt`**:
    *   **作用**: 词表文件。映射每个子词（Subword）到一个唯一的整数 ID。
    *   **重要性**: 没有它，C++ 无法理解 "cat" 或 "dog" 对应的输入数字。
7.  **`merges.txt`**:
    *   **作用**: BPE 合并规则文件。定义了如何将单词拆分成子词。
    *   **重要性**: 确保字符串拆分逻辑与 Python 端模型训练时完全一致。

---

## C++ 集成提示
在 C++ 的 `SAM3Predictor` 构造函数中，你需要提供 `model` 目录的路径。程序会自动寻找上述 7 个文件。如果缺失任何一个，对应的模式（文本模式或点框模式）将无法启动。
