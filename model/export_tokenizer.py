import os
import shutil
from transformers import CLIPTokenizer

def export_tokenizer(output_dir="."):
    """导出 CLIP 分词器所需的 vocab.txt 和 merges.txt"""
    print(f"Exporting CLIP tokenizer files to: {output_dir}...")
    
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载标准的 CLIP 分词器
    # 注意：这需要联网或已缓存相关权重
    try:
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        # 1. 导出 merges.txt
        # transformers 的 save_vocabulary 会导出 vocab.json 和 merges.txt
        # 我们只需要 merges.txt 和 转换后的 vocab.txt
        tokenizer.save_vocabulary(output_dir)
        
        # 2. 准备 vocab.txt
        # C++ 端通常期望一个每行一个 token 的文本文件
        # 我们从词表中提取并按索引排序写入
        vocab = tokenizer.get_vocab()
        ordered_vocab = sorted(vocab.items(), key=lambda x: x[1])
        
        vocab_path = os.path.join(output_dir, "vocab.txt")
        with open(vocab_path, "w", encoding="utf-8") as f:
            for token, index in ordered_vocab:
                f.write(f"{token}\n")
        
        # 清理多余的 vocab.json (如果存在)
        json_path = os.path.join(output_dir, "vocab.json")
        if os.path.exists(json_path):
            os.remove(json_path)
            
        print("Success! Generated files:")
        print(f"  - {os.path.join(output_dir, 'vocab.txt')}")
        print(f"  - {os.path.join(output_dir, 'merges.txt')}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Tip: Make sure you have 'transformers' installed and network access.")

if __name__ == "__main__":
    # 默认在脚本所在目录生成
    script_dir = os.path.dirname(os.path.abspath(__file__))
    export_tokenizer(script_dir)
