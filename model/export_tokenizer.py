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
        from huggingface_hub import hf_hub_download
        
        # 1. 获取 merges.txt
        merges_src = hf_hub_download(repo_id="openai/clip-vit-base-patch32", filename="merges.txt")
        merges_dest = os.path.join(output_dir, "merges.txt")
        shutil.copyfile(merges_src, merges_dest)

        # 2. 获取 vocab.txt (从 CLIPTokenizer 词表导出)
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        vocab = tokenizer.get_vocab()
        ordered_vocab = sorted(vocab.items(), key=lambda x: x[1])
        
        vocab_path = os.path.join(output_dir, "vocab.txt")
        with open(vocab_path, "w", encoding="utf-8") as f:
            for token, index in ordered_vocab:
                f.write(f"{token}\n")
        
        # 清理多余临时文件 (如果存在)
        for extra in ["vocab.json", "tokenizer.model"]:
            p = os.path.join(output_dir, extra)
            if os.path.exists(p):
                os.remove(p)
            
        print("Success! Generated files:")
        print(f"  - {os.path.join(output_dir, 'vocab.txt')}")
        print(f"  - {os.path.join(output_dir, 'merges.txt')}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Tip: Make sure you have 'transformers' and 'huggingface_hub' installed and network access.")

if __name__ == "__main__":
    # 默认在脚本所在目录生成
    script_dir = os.path.dirname(os.path.abspath(__file__))
    export_tokenizer(script_dir)
