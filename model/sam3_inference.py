'''
pip install -U ultralytics
conda activate yolo
'''

import os
from ultralytics import SAM
# 注意：SAM3 的文本提示通常需要专门的语义预测器，或者通过特定参数传递
# 如果 ultralytics 版本更新整合了接口，可能直接用 SAM() 即可，但根据文档，这里使用更明确的导入
try:
    from ultralytics.models.sam import SAM3SemanticPredictor
except ImportError:
    print("Warning: Could not import SAM3SemanticPredictor. Ensure ultralytics is updated.")
    SAM3SemanticPredictor = None

def main():
    # ---------------------------------------------------------
    # 配置
    # ---------------------------------------------------------
    model_name = "sam3.pt"  # 请确保模型文件已下载到当前目录
    image_path = "../assets/i1.png"  # 替换为你的图片路径 
    
    # 检查模型文件是否存在
    if not os.path.exists(model_name):
        print(f"Error: Model file '{model_name}' not found. Please download it from google drive(https://drive.google.com/file/d/1zeiVSlAkVO4Tk2O-R7H3n1gYK2zi43Z3/view?usp=sharing).")
        return

    print(f"Processing image: {image_path}")

    # =========================================================
    # 1. 使用 边界框 (BBoxes) 进行分割
    # =========================================================
    print("\n--- Segmenting with Bounding Boxes ---")
    # 加载标准 SAM 模型接口
    model = SAM(model_name)
    
    # 定义边界框 [x1, y1, x2, y2]
    bbox_prompt = [330, 147, 416, 263] 
    
    # 推理
    results_box = model.predict(image_path, bboxes=[bbox_prompt])
    
    # 展示/保存结果
    for result in results_box:
        result.show()  # 弹窗显示
        result.save(filename="result_bbox.jpg")
        print("BBox segmentation saved to result_bbox.jpg")

    # =========================================================
    # 2. 使用 坐标点 (Points) 进行分割
    # =========================================================
    print("\n--- Segmenting with Points ---")
    # 定义点 [[x, y]] 和 标签 [1] (1代表前景，0代表背景)
    point_prompt = [[360, 257]]
    label_prompt = [1]
    
    # 推理
    results_point = model.predict(image_path, points=point_prompt, labels=label_prompt)
    
    for result in results_point:
        result.show()
        result.save(filename="result_point.jpg")
        print("Point segmentation saved to result_point.jpg")

    # =========================================================
    # 3. 使用 文本 (Text) 进行分割 (Promptable Concept Segmentation)
    # =========================================================
    # 注意：文本提示通常使用 SAM3SemanticPredictor 以实现"概念分割" (找出所有匹配的物体)
    print("\n--- Segmenting with Text Prompts ---")
    
    if SAM3SemanticPredictor:
        # 配置参数
        overrides = dict(conf=0.25, task="segment", mode="predict", model=model_name)
        
        # 初始化语义预测器
        predictor = SAM3SemanticPredictor(overrides=overrides)
        
        # 设置图像
        predictor.set_image(image_path)
        
        # 定义文本提示 
        text_prompt = ["cat","computer"]
        
        # 推理 (注意：根据文档 API，这里传递 text 参数)
        results_text = predictor(text=text_prompt)
        
        # 处理结果 (SAM3SemanticPredictor 返回的可能是列表或结果对象)
        # 如果 results_text 是列表
        if isinstance(results_text, list):
            for i, res in enumerate(results_text):
                # 某些版本可能需要手动绘制，或者 res 本身就是 Results 对象
                if hasattr(res, 'show'):
                    res.show()
                    res.save(filename=f"result_text_{i}.jpg")
                else:
                    print("Text result format needs manual handling depending on version.")
        else:
            # 如果是单个对象
             if hasattr(results_text, 'show'):
                results_text.show()
                results_text.save(filename="result_text.jpg")
        
        print(f"Text segmentation ('{text_prompt[0]}') saved.")
    else:
        print("Skipping Text Segmentation due to missing import.")

if __name__ == "__main__":
    main()

