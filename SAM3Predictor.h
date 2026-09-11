#ifndef SAM3_PREDICTOR_H
#define SAM3_PREDICTOR_H

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#ifdef USE_CUDA
#include <cuda_runtime.h>
extern "C" void launch_preprocess(const unsigned char *d_input, float *d_output,
                                  int width, int height, bool is_bgr,
                                  cudaStream_t stream);
#endif

#include "SimpleTokenizer.h"

// SAM3预测器类
class SAM3Predictor {
public:
  // 推理结果结构体
  struct InferenceResult {
    cv::Mat mask;      // 分割掩码
    float score;       // 置信度分数
    cv::Rect2f box;    // 边界框
    std::string label; // 类别标签
  };

  // 构造函数: 初始化环境和分词器
  SAM3Predictor(const std::string &model_dir, bool use_gpu = false);
  ~SAM3Predictor();

  // 禁用拷贝语义，避免资源与 Session 重复析构
  SAM3Predictor(const SAM3Predictor &) = delete;
  SAM3Predictor &operator=(const SAM3Predictor &) = delete;

  // 单文本提示 -> Grounding Pipeline
  std::vector<InferenceResult> predict_text(const cv::Mat &bgr_img,
                                            const std::string &text,
                                            float threshold = 0.25f,
                                            int max_detections = 0,
                                            float nms_threshold = 0.5f);

  // 批量文本提示 -> Grounding Pipeline (图像只编码一次，大幅提升多标签推理速度)
  std::vector<InferenceResult> predict_texts(const cv::Mat &bgr_img,
                                             const std::vector<std::string> &texts,
                                             float threshold = 0.25f,
                                             int max_detections = 0,
                                             float nms_threshold = 0.5f);

  // 单点提示 -> Interactive Pipeline
  std::vector<InferenceResult> predict_point(const cv::Mat &bgr_img,
                                             const cv::Point2f &point,
                                             float threshold = 0.5f,
                                             int max_detections = 0,
                                             std::string label = "object");

  // 多点提示 -> Interactive Pipeline (支持正负点 labels: 1=前景, 0=背景)
  std::vector<InferenceResult> predict_points(const cv::Mat &bgr_img,
                                              const std::vector<cv::Point2f> &points,
                                              const std::vector<int> &labels,
                                              float threshold = 0.5f,
                                              int max_detections = 0,
                                              std::string label = "object");

  // 单框提示 -> Interactive Pipeline
  std::vector<InferenceResult> predict_box(const cv::Mat &bgr_img,
                                           const cv::Rect2f &box,
                                           float threshold = 0.5f,
                                           int max_detections = 0,
                                           std::string label = "object");

  // 多框提示 -> Interactive Pipeline
  std::vector<InferenceResult> predict_boxes(const cv::Mat &bgr_img,
                                             const std::vector<cv::Rect2f> &boxes,
                                             float threshold = 0.5f,
                                             int max_detections = 0,
                                             std::string label = "object");

  /**
   * @brief 非极大值抑制 (NMS)，抑制重叠的高置信度检测框
   */
  static std::vector<InferenceResult> apply_nms(
      const std::vector<InferenceResult> &candidates,
      float iou_threshold = 0.5f);

  /**
   * @brief 预热模型 (主动加载程序)
   * 在生产环境下，建议在初始化后显式调用此函数。
   * 这将触发 ONNX Runtime 的会话创建和 GPU 显存分配，
   * 从而避免首次推理时的显著延迟。
   */
  void warmup();

private:
  // Interactive Pipeline 内部结果结构体
  struct InteractiveResult {
    cv::Mat mask;
    float score;
  };

  Ort::Env env;
  std::string model_dir;
  bool use_gpu;

  // Grounding Pipeline Models
  std::unique_ptr<Ort::Session> g_encoder_session;
  std::unique_ptr<Ort::Session> lang_session;
  std::unique_ptr<Ort::Session> g_decoder_session;

  // Interactive Pipeline Models
  std::unique_ptr<Ort::Session> i_encoder_session;
  std::unique_ptr<Ort::Session> i_decoder_session;

  // GPU 预处理与流管理
  void *d_input = nullptr;
  void *d_output = nullptr;
  void *cuda_stream = nullptr;

  // CPU 预处理预分配内存
  std::vector<float> cpu_input_buffer;

  SimpleTokenizer tokenizer;

  // 创建会话选项
  Ort::SessionOptions get_session_options();

  // 图像预处理 (统一处理并返回 ORT 输入 Tensor)
  Ort::Value preprocess_image(const cv::Mat &bgr_img);

  // 确保加载 Grounding 模型
  void ensure_grounding_models();

  // 确保加载 Interactive 模型
  void ensure_interactive_models();

  // 运行 Interactive Pipeline 推理
  InteractiveResult run_interactive_inference(
      const cv::Mat &bgr_img, const std::vector<cv::Point2f> &points,
      const std::vector<int> &labels, const std::vector<cv::Rect2f> &boxes);

  // 运行 Grounding Pipeline 推理 (支持传入已计算的图像特征)
  std::vector<InferenceResult>
  run_grounding_inference(const cv::Mat &bgr_img, const std::string &text,
                          const std::vector<float> &box_coords_in,
                          const std::vector<int64_t> &box_labels_in,
                          float threshold, int max_detections,
                          float nms_threshold = 0.5f);

  // 复用已计算图像特征运行 Grounding 解码
  std::vector<InferenceResult>
  run_grounding_with_features(const cv::Mat &bgr_img, const std::string &text,
                              const std::vector<Ort::Value> &encoder_outputs,
                              float threshold, int max_detections,
                              float nms_threshold = 0.5f);
};

#endif // SAM3_PREDICTOR_H

