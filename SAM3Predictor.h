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

  // 文本提示 -> Grounding Pipeline
  std::vector<InferenceResult> predict_text(const cv::Mat &bgr_img,
                                            const std::string &text,
                                            float threshold = 0.25f,
                                            int max_detections = 0);

  // 点提示 -> Interactive Pipeline
  std::vector<InferenceResult> predict_point(const cv::Mat &bgr_img,
                                             const cv::Point2f &point,
                                             float threshold = 0.5f,
                                             int max_detections = 0,
                                             std::string label = "object");

  // 框提示 -> Interactive Pipeline
  std::vector<InferenceResult> predict_box(const cv::Mat &bgr_img,
                                           const cv::Rect2f &box,
                                           float threshold = 0.5f,
                                           int max_detections = 0,
                                           std::string label = "object");

  /**
   * @brief 预热模型 (主动加载程序)
   * 在生产环境下，建议在初始化后显式调用此函数。
   * 这将触发 ONNX Runtime 的会话创建和 GPU 显存分配，
   * 从而避免首次推理时的显著延迟 (约 6 秒)。
   */
  void warmup();

private:
  // Interactive Pipeline 结果结构体
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

  // IOBinding
  std::unique_ptr<Ort::IoBinding> g_encoder_io;
  std::unique_ptr<Ort::IoBinding> i_encoder_io;

  // GPU 预处理缓存
  void *d_input = nullptr;
  void *d_output = nullptr;
  void *cuda_stream = nullptr;

  SimpleTokenizer tokenizer;

  // 创建会话选项
  Ort::SessionOptions get_session_options();

  // 确保加载 Grounding 模型
  void ensure_grounding_models();

  // 确保加载 Interactive 模型
  void ensure_interactive_models();

  // 运行 Interactive Pipeline 推理
  InteractiveResult run_interactive_inference(
      const cv::Mat &bgr_img, const std::vector<cv::Point2f> &points,
      const std::vector<int> &labels, const std::vector<cv::Rect2f> &boxes);

  // 运行 Grounding Pipeline 推理
  std::vector<InferenceResult>
  run_grounding_inference(const cv::Mat &bgr_img, const std::string &text,
                          const std::vector<float> &box_coords_in,
                          const std::vector<int64_t> &box_labels_in,
                          float threshold, int max_detections);
};

#endif // SAM3_PREDICTOR_H
