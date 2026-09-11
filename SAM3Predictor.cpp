#include "SAM3Predictor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

// 构造函数
SAM3Predictor::SAM3Predictor(const std::string &model_dir, bool use_gpu)
    : env(ORT_LOGGING_LEVEL_WARNING, "SAM3_Inference"),
      model_dir(model_dir),
      use_gpu(use_gpu),
      tokenizer(model_dir + "/vocab.txt", model_dir + "/merges.txt") {
  // 预分配 CPU 预处理缓冲区 (1 x 3 x 1008 x 1008)
  cpu_input_buffer.resize(1 * 3 * 1008 * 1008);

#ifdef USE_CUDA
  if (use_gpu) {
    cudaMalloc(&d_input, 1008 * 1008 * 3);
    cudaMalloc(&d_output, 1008 * 1008 * 3 * sizeof(float));
    cudaStreamCreate(reinterpret_cast<cudaStream_t *>(&cuda_stream));
    std::cout << "[Info] GPU preprocessing buffers allocated." << std::endl;
  }
#endif
}

SAM3Predictor::~SAM3Predictor() {
#ifdef USE_CUDA
  if (d_input) {
    cudaFree(d_input);
    d_input = nullptr;
  }
  if (d_output) {
    cudaFree(d_output);
    d_output = nullptr;
  }
  if (cuda_stream) {
    cudaStreamDestroy(static_cast<cudaStream_t>(cuda_stream));
    cuda_stream = nullptr;
  }
#endif
}

void SAM3Predictor::warmup() {
  std::cout << "Warming up SAM3 models..." << std::endl;
  ensure_grounding_models();
  ensure_interactive_models();
  std::cout << "Warmup completed. Models are ready in memory." << std::endl;
}

// 创建会话选项
Ort::SessionOptions SAM3Predictor::get_session_options() {
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(0);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);

  if (use_gpu) {
#ifdef USE_CUDA
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = 0;
    cuda_options.arena_extend_strategy = 1;
    session_options.AppendExecutionProvider_CUDA(cuda_options);
#endif
  }
  return session_options;
}

// 确保加载 Grounding 模型
void SAM3Predictor::ensure_grounding_models() {
  if (g_encoder_session && lang_session && g_decoder_session)
    return;
  std::cout << "Loading Grounding/Text models from: " << model_dir << "..." << std::endl;
  auto start_load = std::chrono::high_resolution_clock::now();

  auto opts = get_session_options();
  g_encoder_session = std::make_unique<Ort::Session>(
      env, (model_dir + "/sam3_grounding_encoder.onnx").c_str(), opts);
  lang_session = std::make_unique<Ort::Session>(
      env, (model_dir + "/sam3_language_encoder.onnx").c_str(), opts);
  g_decoder_session = std::make_unique<Ort::Session>(
      env, (model_dir + "/sam3_grounding_decoder.onnx").c_str(), opts);

  auto end_load = std::chrono::high_resolution_clock::now();
  std::cout << "Model loading time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_load -
                                                                     start_load)
                   .count()
            << " ms" << std::endl;
}

// 确保加载 Interactive 模型
void SAM3Predictor::ensure_interactive_models() {
  if (i_encoder_session && i_decoder_session)
    return;
  std::cout << "Loading Interactive (Point/Box) models from: " << model_dir << "..." << std::endl;
  auto start_load = std::chrono::high_resolution_clock::now();

  auto opts = get_session_options();
  i_encoder_session = std::make_unique<Ort::Session>(
      env, (model_dir + "/sam3_encoder.onnx").c_str(), opts);
  i_decoder_session = std::make_unique<Ort::Session>(
      env, (model_dir + "/sam3_decoder.onnx").c_str(), opts);

  auto end_load = std::chrono::high_resolution_clock::now();
  std::cout << "Interactive model loading time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_load -
                                                                     start_load)
                   .count()
            << " ms" << std::endl;
}

// 统一图像预处理 (BGR -> RGB 归一化 [-1, 1], CHW 布局 1008x1008)
Ort::Value SAM3Predictor::preprocess_image(const cv::Mat &bgr_img) {
  std::vector<int64_t> encoder_input_shape = {1, 3, 1008, 1008};

#ifdef USE_CUDA
  if (use_gpu && d_input && d_output) {
    cv::Mat resized;
    if (bgr_img.cols != 1008 || bgr_img.rows != 1008) {
      cv::resize(bgr_img, resized, cv::Size(1008, 1008));
    } else {
      resized = bgr_img;
    }
    if (!resized.isContinuous()) {
      resized = resized.clone();
    }

    cudaMemcpyAsync(d_input, resized.data, 1008 * 1008 * 3,
                    cudaMemcpyHostToDevice, (cudaStream_t)cuda_stream);
    launch_preprocess((const unsigned char *)d_input, (float *)d_output,
                      1008, 1008, true, (cudaStream_t)cuda_stream);
    cudaStreamSynchronize((cudaStream_t)cuda_stream);

    Ort::MemoryInfo mem_info_cuda("Cuda", OrtAllocatorType::OrtArenaAllocator,
                                  0, OrtMemTypeDefault);
    return Ort::Value::CreateTensor<float>(
        mem_info_cuda, (float *)d_output, 1008 * 1008 * 3,
        encoder_input_shape.data(), encoder_input_shape.size());
  }
#endif

  // CPU 路径：直接缩放并转换为 RGB CHW 格式，归一化到 [-1, 1]
  cv::Mat resized;
  if (bgr_img.cols != 1008 || bgr_img.rows != 1008) {
    cv::resize(bgr_img, resized, cv::Size(1008, 1008));
  } else {
    resized = bgr_img;
  }

  if (cpu_input_buffer.size() != 1 * 3 * 1008 * 1008) {
    cpu_input_buffer.resize(1 * 3 * 1008 * 1008);
  }

  const int pixel_count = 1008 * 1008;
  float *r_channel = cpu_input_buffer.data();
  float *g_channel = r_channel + pixel_count;
  float *b_channel = g_channel + pixel_count;

  const float inv_scale = 1.0f / 127.5f;
  for (int y = 0; y < 1008; ++y) {
    const uint8_t *row_ptr = resized.ptr<uint8_t>(y);
    int idx = y * 1008;
    for (int x = 0; x < 1008; ++x) {
      uint8_t b = row_ptr[3 * x];
      uint8_t g = row_ptr[3 * x + 1];
      uint8_t r = row_ptr[3 * x + 2];
      r_channel[idx + x] = static_cast<float>(r) * inv_scale - 1.0f;
      g_channel[idx + x] = static_cast<float>(g) * inv_scale - 1.0f;
      b_channel[idx + x] = static_cast<float>(b) * inv_scale - 1.0f;
    }
  }

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  return Ort::Value::CreateTensor<float>(
      memory_info, cpu_input_buffer.data(), cpu_input_buffer.size(),
      encoder_input_shape.data(), encoder_input_shape.size());
}

// 非极大值抑制 (NMS)
std::vector<SAM3Predictor::InferenceResult>
SAM3Predictor::apply_nms(const std::vector<InferenceResult> &candidates,
                         float iou_threshold) {
  if (candidates.empty() || iou_threshold <= 0.0f || iou_threshold >= 1.0f) {
    return candidates;
  }

  std::vector<InferenceResult> kept;
  std::vector<bool> suppressed(candidates.size(), false);

  for (size_t i = 0; i < candidates.size(); ++i) {
    if (suppressed[i])
      continue;
    kept.push_back(candidates[i]);

    for (size_t j = i + 1; j < candidates.size(); ++j) {
      if (suppressed[j])
        continue;
      if (candidates[i].label != candidates[j].label)
        continue;

      float xx1 = std::max(candidates[i].box.x, candidates[j].box.x);
      float yy1 = std::max(candidates[i].box.y, candidates[j].box.y);
      float xx2 = std::min(candidates[i].box.x + candidates[i].box.width,
                           candidates[j].box.x + candidates[j].box.width);
      float yy2 = std::min(candidates[i].box.y + candidates[i].box.height,
                           candidates[j].box.y + candidates[j].box.height);

      float w = std::max(0.0f, xx2 - xx1);
      float h = std::max(0.0f, yy2 - yy1);
      float inter = w * h;
      float area_i = candidates[i].box.width * candidates[i].box.height;
      float area_j = candidates[j].box.width * candidates[j].box.height;
      float union_area = area_i + area_j - inter;

      if (union_area > 0.0f && (inter / union_area) > iou_threshold) {
        suppressed[j] = true;
      }
    }
  }
  return kept;
}

// 文本提示 -> Grounding Pipeline
std::vector<SAM3Predictor::InferenceResult>
SAM3Predictor::predict_text(const cv::Mat &bgr_img, const std::string &text,
                            float threshold, int max_detections,
                            float nms_threshold) {
  return run_grounding_inference(bgr_img, text, {}, {}, threshold,
                                 max_detections, nms_threshold);
}

// 批量文本提示 -> 图像仅编码一次，显著加速
std::vector<SAM3Predictor::InferenceResult>
SAM3Predictor::predict_texts(const cv::Mat &bgr_img,
                             const std::vector<std::string> &texts,
                             float threshold, int max_detections,
                             float nms_threshold) {
  if (texts.empty())
    return {};

  ensure_grounding_models();

  auto total_start = std::chrono::high_resolution_clock::now();

  // 1. 图像预处理
  auto start_pre = std::chrono::high_resolution_clock::now();
  Ort::Value encoder_input_tensor = preprocess_image(bgr_img);
  auto end_pre = std::chrono::high_resolution_clock::now();
  std::cout << "[Preprocess] "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_pre - start_pre).count()
            << " ms" << std::endl;

  // 2. 图像编码器 (只运行一次！)
  const char *encoder_input_names[] = {"images"};
  const char *encoder_output_names[] = {"feat0", "feat1", "feat2",
                                        "vpe0",  "vpe1",  "vpe2"};
  std::vector<Ort::Value> encoder_outputs;

  auto start_enc = std::chrono::high_resolution_clock::now();
  try {
    encoder_outputs = g_encoder_session->Run(
        Ort::RunOptions{nullptr}, encoder_input_names, &encoder_input_tensor, 1,
        encoder_output_names, 6);
  } catch (const Ort::Exception &e) {
    if (use_gpu) {
      std::cerr << "[Warning] GPU Inference failed: " << e.what()
                << ". Falling back to CPU..." << std::endl;
      g_encoder_session.reset();
      lang_session.reset();
      g_decoder_session.reset();
      use_gpu = false;
      return predict_texts(bgr_img, texts, threshold, max_detections, nms_threshold);
    }
    throw;
  }
  auto end_enc = std::chrono::high_resolution_clock::now();
  std::cout << "[Image Encoder] "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_enc - start_enc).count()
            << " ms" << std::endl;

  // 3. 对每个文本标签复用图像特征运行解码
  std::vector<InferenceResult> all_results;
  for (const auto &text : texts) {
    auto cls_results = run_grounding_with_features(
        bgr_img, text, encoder_outputs, threshold, max_detections, nms_threshold);
    all_results.insert(all_results.end(), cls_results.begin(), cls_results.end());
  }

  auto total_end = std::chrono::high_resolution_clock::now();
  std::cout << "[Total Batch Inference] "
            << std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count()
            << " ms for " << texts.size() << " classes." << std::endl;

  return all_results;
}

// 点提示 -> Interactive Pipeline
std::vector<SAM3Predictor::InferenceResult>
SAM3Predictor::predict_point(const cv::Mat &bgr_img, const cv::Point2f &point,
                             float threshold, int max_detections,
                             std::string label) {
  return predict_points(bgr_img, {point}, {1}, threshold, max_detections, label);
}

// 多点提示 -> Interactive Pipeline (支持正负点 labels: 1=前景, 0=背景)
std::vector<SAM3Predictor::InferenceResult>
SAM3Predictor::predict_points(const cv::Mat &bgr_img,
                              const std::vector<cv::Point2f> &points,
                              const std::vector<int> &labels,
                              float threshold, int max_detections,
                              std::string label) {
  ensure_interactive_models();

  auto start = std::chrono::high_resolution_clock::now();
  auto res = run_interactive_inference(bgr_img, points, labels, {});

  std::vector<InferenceResult> results;
  if (res.score > threshold) {
    cv::Rect2f box = cv::boundingRect(res.mask);
    if (box.width > 0 && box.height > 0) {
      results.push_back({res.mask, res.score, box, label});
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "Interactive Inference time: " << duration << " ms" << std::endl;
  return results;
}

// 框提示 -> Interactive Pipeline
std::vector<SAM3Predictor::InferenceResult>
SAM3Predictor::predict_box(const cv::Mat &bgr_img, const cv::Rect2f &box,
                           float threshold, int max_detections,
                           std::string label) {
  return predict_boxes(bgr_img, {box}, threshold, max_detections, label);
}

// 多框提示 -> Interactive Pipeline
std::vector<SAM3Predictor::InferenceResult>
SAM3Predictor::predict_boxes(const cv::Mat &bgr_img,
                             const std::vector<cv::Rect2f> &boxes,
                             float threshold, int max_detections,
                             std::string label) {
  ensure_interactive_models();

  auto start = std::chrono::high_resolution_clock::now();
  auto res = run_interactive_inference(bgr_img, {}, {}, boxes);

  std::vector<InferenceResult> results;
  if (res.score > threshold) {
    cv::Rect2f res_box = cv::boundingRect(res.mask);
    if (res_box.width > 0 && res_box.height > 0) {
      results.push_back({res.mask, res.score, res_box, label});
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "Interactive Inference time: " << duration << " ms" << std::endl;
  return results;
}

// 运行 Interactive Pipeline 推理
SAM3Predictor::InteractiveResult SAM3Predictor::run_interactive_inference(
    const cv::Mat &bgr_img, const std::vector<cv::Point2f> &points,
    const std::vector<int> &labels, const std::vector<cv::Rect2f> &boxes) {
  ensure_interactive_models();

  // 1. 图像预处理
  Ort::Value encoder_input_tensor = preprocess_image(bgr_img);

  const char *encoder_input_names[] = {"images"};
  const char *encoder_output_names[] = {"pix_feat", "high_res_0", "high_res_1"};
  auto encoder_outputs =
      i_encoder_session->Run(Ort::RunOptions{nullptr}, encoder_input_names,
                             &encoder_input_tensor, 1, encoder_output_names, 3);

  // 2. 准备 Decoder 输入 (点+框)
  float scale_x = 1008.0f / bgr_img.cols;
  float scale_y = 1008.0f / bgr_img.rows;
  std::vector<float> final_coords;
  std::vector<int32_t> final_labels;

  for (size_t i = 0; i < points.size(); ++i) {
    final_coords.push_back(points[i].x * scale_x);
    final_coords.push_back(points[i].y * scale_y);
    final_labels.push_back(labels[i]);
  }
  for (const auto &box : boxes) {
    final_coords.push_back(box.x * scale_x);
    final_coords.push_back(box.y * scale_y);
    final_labels.push_back(2);
    final_coords.push_back((box.x + box.width) * scale_x);
    final_coords.push_back((box.y + box.height) * scale_y);
    final_labels.push_back(3);
  }
  if (final_labels.empty()) {
    final_coords.push_back(0.0f);
    final_coords.push_back(0.0f);
    final_labels.push_back(-1);
  }

  std::vector<int64_t> coords_shape = {
      1, static_cast<int64_t>(final_labels.size()), 2};
  std::vector<int64_t> labels_shape = {
      1, static_cast<int64_t>(final_labels.size())};

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value coords_tensor = Ort::Value::CreateTensor<float>(
      memory_info, final_coords.data(), final_coords.size(),
      coords_shape.data(), coords_shape.size());
  Ort::Value labels_tensor = Ort::Value::CreateTensor<int32_t>(
      memory_info, final_labels.data(), final_labels.size(),
      labels_shape.data(), labels_shape.size());

  std::vector<Ort::Value> decoder_inputs;
  decoder_inputs.push_back(std::move(encoder_outputs[0]));
  decoder_inputs.push_back(std::move(encoder_outputs[1]));
  decoder_inputs.push_back(std::move(encoder_outputs[2]));
  decoder_inputs.push_back(std::move(coords_tensor));
  decoder_inputs.push_back(std::move(labels_tensor));

  const char *decoder_input_names[] = {"pix_feat", "high_res_0", "high_res_1",
                                       "point_coords", "point_labels"};
  const char *decoder_output_names[] = {"masks", "ious"};

  auto decoder_outputs = i_decoder_session->Run(
      Ort::RunOptions{nullptr}, decoder_input_names, decoder_inputs.data(),
      decoder_inputs.size(), decoder_output_names, 2);

  float *masks_data = decoder_outputs[0].GetTensorMutableData<float>();
  float *ious_data = decoder_outputs[1].GetTensorMutableData<float>();

  // 找到最佳候选掩码 (最大 IoU)
  int best_idx = 0;
  float max_iou = ious_data[0];
  for (int i = 1; i < 3; ++i) {
    if (ious_data[i] > max_iou) {
      max_iou = ious_data[i];
      best_idx = i;
    }
  }

  auto mask_shape = decoder_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
  int mask_h = static_cast<int>(mask_shape[2]);
  int mask_w = static_cast<int>(mask_shape[3]);

  cv::Mat mask_logit(mask_h, mask_w, CV_32F,
                     masks_data + best_idx * mask_h * mask_w);
  cv::Mat resized_mask;
  cv::resize(mask_logit, resized_mask, bgr_img.size());
  cv::Mat binary_mask;
  cv::threshold(resized_mask, binary_mask, 0.0, 255, cv::THRESH_BINARY);
  binary_mask.convertTo(binary_mask, CV_8U);

  return {binary_mask, max_iou};
}

// 运行 Grounding Pipeline 推理
std::vector<SAM3Predictor::InferenceResult>
SAM3Predictor::run_grounding_inference(
    const cv::Mat &bgr_img, const std::string &text,
    const std::vector<float> &box_coords_in,
    const std::vector<int64_t> &box_labels_in, float threshold,
    int max_detections, float nms_threshold) {
  ensure_grounding_models();

  auto total_start = std::chrono::high_resolution_clock::now();

  // 1. 图像预处理
  auto start_pre = std::chrono::high_resolution_clock::now();
  Ort::Value encoder_input_tensor = preprocess_image(bgr_img);
  auto end_pre = std::chrono::high_resolution_clock::now();
  std::cout << "Preprocessing time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_pre - start_pre).count()
            << " ms" << std::endl;

  // 2. 图像编码器推理
  const char *encoder_input_names[] = {"images"};
  const char *encoder_output_names[] = {"feat0", "feat1", "feat2",
                                        "vpe0",  "vpe1",  "vpe2"};
  std::vector<Ort::Value> encoder_outputs;

  auto start_enc = std::chrono::high_resolution_clock::now();
  try {
    encoder_outputs = g_encoder_session->Run(
        Ort::RunOptions{nullptr}, encoder_input_names, &encoder_input_tensor, 1,
        encoder_output_names, 6);
  } catch (const Ort::Exception &e) {
    if (use_gpu) {
      std::cerr << "[Warning] GPU Inference failed: " << e.what()
                << ". Falling back to CPU..." << std::endl;
      g_encoder_session.reset();
      lang_session.reset();
      g_decoder_session.reset();
      use_gpu = false;
      return run_grounding_inference(bgr_img, text, box_coords_in,
                                     box_labels_in, threshold, max_detections,
                                     nms_threshold);
    }
    throw;
  }
  auto end_enc = std::chrono::high_resolution_clock::now();
  std::cout << "Image encoder time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_enc - start_enc).count()
            << " ms" << std::endl;

  auto results = run_grounding_with_features(
      bgr_img, text, encoder_outputs, threshold, max_detections, nms_threshold);

  auto total_end = std::chrono::high_resolution_clock::now();
  std::cout << "Pure Inference time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count()
            << " ms" << std::endl;

  return results;
}

// 辅助函数：创建对已有 Tensor 缓冲区的零拷贝视图，避免搬移所有权
static Ort::Value create_tensor_view(const Ort::Value &tensor) {
  auto type_shape = tensor.GetTensorTypeAndShapeInfo();
  auto shape = type_shape.GetShape();
  size_t element_count = type_shape.GetElementCount();
  const OrtMemoryInfo *mem_info = tensor.GetTensorMemoryInfo();
  const float *data = tensor.GetTensorData<float>();
  return Ort::Value::CreateTensor<float>(
      mem_info, const_cast<float *>(data), element_count,
      shape.data(), shape.size());
}

// 复用图像特征运行 Grounding 解码
std::vector<SAM3Predictor::InferenceResult>
SAM3Predictor::run_grounding_with_features(
    const cv::Mat &bgr_img, const std::string &text,
    const std::vector<Ort::Value> &encoder_outputs,
    float threshold, int max_detections, float nms_threshold) {
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // 1. 语言编码器推理
  auto tokenized = tokenizer.tokenize({text}, 32);
  std::vector<int64_t> tokens_data;
  tokens_data.reserve(32);
  for (int t : tokenized[0])
    tokens_data.push_back(static_cast<int64_t>(t));

  std::vector<int64_t> tokens_shape = {1, 32};
  Ort::Value tokens_tensor = Ort::Value::CreateTensor<int64_t>(
      memory_info, tokens_data.data(), tokens_data.size(), tokens_shape.data(),
      tokens_shape.size());

  const char *lang_input_names[] = {"tokens"};
  const char *lang_output_names[] = {"text_attention_mask", "text_memory",
                                     "text_embeds"};
  auto start_lang = std::chrono::high_resolution_clock::now();
  auto lang_outputs =
      lang_session->Run(Ort::RunOptions{nullptr}, lang_input_names,
                        &tokens_tensor, 1, lang_output_names, 3);
  auto end_lang = std::chrono::high_resolution_clock::now();
  std::cout << "Language encoder time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_lang - start_lang).count()
            << " ms" << std::endl;

  // 2. 准备 Grounding 解码器输入
  // feat0, feat1, feat2, vpe2 通过零拷贝视图传递，保持 encoder_outputs 可复用
  std::vector<Ort::Value> decoder_inputs;
  decoder_inputs.reserve(9);
  decoder_inputs.push_back(create_tensor_view(encoder_outputs[0]));
  decoder_inputs.push_back(create_tensor_view(encoder_outputs[1]));
  decoder_inputs.push_back(create_tensor_view(encoder_outputs[2]));
  decoder_inputs.push_back(create_tensor_view(encoder_outputs[5]));
  decoder_inputs.push_back(std::move(lang_outputs[0]));
  decoder_inputs.push_back(std::move(lang_outputs[1]));

  // 默认框提示
  std::vector<float> box_coords_data = {0.0f, 0.0f, 0.0f, 0.0f};
  std::vector<int64_t> box_labels_data = {1};
  std::vector<uint8_t> box_masks_data = {1}; // 1 = 屏蔽框提示

  std::vector<int64_t> box_coords_shape = {1, 1, 4};
  Ort::Value box_coords_tensor = Ort::Value::CreateTensor<float>(
      memory_info, box_coords_data.data(), box_coords_data.size(),
      box_coords_shape.data(), box_coords_shape.size());

  std::vector<int64_t> box_labels_shape = {1, 1};
  Ort::Value box_labels_tensor = Ort::Value::CreateTensor<int64_t>(
      memory_info, box_labels_data.data(), box_labels_data.size(),
      box_labels_shape.data(), box_labels_shape.size());

  std::vector<int64_t> box_masks_shape = {1, 1};
  Ort::Value box_masks_tensor = Ort::Value::CreateTensor<bool>(
      memory_info, reinterpret_cast<bool *>(box_masks_data.data()),
      box_masks_data.size(), box_masks_shape.data(), box_masks_shape.size());

  decoder_inputs.push_back(std::move(box_coords_tensor));
  decoder_inputs.push_back(std::move(box_labels_tensor));
  decoder_inputs.push_back(std::move(box_masks_tensor));

  const char *decoder_input_names[] = {"feat0",      "feat1",      "feat2",
                                       "vpe2",       "lang_mask",  "lang_feat",
                                       "box_coords", "box_labels", "box_masks"};
  const char *decoder_output_names[] = {"boxes", "scores", "masks", "presence"};

  auto start_dec = std::chrono::high_resolution_clock::now();
  auto decoder_outputs = g_decoder_session->Run(
      Ort::RunOptions{nullptr}, decoder_input_names, decoder_inputs.data(),
      decoder_inputs.size(), decoder_output_names, 4);
  auto end_dec = std::chrono::high_resolution_clock::now();
  std::cout << "Grounding decoder time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_dec - start_dec).count()
            << " ms" << std::endl;

  // 3. 后处理解码结果
  float *boxes_ptr = decoder_outputs[0].GetTensorMutableData<float>();
  float *scores_ptr = decoder_outputs[1].GetTensorMutableData<float>();
  float *masks_ptr = decoder_outputs[2].GetTensorMutableData<float>();
  float *presence_ptr = decoder_outputs[3].GetTensorMutableData<float>();

  auto boxes_shape = decoder_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
  int num_prompts = static_cast<int>(boxes_shape[0]);
  int num_queries = static_cast<int>(boxes_shape[1]);
  int mask_h = static_cast<int>(decoder_outputs[2].GetTensorTypeAndShapeInfo().GetShape()[2]);
  int mask_w = static_cast<int>(decoder_outputs[2].GetTensorTypeAndShapeInfo().GetShape()[3]);

  std::vector<InferenceResult> candidates;
  for (int i = 0; i < num_prompts; ++i) {
    float presence_score = 1.0f / (1.0f + std::exp(-presence_ptr[i]));
    for (int q = 0; q < num_queries; ++q) {
      float logit = scores_ptr[i * num_queries + q];
      float score = (1.0f / (1.0f + std::exp(-logit))) * presence_score;
      if (score < threshold)
        continue;

      float *box = &boxes_ptr[i * num_queries * 4 + q * 4];
      float *mask =
          &masks_ptr[i * num_queries * mask_h * mask_w + q * mask_h * mask_w];

      cv::Mat mask_logit(mask_h, mask_w, CV_32F, mask);
      cv::Mat binary_mask;
      cv::resize(mask_logit, binary_mask, bgr_img.size());
      cv::threshold(binary_mask, binary_mask, 0.0, 255, cv::THRESH_BINARY);
      binary_mask.convertTo(binary_mask, CV_8U);

      // 安全夹取边界框到图像像素范围
      float x1 = std::clamp(box[0], 0.0f, 1.0f) * bgr_img.cols;
      float y1 = std::clamp(box[1], 0.0f, 1.0f) * bgr_img.rows;
      float x2 = std::clamp(box[2], 0.0f, 1.0f) * bgr_img.cols;
      float y2 = std::clamp(box[3], 0.0f, 1.0f) * bgr_img.rows;
      float bw = std::max(0.0f, x2 - x1);
      float bh = std::max(0.0f, y2 - y1);

      candidates.push_back({binary_mask, score,
                            cv::Rect2f(x1, y1, bw, bh),
                            text == "visual" ? "object" : text});
    }
  }

  // 按置信度降序排序
  std::sort(candidates.begin(), candidates.end(),
            [](const InferenceResult &a, const InferenceResult &b) {
              return a.score > b.score;
            });

  // 应用 NMS 抑制重叠框
  std::vector<InferenceResult> filtered = apply_nms(candidates, nms_threshold);

  // 截断到 max_detections
  if (max_detections > 0 && filtered.size() > static_cast<size_t>(max_detections)) {
    filtered.resize(max_detections);
  }

  return filtered;
}

