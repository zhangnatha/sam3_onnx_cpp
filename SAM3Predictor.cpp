#include "SAM3Predictor.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>

// 构造函数
SAM3Predictor::SAM3Predictor(const std::string &model_dir, bool use_gpu)
    : env(ORT_LOGGING_LEVEL_WARNING, "SAM3_Inference"),
      tokenizer(model_dir + "/vocab.txt", model_dir + "/merges.txt") {
  // 保存配置供延迟加载使用
  this->model_dir = model_dir;
  this->use_gpu = use_gpu;
}

// 创建会话选项
Ort::SessionOptions SAM3Predictor::get_session_options() {
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(4);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);

  if (use_gpu) {
#ifdef USE_CUDA
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = 0;
    cuda_options.arena_extend_strategy =
        1; // kSameAsRequested (Less fragmentation)
    session_options.AppendExecutionProvider_CUDA(cuda_options);
#endif
  }
  return session_options;
}

// 确保加载 Grounding 模型
void SAM3Predictor::ensure_grounding_models() {
  if (g_encoder_session)
    return;
  std::cout << "Loading Grounding/Text models..." << std::endl;
  auto opts = get_session_options();
  g_encoder_session = std::make_unique<Ort::Session>(
      env, (model_dir + "/sam3_grounding_encoder.onnx").c_str(), opts);
  lang_session = std::make_unique<Ort::Session>(
      env, (model_dir + "/sam3_language_encoder.onnx").c_str(), opts);
  g_decoder_session = std::make_unique<Ort::Session>(
      env, (model_dir + "/sam3_grounding_decoder.onnx").c_str(), opts);
}

// 确保加载 Interactive 模型
void SAM3Predictor::ensure_interactive_models() {
  if (i_encoder_session)
    return;
  std::cout << "Loading Interactive (Point/Box) models..." << std::endl;
  auto opts = get_session_options();
  i_encoder_session = std::make_unique<Ort::Session>(
      env, (model_dir + "/sam3_encoder.onnx").c_str(), opts);
  i_decoder_session = std::make_unique<Ort::Session>(
      env, (model_dir + "/sam3_decoder.onnx").c_str(), opts);
}

// 文本提示 -> Grounding Pipeline
std::vector<SAM3Predictor::InferenceResult>
SAM3Predictor::predict_text(const cv::Mat &bgr_img, const std::string &text,
                            float threshold, int max_detections) {
  auto start = std::chrono::high_resolution_clock::now();
  auto result =
      run_grounding_inference(bgr_img, text, {}, {}, threshold, max_detections);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  std::cout << "Inference time: " << duration << " ms" << std::endl;
  return result;
}

// 点提示 -> Interactive Pipeline
std::vector<SAM3Predictor::InferenceResult>
SAM3Predictor::predict_point(const cv::Mat &bgr_img, const cv::Point2f &point,
                             float threshold, int max_detections,
                             std::string label) {
  auto start = std::chrono::high_resolution_clock::now();
  std::vector<cv::Point2f> points = {point};
  std::vector<int> labels = {1}; // 1 = 前景
  auto res = run_interactive_inference(bgr_img, points, labels, {});

  // 转换结果格式
  std::vector<InferenceResult> results;
  if (res.score > threshold) {
    cv::Rect2f box = cv::boundingRect(res.mask);
    results.push_back({res.mask, res.score, box, label});
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  std::cout << "Inference time: " << duration << " ms" << std::endl;
  return results;
}

// 框提示 -> Interactive Pipeline
std::vector<SAM3Predictor::InferenceResult>
SAM3Predictor::predict_box(const cv::Mat &bgr_img, const cv::Rect2f &box,
                           float threshold, int max_detections,
                           std::string label) {
  auto start = std::chrono::high_resolution_clock::now();
  std::vector<cv::Rect2f> boxes = {box};
  auto res = run_interactive_inference(bgr_img, {}, {}, boxes);

  std::vector<InferenceResult> results;
  if (res.score > threshold) {
    cv::Rect2f res_box = cv::boundingRect(res.mask); // 重新计算精确包围盒
    results.push_back({res.mask, res.score, res_box, label});
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  std::cout << "Inference time: " << duration << " ms" << std::endl;
  return results;
}

// 运行 Interactive Pipeline 推理
SAM3Predictor::InteractiveResult SAM3Predictor::run_interactive_inference(
    const cv::Mat &bgr_img, const std::vector<cv::Point2f> &points,
    const std::vector<int> &labels, const std::vector<cv::Rect2f> &boxes) {
  // 0. 延迟加载 Interactive Models
  ensure_interactive_models();

  // 1. 图像预处理 (1008x1008)
  cv::Mat rgb_img;
  cv::cvtColor(bgr_img, rgb_img, cv::COLOR_BGR2RGB);
  cv::Mat resized_img;
  cv::resize(rgb_img, resized_img, cv::Size(1008, 1008));
  // 2. 归一化 (使用 0.5/0.5 归一化，与 Grounding 管道保持一致)
  cv::Mat float_img;
  resized_img.convertTo(float_img, CV_32F, 1.0 / 255.0);
  cv::subtract(float_img, cv::Scalar(0.5, 0.5, 0.5), float_img);
  cv::divide(float_img, cv::Scalar(0.5, 0.5, 0.5), float_img);

  std::vector<float> input_tensor_values(1 * 3 * 1008 * 1008);
  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < 1008; ++h) {
      for (int w = 0; w < 1008; ++w) {
        input_tensor_values[c * 1008 * 1008 + h * 1008 + w] =
            float_img.at<cv::Vec3f>(h, w)[c];
      }
    }
  }

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<int64_t> encoder_input_shape = {1, 3, 1008, 1008};
  Ort::Value encoder_input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, input_tensor_values.data(), input_tensor_values.size(),
      encoder_input_shape.data(), encoder_input_shape.size());

  const char *encoder_input_names[] = {"images"};
  const char *encoder_output_names[] = {"pix_feat", "high_res_0", "high_res_1"};
  auto encoder_outputs =
      i_encoder_session->Run(Ort::RunOptions{nullptr}, encoder_input_names,
                             &encoder_input_tensor, 1, encoder_output_names, 3);

  // 2. 准备Decoder输入 (点+框)
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

  // 12. 找到最佳掩码 (3个候选中的最大IoU)
  int best_idx = 0;
  float max_iou = ious_data[0];
  for (int i = 1; i < 3; ++i) {
    if (ious_data[i] > max_iou) {
      max_iou = ious_data[i];
      best_idx = i;
    }
  }

  // 动态获取掩码尺寸
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
    int max_detections) {
  // 0. 延迟加载 Grounding Models
  ensure_grounding_models();

  // 1. 图像预处理 (1008x1008)
  cv::Mat rgb_img;
  cv::cvtColor(bgr_img, rgb_img, cv::COLOR_BGR2RGB);
  cv::Mat resized_img;
  cv::resize(rgb_img, resized_img, cv::Size(1008, 1008));
  cv::Mat float_img;
  resized_img.convertTo(float_img, CV_32F, 1.0 / 255.0);

  // CLIP风格归一化
  cv::subtract(float_img, cv::Scalar(0.5, 0.5, 0.5), float_img);
  cv::divide(float_img, cv::Scalar(0.5, 0.5, 0.5), float_img);

  std::vector<float> input_tensor_values(1 * 3 * 1008 * 1008);
  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < 1008; ++h) {
      for (int w = 0; w < 1008; ++w) {
        input_tensor_values[c * 1008 * 1008 + h * 1008 + w] =
            float_img.at<cv::Vec3f>(h, w)[c];
      }
    }
  }

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<int64_t> encoder_input_shape = {1, 3, 1008, 1008};
  Ort::Value encoder_input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, input_tensor_values.data(), input_tensor_values.size(),
      encoder_input_shape.data(), encoder_input_shape.size());

  // 2. Grounding可用的编码器推理
  const char *encoder_input_names[] = {"images"};
  const char *encoder_output_names[] = {"feat0", "feat1", "feat2",
                                        "vpe0",  "vpe1",  "vpe2"};
  std::vector<Ort::Value> encoder_outputs;

  try {
    encoder_outputs = g_encoder_session->Run(
        Ort::RunOptions{nullptr}, encoder_input_names, &encoder_input_tensor, 1,
        encoder_output_names, 6);
  } catch (const Ort::Exception &e) {
    if (use_gpu) {
      std::cerr << "[Warning] GPU Inference failed (likely OOM: " << e.what()
                << "). Falling back to CPU..." << std::endl;
      // 释放GPU会话
      g_encoder_session.reset();
      lang_session.reset();
      g_decoder_session.reset();

      // 切换到CPU
      use_gpu = false;

      // 递归重试
      return run_grounding_inference(bgr_img, text, box_coords_in,
                                     box_labels_in, threshold, max_detections);
    }
    throw e; // 如果已经是CPU或无法处理，则抛出异常
  }

  // 3. 语言编码器推理
  auto tokenized = tokenizer.tokenize({text}, 32);
  std::vector<int64_t> tokens_data;
  for (int t : tokenized[0])
    tokens_data.push_back(static_cast<int64_t>(t));

  std::vector<int64_t> tokens_shape = {1, 32};
  Ort::Value tokens_tensor = Ort::Value::CreateTensor<int64_t>(
      memory_info, tokens_data.data(), tokens_data.size(), tokens_shape.data(),
      tokens_shape.size());

  const char *lang_input_names[] = {"tokens"};
  const char *lang_output_names[] = {"text_attention_mask", "text_memory",
                                     "text_embeds"};
  auto lang_outputs =
      lang_session->Run(Ort::RunOptions{nullptr}, lang_input_names,
                        &tokens_tensor, 1, lang_output_names, 3);

  // 4. Grounding解码器推理
  std::vector<Ort::Value> decoder_inputs;
  decoder_inputs.push_back(std::move(encoder_outputs[0])); // feat0
  decoder_inputs.push_back(std::move(encoder_outputs[1])); // feat1
  decoder_inputs.push_back(std::move(encoder_outputs[2])); // feat2
  decoder_inputs.push_back(std::move(encoder_outputs[5])); // vpe2
  decoder_inputs.push_back(std::move(lang_outputs[0]));    // lang_mask
  decoder_inputs.push_back(std::move(lang_outputs[1]));    // lang_feat

  // 准备框提示
  std::vector<float> box_coords_data = {0.0f, 0.0f, 0.0f, 0.0f};
  std::vector<int64_t> box_labels_data = {1};
  std::vector<uint8_t> box_masks_data = {1};

  if (!box_coords_in.empty()) {
    box_coords_data = {
        box_coords_in[0] / bgr_img.cols, box_coords_in[1] / bgr_img.rows,
        box_coords_in[2] / bgr_img.cols, box_coords_in[3] / bgr_img.rows};
    box_labels_data = box_labels_in;
    box_masks_data = {0};
  }

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
      memory_info, (bool *)box_masks_data.data(), box_masks_data.size(),
      box_masks_shape.data(), box_masks_shape.size());

  decoder_inputs.push_back(std::move(box_coords_tensor));
  decoder_inputs.push_back(std::move(box_labels_tensor));
  decoder_inputs.push_back(std::move(box_masks_tensor));

  const char *decoder_input_names[] = {"feat0",      "feat1",      "feat2",
                                       "vpe2",       "lang_mask",  "lang_feat",
                                       "box_coords", "box_labels", "box_masks"};
  const char *decoder_output_names[] = {"boxes", "scores", "masks", "presence"};

  auto decoder_outputs = g_decoder_session->Run(
      Ort::RunOptions{nullptr}, decoder_input_names, decoder_inputs.data(),
      decoder_inputs.size(), decoder_output_names, 4);

  // 5. 后处理
  float *boxes_ptr = decoder_outputs[0].GetTensorMutableData<float>();
  float *scores_ptr = decoder_outputs[1].GetTensorMutableData<float>();
  float *masks_ptr = decoder_outputs[2].GetTensorMutableData<float>();
  float *presence_ptr = decoder_outputs[3].GetTensorMutableData<float>();

  auto boxes_shape = decoder_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
  int num_prompts = boxes_shape[0];
  int num_queries = boxes_shape[1];
  int mask_h = decoder_outputs[2].GetTensorTypeAndShapeInfo().GetShape()[2];
  int mask_w = decoder_outputs[2].GetTensorTypeAndShapeInfo().GetShape()[3];

  std::vector<std::pair<float, InferenceResult>> candidates;
  for (int i = 0; i < num_prompts; ++i) {
    float presence_score = 1.0f / (1.0f + exp(-presence_ptr[i]));
    for (int q = 0; q < num_queries; ++q) {
      float logit = scores_ptr[i * num_queries + q];
      float score = (1.0f / (1.0f + exp(-logit))) * presence_score;
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

      candidates.push_back(
          {score,
           {binary_mask, score,
            cv::Rect2f(box[0] * bgr_img.cols, box[1] * bgr_img.rows,
                       (box[2] - box[0]) * bgr_img.cols,
                       (box[3] - box[1]) * bgr_img.rows),
            text == "visual" ? "object" : text}});
    }
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });

  // 提取结果 (Text模式 不需要 NMS，或由调用者控制; 这里简单提取top k)
  std::vector<InferenceResult> results;
  int count = (max_detections > 0)
                  ? std::min(max_detections, (int)candidates.size())
                  : candidates.size();
  for (int i = 0; i < count; ++i) {
    results.push_back(candidates[i].second);
  }
  return results;
}
