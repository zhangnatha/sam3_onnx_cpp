#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "SAM3Predictor.h"

// 获取调色板颜色
cv::Scalar get_color(int id) {
  const std::vector<cv::Scalar> palette = {
      {255, 56, 56},   {255, 157, 151}, {255, 112, 31},  {255, 178, 29},
      {207, 210, 49},  {72, 249, 10},   {146, 204, 23},  {61, 219, 134},
      {26, 147, 52},   {0, 212, 187},   {44, 153, 168},  {0, 194, 255},
      {52, 69, 147},   {100, 115, 255}, {0, 24, 236},   {132, 56, 255},
      {82, 0, 133},    {203, 56, 255},  {255, 149, 200}, {255, 55, 199}};
  return palette[id % palette.size()];
}

// 自动定位模型目录
std::string resolve_model_dir(const std::string &user_path) {
  if (!user_path.empty()) {
    return user_path;
  }
  std::vector<std::string> candidates = {"./model", "../model", "model"};
  for (const auto &p : candidates) {
    std::ifstream v(p + "/vocab.txt");
    if (v.is_open()) {
      return p;
    }
  }
  return "./model";
}

// 保存可视化结果
void save_visualization(
    const cv::Mat &img,
    const std::vector<SAM3Predictor::InferenceResult> &results,
    const std::string &filename,
    const std::vector<cv::Point2f> &prompt_points = {},
    const std::vector<cv::Rect2f> &prompt_boxes = {}) {
  cv::Mat overlay = img.clone();
  for (size_t i = 0; i < results.size(); ++i) {
    auto color = get_color(i);

    // 1. 绘制半透明彩色掩码
    if (!results[i].mask.empty()) {
      cv::Mat colored_mask = cv::Mat::zeros(img.size(), img.type());
      colored_mask.setTo(color, results[i].mask);
      cv::addWeighted(overlay, 1.0, colored_mask, 0.45, 0, overlay);
    }

    // 2. 绘制边界框
    if (results[i].box.width > 0 && results[i].box.height > 0) {
      cv::rectangle(overlay, results[i].box, color, 2);
    }

    // 3. 绘制标签
    std::ostringstream ss;
    ss << results[i].label << " " << std::fixed << std::setprecision(2)
       << results[i].score;
    std::string text = ss.str();

    int baseline = 0;
    cv::Size text_size =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseline);

    int label_w = std::min(text_size.width + 8, overlay.cols);
    int label_h = text_size.height + 6;
    int label_x = std::clamp(static_cast<int>(results[i].box.x), 0,
                             std::max(0, overlay.cols - label_w));
    int label_y = static_cast<int>(results[i].box.y);
    if (label_y - label_h < 0) {
      label_y = label_h;
    }
    if (label_y > overlay.rows) {
      label_y = overlay.rows;
    }

    if (label_w > 0 && label_h > 0) {
      cv::Rect label_rect(label_x, label_y - label_h, label_w, label_h);
      cv::Mat label_bg = overlay(label_rect).clone();
      cv::rectangle(label_bg, cv::Point(0, 0), cv::Point(label_w, label_h),
                    color, -1);
      cv::addWeighted(overlay(label_rect), 0.35, label_bg, 0.65, 0,
                      overlay(label_rect));
      cv::putText(overlay, text, cv::Point(label_x + 4, label_y - 4),
                  cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1,
                  cv::LINE_AA);
    }
  }

  // 绘制原始提示词 (提示点)
  for (const auto &p : prompt_points) {
    cv::circle(overlay, p, 6, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    cv::circle(overlay, p, 5, cv::Scalar(255, 255, 255), -1, cv::LINE_AA);
  }

  // 绘制原始提示词 (提示框)
  for (const auto &box : prompt_boxes) {
    cv::rectangle(overlay, box, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
  }

  cv::imwrite(filename, overlay);
  std::cout << "Saved: " << filename << " (Found " << results.size()
            << " objects)" << std::endl;
}

// 打印使用说明
void print_usage(const char *program_name) {
  std::cout << "Usage: " << program_name << " [options]" << std::endl;
  std::cout << "\nOptions:" << std::endl;
  std::cout << "  --image <path>          Input image path (required)"
            << std::endl;
  std::cout
      << "  --output <path>         Output image path (default: result.jpg)"
      << std::endl;
  std::cout << "  --model <path>          Model directory (default: auto-detect "
               "'./model' or '../model')"
            << std::endl;
  std::cout << "  --mode <type>           Prompt mode: texts|points|boxes "
               "(default: texts)"
            << std::endl;
  std::cout << "  --prompt <value>        Prompt value:" << std::endl;
  std::cout << "                          - texts: text string or comma-separated "
               "(e.g., 'cat,computer')"
            << std::endl;
  std::cout
      << "                          - points: x,y coordinates (e.g., '100,200')"
      << std::endl;
  std::cout << "                          - boxes: x1,y1,x2,y2 coordinates "
               "(e.g., '100,100,200,200')"
            << std::endl;
  std::cout << "  --threshold <value>     Confidence threshold (default: 0.25)"
            << std::endl;
  std::cout << "  --nms <value>           IoU threshold for NMS (default: 0.5, 0 "
               "to disable)"
            << std::endl;
  std::cout << "  --max-detections <num>  Maximum number of detections "
               "(default: 0 = unlimited)"
            << std::endl;
  std::cout << "  --label <name>          Class label for points/boxes "
               "(default: 'object')"
            << std::endl;
  std::cout << "  --gpu                   Use GPU acceleration (if available)"
            << std::endl;
  std::cout << "  --warmup                Pre-load models before timing inference"
            << std::endl;
  std::cout << "  --help                  Show this help message" << std::endl;
  std::cout << "\nExamples:" << std::endl;
  std::cout
      << "  " << program_name
      << " --image ../assets/i1.png --mode texts --prompt 'cat,computer' --gpu"
      << std::endl;
  std::cout << "  " << program_name
            << " --image ../assets/i3.png --mode points --prompt '558,724' "
               "--label 'pillow' --gpu"
            << std::endl;
  std::cout << "  " << program_name
            << " --image ../assets/i3.png --mode boxes --prompt "
               "'1276,484,1630,820' --label 'potting' --gpu"
            << std::endl;
}

int main(int argc, char *argv[]) {
  std::string image_path;
  std::string output_path = "result.jpg";
  std::string model_path = "";
  std::string mode = "texts";
  std::string prompt;
  std::string label = "object";
  float threshold = 0.25f;
  float nms_threshold = 0.5f;
  int max_detections = 0;
  bool use_gpu = false;
  bool do_warmup = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    } else if (arg == "--image" && i + 1 < argc) {
      image_path = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      output_path = argv[++i];
    } else if (arg == "--model" && i + 1 < argc) {
      model_path = argv[++i];
    } else if (arg == "--mode" && i + 1 < argc) {
      mode = argv[++i];
    } else if (arg == "--prompt" && i + 1 < argc) {
      prompt = argv[++i];
    } else if (arg == "--threshold" && i + 1 < argc) {
      threshold = std::stof(argv[++i]);
    } else if (arg == "--nms" && i + 1 < argc) {
      nms_threshold = std::stof(argv[++i]);
    } else if (arg == "--max-detections" && i + 1 < argc) {
      max_detections = std::stoi(argv[++i]);
    } else if (arg == "--label" && i + 1 < argc) {
      label = argv[++i];
    } else if (arg == "--gpu") {
      use_gpu = true;
    } else if (arg == "--warmup") {
      do_warmup = true;
    }
  }

  if (image_path.empty() || prompt.empty()) {
    std::cerr << "Error: --image and --prompt are required" << std::endl;
    print_usage(argv[0]);
    return -1;
  }

  cv::Mat img = cv::imread(image_path);
  if (img.empty()) {
    std::cerr << "Error: Could not read image: " << image_path << std::endl;
    return -1;
  }

  std::string resolved_model_dir = resolve_model_dir(model_path);

  try {
    SAM3Predictor predictor(resolved_model_dir, use_gpu);
    if (do_warmup) {
      predictor.warmup();
    }

    std::vector<SAM3Predictor::InferenceResult> results;
    std::vector<cv::Point2f> prompt_points;
    std::vector<cv::Rect2f> prompt_boxes;

    if (mode == "texts") {
      std::cout << "Running text segmentation with prompt: \"" << prompt << "\""
                << std::endl;

      // 分割逗号分隔的多类别提示词
      std::vector<std::string> class_names;
      std::stringstream ss(prompt);
      std::string class_name;
      while (std::getline(ss, class_name, ',')) {
        class_name.erase(0, class_name.find_first_not_of(" \t"));
        class_name.erase(class_name.find_last_not_of(" \t") + 1);
        if (!class_name.empty()) {
          class_names.push_back(class_name);
        }
      }

      // 批量处理：图像仅编码一次，显著加速多类别检测
      results = predictor.predict_texts(img, class_names, threshold,
                                        max_detections, nms_threshold);
    } else if (mode == "points") {
      // 鲁棒解析点坐标 (支持 "x,y", "x, y", "x y")
      std::string clean_prompt = prompt;
      std::replace(clean_prompt.begin(), clean_prompt.end(), ',', ' ');
      std::stringstream ss(clean_prompt);
      float x = 0.0f, y = 0.0f;
      if (!(ss >> x >> y)) {
        std::cerr << "Error: Invalid point prompt format: " << prompt << std::endl;
        return -1;
      }
      std::cout << "Running point segmentation at: (" << x << ", " << y
                << ") with label: " << label << std::endl;
      prompt_points.push_back(cv::Point2f(x, y));
      results = predictor.predict_point(img, cv::Point2f(x, y), threshold,
                                        max_detections, label);
    } else if (mode == "boxes") {
      // 鲁棒解析框坐标 (支持 "x1,y1,x2,y2", "x1, y1, x2, y2", "x1 y1 x2 y2")
      std::string clean_prompt = prompt;
      std::replace(clean_prompt.begin(), clean_prompt.end(), ',', ' ');
      std::stringstream ss(clean_prompt);
      float x1 = 0.0f, y1 = 0.0f, x2 = 0.0f, y2 = 0.0f;
      if (!(ss >> x1 >> y1 >> x2 >> y2)) {
        std::cerr << "Error: Invalid box prompt format: " << prompt << std::endl;
        return -1;
      }
      std::cout << "Running box segmentation: [" << x1 << ", " << y1 << ", "
                << x2 << ", " << y2 << "] with label: " << label << std::endl;
      prompt_boxes.push_back(cv::Rect2f(x1, y1, x2 - x1, y2 - y1));
      results = predictor.predict_box(img, cv::Rect2f(x1, y1, x2 - x1, y2 - y1),
                                      threshold, max_detections, label);
    } else {
      std::cerr << "Error: Invalid mode: " << mode
                << ". Use 'texts', 'points', or 'boxes'" << std::endl;
      return -1;
    }

    save_visualization(img, results, output_path, prompt_points, prompt_boxes);
    std::cout << "Done! Found " << results.size() << " objects." << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}
