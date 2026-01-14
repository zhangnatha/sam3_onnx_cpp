#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "SAM3Predictor.h"

// 获取调色板颜色
cv::Scalar get_color(int id) {
  const std::vector<cv::Scalar> palette = {
      {255, 0, 0},   {0, 255, 0},   {0, 0, 255},   {255, 255, 0},
      {255, 0, 255}, {0, 255, 255}, {255, 128, 0}, {255, 0, 128},
      {128, 255, 0}, {0, 255, 128}, {128, 0, 255}, {0, 128, 255}};
  return palette[id % palette.size()];
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

    // 绘制彩色掩码
    cv::Mat colored_mask = cv::Mat::zeros(img.size(), img.type());
    colored_mask.setTo(color, results[i].mask);
    cv::addWeighted(overlay, 1.0, colored_mask, 0.5, 0, overlay);

    // 绘制边界框
    cv::rectangle(overlay, results[i].box, color, 2);

    // 绘制标签(白色文字,半透明彩色背景)
    std::string text =
        results[i].label + " " + std::to_string(results[i].score).substr(0, 4);
    int baseline;
    cv::Size text_size =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);

    // 确保标签在图像范围内
    int label_x = std::max(0, (int)results[i].box.x);
    int label_y = std::max(text_size.height + 5, (int)results[i].box.y);
    int label_w = std::min(text_size.width, overlay.cols - label_x);
    int label_h = text_size.height + 5;

    if (label_w > 0 && label_h > 0 && label_x + label_w <= overlay.cols &&
        label_y >= label_h) {
      // 创建半透明背景
      cv::Mat label_bg =
          overlay(cv::Rect(label_x, label_y - label_h, label_w, label_h))
              .clone();
      cv::rectangle(label_bg, cv::Point(0, 0), cv::Point(label_w, label_h),
                    color, -1);
      cv::addWeighted(
          overlay(cv::Rect(label_x, label_y - label_h, label_w, label_h)), 0.6,
          label_bg, 0.4, 0,
          overlay(cv::Rect(label_x, label_y - label_h, label_w, label_h)));

      // 绘制白色文字
      cv::putText(overlay, text, cv::Point(label_x, label_y - 5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1,
                  cv::LINE_AA);
    }
  }

  // 绘制原始提示词 (提示点)
  for (const auto &p : prompt_points) {
    // 绘制外部黑边
    cv::circle(overlay, p, 6, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    // 绘制内部白点
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
  std::cout << "  --mode <type>           Prompt mode: texts|points|boxes "
               "(default: texts)"
            << std::endl;
  std::cout << "  --prompt <value>        Prompt value:" << std::endl;
  std::cout << "                          - texts: text string (e.g., 'person')"
            << std::endl;
  std::cout
      << "                          - points: x,y coordinates (e.g., '100,200')"
      << std::endl;
  std::cout << "                          - boxes: x1,y1,x2,y2 coordinates "
               "(e.g., '100,100,200,200')"
            << std::endl;
  std::cout << "  --threshold <value>     Confidence threshold (default: 0.25)"
            << std::endl;
  std::cout << "  --max-detections <num>  Maximum number of detections "
               "(default: 0 = unlimited)"
            << std::endl;
  std::cout << "  --label <name>          Class label for points/boxes "
               "(default: 'object')"
            << std::endl;
  std::cout << "  --gpu                   Use GPU acceleration (if available)"
            << std::endl;
  std::cout << "  --help                  Show this help message" << std::endl;
  std::cout << "\nExamples:" << std::endl;
  std::cout
      << "  " << program_name
      << " --image demo.png --mode texts --prompt 'person' --max-detections 5"
      << std::endl;
  std::cout << "  " << program_name
            << " --image demo.png --mode points --prompt '384,179' --gpu"
            << std::endl;
  std::cout << "  " << program_name
            << " --image demo.png --mode boxes --prompt '100,100,300,300' "
               "--threshold 0.3"
            << std::endl;
}

int main(int argc, char *argv[]) {
  // 默认参数
  std::string image_path;
  std::string output_path = "result.jpg";
  std::string mode = "texts";
  std::string prompt;
  std::string label = "object";
  float threshold = 0.25f;
  int max_detections = 0;
  bool use_gpu = false;

  // 解析命令行参数
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    } else if (arg == "--image" && i + 1 < argc) {
      image_path = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      output_path = argv[++i];
    } else if (arg == "--mode" && i + 1 < argc) {
      mode = argv[++i];
    } else if (arg == "--prompt" && i + 1 < argc) {
      prompt = argv[++i];
    } else if (arg == "--threshold" && i + 1 < argc) {
      threshold = std::stof(argv[++i]);
    } else if (arg == "--max-detections" && i + 1 < argc) {
      max_detections = std::stoi(argv[++i]);
    } else if (arg == "--label" && i + 1 < argc) {
      label = argv[++i];
    } else if (arg == "--gpu") {
      use_gpu = true;
    }
  }

  // 验证必需参数
  if (image_path.empty() || prompt.empty()) {
    std::cerr << "Error: --image and --prompt are required" << std::endl;
    print_usage(argv[0]);
    return -1;
  }

  // 加载图像
  cv::Mat img = cv::imread(image_path);
  if (img.empty()) {
    std::cerr << "Error: Could not read image: " << image_path << std::endl;
    return -1;
  }

  try {
    SAM3Predictor predictor("../model", use_gpu);
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
        // 去除前后空格
        class_name.erase(0, class_name.find_first_not_of(" \t"));
        class_name.erase(class_name.find_last_not_of(" \t") + 1);
        if (!class_name.empty()) {
          class_names.push_back(class_name);
        }
      }

      // 为每个类别单独运行推理
      for (const auto &cls : class_names) {
        auto cls_results =
            predictor.predict_text(img, cls, threshold, max_detections);
        results.insert(results.end(), cls_results.begin(), cls_results.end());
      }
    } else if (mode == "points") {
      // 解析点坐标 "x,y"
      std::stringstream ss(prompt);
      float x, y;
      char comma;
      ss >> x >> comma >> y;
      std::cout << "Running point segmentation at: (" << x << ", " << y
                << ") with label: " << label << std::endl;
      prompt_points.push_back(cv::Point2f(x, y));
      results = predictor.predict_point(img, cv::Point2f(x, y), threshold,
                                        max_detections, label);
    } else if (mode == "boxes") {
      // 解析框坐标 "x1,y1,x2,y2"
      std::stringstream ss(prompt);
      float x1, y1, x2, y2;
      char comma;
      ss >> x1 >> comma >> y1 >> comma >> x2 >> comma >> y2;
      std::cout << "Running box segmentation: [" << x1 << "," << y1 << "," << x2
                << "," << y2 << "] with label: " << label << std::endl;
      prompt_boxes.push_back(cv::Rect2f(x1, y1, x2 - x1, y2 - y1));
      results = predictor.predict_box(img, cv::Rect2f(x1, y1, x2 - x1, y2 - y1),
                                      threshold, max_detections, label);
    } else {
      std::cerr << "Error: Invalid mode. Use 'texts', 'points', or 'boxes'"
                << std::endl;
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
