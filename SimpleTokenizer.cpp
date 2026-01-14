#include "SimpleTokenizer.h"
#include <algorithm>
#include <fstream>
#include <regex>
#include <sstream>

// 模拟 Python ord() 函数的辅助函数
static int ord(char c) { return (unsigned char)c; }

SimpleTokenizer::SimpleTokenizer(const std::string &vocab_path,
                                 const std::string &merges_path) {
  init_byte_encoder();

  // 加载词表
  std::ifstream vocab_file(vocab_path);
  std::string line;
  int current_id = 0;
  while (std::getline(vocab_file, line)) {
    if (line.empty() && vocab_file.eof())
      break;
    // 来自 export_tokenizer.py 的词表文件每行包含一个 token
    // 行索引即为 ID。
    encoder[line] = current_id;
    decoder[current_id] = line;
    current_id++;
  }

  // 加载合并规则 (merges)
  std::ifstream merges_file(merges_path);
  int rank = 0;
  while (std::getline(merges_file, line)) {
    if (line.empty())
      continue;
    std::istringstream iss(line);
    std::string first, second;
    if (iss >> first >> second) {
      bpe_ranks[{first, second}] = rank++;
    }
  }

  if (encoder.count("<|startoftext|>"))
    sot_token_id = encoder["<|startoftext|>"];
  else
    sot_token_id = 49406; // 默认 CLIP SOT

  if (encoder.count("<|endoftext|>"))
    eot_token_id = encoder["<|endoftext|>"];
  else
    eot_token_id = 49407; // 默认 CLIP EOT
}

void SimpleTokenizer::init_byte_encoder() {
  std::vector<int> bs;
  // bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") +
  // 1)) + list(range(ord("®"), ord("ÿ") + 1))
  for (int b = ord('!'); b <= ord('~'); ++b)
    bs.push_back(b);
  for (int b = 0xA1; b <= 0xAC; ++b)
    bs.push_back(b);
  for (int b = 0xAE; b <= 0xFF; ++b)
    bs.push_back(b);

  std::vector<int> cs = bs;
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
      bs.push_back(b);
      cs.push_back(256 + n);
      n++;
    }
  }

  // 转换为 UTF-8 字符串
  auto to_utf8 = [](int c) {
    std::string s;
    if (c < 128) {
      s += (char)c;
    } else if (c < 2048) {
      s += (char)(0xC0 | (c >> 6));
      s += (char)(0x80 | (c & 0x3F));
    } else if (c < 65536) {
      s += (char)(0xE0 | (c >> 12));
      s += (char)(0x80 | ((c >> 6) & 0x3F));
      s += (char)(0x80 | (c & 0x3F));
    }
    return s;
  };

  for (size_t i = 0; i < bs.size(); ++i) {
    std::string utf8_c = to_utf8(cs[i]);
    byte_encoder[bs[i]] = utf8_c;
    byte_decoder[utf8_c] = bs[i];
  }
}

std::string SimpleTokenizer::bpe(const std::string &token) {
  if (cache.count(token))
    return cache[token];

  std::vector<std::string> word;
  for (size_t i = 0; i < token.length();) {
    size_t len = 1;
    unsigned char c = (unsigned char)token[i];
    if (c >= 0xf0)
      len = 4;
    else if (c >= 0xe0)
      len = 3;
    else if (c >= 0xc0)
      len = 2;
    word.push_back(token.substr(i, len));
    i += len;
  }

  word.back() += "</w>";

  auto get_pairs = [](const std::vector<std::string> &word) {
    std::set<std::pair<std::string, std::string>> pairs;
    for (size_t i = 0; i < word.size() - 1; ++i) {
      pairs.insert({word[i], word[i + 1]});
    }
    return pairs;
  };

  std::set<std::pair<std::string, std::string>> pairs = get_pairs(word);
  if (pairs.empty())
    return token + "</w>";

  while (true) {
    std::pair<std::string, std::string> bigram = {"", ""};
    int min_rank = -1;

    for (auto const &pair : pairs) {
      if (bpe_ranks.count(pair)) {
        int r = bpe_ranks[pair];
        if (min_rank == -1 || r < min_rank) {
          min_rank = r;
          bigram = pair;
        }
      }
    }

    if (min_rank == -1)
      break;

    std::vector<std::string> new_word;
    for (size_t i = 0; i < word.size();) {
      if (i < word.size() - 1 && word[i] == bigram.first &&
          word[i + 1] == bigram.second) {
        new_word.push_back(bigram.first + bigram.second);
        i += 2;
      } else {
        new_word.push_back(word[i]);
        i += 1;
      }
    }
    word = new_word;
    if (word.size() == 1)
      break;
    pairs = get_pairs(word);
  }

  std::string result = "";
  for (size_t i = 0; i < word.size(); ++i) {
    result += word[i] + (i == word.size() - 1 ? "" : " ");
  }
  cache[token] = result;
  return result;
}

std::vector<int> SimpleTokenizer::encode(const std::string &text) {
  std::vector<int> bpe_tokens;

  std::string cleaned = text;
  std::transform(cleaned.begin(), cleaned.end(), cleaned.begin(), ::tolower);

  // 简化版的正则表达式
  std::regex pat("<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'"
                 "d|[a-z]+|[0-9]+|[^\\s a-z0-9]+",
                 std::regex::icase);

  auto words_begin = std::sregex_iterator(cleaned.begin(), cleaned.end(), pat);
  auto words_end = std::sregex_iterator();

  for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
    std::string token = i->str();
    std::string encoded_token = "";
    for (unsigned char b : token) {
      encoded_token += byte_encoder[b];
    }

    std::string bpe_res = bpe(encoded_token);
    std::stringstream ss(bpe_res);
    std::string part;
    while (ss >> part) {
      if (encoder.count(part)) {
        bpe_tokens.push_back(encoder[part]);
      }
    }
  }

  return bpe_tokens;
}

std::vector<std::vector<int>>
SimpleTokenizer::tokenize(const std::vector<std::string> &texts,
                          int context_length) {
  std::vector<std::vector<int>> results;
  for (const auto &text : texts) {
    std::vector<int> tokens = encode(text);
    std::vector<int> result(context_length, 0);

    result[0] = sot_token_id;
    int count = 1;
    for (int t : tokens) {
      if (count >= context_length - 1)
        break;
      result[count++] = t;
    }
    result[count] = eot_token_id;
    results.push_back(result);
  }
  return results;
}
