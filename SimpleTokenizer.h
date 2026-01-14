#ifndef SIMPLE_TOKENIZER_H
#define SIMPLE_TOKENIZER_H

#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

class SimpleTokenizer {
public:
  SimpleTokenizer(const std::string &vocab_path,
                  const std::string &merges_path);

  std::vector<int> encode(const std::string &text);
  std::string decode(const std::vector<int> &tokens);

  // 用于 SAM3 风格的调用 [batch, context_length]
  std::vector<std::vector<int>> tokenize(const std::vector<std::string> &texts,
                                         int context_length = 77);

private:
  std::map<int, std::string> byte_encoder;
  std::map<std::string, int> byte_decoder;
  std::unordered_map<std::string, int> encoder;
  std::unordered_map<int, std::string> decoder;
  std::map<std::pair<std::string, std::string>, int> bpe_ranks;
  std::unordered_map<std::string, std::string> cache;

  int sot_token_id;
  int eot_token_id;

  void init_byte_encoder();
  std::string bpe(const std::string &token);
  std::vector<std::string> whitespace_clean(const std::string &text);
};

#endif // SIMPLE_TOKENIZER_H
