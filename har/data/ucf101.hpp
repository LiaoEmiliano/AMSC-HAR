#pragma once

#include "har/data/clip_io.hpp"
#include "har/data/video.hpp"
#include "har/tensor.hpp"

#include <algorithm>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace har::data {

struct UCF101Options {
  FrameExtractOptions frames{};
  float val_ratio{0.1f};
  int split{1};
  size_t max_per_class{0};
  bool use_cache{true};
  int cache_frames{16};
  unsigned seed{42};
  std::string cache_dir{"data/ucf101_cache"};
  std::string split_dir{};

  UCF101Options() { frames.uniform = true; }
};

// UCF101 with the official recognition train/test split (1, 2, or 3).
// Validation is carved from the official training list.
class UCF101Dataset {
public:
  struct Sample {
    std::filesystem::path path;
    size_t label{0};
  };

  UCF101Dataset(std::string root, UCF101Options options = {})
      : root_(resolve_video_root(std::filesystem::path(std::move(root)))),
        options_(std::move(options)),
        cache_(options_.cache_dir, make_cache_frames(options_.frames,
                                                     options_.cache_frames)) {
    if (options_.val_ratio < 0.0f || options_.val_ratio >= 1.0f) {
      throw std::invalid_argument("UCF101Options val_ratio must be in [0, 1)");
    }
    if (options_.frames.width <= 0 || options_.frames.height <= 0) {
      throw std::invalid_argument("frame size must be positive");
    }
    if (options_.split < 1 || options_.split > 3) {
      throw std::invalid_argument("UCF101 split must be 1, 2, or 3");
    }
    split_dir_ = resolve_split_dir(root_, options_.split_dir);
    load_class_index();
    load_official_split();

    std::cout << std::format(
        "UCF101: root={}  split={}  classes={}  videos={}  train={}  val={}  "
        "test={}\n",
        root_.string(), options_.split, class_names_.size(), samples_.size(),
        train_indices_.size(), val_indices_.size(), test_indices_.size());
  }

  auto load(size_t index, bool augment = false) const
      -> std::pair<Tensor<float>, float> {
    if (index >= samples_.size()) {
      throw std::out_of_range("UCF101Dataset::load index out of range");
    }
    const auto &sample = samples_[index];
    const auto cache_opt =
        make_cache_frames(options_.frames, options_.cache_frames);
    auto x = load_video_clip(sample.path, cache_opt, options_.use_cache,
                             options_.use_cache ? &cache_ : nullptr);
    x = sample_model_clip(std::move(x),
                          static_cast<size_t>(std::max(1, options_.frames.max_frames)),
                          augment);
    return {std::move(x), static_cast<float>(sample.label)};
  }

  auto train_size() const -> size_t { return train_indices_.size(); }
  auto val_size() const -> size_t { return val_indices_.size(); }
  auto test_size() const -> size_t { return test_indices_.size(); }
  auto size() const -> size_t { return samples_.size(); }
  auto num_classes() const -> size_t { return class_names_.size(); }

  auto options() const -> const UCF101Options & { return options_; }
  auto root() const -> const std::filesystem::path & { return root_; }
  auto class_names() const -> const std::vector<std::string> & {
    return class_names_;
  }

  auto train_indices() const -> const std::vector<size_t> & {
    return train_indices_;
  }
  auto val_indices() const -> const std::vector<size_t> & {
    return val_indices_;
  }
  auto test_indices() const -> const std::vector<size_t> & {
    return test_indices_;
  }
  auto samples() const -> const std::vector<Sample> & { return samples_; }

private:
  static auto directory_has_videos(const std::filesystem::path &dir) -> bool {
    std::error_code ec;
    for (auto it = std::filesystem::recursive_directory_iterator(dir, ec);
         it != std::filesystem::recursive_directory_iterator();
         it.increment(ec)) {
      if (ec) {
        break;
      }
      if (it->is_directory() && is_skipped_dir(it->path())) {
        it.disable_recursion_pending();
        continue;
      }
      if (it->is_regular_file() && is_video_file(it->path())) {
        return true;
      }
    }
    return false;
  }

  static auto looks_like_ucf101(const std::filesystem::path &dir) -> bool {
    if (!std::filesystem::is_directory(dir)) {
      return false;
    }
    size_t classes = 0;
    for (const auto &entry : std::filesystem::directory_iterator(dir)) {
      if (entry.is_directory() && !is_skipped_dir(entry.path()) &&
          directory_has_videos(entry.path())) {
        ++classes;
        if (classes >= 50) {
          return true;
        }
      }
    }
    return false;
  }

  static auto resolve_video_root(std::filesystem::path root)
      -> std::filesystem::path {
    if (!std::filesystem::exists(root)) {
      throw std::runtime_error("UCF101 root not found: " + root.string());
    }
    for (const char *nested : {"UCF-101", "UCF101", "ucf101", "UCF101_videos"}) {
      const auto candidate = root / nested;
      if (looks_like_ucf101(candidate)) {
        return candidate;
      }
    }
    if (looks_like_ucf101(root)) {
      return root;
    }
    if (std::filesystem::is_regular_file(root)) {
      throw std::runtime_error(
          "UCF101 root is a file, expected extracted video directory: " +
          root.string());
    }
    return root;
  }

  static auto resolve_split_dir(const std::filesystem::path &video_root,
                                const std::string &explicit_dir)
      -> std::filesystem::path {
    std::vector<std::filesystem::path> candidates;
    if (!explicit_dir.empty()) {
      candidates.emplace_back(explicit_dir);
    }
    candidates.push_back(video_root / "ucfTrainTestlist");
    candidates.push_back(video_root.parent_path() / "ucfTrainTestlist");
    candidates.emplace_back("data/ucfTrainTestlist");
    for (const auto &dir : candidates) {
      if (std::filesystem::exists(dir / "classInd.txt")) {
        return dir;
      }
    }
    throw std::runtime_error(
        "UCF101 splits not found (need ucfTrainTestlist/classInd.txt). "
        "Run scripts/download_ucf101.ps1");
  }

  static auto trim(std::string s) -> std::string {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
      return {};
    }
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
  }

  static auto normalize_rel(std::string rel) -> std::string {
    std::replace(rel.begin(), rel.end(), '\\', '/');
    return rel;
  }

  void load_class_index() {
    const auto path = split_dir_ / "classInd.txt";
    std::ifstream in(path);
    if (!in) {
      throw std::runtime_error("cannot read " + path.string());
    }
    class_names_.clear();
    class_to_label_.clear();
    std::string line;
    while (std::getline(in, line)) {
      line = trim(std::move(line));
      if (line.empty()) {
        continue;
      }
      std::istringstream iss(line);
      int id = 0;
      std::string name;
      if (!(iss >> id >> name) || id < 1) {
        continue;
      }
      const size_t label = static_cast<size_t>(id - 1);
      if (class_names_.size() <= label) {
        class_names_.resize(label + 1);
      }
      class_names_[label] = name;
      class_to_label_[name] = label;
    }
    if (class_names_.size() < 2) {
      throw std::runtime_error("UCF101 classInd.txt has too few classes");
    }
  }

  auto find_video(const std::string &rel) const -> std::filesystem::path {
    const auto direct = root_ / rel;
    if (std::filesystem::exists(direct)) {
      return direct;
    }
    const auto stem = std::filesystem::path(rel).stem();
    const auto parent = std::filesystem::path(rel).parent_path();
    const auto dir = root_ / parent;
    if (!std::filesystem::is_directory(dir)) {
      return {};
    }
    std::error_code ec;
    for (const auto &entry : std::filesystem::directory_iterator(dir, ec)) {
      if (entry.is_regular_file() && is_video_file(entry.path()) &&
          entry.path().stem() == stem) {
        return entry.path();
      }
    }
    return {};
  }

  auto add_sample(const std::string &rel, size_t label) -> size_t {
    auto path = find_video(rel);
    if (path.empty()) {
      std::cerr << "Warning: missing UCF101 video " << rel << "\n";
      return static_cast<size_t>(-1);
    }
    const auto key = path.generic_string();
    auto it = path_to_index_.find(key);
    if (it != path_to_index_.end()) {
      return it->second;
    }
    const size_t idx = samples_.size();
    samples_.push_back(Sample{std::move(path), label});
    path_to_index_[key] = idx;
    return idx;
  }

  auto read_list(const std::filesystem::path &path, bool has_label)
      -> std::vector<size_t> {
    std::ifstream in(path);
    if (!in) {
      throw std::runtime_error("cannot read " + path.string());
    }
    std::vector<size_t> indices;
    std::unordered_map<size_t, size_t> per_class;
    std::string line;
    while (std::getline(in, line)) {
      line = trim(std::move(line));
      if (line.empty()) {
        continue;
      }
      std::istringstream iss(line);
      std::string rel;
      int id = 0;
      iss >> rel;
      rel = normalize_rel(rel);
      if (rel.empty()) {
        continue;
      }
      size_t label = 0;
      if (has_label && (iss >> id) && id >= 1) {
        label = static_cast<size_t>(id - 1);
      } else {
        const auto cls = std::filesystem::path(rel).parent_path().string();
        auto it = class_to_label_.find(cls);
        if (it == class_to_label_.end()) {
          std::cerr << "Warning: unknown class for " << rel << "\n";
          continue;
        }
        label = it->second;
      }
      if (options_.max_per_class > 0 &&
          per_class[label] >= options_.max_per_class) {
        continue;
      }
      const size_t idx = add_sample(rel, label);
      if (idx == static_cast<size_t>(-1)) {
        continue;
      }
      indices.push_back(idx);
      ++per_class[label];
    }
    return indices;
  }

  void load_official_split() {
    samples_.clear();
    path_to_index_.clear();
    train_indices_.clear();
    val_indices_.clear();
    test_indices_.clear();

    const auto train_file = split_dir_ / std::format("trainlist{:02d}.txt",
                                                     options_.split);
    const auto test_file =
        split_dir_ / std::format("testlist{:02d}.txt", options_.split);

    auto train_all = read_list(train_file, true);
    test_indices_ = read_list(test_file, false);

    std::unordered_map<size_t, std::vector<size_t>> by_class;
    for (size_t idx : train_all) {
      by_class[samples_[idx].label].push_back(idx);
    }
    std::mt19937 rng{options_.seed};
    for (size_t c = 0; c < class_names_.size(); ++c) {
      auto &idx = by_class[c];
      std::shuffle(idx.begin(), idx.end(), rng);
      size_t n_val = static_cast<size_t>(static_cast<float>(idx.size()) *
                                         options_.val_ratio);
      if (idx.size() > 1) {
        n_val = std::min(n_val, idx.size() - 1);
      } else {
        n_val = 0;
      }
      val_indices_.insert(val_indices_.end(), idx.begin(),
                          idx.begin() + static_cast<std::ptrdiff_t>(n_val));
      train_indices_.insert(train_indices_.end(),
                            idx.begin() + static_cast<std::ptrdiff_t>(n_val),
                            idx.end());
    }
    std::ranges::sort(train_indices_);
    std::ranges::sort(val_indices_);
    std::ranges::sort(test_indices_);

    if (train_indices_.empty()) {
      throw std::runtime_error("UCF101 train split is empty under " +
                               root_.string());
    }
  }

  std::filesystem::path root_;
  std::filesystem::path split_dir_;
  UCF101Options options_;
  ClipCache cache_;
  std::vector<std::string> class_names_;
  std::unordered_map<std::string, size_t> class_to_label_;
  std::vector<Sample> samples_;
  std::unordered_map<std::string, size_t> path_to_index_;
  std::vector<size_t> train_indices_;
  std::vector<size_t> val_indices_;
  std::vector<size_t> test_indices_;
};

} // namespace har::data
