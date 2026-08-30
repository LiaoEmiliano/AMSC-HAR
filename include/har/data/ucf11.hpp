#pragma once

#include "har/data/clip_io.hpp"
#include "har/data/video.hpp"
#include "har/tensor.hpp"

#include <algorithm>
#include <filesystem>
#include <format>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace har::data {

struct UCF11Options {
  FrameExtractOptions frames{};
  float val_ratio{0.1f};
  float test_ratio{0.1f};
  size_t max_per_class{0};
  bool use_cache{true};
  int cache_frames{16};
  unsigned seed{42};
  std::string cache_dir{"data/ucf11_cache"};

  UCF11Options() { frames.uniform = true; }
};

class UCF11Dataset {
public:
  struct Sample {
    std::filesystem::path path;
    size_t label{0};
  };

  UCF11Dataset(std::string root, UCF11Options options = {})
      : root_(resolve_root(std::filesystem::path(std::move(root)))),
        options_(std::move(options)),
        cache_(options_.cache_dir, make_cache_frames(options_.frames,
                                                     options_.cache_frames)) {
    if (options_.val_ratio < 0.0f || options_.test_ratio < 0.0f ||
        options_.val_ratio + options_.test_ratio >= 1.0f) {
      throw std::invalid_argument(
          "UCF11Options val_ratio + test_ratio must be in [0, 1)");
    }
    if (options_.frames.width <= 0 || options_.frames.height <= 0) {
      throw std::invalid_argument("frame size must be positive");
    }
    scan();
    split();

    std::cout << std::format(
        "UCF11: root={}  classes={}  videos={}  train={}  val={}  test={}\n",
        root_.string(), class_names_.size(), samples_.size(),
        train_indices_.size(), val_indices_.size(), test_indices_.size());
  }

  auto load(size_t index, bool augment = false) const
      -> std::pair<Tensor<float>, float> {
    if (index >= samples_.size()) {
      throw std::out_of_range("UCF11Dataset::load index out of range");
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

  auto options() const -> const UCF11Options & { return options_; }
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

  static auto looks_like_ucf11(const std::filesystem::path &dir) -> bool {
    if (!std::filesystem::is_directory(dir)) {
      return false;
    }
    size_t classes = 0;
    for (const auto &entry : std::filesystem::directory_iterator(dir)) {
      if (entry.is_directory() && !is_skipped_dir(entry.path()) &&
          directory_has_videos(entry.path())) {
        ++classes;
      }
    }
    return classes >= 2;
  }

  static auto resolve_root(std::filesystem::path root) -> std::filesystem::path {
    if (!std::filesystem::exists(root)) {
      throw std::runtime_error("UCF11 root not found: " + root.string());
    }
    if (looks_like_ucf11(root)) {
      return root;
    }
    for (const char *nested :
         {"UCF11_updated_mpg", "action_youtube_naudio", "UCF11"}) {
      const auto candidate = root / nested;
      if (looks_like_ucf11(candidate)) {
        return candidate;
      }
    }
    return root;
  }

  void scan() {
    class_names_.clear();
    samples_.clear();

    std::vector<std::filesystem::path> class_dirs;
    for (const auto &entry : std::filesystem::directory_iterator(root_)) {
      if (entry.is_directory() && !is_skipped_dir(entry.path())) {
        class_dirs.push_back(entry.path());
      }
    }
    std::ranges::sort(class_dirs);

    for (const auto &class_dir : class_dirs) {
      std::vector<std::filesystem::path> videos;
      std::error_code ec;
      for (auto it =
               std::filesystem::recursive_directory_iterator(class_dir, ec);
           it != std::filesystem::recursive_directory_iterator();
           it.increment(ec)) {
        if (ec) {
          continue;
        }
        if (it->is_directory() && is_skipped_dir(it->path())) {
          it.disable_recursion_pending();
          continue;
        }
        if (it->is_regular_file() && is_video_file(it->path())) {
          videos.push_back(it->path());
        }
      }
      if (videos.empty()) {
        continue;
      }
      std::ranges::sort(videos);
      if (options_.max_per_class > 0 && videos.size() > options_.max_per_class) {
        videos.resize(options_.max_per_class);
      }

      const size_t label = class_names_.size();
      class_names_.push_back(class_dir.filename().string());
      for (auto &video : videos) {
        samples_.push_back(Sample{std::move(video), label});
      }
    }

    if (class_names_.empty()) {
      throw std::runtime_error(
          "No UCF11 action classes found under " + root_.string() +
          ". Pass the extracted UCF11_updated_mpg (or action_youtube_naudio) directory.");
    }
  }

  void split() {
    train_indices_.clear();
    val_indices_.clear();
    test_indices_.clear();

    std::unordered_map<size_t, std::vector<size_t>> by_class;
    for (size_t i = 0; i < samples_.size(); ++i) {
      by_class[samples_[i].label].push_back(i);
    }

    std::mt19937 rng{options_.seed};
    for (size_t c = 0; c < class_names_.size(); ++c) {
      auto &idx = by_class[c];
      std::shuffle(idx.begin(), idx.end(), rng);
      size_t n_test = static_cast<size_t>(static_cast<float>(idx.size()) *
                                          options_.test_ratio);
      size_t n_val = static_cast<size_t>(static_cast<float>(idx.size()) *
                                         options_.val_ratio);
      if (idx.size() > 1) {
        n_test = std::min(n_test, idx.size() - 1);
        const size_t remain = idx.size() - n_test;
        n_val = std::min(n_val, remain > 0 ? remain - 1 : 0);
      } else {
        n_test = 0;
        n_val = 0;
      }
      auto it = idx.begin();
      test_indices_.insert(test_indices_.end(), it,
                           it + static_cast<std::ptrdiff_t>(n_test));
      it += static_cast<std::ptrdiff_t>(n_test);
      val_indices_.insert(val_indices_.end(), it,
                          it + static_cast<std::ptrdiff_t>(n_val));
      it += static_cast<std::ptrdiff_t>(n_val);
      train_indices_.insert(train_indices_.end(), it, idx.end());
    }
    std::ranges::sort(train_indices_);
    std::ranges::sort(val_indices_);
    std::ranges::sort(test_indices_);
  }

  std::filesystem::path root_;
  UCF11Options options_;
  ClipCache cache_;
  std::vector<std::string> class_names_;
  std::vector<Sample> samples_;
  std::vector<size_t> train_indices_;
  std::vector<size_t> val_indices_;
  std::vector<size_t> test_indices_;
};

} // namespace har::data
