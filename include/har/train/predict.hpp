#pragma once

#include "har/data/clip_io.hpp"
#include "har/data/video.hpp"
#include "har/network/video_cnn.hpp"
#include "har/train/checkpoint.hpp"
#include "har/train/trainer.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <format>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace har::train {

struct PredictConfig {
  std::string dir{"checkAcc"};
  std::string weights_path{"Weights/ucf11_videocnn.harw"};
  size_t top_k{3};
};

namespace detail {

inline auto normalize_label(std::string s) -> std::string {
  std::string out;
  out.reserve(s.size());
  for (unsigned char c : s) {
    if (c == ' ' || c == '-' || c == '.') {
      out.push_back('_');
    } else {
      out.push_back(static_cast<char>(std::tolower(c)));
    }
  }
  return out;
}

inline auto list_predict_videos(const std::filesystem::path &root)
    -> std::vector<std::filesystem::path> {
  std::vector<std::filesystem::path> videos;
  std::error_code ec;
  if (!std::filesystem::exists(root, ec)) {
    return videos;
  }
  for (auto it = std::filesystem::recursive_directory_iterator(root, ec);
       it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
    if (ec) {
      continue;
    }
    if (it->is_regular_file() && data::is_video_file(it->path())) {
      videos.push_back(it->path());
    }
  }
  std::ranges::sort(videos);
  return videos;
}

inline auto guess_label(const std::filesystem::path &video,
                        const std::filesystem::path &root,
                        const std::vector<std::string> &class_names) -> int {
  auto parent = video.parent_path();
  if (parent.empty() || parent == root) {
    return -1;
  }
  const auto folder = normalize_label(parent.filename().string());
  for (size_t i = 0; i < class_names.size(); ++i) {
    if (normalize_label(class_names[i]) == folder) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

inline auto softmax_topk(const Tensor<float> &logits, size_t row, size_t k)
    -> std::vector<std::pair<size_t, float>> {
  logits.sync();
  const size_t classes = logits.shape()[1];
  k = std::min(k, classes);
  std::vector<float> expv(classes);
  float max_logit = logits.at(row, 0);
  for (size_t c = 1; c < classes; ++c) {
    max_logit = std::max(max_logit, logits.at(row, c));
  }
  float sum = 0.0f;
  for (size_t c = 0; c < classes; ++c) {
    expv[c] = std::exp(logits.at(row, c) - max_logit);
    sum += expv[c];
  }
  std::vector<std::pair<size_t, float>> ranked(classes);
  for (size_t c = 0; c < classes; ++c) {
    ranked[c] = {c, expv[c] / sum};
  }
  std::ranges::partial_sort(
      ranked, ranked.begin() + static_cast<std::ptrdiff_t>(k),
      [](const auto &a, const auto &b) { return a.second > b.second; });
  ranked.resize(k);
  return ranked;
}

inline auto class_name(const std::vector<std::string> &names, size_t index)
    -> std::string {
  if (index < names.size() && !names[index].empty()) {
    return names[index];
  }
  return std::format("class_{}", index);
}

} // namespace detail

inline auto run_predict(const PredictConfig &cfg) -> int {
  const auto dir = std::filesystem::path(cfg.dir);
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);

  if (!std::filesystem::exists(cfg.weights_path)) {
    throw std::runtime_error("weights not found: " + cfg.weights_path);
  }

  const auto peek = peek_checkpoint_meta(cfg.weights_path);
  if (peek.image_size < 8 || peek.max_frames < 1 || peek.num_classes < 1) {
    throw std::runtime_error("checkpoint metadata is invalid: " +
                             cfg.weights_path);
  }

  auto model = build_video_cnn(peek.num_classes, peek.image_size,
                               peek.in_channels, 0.0f);
  const auto meta = load_checkpoint(cfg.weights_path, model);
  model.eval();

  data::FrameExtractOptions frames;
  frames.width = static_cast<int>(meta.image_size);
  frames.height = static_cast<int>(meta.image_size);
  frames.max_frames = static_cast<int>(meta.max_frames);
  frames.rgb = meta.in_channels != 1;
  frames.uniform = true;

  auto class_names = meta.class_names;
  if (class_names.size() != meta.num_classes) {
    class_names.resize(meta.num_classes);
  }

  std::cout << std::format(
      "Predict: dir={}  weights={}  size={}x{}  frames={}  classes={}\n",
      dir.string(), cfg.weights_path, meta.image_size, meta.image_size,
      meta.max_frames, meta.num_classes);

  const auto videos = detail::list_predict_videos(dir);
  if (videos.empty()) {
    std::cout << std::format(
        "No videos in '{}'. Drop .mp4/.avi/.mpg clips here "
        "(optionally in class-named subfolders) and rerun:\n"
        "  har_cnn predict --dir {} --weights {}\n",
        dir.string(), dir.string(), cfg.weights_path);
    return 0;
  }

  size_t labeled = 0;
  size_t correct = 0;
  size_t failed = 0;

  for (const auto &video : videos) {
    const auto rel = std::filesystem::relative(video, dir, ec);
    const auto shown = ec ? video.filename().string() : rel.string();
    Tensor<float> clip;
    try {
      auto decoded = data::extract_video_frames(video.string(), frames);
      clip = data::pad_clip(std::move(decoded), meta.max_frames);
    } catch (const std::exception &ex) {
      std::cout << std::format("  {}  [skip: {}]\n", shown, ex.what());
      ++failed;
      continue;
    }

    std::vector<Tensor<float>> items;
    items.push_back(std::move(clip));
    auto logits = model.forward(stack_batch(items));
    const auto top = detail::softmax_topk(logits, 0, cfg.top_k);
    const int truth = detail::guess_label(video, dir, class_names);
    const bool has_label = truth >= 0;
    const bool ok =
        has_label && !top.empty() &&
        static_cast<size_t>(truth) == top.front().first;
    if (has_label) {
      ++labeled;
      if (ok) {
        ++correct;
      }
    }

    std::cout << std::format("  {}\n", shown);
    for (size_t i = 0; i < top.size(); ++i) {
      const auto mark =
          (i == 0 && has_label) ? (ok ? "  ok" : "  miss") : "";
      std::cout << std::format("    {:>2}. {:<22} {:>6.1f}%{}\n", i + 1,
                               detail::class_name(class_names, top[i].first),
                               top[i].second * 100.0f, mark);
    }
    if (has_label && !ok) {
      std::cout << std::format("       label: {}\n",
                               detail::class_name(class_names,
                                                  static_cast<size_t>(truth)));
    }
  }

  std::cout << std::format("done: {} video(s)", videos.size());
  if (failed > 0) {
    std::cout << std::format(", {} failed to decode", failed);
  }
  if (labeled > 0) {
    std::cout << std::format(", labeled {}/{} ({:.1f}%)", correct, labeled,
                             100.0f * static_cast<float>(correct) /
                                 static_cast<float>(labeled));
  }
  std::cout << '\n';
  return failed > 0 && failed == videos.size() ? 1 : 0;
}

} // namespace har::train

namespace har::train {
struct EvalConfig {
  std::string dataset{"ucf11"};
  std::string data_root;
  std::string weights_path;
  std::string split{"test"};
  size_t batch_size{16};
  size_t num_workers{0};
  unsigned seed{42};
};

inline auto run_eval(const EvalConfig &) -> int {
  std::cerr << "eval is not implemented yet.\n";
  return 1;
}
}
