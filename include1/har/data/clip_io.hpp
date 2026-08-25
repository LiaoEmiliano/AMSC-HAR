#pragma once

#include "har/data/video.hpp"
#include "har/tensor.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace har::data {

inline auto is_video_file(const std::filesystem::path &path) -> bool {
  auto ext = path.extension().string();
  std::ranges::transform(ext, ext.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return ext == ".mpg" || ext == ".mpeg" || ext == ".avi" || ext == ".mp4" ||
         ext == ".mov" || ext == ".mkv" || ext == ".wmv" || ext == ".webm" ||
         ext == ".ogv" || ext == ".ogg";
}

inline auto is_skipped_dir(const std::filesystem::path &path) -> bool {
  const auto name = path.filename().string();
  return name == "Annotation" || name == "annotation" || name == "." ||
         name == ".." || name == "ucfTrainTestlist";
}

inline auto pad_clip(Tensor<float> frames, size_t max_frames) -> Tensor<float> {
  if (frames.dims() != 4) {
    throw std::invalid_argument("pad_clip: expected [T,C,H,W]");
  }
  const size_t F = frames.shape()[0];
  const size_t C = frames.shape()[1];
  const size_t H = frames.shape()[2];
  const size_t W = frames.shape()[3];
  if (F == max_frames) {
    return frames;
  }
  Tensor<float> out({max_frames, C, H, W}, frames.device());
  if (F == 0) {
    return out;
  }
  const size_t plane = C * H * W;
  const size_t copy_n = std::min(F, max_frames);
  frames.sync();
#ifdef HAR_HAS_CUDA
  cuda_check(cudaMemcpy(out.data(), frames.data(),
                        copy_n * plane * sizeof(float), cudaMemcpyDefault),
             "pad_clip copy");
#else
  std::memcpy(out.data(), frames.data(), copy_n * plane * sizeof(float));
#endif
  if (F < max_frames) {
    const float *last = out.data() + (F - 1) * plane;
    for (size_t t = F; t < max_frames; ++t) {
#ifdef HAR_HAS_CUDA
      cuda_check(cudaMemcpy(out.data() + t * plane, last, plane * sizeof(float),
                            cudaMemcpyDefault),
                 "pad_clip repeat");
#else
      std::memcpy(out.data() + t * plane, last, plane * sizeof(float));
#endif
    }
  }
  return out;
}

inline auto make_cache_frames(const FrameExtractOptions &frames,
                              int cache_frames) -> FrameExtractOptions {
  FrameExtractOptions out = frames;
  out.uniform = true;
  const int want = cache_frames > 0 ? cache_frames : frames.max_frames;
  out.max_frames = std::max(frames.max_frames, want);
  return out;
}

inline auto gather_frames(const Tensor<float> &src,
                          const std::vector<size_t> &idx) -> Tensor<float> {
  if (src.dims() != 4) {
    throw std::invalid_argument("gather_frames: expected [T,C,H,W]");
  }
  const size_t C = src.shape()[1];
  const size_t H = src.shape()[2];
  const size_t W = src.shape()[3];
  const size_t plane = C * H * W;
  Tensor<float> out({idx.size(), C, H, W}, Device::CPU);
  if (src.shape()[0] == 0) {
    return out;
  }
  src.sync();
  for (size_t i = 0; i < idx.size(); ++i) {
    const size_t t = std::min(idx[i], src.shape()[0] - 1);
    std::memcpy(out.data() + i * plane, src.data() + t * plane,
                plane * sizeof(float));
  }
  return out;
}

inline void flip_clip_horizontal(Tensor<float> &clip) {
  if (clip.dims() != 4) {
    throw std::invalid_argument("flip_clip_horizontal: expected [T,C,H,W]");
  }
  clip.sync();
  const size_t T = clip.shape()[0];
  const size_t C = clip.shape()[1];
  const size_t H = clip.shape()[2];
  const size_t W = clip.shape()[3];
  for (size_t t = 0; t < T; ++t) {
    for (size_t c = 0; c < C; ++c) {
      for (size_t h = 0; h < H; ++h) {
        for (size_t w = 0; w < W / 2; ++w) {
          std::swap(clip.at(t, c, h, w), clip.at(t, c, h, W - 1 - w));
        }
      }
    }
  }
}

inline void reverse_clip_time(Tensor<float> &clip) {
  if (clip.dims() != 4) {
    throw std::invalid_argument("reverse_clip_time: expected [T,C,H,W]");
  }
  clip.sync();
  const size_t T = clip.shape()[0];
  const size_t plane = clip.shape()[1] * clip.shape()[2] * clip.shape()[3];
  std::vector<float> tmp(plane);
  for (size_t i = 0, j = T - 1; i < j; ++i, --j) {
    float *a = clip.data() + i * plane;
    float *b = clip.data() + j * plane;
    std::memcpy(tmp.data(), a, plane * sizeof(float));
    std::memcpy(a, b, plane * sizeof(float));
    std::memcpy(b, tmp.data(), plane * sizeof(float));
  }
}

inline auto sample_model_clip(Tensor<float> src, size_t out_frames, bool augment)
    -> Tensor<float> {
  if (src.dims() != 4) {
    throw std::invalid_argument("sample_model_clip: expected [T,C,H,W]");
  }
  const size_t t_in = src.shape()[0];
  const size_t t_out = std::max<size_t>(1, out_frames);
  thread_local std::mt19937 rng{std::random_device{}()};
  std::vector<size_t> idx(t_out);
  if (!augment || t_in <= t_out) {
    if (t_in <= 1 || t_out == 1) {
      std::fill(idx.begin(), idx.end(), 0);
    } else {
      for (size_t i = 0; i < t_out; ++i) {
        idx[i] = i * (t_in - 1) / (t_out - 1);
      }
    }
  } else {
    std::uniform_int_distribution<size_t> dist(0, t_in - t_out);
    const size_t start = dist(rng);
    for (size_t i = 0; i < t_out; ++i) {
      idx[i] = start + i;
    }
  }
  auto out = (t_in == t_out && !augment) ? std::move(src) : gather_frames(src, idx);
  if (augment) {
    std::bernoulli_distribution coin(0.5);
    if (coin(rng)) {
      flip_clip_horizontal(out);
    }
    if (std::bernoulli_distribution(0.3)(rng)) {
      reverse_clip_time(out);
    }
  }
  return out;
}

class ClipCache {
public:
  ClipCache(std::string cache_dir, FrameExtractOptions frames)
      : cache_dir_(std::move(cache_dir)), frames_(std::move(frames)) {}

  auto path_for(const std::filesystem::path &video) const
      -> std::filesystem::path {
    const auto key =
        std::format("{}|{}x{}|f{}|rgb{}|u{}|clip", video.generic_string(),
                    frames_.width, frames_.height, frames_.max_frames,
                    frames_.rgb ? 1 : 0, frames_.uniform ? 1 : 0);
    const auto hash = std::hash<std::string>{}(key);
    return std::filesystem::path(cache_dir_) /
           std::format("{}_{:016x}.bin", video.stem().string(),
                       static_cast<unsigned long long>(hash));
  }

  auto read(const std::filesystem::path &video, Tensor<float> &out) const
      -> bool {
    const auto path = path_for(video);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      return false;
    }
    char magic[8]{};
    std::uint32_t T = 0;
    std::uint32_t C = 0;
    std::uint32_t H = 0;
    std::uint32_t W = 0;
    in.read(magic, 8);
    in.read(reinterpret_cast<char *>(&T), sizeof(T));
    in.read(reinterpret_cast<char *>(&C), sizeof(C));
    in.read(reinterpret_cast<char *>(&H), sizeof(H));
    in.read(reinterpret_cast<char *>(&W), sizeof(W));
    if (!in || std::string_view(magic, 7) != "HARF002") {
      return false;
    }
    const std::uint32_t expect_c = frames_.rgb ? 3u : 1u;
    const std::uint32_t expect_t =
        static_cast<std::uint32_t>(std::max(1, frames_.max_frames));
    if (T != expect_t || C != expect_c ||
        H != static_cast<std::uint32_t>(frames_.height) ||
        W != static_cast<std::uint32_t>(frames_.width)) {
      return false;
    }
    out = Tensor<float>({T, C, H, W}, Device::CPU);
    in.read(reinterpret_cast<char *>(out.data()),
            static_cast<std::streamsize>(out.size() * sizeof(float)));
    if (!static_cast<bool>(in)) {
      return false;
    }
    out.sync();
    return true;
  }

  void write(const std::filesystem::path &video, const Tensor<float> &x) const {
    const auto path = path_for(video);
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    std::ofstream out(path, std::ios::binary);
    if (!out) {
      return;
    }
    x.sync();
    const char magic[8] = {'H', 'A', 'R', 'F', '0', '0', '2', '\0'};
    const std::uint32_t T = static_cast<std::uint32_t>(x.shape()[0]);
    const std::uint32_t C = static_cast<std::uint32_t>(x.shape()[1]);
    const std::uint32_t H = static_cast<std::uint32_t>(x.shape()[2]);
    const std::uint32_t W = static_cast<std::uint32_t>(x.shape()[3]);
    out.write(magic, 8);
    out.write(reinterpret_cast<const char *>(&T), sizeof(T));
    out.write(reinterpret_cast<const char *>(&C), sizeof(C));
    out.write(reinterpret_cast<const char *>(&H), sizeof(H));
    out.write(reinterpret_cast<const char *>(&W), sizeof(W));
    out.write(reinterpret_cast<const char *>(x.data()),
              static_cast<std::streamsize>(x.size() * sizeof(float)));
  }

private:
  std::string cache_dir_;
  FrameExtractOptions frames_;
};

inline auto load_video_clip(const std::filesystem::path &path,
                            const FrameExtractOptions &frames, bool use_cache,
                            const ClipCache *cache) -> Tensor<float> {
  Tensor<float> x;
  if (use_cache && cache != nullptr && cache->read(path, x)) {
    return x;
  }
  FrameExtractOptions opt = frames;
  opt.uniform = true;
  const size_t T = static_cast<size_t>(std::max(1, frames.max_frames));
  const size_t C = frames.rgb ? 3u : 1u;
  const size_t H = static_cast<size_t>(frames.height);
  const size_t W = static_cast<size_t>(frames.width);
  try {
    static std::mutex decode_mu;
    std::lock_guard<std::mutex> lock(decode_mu);
    auto decoded = extract_video_frames(path.string(), opt);
    x = pad_clip(std::move(decoded), T);
  } catch (const std::exception &e) {
    std::cerr << "Warning: skip " << path.string() << " (" << e.what() << ")\n";
    x = Tensor<float>({T, C, H, W}, Device::CPU);
  }
  if (use_cache && cache != nullptr) {
    cache->write(path, x);
  }
  return x;
}

} // namespace har::data
