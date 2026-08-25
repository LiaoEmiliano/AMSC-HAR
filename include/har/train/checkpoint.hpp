#pragma once

#include "har/layers/norm.hpp"
#include "har/network/video_cnn.hpp"
#include "har/tensor.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace har::train {

struct CheckpointMeta {
  std::uint32_t image_size{64};
  std::uint32_t max_frames{8};
  std::uint32_t in_channels{3};
  std::uint32_t num_classes{11};
  std::vector<std::string> class_names;
};

namespace detail {

inline constexpr char kMagic[8] = {'H', 'A', 'R', 'W', '0', '0', '1', '\0'};

inline void write_u32(std::ostream &out, std::uint32_t value) {
  out.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

inline auto read_u32(std::istream &in) -> std::uint32_t {
  std::uint32_t value = 0;
  in.read(reinterpret_cast<char *>(&value), sizeof(value));
  if (!in) {
    throw std::runtime_error("checkpoint: unexpected EOF");
  }
  return value;
}

inline void write_tensor(std::ostream &out, Tensor<float> &tensor) {
  tensor.sync();
  const auto &shape = tensor.shape();
  write_u32(out, static_cast<std::uint32_t>(shape.size()));
  for (size_t dim : shape) {
    write_u32(out, static_cast<std::uint32_t>(dim));
  }
  out.write(reinterpret_cast<const char *>(tensor.data()),
            static_cast<std::streamsize>(tensor.size() * sizeof(float)));
}

inline void read_tensor(std::istream &in, Tensor<float> &tensor) {
  const auto rank = read_u32(in);
  std::vector<size_t> shape(rank);
  for (std::uint32_t i = 0; i < rank; ++i) {
    shape[i] = read_u32(in);
  }
  if (shape != tensor.shape()) {
    throw std::runtime_error("checkpoint: tensor shape mismatch");
  }
  tensor.sync();
  in.read(reinterpret_cast<char *>(tensor.data()),
          static_cast<std::streamsize>(tensor.size() * sizeof(float)));
  if (!in) {
    throw std::runtime_error("checkpoint: truncated tensor data");
  }
  tensor.sync();
}

inline auto read_checkpoint_header(std::istream &in, const std::string &path)
    -> CheckpointMeta {
  char magic[8]{};
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
    throw std::runtime_error("invalid checkpoint magic: " + path);
  }

  CheckpointMeta meta;
  meta.image_size = read_u32(in);
  meta.max_frames = read_u32(in);
  meta.in_channels = read_u32(in);
  meta.num_classes = read_u32(in);
  const auto n_names = read_u32(in);
  meta.class_names.resize(n_names);
  for (auto &name : meta.class_names) {
    const auto len = read_u32(in);
    name.resize(len);
    in.read(name.data(), static_cast<std::streamsize>(len));
    if (!in) {
      throw std::runtime_error("checkpoint: truncated class name");
    }
  }
  return meta;
}

} // namespace detail

inline auto peek_checkpoint_meta(const std::filesystem::path &path)
    -> CheckpointMeta {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot read checkpoint: " + path.string());
  }
  return detail::read_checkpoint_header(in, path.string());
}

inline void save_checkpoint(const std::filesystem::path &path,
                            network::VideoCNN<float> &model,
                            const CheckpointMeta &meta) {
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("cannot write checkpoint: " + path.string());
  }

  out.write(detail::kMagic, sizeof(detail::kMagic));
  detail::write_u32(out, meta.image_size);
  detail::write_u32(out, meta.max_frames);
  detail::write_u32(out, meta.in_channels);
  detail::write_u32(out, meta.num_classes);
  detail::write_u32(out, static_cast<std::uint32_t>(meta.class_names.size()));
  for (const auto &name : meta.class_names) {
    detail::write_u32(out, static_cast<std::uint32_t>(name.size()));
    out.write(name.data(), static_cast<std::streamsize>(name.size()));
  }

  auto tensors = model.state_tensors();
  detail::write_u32(out, static_cast<std::uint32_t>(tensors.size()));
  for (auto *tensor : tensors) {
    detail::write_tensor(out, *tensor);
  }
  if (!out) {
    throw std::runtime_error("failed writing checkpoint: " + path.string());
  }
}

inline auto load_checkpoint(const std::filesystem::path &path,
                            network::VideoCNN<float> &model)
    -> CheckpointMeta {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot read checkpoint: " + path.string());
  }

  auto meta = detail::read_checkpoint_header(in, path.string());
  auto tensors = model.state_tensors();
  const auto n_tensors = detail::read_u32(in);
  if (n_tensors != tensors.size()) {
    throw std::runtime_error("checkpoint: tensor count mismatch");
  }
  for (auto *tensor : tensors) {
    detail::read_tensor(in, *tensor);
  }
  return meta;
}

} // namespace har::train
