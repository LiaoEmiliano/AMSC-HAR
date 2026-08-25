#pragma once

#include "har/data/ucf11.hpp"
#include "har/data/ucf101.hpp"
#include "har/loss/cross_entropy.hpp"
#include "har/network/video_cnn.hpp"
#include "har/optim/sgd.hpp"
#include "har/tensor.hpp"
#include "har/train/checkpoint.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <future>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace har::train {

struct TrainConfig {
  size_t epochs{10};
  size_t batch_size{32};
  float learning_rate{0.003f};
  float momentum{0.9f};
  float weight_decay{1e-4f};
  float dropout{0.3f};
  size_t lr_step{4};
  float lr_gamma{0.5f};
  size_t num_workers{8};
  unsigned seed{42};
  bool verbose{true};
  float target_test_acc{0};
  std::string weights_path{"models/ucf101_videocnn.harw"};
  std::string resume_path{};
};

inline auto build_video_cnn(size_t num_classes, size_t image_size,
                            size_t in_channels = 3, float dropout = 0.3f)
    -> network::VideoCNN<float> {
  if (image_size < 8 || (image_size % 4) != 0) {
    throw std::invalid_argument(
        "image_size must be >= 8 and divisible by 4 (got " +
        std::to_string(image_size) + ")");
  }
  return network::VideoCNN<float>(in_channels, num_classes, dropout);
}

// Cap N*T*H*W so 112x112 T=16 can use batch 16 on a 12 GB card.
// Larger requests still get clipped to avoid managed-memory oversubscribe.
inline auto activation_batch_cap(size_t requested, size_t frames, size_t height,
                                 size_t width) -> size_t {
  constexpr size_t kBudget = 16ull * 16ull * 112ull * 112ull;
  const size_t per = std::max<size_t>(1, frames * height * width);
  return std::max<size_t>(1, std::min(requested, kBudget / per));
}

inline auto stack_batch(const std::vector<Tensor<float>> &items)
    -> Tensor<float> {
  if (items.empty()) {
    throw std::invalid_argument("stack_batch: empty");
  }
  if (items[0].dims() != 4) {
    throw std::invalid_argument("stack_batch: expected clips [T,C,H,W]");
  }
  const size_t N = items.size();
  const size_t T = items[0].shape()[0];
  const size_t C = items[0].shape()[1];
  const size_t H = items[0].shape()[2];
  const size_t W = items[0].shape()[3];
  Tensor<float> batch({N, T, C, H, W});
  const size_t plane = T * C * H * W;
  for (size_t n = 0; n < N; ++n) {
    if (items[n].shape() != items[0].shape()) {
      throw std::invalid_argument("stack_batch: shape mismatch");
    }
#ifdef HAR_HAS_CUDA
    cuda_check(cudaMemcpy(batch.data() + n * plane, items[n].data(),
                          plane * sizeof(float), cudaMemcpyDefault),
               "stack_batch");
#else
    std::memcpy(batch.data() + n * plane, items[n].data(),
                plane * sizeof(float));
#endif
  }
  return batch;
}

inline auto count_correct(const Tensor<float> &logits, const Tensor<float> &ys)
    -> size_t {
  logits.sync();
  ys.sync();
  const size_t n = ys.size();
  const size_t classes = logits.shape()[1];
  size_t correct = 0;
  for (size_t i = 0; i < n; ++i) {
    size_t best = 0;
    float best_v = logits.at(i, 0);
    for (size_t c = 1; c < classes; ++c) {
      const float v = logits.at(i, c);
      if (v > best_v) {
        best_v = v;
        best = c;
      }
    }
    if (best == static_cast<size_t>(ys[i])) {
      ++correct;
    }
  }
  return correct;
}

template <typename Dataset>
inline auto load_host_batch(const Dataset &dataset,
                            const std::vector<size_t> &indices, size_t start,
                            size_t end, bool augment = false)
    -> std::pair<std::vector<Tensor<float>>, Tensor<float>> {
  const size_t n = end - start;
  std::vector<Tensor<float>> xs(n);
  Tensor<float> ys({n}, Device::CPU);
  for (size_t i = 0; i < n; ++i) {
    auto sample = dataset.load(indices[start + i], augment);
    xs[i] = std::move(sample.first);
    ys[i] = sample.second;
  }
  return {std::move(xs), std::move(ys)};
}

struct EpochStats {
  float loss{0};
  float accuracy{0};
  size_t samples{0};
};

template <typename Dataset>
inline auto evaluate(network::VideoCNN<float> &model, const Dataset &dataset,
                     const std::vector<size_t> &indices, size_t batch_size,
                     size_t workers) -> EpochStats {
  model.eval();
  loss::CrossEntropyLoss<float> criterion;

  float total_loss = 0;
  size_t correct = 0;
  size_t seen = 0;
  (void)workers;

  for (size_t start = 0; start < indices.size(); start += batch_size) {
    const size_t end = std::min(start + batch_size, indices.size());
    auto [xs, ys] = load_host_batch(dataset, indices, start, end);
    auto batch = stack_batch(xs);
    auto logits = model.forward(std::move(batch));
    total_loss +=
        criterion.forward(logits, ys) * static_cast<float>(end - start);
    correct += count_correct(logits, ys);
    seen += end - start;
  }

  EpochStats stats;
  stats.samples = seen;
  stats.loss = seen ? total_loss / static_cast<float>(seen) : 0;
  stats.accuracy =
      seen ? static_cast<float>(correct) / static_cast<float>(seen) : 0;
  return stats;
}

template <typename Dataset>
inline auto train_one_epoch(network::VideoCNN<float> &model,
                            const Dataset &dataset, std::vector<size_t> indices,
                            size_t batch_size, size_t workers,
                            optim::SGD<float> &optimizer, std::mt19937 &rng)
    -> EpochStats {
  model.train();
  loss::CrossEntropyLoss<float> criterion;
  std::shuffle(indices.begin(), indices.end(), rng);

  float total_loss = 0;
  size_t correct = 0;
  size_t seen = 0;
  using HostBatch = std::pair<std::vector<Tensor<float>>, Tensor<float>>;
  std::future<HostBatch> pending;

  for (size_t start = 0; start < indices.size(); start += batch_size) {
    const size_t end = std::min(start + batch_size, indices.size());
    HostBatch host;
    if (pending.valid()) {
      host = pending.get();
    } else {
      host = load_host_batch(dataset, indices, start, end, true);
    }
    if (end < indices.size() && workers > 0) {
      const size_t nstart = end;
      const size_t nend = std::min(end + batch_size, indices.size());
      pending = std::async(std::launch::async, [&dataset, &indices, nstart,
                                                nend] {
        return load_host_batch(dataset, indices, nstart, nend, true);
      });
    }

    auto batch = stack_batch(host.first);
    auto &ys = host.second;
    if (start == 0) {
      batch.sync();
      double sum = 0;
      double sq = 0;
      for (size_t i = 0; i < batch.size(); ++i) {
        const double v = batch[i];
        sum += v;
        sq += v * v;
      }
      const double mean = sum / static_cast<double>(batch.size());
      const double var =
          sq / static_cast<double>(batch.size()) - mean * mean;
      std::cout << std::format("  first batch mean={:.4f} std={:.4f}\n", mean,
                               std::sqrt(std::max(0.0, var)));
      if (std::abs(mean) < 1e-5 && var < 1e-6) {
        std::cout << "WARNING: first batch is ~0; video clips may be empty\n";
      }
    }
    model.zero_grad();
    auto logits = model.forward(std::move(batch));
    const float loss = criterion.forward(logits, ys);
    auto grad = criterion.backward();
    model.backward(grad);
    optimizer.step(model.parameters());

    total_loss += loss * static_cast<float>(end - start);
    correct += count_correct(logits, ys);
    seen += end - start;
    if (end == indices.size() || end % 256 == 0) {
      std::cout << std::format("  processed {}/{}\n", end, indices.size());
    }
  }

  EpochStats stats;
  stats.samples = seen;
  stats.loss = seen ? total_loss / static_cast<float>(seen) : 0;
  stats.accuracy =
      seen ? static_cast<float>(correct) / static_cast<float>(seen) : 0;
  return stats;
}

template <typename Dataset>
inline auto train_video_cnn(Dataset &dataset, const TrainConfig &cfg,
                            std::string_view dataset_name = "HAR") -> int {
  if (dataset.train_size() == 0) {
    throw std::runtime_error(std::string(dataset_name) + " train set is empty");
  }

  const size_t image_size =
      static_cast<size_t>(dataset.options().frames.width);
  const size_t clip_frames =
      static_cast<size_t>(dataset.options().frames.max_frames);
  const size_t batch_size =
      activation_batch_cap(cfg.batch_size, clip_frames, image_size, image_size);
  auto model = build_video_cnn(dataset.num_classes(), image_size,
                               dataset.options().frames.rgb ? 3 : 1,
                               cfg.dropout);
  optim::SGD<float> optimizer(cfg.learning_rate, cfg.momentum, cfg.weight_decay);
  std::mt19937 rng{cfg.seed};

  if (batch_size != cfg.batch_size) {
    std::cout << std::format(
        "  batch {} -> {} to fit {}x{} T={} in GPU memory\n", cfg.batch_size,
        batch_size, image_size, image_size, clip_frames);
  }

  std::cout << std::format(
      "Training {} VideoCNN: device={}  epochs={}  batch={}  workers={}  lr={}  "
      "lr_step={}  lr_gamma={}  momentum={}  wd={}  dropout={}  size={}x{}  "
      "frames={}  classes={}  target_test={}\n",
      dataset_name, device_name(), cfg.epochs, batch_size, cfg.num_workers,
      cfg.learning_rate, cfg.lr_step, cfg.lr_gamma, cfg.momentum,
      cfg.weight_decay, cfg.dropout, image_size, image_size,
      dataset.options().frames.max_frames, dataset.num_classes(),
      cfg.target_test_acc > 0
          ? std::format("{:.1f}%", cfg.target_test_acc * 100.0f)
          : std::string("off"));
  std::cout << "Architecture: per-frame Conv32-BN-ReLU-Pool, Conv64-BN-ReLU-Pool, "
               "Conv128-BN-ReLU, Conv256-BN-ReLU-GAP, TemporalConv1D(k=3, residual), "
               "temporal attention pool, Dropout, Linear(256, classes)\n";
  std::cout << "Augment: train-time hflip / time-reverse / 16->8 temporal crop; "
               "eval uses uniform 8 frames\n";

  CheckpointMeta ckpt_meta;
  ckpt_meta.image_size = static_cast<std::uint32_t>(image_size);
  ckpt_meta.max_frames =
      static_cast<std::uint32_t>(dataset.options().frames.max_frames);
  ckpt_meta.in_channels = dataset.options().frames.rgb ? 3u : 1u;
  ckpt_meta.num_classes = static_cast<std::uint32_t>(dataset.num_classes());
  ckpt_meta.class_names = dataset.class_names();

  float best_val = -1.0f;
  size_t best_epoch = 0;

  if (!cfg.resume_path.empty()) {
    if (!std::filesystem::exists(cfg.resume_path)) {
      throw std::runtime_error("resume checkpoint not found: " +
                               cfg.resume_path);
    }
    const auto loaded = load_checkpoint(cfg.resume_path, model);
    if (loaded.num_classes != ckpt_meta.num_classes) {
      throw std::runtime_error(std::format(
          "resume class count {} does not match dataset {}",
          loaded.num_classes, ckpt_meta.num_classes));
    }
    std::cout << std::format("resumed weights from {}\n", cfg.resume_path);
    if (cfg.target_test_acc > 0 && dataset.test_size() > 0) {
      const auto test_stats =
          evaluate(model, dataset, dataset.test_indices(), batch_size,
                   cfg.num_workers);
      best_val = test_stats.accuracy;
      if (cfg.verbose) {
        std::cout << std::format("  resume test loss={:.4f} acc={:.1f}%\n",
                                 test_stats.loss, test_stats.accuracy * 100.0f);
      }
    } else if (dataset.val_size() > 0) {
      const auto val_stats =
          evaluate(model, dataset, dataset.val_indices(), batch_size,
                   cfg.num_workers);
      best_val = val_stats.accuracy;
      if (cfg.verbose) {
        std::cout << std::format("  resume val loss={:.4f} acc={:.1f}%\n",
                                 val_stats.loss, val_stats.accuracy * 100.0f);
      }
    }
  }

  for (size_t epoch = 1; epoch <= cfg.epochs; ++epoch) {
    if (cfg.lr_step > 0 && epoch > 1 && (epoch - 1) % cfg.lr_step == 0) {
      const float next_lr = optimizer.learning_rate() * cfg.lr_gamma;
      optimizer.set_learning_rate(next_lr);
      if (cfg.verbose) {
        std::cout << std::format("  lr -> {:.6f}\n", next_lr);
      }
    }

    const auto t0 = std::chrono::steady_clock::now();
    const auto train_stats = train_one_epoch(
        model, dataset, dataset.train_indices(), batch_size,
        cfg.num_workers, optimizer, rng);
    const auto val_stats =
        dataset.val_size() > 0
            ? evaluate(model, dataset, dataset.val_indices(), batch_size,
                       cfg.num_workers)
            : EpochStats{};
    const bool track_test =
        cfg.target_test_acc > 0 && dataset.test_size() > 0;
    const auto test_stats =
        track_test ? evaluate(model, dataset, dataset.test_indices(),
                              batch_size, cfg.num_workers)
                   : EpochStats{};
    const auto t1 = std::chrono::steady_clock::now();
    const double sec =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
            .count() /
        1000.0;

    if (cfg.verbose) {
      if (track_test) {
        std::cout << std::format(
            "epoch {:>3}/{:<3}  train loss={:.4f} acc={:.1f}%  "
            "val loss={:.4f} acc={:.1f}%  test loss={:.4f} acc={:.1f}%  "
            "({:.1f}s)\n",
            epoch, cfg.epochs, train_stats.loss, train_stats.accuracy * 100.0f,
            val_stats.loss, val_stats.accuracy * 100.0f, test_stats.loss,
            test_stats.accuracy * 100.0f, sec);
      } else {
        std::cout << std::format(
            "epoch {:>3}/{:<3}  train loss={:.4f} acc={:.1f}%  "
            "val loss={:.4f} acc={:.1f}%  ({:.1f}s)\n",
            epoch, cfg.epochs, train_stats.loss, train_stats.accuracy * 100.0f,
            val_stats.loss, val_stats.accuracy * 100.0f, sec);
      }
    }

    const float score =
        track_test ? test_stats.accuracy
                   : (dataset.val_size() > 0 ? val_stats.accuracy
                                             : train_stats.accuracy);
    if (score >= best_val) {
      best_val = score;
      best_epoch = epoch;
      if (!cfg.weights_path.empty()) {
        save_checkpoint(cfg.weights_path, model, ckpt_meta);
        if (cfg.verbose) {
          std::cout << std::format("  saved weights -> {}  (best {:.1f}%)\n",
                                   cfg.weights_path, best_val * 100.0f);
        }
      }
    }

    if (track_test && test_stats.accuracy >= cfg.target_test_acc) {
      std::cout << std::format(
          "reached target test accuracy {:.2f}% at epoch {} (stop)\n",
          test_stats.accuracy * 100.0f, epoch);
      break;
    }
  }

  if (!cfg.weights_path.empty() &&
      (best_epoch > 0 || !cfg.resume_path.empty())) {
    load_checkpoint(cfg.weights_path, model);
    if (cfg.verbose) {
      if (best_epoch > 0) {
        std::cout << std::format("loaded best weights from epoch {} for test\n",
                                 best_epoch);
      } else {
        std::cout << "loaded resumed best weights for test\n";
      }
    }
  }

  const auto test_stats =
      dataset.test_size() > 0
          ? evaluate(model, dataset, dataset.test_indices(), batch_size,
                     cfg.num_workers)
          : evaluate(model, dataset, dataset.val_indices(), batch_size,
                     cfg.num_workers);
  std::cout << std::format(
      "test accuracy: {:.2f}%  ({:d} samples, loss={:.4f})\n",
      test_stats.accuracy * 100.0f, test_stats.samples, test_stats.loss);

  if (!cfg.weights_path.empty()) {
    const char *metric =
        cfg.target_test_acc > 0 && dataset.test_size() > 0
            ? "test"
            : (dataset.val_size() > 0 ? "val" : "train");
    std::cout << std::format(
        "weights kept at {}  (best epoch {}, {:.1f}% {})\n",
        cfg.weights_path, best_epoch, best_val * 100.0f, metric);
  }

  return 0;
}

} // namespace har::train
