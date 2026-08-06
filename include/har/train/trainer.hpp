#pragma once

#include "har/data/ucf11.hpp"
#include "har/layers/activation.hpp"
#include "har/layers/conv2d.hpp"
#include "har/layers/flatten.hpp"
#include "har/layers/linear.hpp"
#include "har/layers/pool.hpp"
#include "har/loss/cross_entropy.hpp"
#include "har/network/sequential.hpp"
#include "har/optim/sgd.hpp"
#include "har/tensor.hpp"

#include <algorithm>
#include <chrono>
#include <format>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace har::train {

struct TrainConfig {
  size_t epochs{10};
  size_t batch_size{8};
  float learning_rate{0.01f};
  unsigned seed{42};
  bool verbose{true};
};

inline auto build_ucf11_cnn(size_t num_classes, size_t image_size,
                            size_t in_channels = 3)
    -> network::Sequential<float> {
  // Input: [N, C, S, S]
  // Conv3x3(pad1) -> S, Pool2 -> S/2
  // Conv3x3(pad1) -> S/2, Pool2 -> S/4
  if (image_size < 8 || (image_size % 4) != 0) {
    throw std::invalid_argument(
        "image_size must be >= 8 and divisible by 4 (got " +
        std::to_string(image_size) + ")");
  }

  const size_t feat = image_size / 4;
  const size_t flat = 16 * feat * feat;

  network::Sequential<float> model;
  model.add(std::make_unique<layers::Conv2D<float>>(in_channels, 8, 3, 1, 1));
  model.add(std::make_unique<layers::ReLU<float>>());
  model.add(std::make_unique<layers::MaxPool2D<float>>(2));
  model.add(std::make_unique<layers::Conv2D<float>>(8, 16, 3, 1, 1));
  model.add(std::make_unique<layers::ReLU<float>>());
  model.add(std::make_unique<layers::MaxPool2D<float>>(2));
  model.add(std::make_unique<layers::Flatten<float>>());
  model.add(std::make_unique<layers::Linear<float>>(flat, 64));
  model.add(std::make_unique<layers::ReLU<float>>());
  model.add(std::make_unique<layers::Linear<float>>(64, num_classes));
  return model;
}

inline auto stack_batch(const std::vector<Tensor<float>> &items)
    -> Tensor<float> {
  if (items.empty()) {
    throw std::invalid_argument("stack_batch: empty");
  }
  const size_t N = items.size();
  const size_t C = items[0].shape()[1];
  const size_t H = items[0].shape()[2];
  const size_t W = items[0].shape()[3];
  Tensor<float> batch({N, C, H, W});
  const size_t plane = C * H * W;
  for (size_t n = 0; n < N; ++n) {
    if (items[n].shape() != items[0].shape()) {
      throw std::invalid_argument("stack_batch: shape mismatch");
    }
    for (size_t i = 0; i < plane; ++i) {
      batch[n * plane + i] = items[n][i];
    }
  }
  return batch;
}

inline auto argmax_row(const Tensor<float> &logits, size_t row) -> size_t {
  const size_t classes = logits.shape()[1];
  size_t best = 0;
  float best_v = logits.at(row, 0);
  for (size_t c = 1; c < classes; ++c) {
    const float v = logits.at(row, c);
    if (v > best_v) {
      best_v = v;
      best = c;
    }
  }
  return best;
}

struct EpochStats {
  float loss{0};
  float accuracy{0};
  size_t samples{0};
};

inline auto evaluate(network::Sequential<float> &model,
                     const data::UCF11Dataset &dataset,
                     const std::vector<size_t> &indices, size_t batch_size)
    -> EpochStats {
  model.eval();
  loss::CrossEntropyLoss<float> criterion;

  float total_loss = 0;
  size_t correct = 0;
  size_t seen = 0;

  for (size_t start = 0; start < indices.size(); start += batch_size) {
    const size_t end = std::min(start + batch_size, indices.size());
    std::vector<Tensor<float>> xs;
    Tensor<float> ys({end - start});
    xs.reserve(end - start);

    for (size_t i = start; i < end; ++i) {
      auto [x, y] = dataset.load(indices[i]);
      xs.push_back(std::move(x));
      ys[i - start] = y;
    }

    auto batch = stack_batch(xs);
    auto logits = model.forward(batch);
    total_loss +=
        criterion.forward(logits, ys) * static_cast<float>(end - start);

    for (size_t i = 0; i < end - start; ++i) {
      if (argmax_row(logits, i) == static_cast<size_t>(ys[i])) {
        ++correct;
      }
    }
    seen += end - start;
  }

  EpochStats stats;
  stats.samples = seen;
  stats.loss = seen ? total_loss / static_cast<float>(seen) : 0;
  stats.accuracy =
      seen ? static_cast<float>(correct) / static_cast<float>(seen) : 0;
  return stats;
}

inline auto train_one_epoch(network::Sequential<float> &model,
                            const data::UCF11Dataset &dataset,
                            std::vector<size_t> indices, size_t batch_size,
                            optim::SGD<float> &optimizer, std::mt19937 &rng)
    -> EpochStats {
  model.train();
  loss::CrossEntropyLoss<float> criterion;
  std::shuffle(indices.begin(), indices.end(), rng);

  float total_loss = 0;
  size_t correct = 0;
  size_t seen = 0;

  for (size_t start = 0; start < indices.size(); start += batch_size) {
    const size_t end = std::min(start + batch_size, indices.size());
    std::vector<Tensor<float>> xs;
    Tensor<float> ys({end - start});
    xs.reserve(end - start);

    for (size_t i = start; i < end; ++i) {
      auto [x, y] = dataset.load(indices[i]);
      xs.push_back(std::move(x));
      ys[i - start] = y;
    }

    auto batch = stack_batch(xs);
    model.zero_grad();
    auto logits = model.forward(batch);
    const float loss = criterion.forward(logits, ys);
    auto grad = criterion.backward();
    model.backward(grad);
    optimizer.step(model.parameters());

    total_loss += loss * static_cast<float>(end - start);
    for (size_t i = 0; i < end - start; ++i) {
      if (argmax_row(logits, i) == static_cast<size_t>(ys[i])) {
        ++correct;
      }
    }
    seen += end - start;
  }

  EpochStats stats;
  stats.samples = seen;
  stats.loss = seen ? total_loss / static_cast<float>(seen) : 0;
  stats.accuracy =
      seen ? static_cast<float>(correct) / static_cast<float>(seen) : 0;
  return stats;
}

inline auto train_ucf11(data::UCF11Dataset &dataset, const TrainConfig &cfg)
    -> int {
  if (dataset.train_size() == 0) {
    throw std::runtime_error("UCF11 train set is empty");
  }

  const size_t image_size =
      static_cast<size_t>(dataset.options().frames.width);
  auto model = build_ucf11_cnn(dataset.num_classes(), image_size,
                               dataset.options().frames.rgb ? 3 : 1);
  optim::SGD<float> optimizer(cfg.learning_rate);
  std::mt19937 rng{cfg.seed};

  std::cout << std::format(
      "Training UCF11 CNN: epochs={}, batch={}, lr={}, size={}x{}, classes={}\n",
      cfg.epochs, cfg.batch_size, cfg.learning_rate, image_size, image_size,
      dataset.num_classes());

  for (size_t epoch = 1; epoch <= cfg.epochs; ++epoch) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto train_stats = train_one_epoch(
        model, dataset, dataset.train_indices(), cfg.batch_size, optimizer, rng);
    const auto val_stats =
        dataset.val_size() > 0
            ? evaluate(model, dataset, dataset.val_indices(), cfg.batch_size)
            : EpochStats{};
    const auto t1 = std::chrono::steady_clock::now();
    const double sec =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
            .count() /
        1000.0;

    if (cfg.verbose) {
      std::cout << std::format(
          "epoch {:>3}/{:<3}  train loss={:.4f} acc={:.1f}%  "
          "val loss={:.4f} acc={:.1f}%  ({:.1f}s)\n",
          epoch, cfg.epochs, train_stats.loss, train_stats.accuracy * 100.0f,
          val_stats.loss, val_stats.accuracy * 100.0f, sec);
    }
  }

  return 0;
}

} // namespace har::train
