#pragma once

#include "har/loss/loss.hpp"

#include <cmath>
#include <stdexcept>

namespace har::loss {

// Softmax + NLL over logits.
// predictions: [batch, num_classes] (logits)
// targets: [batch] class indices stored as floating values
template <std::floating_point T = float>
class CrossEntropyLoss : public Loss<T> {
public:
  auto forward(const Tensor<T> &predictions, const Tensor<T> &targets)
      -> T override {
    if (predictions.dims() != 2) {
      throw std::invalid_argument("CrossEntropy: predictions must be 2D");
    }
    if (targets.dims() != 1 || targets.size() != predictions.shape()[0]) {
      throw std::invalid_argument(
          "CrossEntropy: targets must be [batch] class indices");
    }

    this->predictions_cache_ = predictions;
    this->targets_cache_ = targets;
    predictions.sync();
    targets.sync();

    const size_t batch = predictions.shape()[0];
    const size_t classes = predictions.shape()[1];
    probs_ = Tensor<T>({batch, classes}, Device::CPU);

    T total{0};
    for (size_t i = 0; i < batch; ++i) {
      T max_logit = predictions.at(i, 0);
      for (size_t j = 1; j < classes; ++j) {
        max_logit = std::max(max_logit, predictions.at(i, j));
      }

      T sum_exp{0};
      for (size_t j = 0; j < classes; ++j) {
        const T e = std::exp(predictions.at(i, j) - max_logit);
        probs_.at(i, j) = e;
        sum_exp += e;
      }

      for (size_t j = 0; j < classes; ++j) {
        probs_.at(i, j) /= sum_exp;
      }

      const auto cls = static_cast<size_t>(targets[i]);
      if (cls >= classes) {
        throw std::invalid_argument("CrossEntropy: target class out of range");
      }
      total += -std::log(probs_.at(i, cls) + T{1e-12});
    }

    return total / static_cast<T>(batch);
  }

  auto backward() -> Tensor<T> override {
    const size_t batch = probs_.shape()[0];
    Tensor<T> grad = probs_;

    for (size_t i = 0; i < batch; ++i) {
      const auto cls = static_cast<size_t>(this->targets_cache_[i]);
      grad.at(i, cls) -= T{1};
    }

    auto scaled = grad * (T{1} / static_cast<T>(batch));
    scaled.to_(this->predictions_cache_.device());
    return scaled;
  }

  auto name() const -> std::string override { return "CrossEntropyLoss"; }

private:
  Tensor<T> probs_;
};

} // namespace har::loss
