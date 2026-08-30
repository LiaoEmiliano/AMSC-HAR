#pragma once

#include "har/layers/layer.hpp"

#include <concepts>
#include <vector>

namespace har::optim {

template <std::floating_point T = float> class SGD {
public:
  explicit SGD(T learning_rate = T{0.01}, T momentum = T{0.9},
               T weight_decay = T{0})
      : lr_(learning_rate), momentum_(momentum), weight_decay_(weight_decay) {}

  void step(const std::vector<Parameter<T> *> &params) {
    ensure_velocity(params);
    for (size_t i = 0; i < params.size(); ++i) {
      auto *param = params[i];
      if (param == nullptr || !param->requires_grad) {
        continue;
      }

      if (weight_decay_ != T{0}) {
        param->grad += param->data * weight_decay_;
      }

      if (momentum_ != T{0}) {
        velocity_[i] *= momentum_;
        velocity_[i] += param->grad;
        param->data -= velocity_[i] * lr_;
      } else {
        param->data -= param->grad * lr_;
      }
    }
  }

  auto learning_rate() const -> T { return lr_; }
  void set_learning_rate(T lr) { lr_ = lr; }
  auto momentum() const -> T { return momentum_; }
  auto weight_decay() const -> T { return weight_decay_; }

private:
  void ensure_velocity(const std::vector<Parameter<T> *> &params) {
    bool reset = velocity_.size() != params.size();
    if (!reset) {
      for (size_t i = 0; i < params.size(); ++i) {
        if (params[i] == nullptr) {
          continue;
        }
        if (velocity_[i].shape() != params[i]->data.shape() ||
            velocity_[i].device() != params[i]->data.device()) {
          reset = true;
          break;
        }
      }
    }
    if (!reset) {
      return;
    }
    velocity_.clear();
    velocity_.reserve(params.size());
    for (auto *param : params) {
      if (param == nullptr) {
        velocity_.emplace_back();
        continue;
      }
      velocity_.emplace_back(param->data.shape(), T{0}, param->data.device());
    }
  }

  T lr_;
  T momentum_;
  T weight_decay_;
  std::vector<Tensor<T>> velocity_;
};

} // namespace har::optim
