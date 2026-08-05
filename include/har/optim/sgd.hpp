#pragma once

#include "har/layers/layer.hpp"

#include <concepts>
#include <vector>

namespace har::optim {

template <std::floating_point T = float> class SGD {
public:
  explicit SGD(T learning_rate = T{0.01}) : lr_(learning_rate) {}

  void step(const std::vector<Parameter<T> *> &params) {
    for (auto *param : params) {
      if (param == nullptr || !param->requires_grad) {
        continue;
      }
      param->data -= param->grad * lr_;
    }
  }

  auto learning_rate() const -> T { return lr_; }
  void set_learning_rate(T lr) { lr_ = lr; }

private:
  T lr_;
};

} // namespace har::optim
