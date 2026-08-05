#pragma once

#include "har/tensor.hpp"

#include <concepts>
#include <string>

namespace har::loss {

template <std::floating_point T = float> class Loss {
public:
  virtual ~Loss() = default;

  virtual auto forward(const Tensor<T> &predictions, const Tensor<T> &targets)
      -> T = 0;

  virtual auto backward() -> Tensor<T> = 0;

  virtual auto name() const -> std::string = 0;

protected:
  Tensor<T> predictions_cache_;
  Tensor<T> targets_cache_;
};

} // namespace har::loss
