#pragma once

#include "har/layers/layer.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace har::layers {

// BatchNorm over NCHW: statistics are computed per channel across N*H*W.
template <std::floating_point T = float> class BatchNorm2D : public Layer<T> {
public:
  explicit BatchNorm2D(size_t num_features, T momentum = T{0.1}, T eps = T{1e-5})
      : num_features_(num_features), momentum_(momentum), eps_(eps),
        gamma_({num_features}, T{1}), beta_({num_features}, T{0}),
        running_mean_({num_features}, T{0}),
        running_var_({num_features}, T{1}) {}

  auto forward(const Tensor<T> &input) -> Tensor<T> override {
    if (input.dims() != 4 || input.shape()[1] != num_features_) {
      throw std::invalid_argument("BatchNorm2D: expected NCHW with C=" +
                                  std::to_string(num_features_));
    }
    this->input_cache_ = input;
    const size_t N = input.shape()[0];
    const size_t C = num_features_;
    const size_t H = input.shape()[2];
    const size_t W = input.shape()[3];
    this->output_ = Tensor<T>(input.shape(), T{0}, input.device());
    xhat_ = Tensor<T>(input.shape(), T{0}, input.device());
    batch_mean_ = Tensor<T>({C}, T{0}, input.device());
    inv_std_ = Tensor<T>({C}, T{0}, input.device());

#ifdef HAR_HAS_CUDA
    if constexpr (std::is_same_v<T, float>) {
      if (input.device() == Device::CUDA && cuda_ops::active()) {
        cuda_ops::bn2d_forward(
            input.data(), this->output_.data(), xhat_.data(),
            gamma_.data.data(), beta_.data.data(), batch_mean_.data(),
            inv_std_.data(), running_mean_.data(), running_var_.data(),
            static_cast<int>(N), static_cast<int>(C), static_cast<int>(H),
            static_cast<int>(W), this->training_ ? 1 : 0,
            static_cast<float>(momentum_), static_cast<float>(eps_));
        return this->output_;
      }
    }
#endif

    input.sync();
    const T m = static_cast<T>(N * H * W);
    std::vector<T> mean(C, T{0});
    std::vector<T> var(C, T{0});

    if (this->training_) {
      for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
          for (size_t h = 0; h < H; ++h) {
            for (size_t w = 0; w < W; ++w) {
              mean[c] += input.at(n, c, h, w);
            }
          }
        }
      }
      for (size_t c = 0; c < C; ++c) {
        mean[c] /= m;
      }
      for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
          for (size_t h = 0; h < H; ++h) {
            for (size_t w = 0; w < W; ++w) {
              const T d = input.at(n, c, h, w) - mean[c];
              var[c] += d * d;
            }
          }
        }
      }
      for (size_t c = 0; c < C; ++c) {
        var[c] /= m;
        running_mean_[c] =
            (T{1} - momentum_) * running_mean_[c] + momentum_ * mean[c];
        running_var_[c] =
            (T{1} - momentum_) * running_var_[c] + momentum_ * var[c];
      }
    } else {
      running_mean_.sync();
      running_var_.sync();
      for (size_t c = 0; c < C; ++c) {
        mean[c] = running_mean_[c];
        var[c] = running_var_[c];
      }
    }

    for (size_t c = 0; c < C; ++c) {
      batch_mean_[c] = mean[c];
      inv_std_[c] = T{1} / std::sqrt(var[c] + eps_);
    }

    for (size_t n = 0; n < N; ++n) {
      for (size_t c = 0; c < C; ++c) {
        for (size_t h = 0; h < H; ++h) {
          for (size_t w = 0; w < W; ++w) {
            const T hat = (input.at(n, c, h, w) - mean[c]) * inv_std_[c];
            xhat_.at(n, c, h, w) = hat;
            this->output_.at(n, c, h, w) = gamma_.data[c] * hat + beta_.data[c];
          }
        }
      }
    }
    this->output_.sync();
    return this->output_;
  }

  auto backward(const Tensor<T> &grad_output) -> Tensor<T> override {
    const auto &input = this->input_cache_;
    const size_t N = input.shape()[0];
    const size_t C = num_features_;
    const size_t H = input.shape()[2];
    const size_t W = input.shape()[3];
    Tensor<T> grad_input(input.shape(), T{0}, input.device());

#ifdef HAR_HAS_CUDA
    if constexpr (std::is_same_v<T, float>) {
      if (input.device() == Device::CUDA && cuda_ops::active()) {
        cuda_ops::bn2d_backward(
            xhat_.data(), gamma_.data.data(), inv_std_.data(),
            grad_output.data(), grad_input.data(), gamma_.grad.data(),
            beta_.grad.data(), static_cast<int>(N), static_cast<int>(C),
            static_cast<int>(H), static_cast<int>(W));
        return grad_input;
      }
    }
#endif

    grad_output.sync();
    xhat_.sync();
    inv_std_.sync();
    const T m = static_cast<T>(N * H * W);
    std::vector<T> dgamma(C, T{0});
    std::vector<T> dbeta(C, T{0});
    std::vector<T> dxhat_sum(C, T{0});
    std::vector<T> dxhat_xhat_sum(C, T{0});

    for (size_t n = 0; n < N; ++n) {
      for (size_t c = 0; c < C; ++c) {
        for (size_t h = 0; h < H; ++h) {
          for (size_t w = 0; w < W; ++w) {
            const T dy = grad_output.at(n, c, h, w);
            const T hat = xhat_.at(n, c, h, w);
            dbeta[c] += dy;
            dgamma[c] += dy * hat;
            const T dxhat = dy * gamma_.data[c];
            dxhat_sum[c] += dxhat;
            dxhat_xhat_sum[c] += dxhat * hat;
          }
        }
      }
    }

    for (size_t c = 0; c < C; ++c) {
      gamma_.grad[c] += dgamma[c];
      beta_.grad[c] += dbeta[c];
    }

    const T inv_m = T{1} / m;
    for (size_t n = 0; n < N; ++n) {
      for (size_t c = 0; c < C; ++c) {
        for (size_t h = 0; h < H; ++h) {
          for (size_t w = 0; w < W; ++w) {
            const T dy = grad_output.at(n, c, h, w);
            const T hat = xhat_.at(n, c, h, w);
            const T dxhat = dy * gamma_.data[c];
            grad_input.at(n, c, h, w) =
                inv_std_[c] * inv_m *
                (m * dxhat - dxhat_sum[c] - hat * dxhat_xhat_sum[c]);
          }
        }
      }
    }
    grad_input.sync();
    return grad_input;
  }

  auto name() const -> std::string override { return "BatchNorm2D"; }

  auto parameters() -> std::vector<Parameter<T> *> override {
    return {&gamma_, &beta_};
  }

  auto running_mean() -> Tensor<T> & { return running_mean_; }
  auto running_var() -> Tensor<T> & { return running_var_; }
  auto running_mean() const -> const Tensor<T> & { return running_mean_; }
  auto running_var() const -> const Tensor<T> & { return running_var_; }

private:
  size_t num_features_;
  T momentum_;
  T eps_;
  Parameter<T> gamma_;
  Parameter<T> beta_;
  Tensor<T> running_mean_;
  Tensor<T> running_var_;
  Tensor<T> xhat_;
  Tensor<T> batch_mean_;
  Tensor<T> inv_std_;
};

} // namespace har::layers
