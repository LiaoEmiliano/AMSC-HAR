#pragma once

#include "har/layers/layer.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace har::layers {

// Softmax attention over time: score_t = h_t · w, then weighted sum.
// Zero-initialized w makes the first step identical to temporal mean.
// input:  [N*T, C]
// output: [N, C]
template <std::floating_point T = float> class TemporalAttentionPool {
public:
  explicit TemporalAttentionPool(size_t channels)
      : channels_(channels), query_({channels}, T{0}) {}

  auto forward(const Tensor<T> &feat, size_t n, size_t t) -> Tensor<T> {
    if (feat.dims() != 2 || feat.shape()[0] != n * t ||
        feat.shape()[1] != channels_) {
      throw std::invalid_argument("TemporalAttentionPool: expected [N*T, C]");
    }
    n_ = n;
    t_ = t;
    feat.sync();
    query_.data.sync();

    hfeat_.assign(feat.data(), feat.data() + feat.size());
    hw_.assign(query_.data.data(), query_.data.data() + channels_);
    alpha_.assign(n * t, T{0});

    Tensor<T> out({n, channels_}, T{0}, feat.device());
    std::vector<T> hout(n * channels_, T{0});

    for (size_t i = 0; i < n; ++i) {
      T max_s = T{0};
      std::vector<T> scores(t, T{0});
      for (size_t k = 0; k < t; ++k) {
        T s = T{0};
        const T *row = hfeat_.data() + (i * t + k) * channels_;
        for (size_t c = 0; c < channels_; ++c) {
          s += row[c] * hw_[c];
        }
        scores[k] = s;
        if (k == 0 || s > max_s) {
          max_s = s;
        }
      }
      T sum = T{0};
      for (size_t k = 0; k < t; ++k) {
        const T e = std::exp(scores[k] - max_s);
        alpha_[i * t + k] = e;
        sum += e;
      }
      const T inv = T{1} / sum;
      T *dst = hout.data() + i * channels_;
      for (size_t k = 0; k < t; ++k) {
        const T a = alpha_[i * t + k] * inv;
        alpha_[i * t + k] = a;
        const T *row = hfeat_.data() + (i * t + k) * channels_;
        for (size_t c = 0; c < channels_; ++c) {
          dst[c] += a * row[c];
        }
      }
    }

    out.sync();
    std::memcpy(out.data(), hout.data(), hout.size() * sizeof(T));
    out.sync();
    return out;
  }

  auto backward(const Tensor<T> &grad) -> Tensor<T> {
    if (grad.dims() != 2 || grad.shape()[0] != n_ ||
        grad.shape()[1] != channels_) {
      throw std::invalid_argument("TemporalAttentionPool: grad shape mismatch");
    }
    grad.sync();
    query_.grad.sync();

    std::vector<T> hgrad(grad.size());
    std::memcpy(hgrad.data(), grad.data(), grad.size() * sizeof(T));
    std::vector<T> dfeat(n_ * t_ * channels_, T{0});
    std::vector<T> dw(channels_, T{0});

    for (size_t i = 0; i < n_; ++i) {
      T weighted = T{0};
      std::vector<T> slot(t_, T{0});
      const T *gout = hgrad.data() + i * channels_;
      for (size_t k = 0; k < t_; ++k) {
        const T *row = hfeat_.data() + (i * t_ + k) * channels_;
        T s = T{0};
        for (size_t c = 0; c < channels_; ++c) {
          s += gout[c] * row[c];
        }
        slot[k] = s;
        weighted += alpha_[i * t_ + k] * s;
      }
      for (size_t k = 0; k < t_; ++k) {
        const T a = alpha_[i * t_ + k];
        const T dscore = a * (slot[k] - weighted);
        const T *row = hfeat_.data() + (i * t_ + k) * channels_;
        T *drow = dfeat.data() + (i * t_ + k) * channels_;
        for (size_t c = 0; c < channels_; ++c) {
          drow[c] = a * gout[c] + dscore * hw_[c];
          dw[c] += dscore * row[c];
        }
      }
    }

    Tensor<T> out({n_ * t_, channels_}, T{0}, grad.device());
    out.sync();
    std::memcpy(out.data(), dfeat.data(), dfeat.size() * sizeof(T));
    for (size_t c = 0; c < channels_; ++c) {
      query_.grad[c] += dw[c];
    }
    out.sync();
    query_.grad.sync();
    return out;
  }

  auto parameters() -> std::vector<Parameter<T> *> { return {&query_}; }

  void zero_grad() { query_.zero_grad(); }

private:
  size_t channels_{0};
  Parameter<T> query_;
  std::vector<T> hfeat_;
  std::vector<T> hw_;
  std::vector<T> alpha_;
  size_t n_{0};
  size_t t_{0};
};

} // namespace har::layers
