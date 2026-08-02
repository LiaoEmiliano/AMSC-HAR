#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace har {
template <std::floating_point T = float> class Tensor {
private:
  std::vector<size_t> shape_;
  std::vector<size_t> strides_;
  std::vector<T> data_;

  void compute_strides() {
    strides_.resize(shape_.size());
    if (shape_.empty()) {
      return;
    }

    strides_.back() = 1;
    for (size_t i = shape_.size() - 1; i > 0; --i) {
      strides_[i - 1] = strides_[i] * shape_[i];
    }
  }

  auto compute_size() const -> size_t {
    if (shape_.empty()) {
      return 0;
    }

    return std::accumulate(shape_.begin(), shape_.end(), size_t{1},
                           std::multiplies{});
  }

  template <typename... Indices>
  auto compute_flat_index(Indices... indices) const -> size_t {
    std::array<size_t, sizeof...(Indices)> idx_arr{
        static_cast<size_t>(indices)...};
    if (idx_arr.size() != shape_.size()) {
      throw std::invalid_argument("Index dimension mismatch");
    }

    size_t flat_idx = 0;
    size_t stride = 1;
    for (size_t i = shape_.size(); i-- > 0;) {
      flat_idx += idx_arr[i] * stride;
      stride *= shape_[i];
    }

    return flat_idx;
  }

public:
  Tensor() = default;

  explicit Tensor(std::vector<size_t> shape) : shape_(std::move(shape)) {
    compute_strides();
    data_.resize(compute_size());
  }

  Tensor(std::vector<size_t> shape, T init_value) : shape_(std::move(shape)) {
    compute_strides();
    data_.resize(compute_size(), init_value);
  }

  Tensor(std::vector<size_t> shape, std::vector<T> data)
      : shape_(std::move(shape)), data_(std::move(data)) {
    compute_strides();
    if (data_.size() != compute_size()) {
      throw std::invalid_argument("Data size mismatch with shape");
    }
  }

  Tensor(const Tensor &) = default;
  Tensor(Tensor &&) = default;
  auto operator=(const Tensor &) -> Tensor & = default;
  auto operator=(Tensor &&) -> Tensor & = default;

  auto shape() const -> const std::vector<size_t> & { return shape_; }

  auto size() const -> size_t { return data_.size(); }

  auto data() -> T * { return data_.data(); }

  auto data() const -> const T * { return data_.data(); }

  auto begin() { return data_.begin(); }

  auto end() { return data_.end(); }

  auto begin() const { return data_.begin(); }

  auto end() const { return data_.end(); }

  auto operator[](size_t idx) -> T & { return data_[idx]; }

  auto operator[](size_t idx) const -> const T & { return data_[idx]; }

  auto dims() const -> size_t { return shape_.size(); }

  template <typename... Indices>
    requires(sizeof...(Indices) > 0) &&
            (std::convertible_to<Indices, size_t> && ...)
  auto at(Indices... indices) -> T & {
    return data_[compute_flat_index(indices...)];
  }

  template <typename... Indices>
    requires(sizeof...(Indices) > 0) &&
            (std::convertible_to<Indices, size_t> && ...)
  auto at(Indices... indices) const -> const T & {
    return data_[compute_flat_index(indices...)];
  }

  void reshape(std::vector<size_t> new_shape) {
    auto new_size = std::accumulate(new_shape.begin(), new_shape.end(),
                                    size_t{1}, std::multiplies{});

    if (new_size != data_.size()) {
      throw std::invalid_argument("Reshape size mismatch");
    }

    shape_ = std::move(new_shape);
    compute_strides();
  }

  auto reshaped(std::vector<size_t> new_shape) const -> Tensor {
    Tensor result = *this;
    result.reshape(std::move(new_shape));
    return result;
  }

  auto flatten() const -> Tensor { return reshaped({data_.size()}); }

  void fill(T value) { std::ranges::fill(data_, value); }
  void zero() { fill(T{0}); }

  // Element-wise operations (in-place)
  auto operator+=(const Tensor &other) -> Tensor & {
    if (shape_ != other.shape_) {
      throw std::invalid_argument("Shape mismatch for addition");
    }
    std::ranges::transform(data_, other.data_, data_.begin(), std::plus{});
    return *this;
  }

  auto operator-=(const Tensor &other) -> Tensor & {
    if (shape_ != other.shape_) {
      throw std::invalid_argument("Shape mismatch for subtraction");
    }
    std::ranges::transform(data_, other.data_, data_.begin(), std::minus{});
    return *this;
  }

  auto operator*=(const Tensor &other) -> Tensor & {
    if (shape_ != other.shape_) {
      throw std::invalid_argument("Shape mismatch for multiplication");
    }
    std::ranges::transform(data_, other.data_, data_.begin(),
                           std::multiplies{});
    return *this;
  }

  auto operator/=(const Tensor &other) -> Tensor & {
    if (shape_ != other.shape_) {
      throw std::invalid_argument("Shape mismatch for division");
    }
    std::ranges::transform(data_, other.data_, data_.begin(), std::divides{});
    return *this;
  }
  auto operator+=(T scalar) -> Tensor & {
    std::ranges::for_each(data_, [scalar](T &x) { x += scalar; });
    return *this;
  }

  auto operator-=(T scalar) -> Tensor & {
    std::ranges::for_each(data_, [scalar](T &x) { x -= scalar; });
    return *this;
  }

  auto operator*=(T scalar) -> Tensor & {
    std::ranges::for_each(data_, [scalar](T &x) { x *= scalar; });
    return *this;
  }

  auto operator/=(T scalar) -> Tensor & {
    std::ranges::for_each(data_, [scalar](T &x) { x /= scalar; });
    return *this;
  }

  // Element-wise operations (new tensor)
  auto operator+(const Tensor &other) const -> Tensor {
    Tensor result = *this;
    result += other;
    return result;
  }
  auto operator-(const Tensor &other) const -> Tensor {
    Tensor result = *this;
    result -= other;
    return result;
  }

  auto operator*(const Tensor &other) const -> Tensor {
    Tensor result = *this;
    result *= other;
    return result;
  }

  auto operator/(const Tensor &other) const -> Tensor {
    Tensor result = *this;
    result /= other;
    return result;
  }

  // Scalar operations (new tensor)
  auto operator+(T scalar) const -> Tensor {
    Tensor result = *this;
    result += scalar;
    return result;
  }

  auto operator-(T scalar) const -> Tensor {
    Tensor result = *this;
    result -= scalar;
    return result;
  }

  auto operator*(T scalar) const -> Tensor {
    Tensor result = *this;
    result *= scalar;
    return result;
  }

  auto operator/(T scalar) const -> Tensor {
    Tensor result = *this;
    result /= scalar;
    return result;
  }

  auto operator-() const -> Tensor {
    Tensor result(shape_);
    std::ranges::transform(data_, result.begin(), std::negate{});
    return result;
  }

  auto sum() const -> T {
    return std::accumulate(data_.begin(), data_.end(), T{0});
  }

  auto mean() const -> T { return sum() / static_cast<T>(data_.size()); }

  auto max() const -> T { return *std::ranges::max_element(data_); };

  auto min() const -> T { return *std::ranges::min_element(data_); };

  auto operator==(const Tensor &other) const -> bool {
    return shape_ == other.shape_ && data_ == other.data_;
  }

  static auto zeros(std::vector<size_t> shape) -> Tensor {
    return Tensor(std::move(shape), T{0});
  }

  static auto ones(std::vector<size_t> shape) -> Tensor {
    return Tensor(std::move(shape), T{1});
  }

  static auto fill(std::vector<size_t> shape, T value) -> Tensor {
    return Tensor(std::move(shape), value);
  }
};

// Free function scalar operations (scalar on left)
template <std::floating_point T>
auto operator+(T scalar, const Tensor<T> &tensor) -> Tensor<T> {
  return tensor + scalar;
}

template <std::floating_point T>
auto operator*(T scalar, const Tensor<T> &tensor) -> Tensor<T> {
  return tensor * scalar;
}

template <std::floating_point T>
auto operator-(T scalar, const Tensor<T> &tensor) -> Tensor<T> {
  Tensor<T> result(tensor.shape());
  std::ranges::transform(tensor, result.begin(),
                         [scalar](T x) { return scalar - x; });

  return result;
}

} // namespace har
