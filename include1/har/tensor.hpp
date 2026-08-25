#pragma once

#include "har/cuda/device.hpp"
#include "har/cuda/ops.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstring>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace har {

template <std::floating_point T = float> class Tensor {
private:
  std::vector<size_t> shape_;
  std::vector<size_t> strides_;
  T *data_{nullptr};
  size_t size_{0};
  Device device_{Device::CPU};

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

  void deallocate() {
    if (data_ == nullptr) {
      size_ = 0;
      return;
    }
#ifdef HAR_HAS_CUDA
    if (device_ == Device::CUDA) {
      cudaFree(data_);
      data_ = nullptr;
      size_ = 0;
      return;
    }
#endif
    delete[] data_;
    data_ = nullptr;
    size_ = 0;
  }

  void allocate(size_t n, Device device) {
    deallocate();
    size_ = n;
    device_ = device;
    if (n == 0) {
      data_ = nullptr;
      return;
    }
#ifdef HAR_HAS_CUDA
    if (device == Device::CUDA && cuda_runtime_available()) {
      cuda_check(cudaMallocManaged(&data_, n * sizeof(T)), "cudaMallocManaged");
      cuda_check(cudaMemset(data_, 0, n * sizeof(T)), "cudaMemset");
      return;
    }
#endif
    device_ = Device::CPU;
    data_ = new T[n]{};
  }

  void copy_from(const T *src, size_t n, Device src_device) {
    if (n == 0) {
      return;
    }
#ifdef HAR_HAS_CUDA
    if (device_ == Device::CUDA || src_device == Device::CUDA) {
      cuda_check(cudaMemcpy(data_, src, n * sizeof(T), cudaMemcpyDefault),
                 "cudaMemcpy");
      return;
    }
#endif
    (void)src_device;
    std::memcpy(data_, src, n * sizeof(T));
  }

  auto gpu() const -> bool {
#ifdef HAR_HAS_CUDA
    return device_ == Device::CUDA && cuda_ops::active();
#else
    return false;
#endif
  }

public:
  Tensor() = default;

  explicit Tensor(std::vector<size_t> shape, Device device = default_device())
      : shape_(std::move(shape)) {
    compute_strides();
    allocate(compute_size(), device);
  }

  Tensor(std::vector<size_t> shape, T init_value,
         Device device = default_device())
      : shape_(std::move(shape)) {
    compute_strides();
    allocate(compute_size(), device);
    fill(init_value);
  }

  Tensor(std::vector<size_t> shape, std::vector<T> host,
         Device device = default_device())
      : shape_(std::move(shape)) {
    compute_strides();
    allocate(compute_size(), device);
    if (host.size() != size_) {
      throw std::invalid_argument("Data size mismatch with shape");
    }
    if (size_ == 0) {
      return;
    }
#ifdef HAR_HAS_CUDA
    if (device_ == Device::CUDA) {
      cuda_check(cudaMemcpy(data_, host.data(), size_ * sizeof(T),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy H2D");
      return;
    }
#endif
    std::memcpy(data_, host.data(), size_ * sizeof(T));
  }

  Tensor(const Tensor &other) : shape_(other.shape_), strides_(other.strides_) {
    allocate(other.size_, other.device_);
    copy_from(other.data_, other.size_, other.device_);
  }

  Tensor(Tensor &&other) noexcept
      : shape_(std::move(other.shape_)), strides_(std::move(other.strides_)),
        data_(other.data_), size_(other.size_), device_(other.device_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }

  auto operator=(const Tensor &other) -> Tensor & {
    if (this == &other) {
      return *this;
    }
    shape_ = other.shape_;
    strides_ = other.strides_;
    allocate(other.size_, other.device_);
    copy_from(other.data_, other.size_, other.device_);
    return *this;
  }

  auto operator=(Tensor &&other) noexcept -> Tensor & {
    if (this == &other) {
      return *this;
    }
    deallocate();
    shape_ = std::move(other.shape_);
    strides_ = std::move(other.strides_);
    data_ = other.data_;
    size_ = other.size_;
    device_ = other.device_;
    other.data_ = nullptr;
    other.size_ = 0;
    return *this;
  }

  ~Tensor() { deallocate(); }

  auto device() const -> Device { return device_; }

  void to_(Device device) {
    if (device_ == device || size_ == 0) {
      device_ = device;
      return;
    }
    Tensor tmp(shape_, device);
    tmp.copy_from(data_, size_, device_);
    *this = std::move(tmp);
  }

  auto to(Device device) const -> Tensor {
    Tensor out(*this);
    out.to_(device);
    return out;
  }

  void sync() const {
#ifdef HAR_HAS_CUDA
    if (device_ == Device::CUDA) {
      cuda_check(cudaDeviceSynchronize(), "Tensor::sync");
    }
#endif
  }

  auto shape() const -> const std::vector<size_t> & { return shape_; }

  auto size() const -> size_t { return size_; }

  auto data() -> T * { return data_; }

  auto data() const -> const T * { return data_; }

  auto begin() -> T * { return data_; }
  auto end() -> T * { return data_ + size_; }
  auto begin() const -> const T * { return data_; }
  auto end() const -> const T * { return data_ + size_; }

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
    if (new_size != size_) {
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

  auto flatten() const -> Tensor { return reshaped({size_}); }

  void fill(T value) {
#ifdef HAR_HAS_CUDA
    if constexpr (std::is_same_v<T, float>) {
      if (gpu()) {
        cuda_ops::fill(data_, size_, value);
        return;
      }
    }
#endif
    std::fill(data_, data_ + size_, value);
  }

  void zero() { fill(T{0}); }

  auto operator+=(const Tensor &other) -> Tensor & {
    if (shape_ != other.shape_) {
      throw std::invalid_argument("Shape mismatch for addition");
    }
#ifdef HAR_HAS_CUDA
    if constexpr (std::is_same_v<T, float>) {
      if (gpu()) {
        cuda_ops::add(data_, data_, other.data_, size_);
        return *this;
      }
    }
#endif
    std::transform(data_, data_ + size_, other.data_, data_, std::plus<T>{});
    return *this;
  }

  auto operator-=(const Tensor &other) -> Tensor & {
    if (shape_ != other.shape_) {
      throw std::invalid_argument("Shape mismatch for subtraction");
    }
#ifdef HAR_HAS_CUDA
    if constexpr (std::is_same_v<T, float>) {
      if (gpu()) {
        cuda_ops::sub(data_, data_, other.data_, size_);
        return *this;
      }
    }
#endif
    std::transform(data_, data_ + size_, other.data_, data_, std::minus<T>{});
    return *this;
  }

  auto operator*=(const Tensor &other) -> Tensor & {
    if (shape_ != other.shape_) {
      throw std::invalid_argument("Shape mismatch for multiplication");
    }
#ifdef HAR_HAS_CUDA
    if constexpr (std::is_same_v<T, float>) {
      if (gpu()) {
        cuda_ops::mul(data_, data_, other.data_, size_);
        return *this;
      }
    }
#endif
    std::transform(data_, data_ + size_, other.data_, data_,
                   std::multiplies<T>{});
    return *this;
  }

  auto operator/=(const Tensor &other) -> Tensor & {
    if (shape_ != other.shape_) {
      throw std::invalid_argument("Shape mismatch for division");
    }
#ifdef HAR_HAS_CUDA
    if constexpr (std::is_same_v<T, float>) {
      if (gpu()) {
        cuda_ops::div(data_, data_, other.data_, size_);
        return *this;
      }
    }
#endif
    std::transform(data_, data_ + size_, other.data_, data_, std::divides<T>{});
    return *this;
  }

  auto operator+=(T scalar) -> Tensor & {
#ifdef HAR_HAS_CUDA
    if constexpr (std::is_same_v<T, float>) {
      if (gpu()) {
        cuda_ops::add_scalar(data_, data_, scalar, size_);
        return *this;
      }
    }
#endif
    std::for_each(data_, data_ + size_, [scalar](T &x) { x += scalar; });
    return *this;
  }

  auto operator-=(T scalar) -> Tensor & {
#ifdef HAR_HAS_CUDA
    if constexpr (std::is_same_v<T, float>) {
      if (gpu()) {
        cuda_ops::sub_scalar(data_, data_, scalar, size_);
        return *this;
      }
    }
#endif
    std::for_each(data_, data_ + size_, [scalar](T &x) { x -= scalar; });
    return *this;
  }

  auto operator*=(T scalar) -> Tensor & {
#ifdef HAR_HAS_CUDA
    if constexpr (std::is_same_v<T, float>) {
      if (gpu()) {
        cuda_ops::mul_scalar(data_, data_, scalar, size_);
        return *this;
      }
    }
#endif
    std::for_each(data_, data_ + size_, [scalar](T &x) { x *= scalar; });
    return *this;
  }

  auto operator/=(T scalar) -> Tensor & {
#ifdef HAR_HAS_CUDA
    if constexpr (std::is_same_v<T, float>) {
      if (gpu()) {
        cuda_ops::div_scalar(data_, data_, scalar, size_);
        return *this;
      }
    }
#endif
    std::for_each(data_, data_ + size_, [scalar](T &x) { x /= scalar; });
    return *this;
  }

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
    Tensor result(shape_, device_);
#ifdef HAR_HAS_CUDA
    if constexpr (std::is_same_v<T, float>) {
      if (gpu()) {
        cuda_ops::negate(result.data_, data_, size_);
        return result;
      }
    }
#endif
    std::transform(data_, data_ + size_, result.data_, std::negate<T>{});
    return result;
  }

  auto sum() const -> T {
    sync();
    return std::accumulate(data_, data_ + size_, T{0});
  }

  auto mean() const -> T { return sum() / static_cast<T>(size_); }

  auto max() const -> T {
    sync();
    return *std::max_element(data_, data_ + size_);
  }

  auto min() const -> T {
    sync();
    return *std::min_element(data_, data_ + size_);
  }

  auto operator==(const Tensor &other) const -> bool {
    if (shape_ != other.shape_) {
      return false;
    }
    sync();
    other.sync();
    return std::equal(data_, data_ + size_, other.data_);
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
  Tensor<T> result(tensor.shape(), tensor.device());
#ifdef HAR_HAS_CUDA
  if constexpr (std::is_same_v<T, float>) {
    if (tensor.device() == Device::CUDA && cuda_ops::active()) {
      cuda_ops::rsub_scalar(result.data(), scalar, tensor.data(), tensor.size());
      return result;
    }
  }
#endif
  std::transform(tensor.begin(), tensor.end(), result.begin(),
                 [scalar](T x) { return scalar - x; });
  return result;
}

} // namespace har
