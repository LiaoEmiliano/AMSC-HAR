#pragma once

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

#ifdef HAR_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace har {

enum class Device { CPU, CUDA };

#ifdef HAR_HAS_CUDA
inline void cuda_check(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
  }
}

inline auto cuda_device_count() -> int {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) {
    return 0;
  }
  return n;
}
#else
inline auto cuda_device_count() -> int { return 0; }
#endif

inline auto cuda_runtime_available() -> bool { return cuda_device_count() > 0; }

inline auto default_device() -> Device {
  static const Device device = [] {
    if (const char *env = std::getenv("HAR_DEVICE")) {
      const std::string_view v{env};
      if (v == "cpu" || v == "CPU") {
        return Device::CPU;
      }
    }
    return cuda_runtime_available() ? Device::CUDA : Device::CPU;
  }();
  return device;
}

inline auto device_name() -> std::string {
#ifdef HAR_HAS_CUDA
  if (default_device() == Device::CUDA) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
      return prop.name;
    }
    return "CUDA";
  }
#endif
  return "CPU";
}

} // namespace har
