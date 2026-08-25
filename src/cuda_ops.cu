#include "har/cuda/ops.hpp"
#include "har/cuda/device.hpp"

#ifdef HAR_HAS_CUDA

namespace har::cuda_ops {
namespace {

constexpr int kBlock = 256;

auto grid_1d(std::size_t n) -> int {
  return static_cast<int>((n + static_cast<std::size_t>(kBlock) - 1) /
                          static_cast<std::size_t>(kBlock));
}

void launch_sync() {
  cuda_check(cudaGetLastError(), "CUDA kernel");
}

auto bn_scratch(int count) -> float * {
  static float *p = nullptr;
  static int cap = 0;
  if (count > cap) {
    if (p != nullptr) {
      cudaFree(p);
    }
    cuda_check(cudaMalloc(&p, static_cast<std::size_t>(count) * sizeof(float)),
               "cudaMalloc bn scratch");
    cap = count;
  }
  return p;
}

__global__ void fill_kernel(float *p, std::size_t n, float value) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    p[i] = value;
  }
}

__global__ void copy_kernel(float *dst, const float *src, std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = src[i];
  }
}

__global__ void add_kernel(float *dst, const float *a, const float *b,
                           std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = a[i] + b[i];
  }
}

__global__ void sub_kernel(float *dst, const float *a, const float *b,
                           std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = a[i] - b[i];
  }
}

__global__ void mul_kernel(float *dst, const float *a, const float *b,
                           std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = a[i] * b[i];
  }
}

__global__ void div_kernel(float *dst, const float *a, const float *b,
                           std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = a[i] / b[i];
  }
}

__global__ void add_scalar_kernel(float *dst, const float *a, float s,
                                  std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = a[i] + s;
  }
}

__global__ void sub_scalar_kernel(float *dst, const float *a, float s,
                                  std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = a[i] - s;
  }
}

__global__ void mul_scalar_kernel(float *dst, const float *a, float s,
                                  std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = a[i] * s;
  }
}

__global__ void div_scalar_kernel(float *dst, const float *a, float s,
                                  std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = a[i] / s;
  }
}

__global__ void rsub_scalar_kernel(float *dst, float s, const float *a,
                                   std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = s - a[i];
  }
}

__global__ void negate_kernel(float *dst, const float *src, std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = -src[i];
  }
}

__global__ void matmul_kernel(const float *a, const float *b, float *c, int M,
                              int K, int N) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= M || j >= N) {
    return;
  }
  float sum = 0.0f;
  for (int k = 0; k < K; ++k) {
    sum += a[i * K + k] * b[k * N + j];
  }
  c[i * N + j] = sum;
}

__global__ void transpose_kernel(const float *a, float *b, int rows, int cols) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= rows || j >= cols) {
    return;
  }
  b[j * rows + i] = a[i * cols + j];
}

__global__ void add_bias_rows_kernel(float *out, const float *bias, int batch,
                                     int features) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= batch || j >= features) {
    return;
  }
  out[i * features + j] += bias[j];
}

__global__ void bias_grad_kernel(const float *grad_out, float *grad_bias,
                                 int batch, int features) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= features) {
    return;
  }
  float sum = 0.0f;
  for (int i = 0; i < batch; ++i) {
    sum += grad_out[i * features + j];
  }
  grad_bias[j] += sum;
}

__global__ void conv2d_forward_kernel(const float *input, const float *weight,
                                      const float *bias, float *output, int N,
                                      int Cin, int H, int W, int Cout, int Kh,
                                      int Kw, int Sh, int Sw, int Ph, int Pw,
                                      int Ho, int Wo, int use_bias) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = N * Cout * Ho * Wo;
  if (idx >= total) {
    return;
  }
  const int ow = idx % Wo;
  const int oh = (idx / Wo) % Ho;
  const int oc = (idx / (Wo * Ho)) % Cout;
  const int n = idx / (Wo * Ho * Cout);

  float sum = use_bias ? bias[oc] : 0.0f;
  for (int ic = 0; ic < Cin; ++ic) {
    for (int kh = 0; kh < Kh; ++kh) {
      for (int kw = 0; kw < Kw; ++kw) {
        const int ih = oh * Sh + kh - Ph;
        const int iw = ow * Sw + kw - Pw;
        if (ih < 0 || iw < 0 || ih >= H || iw >= W) {
          continue;
        }
        const float iv =
            input[((n * Cin + ic) * H + ih) * W + iw];
        const float wv =
            weight[((oc * Cin + ic) * Kh + kh) * Kw + kw];
        sum += iv * wv;
      }
    }
  }
  output[((n * Cout + oc) * Ho + oh) * Wo + ow] = sum;
}

__global__ void conv2d_backward_kernel(const float *input, const float *weight,
                                       const float *grad_out, float *grad_in,
                                       float *grad_w, float *grad_b, int N,
                                       int Cin, int H, int W, int Cout, int Kh,
                                       int Kw, int Sh, int Sw, int Ph, int Pw,
                                       int Ho, int Wo, int use_bias) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = N * Cout * Ho * Wo;
  if (idx >= total) {
    return;
  }
  const int ow = idx % Wo;
  const int oh = (idx / Wo) % Ho;
  const int oc = (idx / (Wo * Ho)) % Cout;
  const int n = idx / (Wo * Ho * Cout);
  const float go = grad_out[((n * Cout + oc) * Ho + oh) * Wo + ow];
  if (use_bias) {
    atomicAdd(&grad_b[oc], go);
  }
  for (int ic = 0; ic < Cin; ++ic) {
    for (int kh = 0; kh < Kh; ++kh) {
      for (int kw = 0; kw < Kw; ++kw) {
        const int ih = oh * Sh + kh - Ph;
        const int iw = ow * Sw + kw - Pw;
        if (ih < 0 || iw < 0 || ih >= H || iw >= W) {
          continue;
        }
        const int in_i = ((n * Cin + ic) * H + ih) * W + iw;
        const int w_i = ((oc * Cin + ic) * Kh + kh) * Kw + kw;
        atomicAdd(&grad_w[w_i], go * input[in_i]);
        atomicAdd(&grad_in[in_i], go * weight[w_i]);
      }
    }
  }
}

__global__ void relu_forward_kernel(const float *in, float *out, std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = in[i] > 0.0f ? in[i] : 0.0f;
  }
}

__global__ void relu_backward_kernel(const float *in, const float *grad_out,
                                     float *grad_in, std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    grad_in[i] = in[i] > 0.0f ? grad_out[i] : 0.0f;
  }
}

__global__ void dropout_forward_kernel(const float *in, float *out, float *mask,
                                       std::size_t n, float keep, float scale,
                                       unsigned long long seed) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  unsigned long long x = seed ^ (0x9E3779B97F4A7C15ULL * (i + 1));
  x ^= x >> 30;
  x *= 0xBF58476D1CE4E5B9ULL;
  x ^= x >> 27;
  x *= 0x94D049BB133111EBULL;
  x ^= x >> 31;
  const float u = static_cast<float>(x & 0xFFFFFFULL) / 16777216.0f;
  const float m = u < keep ? scale : 0.0f;
  mask[i] = m;
  out[i] = in[i] * m;
}

__global__ void dropout_backward_kernel(const float *grad_out, const float *mask,
                                        float *grad_in, std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    grad_in[i] = grad_out[i] * mask[i];
  }
}

__global__ void maxpool_forward_kernel(const float *in, float *out, int *indices,
                                       int N, int C, int H, int W, int Kh,
                                       int Kw, int Sh, int Sw, int Ho, int Wo) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = N * C * Ho * Wo;
  if (idx >= total) {
    return;
  }
  const int ow = idx % Wo;
  const int oh = (idx / Wo) % Ho;
  const int c = (idx / (Wo * Ho)) % C;
  const int n = idx / (Wo * Ho * C);

  float best = -1e30f;
  int best_idx = 0;
  for (int kh = 0; kh < Kh; ++kh) {
    for (int kw = 0; kw < Kw; ++kw) {
      const int ih = oh * Sh + kh;
      const int iw = ow * Sw + kw;
      const float val = in[((n * C + c) * H + ih) * W + iw];
      if (val > best) {
        best = val;
        best_idx = ih * W + iw;
      }
    }
  }
  out[((n * C + c) * Ho + oh) * Wo + ow] = best;
  indices[idx] = best_idx;
}

__global__ void gap_forward_kernel(const float *in, float *out, int N, int C,
                                   int H, int W) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = N * C;
  if (idx >= total) {
    return;
  }
  const int c = idx % C;
  const int n = idx / C;
  const int hw = H * W;
  const float *plane = in + (static_cast<long long>(n * C + c) * hw);
  float sum = 0.0f;
  for (int i = 0; i < hw; ++i) {
    sum += plane[i];
  }
  out[idx] = sum / static_cast<float>(hw);
}

__global__ void gap_backward_kernel(const float *grad_out, float *grad_in,
                                    int N, int C, int H, int W) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int hw = H * W;
  const int total = N * C * hw;
  if (idx >= total) {
    return;
  }
  const int nc = idx / hw;
  grad_in[idx] = grad_out[nc] / static_cast<float>(hw);
}

__global__ void temporal_mean_forward_kernel(const float *feat, float *out,
                                             int N, int T, int C) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = N * C;
  if (idx >= total) {
    return;
  }
  const int c = idx % C;
  const int n = idx / C;
  float sum = 0.0f;
  for (int t = 0; t < T; ++t) {
    sum += feat[(n * T + t) * C + c];
  }
  out[idx] = sum / static_cast<float>(T);
}

__global__ void temporal_mean_backward_kernel(const float *grad_out,
                                              float *grad_in, int N, int T,
                                              int C) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = N * T * C;
  if (idx >= total) {
    return;
  }
  const int c = idx % C;
  const int n = (idx / C) / T;
  grad_in[idx] = grad_out[n * C + c] / static_cast<float>(T);
}

__global__ void maxpool_backward_kernel(const float *grad_out, float *grad_in,
                                        const int *indices, int N, int C, int H,
                                        int W, int Ho, int Wo) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = N * C * Ho * Wo;
  if (idx >= total) {
    return;
  }
  const int ow = idx % Wo;
  const int oh = (idx / Wo) % Ho;
  const int c = (idx / (Wo * Ho)) % C;
  const int n = idx / (Wo * Ho * C);
  const int flat = indices[idx];
  const int ih = flat / W;
  const int iw = flat % W;
  (void)H;
  atomicAdd(&grad_in[((n * C + c) * H + ih) * W + iw],
            grad_out[((n * C + c) * Ho + oh) * Wo + ow]);
}

__global__ void bn2d_stats_kernel(const float *x, float *mean, float *var,
                                  int N, int C, int H, int W) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) {
    return;
  }
  const int hw = H * W;
  const int m = N * hw;
  float sum = 0.0f;
  float sq = 0.0f;
  for (int n = 0; n < N; ++n) {
    const float *plane = x + static_cast<long long>(n * C + c) * hw;
    for (int i = 0; i < hw; ++i) {
      const float v = plane[i];
      sum += v;
      sq += v * v;
    }
  }
  const float mu = sum / static_cast<float>(m);
  mean[c] = mu;
  var[c] = sq / static_cast<float>(m) - mu * mu;
}

__global__ void bn2d_apply_kernel(const float *x, float *y, float *xhat,
                                  const float *mean, const float *inv_std,
                                  const float *gamma, const float *beta, int N,
                                  int C, int H, int W) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = N * C * H * W;
  if (idx >= total) {
    return;
  }
  const int hw = H * W;
  const int c = (idx / hw) % C;
  const float hat = (x[idx] - mean[c]) * inv_std[c];
  xhat[idx] = hat;
  y[idx] = gamma[c] * hat + beta[c];
}

__global__ void bn2d_inv_std_kernel(const float *var, float *inv_std, int C,
                                    float eps) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) {
    return;
  }
  inv_std[c] = rsqrtf(var[c] + eps);
}

__global__ void bn2d_running_kernel(float *running_mean, float *running_var,
                                    const float *mean, const float *var, int C,
                                    float momentum) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) {
    return;
  }
  running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean[c];
  running_var[c] = (1.0f - momentum) * running_var[c] + momentum * var[c];
}

__global__ void bn2d_grad_reduce_kernel(const float *xhat, const float *gamma,
                                        const float *grad_out, float *dgamma,
                                        float *dbeta, float *dxhat_sum,
                                        float *dxhat_xhat_sum, int N, int C,
                                        int H, int W) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) {
    return;
  }
  const int hw = H * W;
  float dg = 0.0f;
  float db = 0.0f;
  float s1 = 0.0f;
  float s2 = 0.0f;
  for (int n = 0; n < N; ++n) {
    const int base = (n * C + c) * hw;
    for (int i = 0; i < hw; ++i) {
      const float dy = grad_out[base + i];
      const float hat = xhat[base + i];
      db += dy;
      dg += dy * hat;
      const float dxhat = dy * gamma[c];
      s1 += dxhat;
      s2 += dxhat * hat;
    }
  }
  dgamma[c] += dg;
  dbeta[c] += db;
  dxhat_sum[c] = s1;
  dxhat_xhat_sum[c] = s2;
}

__global__ void bn2d_grad_input_kernel(const float *xhat, const float *inv_std,
                                       const float *gamma, const float *grad_out,
                                       const float *dxhat_sum,
                                       const float *dxhat_xhat_sum,
                                       float *grad_in, int N, int C, int H,
                                       int W) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = N * C * H * W;
  if (idx >= total) {
    return;
  }
  const int hw = H * W;
  const int c = (idx / hw) % C;
  const float m = static_cast<float>(N * hw);
  const float dy = grad_out[idx];
  const float hat = xhat[idx];
  const float dxhat = dy * gamma[c];
  grad_in[idx] = inv_std[c] / m *
                 (m * dxhat - dxhat_sum[c] - hat * dxhat_xhat_sum[c]);
}

} // namespace

auto active() -> bool {
  return default_device() == Device::CUDA && cuda_runtime_available();
}

void synchronize() { cuda_check(cudaDeviceSynchronize(), "synchronize"); }

void fill(float *p, std::size_t n, float value) {
  if (n == 0) {
    return;
  }
  fill_kernel<<<grid_1d(n), kBlock>>>(p, n, value);
  launch_sync();
}

void copy(float *dst, const float *src, std::size_t n) {
  if (n == 0) {
    return;
  }
  copy_kernel<<<grid_1d(n), kBlock>>>(dst, src, n);
  launch_sync();
}

void add(float *dst, const float *a, const float *b, std::size_t n) {
  if (n == 0) {
    return;
  }
  add_kernel<<<grid_1d(n), kBlock>>>(dst, a, b, n);
  launch_sync();
}
void sub(float *dst, const float *a, const float *b, std::size_t n) {
  if (n == 0) {
    return;
  }
  sub_kernel<<<grid_1d(n), kBlock>>>(dst, a, b, n);
  launch_sync();
}
void mul(float *dst, const float *a, const float *b, std::size_t n) {
  if (n == 0) {
    return;
  }
  mul_kernel<<<grid_1d(n), kBlock>>>(dst, a, b, n);
  launch_sync();
}
void div(float *dst, const float *a, const float *b, std::size_t n) {
  if (n == 0) {
    return;
  }
  div_kernel<<<grid_1d(n), kBlock>>>(dst, a, b, n);
  launch_sync();
}

void add_scalar(float *dst, const float *a, float s, std::size_t n) {
  if (n == 0) {
    return;
  }
  add_scalar_kernel<<<grid_1d(n), kBlock>>>(dst, a, s, n);
  launch_sync();
}
void sub_scalar(float *dst, const float *a, float s, std::size_t n) {
  if (n == 0) {
    return;
  }
  sub_scalar_kernel<<<grid_1d(n), kBlock>>>(dst, a, s, n);
  launch_sync();
}
void mul_scalar(float *dst, const float *a, float s, std::size_t n) {
  if (n == 0) {
    return;
  }
  mul_scalar_kernel<<<grid_1d(n), kBlock>>>(dst, a, s, n);
  launch_sync();
}
void div_scalar(float *dst, const float *a, float s, std::size_t n) {
  if (n == 0) {
    return;
  }
  div_scalar_kernel<<<grid_1d(n), kBlock>>>(dst, a, s, n);
  launch_sync();
}
void rsub_scalar(float *dst, float s, const float *a, std::size_t n) {
  if (n == 0) {
    return;
  }
  rsub_scalar_kernel<<<grid_1d(n), kBlock>>>(dst, s, a, n);
  launch_sync();
}
void negate(float *dst, const float *src, std::size_t n) {
  if (n == 0) {
    return;
  }
  negate_kernel<<<grid_1d(n), kBlock>>>(dst, src, n);
  launch_sync();
}

void matmul(const float *a, const float *b, float *c, int M, int K, int N) {
  dim3 block(16, 16);
  dim3 grid((N + 15) / 16, (M + 15) / 16);
  matmul_kernel<<<grid, block>>>(a, b, c, M, K, N);
  launch_sync();
}

void transpose(const float *a, float *b, int rows, int cols) {
  dim3 block(16, 16);
  dim3 grid((cols + 15) / 16, (rows + 15) / 16);
  transpose_kernel<<<grid, block>>>(a, b, rows, cols);
  launch_sync();
}

void add_bias_rows(float *out, const float *bias, int batch, int features) {
  dim3 block(16, 16);
  dim3 grid((features + 15) / 16, (batch + 15) / 16);
  add_bias_rows_kernel<<<grid, block>>>(out, bias, batch, features);
  launch_sync();
}

void bias_grad(const float *grad_out, float *grad_bias, int batch,
               int features) {
  bias_grad_kernel<<<(features + kBlock - 1) / kBlock, kBlock>>>(
      grad_out, grad_bias, batch, features);
  launch_sync();
}

void conv2d_forward(const float *input, const float *weight, const float *bias,
                    float *output, int N, int Cin, int H, int W, int Cout,
                    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw, int Ho,
                    int Wo, int use_bias) {
  const int total = N * Cout * Ho * Wo;
  if (total <= 0) {
    return;
  }
  conv2d_forward_kernel<<<grid_1d(static_cast<std::size_t>(total)), kBlock>>>(
      input, weight, bias, output, N, Cin, H, W, Cout, Kh, Kw, Sh, Sw, Ph, Pw,
      Ho, Wo, use_bias);
  launch_sync();
}

void conv2d_backward(const float *input, const float *weight,
                     const float *grad_out, float *grad_in, float *grad_w,
                     float *grad_b, int N, int Cin, int H, int W, int Cout,
                     int Kh, int Kw, int Sh, int Sw, int Ph, int Pw, int Ho,
                     int Wo, int use_bias) {
  const int total = N * Cout * Ho * Wo;
  if (total <= 0) {
    return;
  }
  conv2d_backward_kernel<<<grid_1d(static_cast<std::size_t>(total)), kBlock>>>(
      input, weight, grad_out, grad_in, grad_w, grad_b, N, Cin, H, W, Cout, Kh,
      Kw, Sh, Sw, Ph, Pw, Ho, Wo, use_bias);
  launch_sync();
}

void relu_forward(const float *in, float *out, std::size_t n) {
  if (n == 0) {
    return;
  }
  relu_forward_kernel<<<grid_1d(n), kBlock>>>(in, out, n);
  launch_sync();
}

void relu_backward(const float *in, const float *grad_out, float *grad_in,
                   std::size_t n) {
  if (n == 0) {
    return;
  }
  relu_backward_kernel<<<grid_1d(n), kBlock>>>(in, grad_out, grad_in, n);
  launch_sync();
}

void dropout_forward(const float *in, float *out, float *mask, std::size_t n,
                     float keep, unsigned long long seed) {
  if (n == 0) {
    return;
  }
  const float scale = keep > 0.0f ? 1.0f / keep : 0.0f;
  dropout_forward_kernel<<<grid_1d(n), kBlock>>>(in, out, mask, n, keep, scale,
                                                 seed);
  launch_sync();
}

void dropout_backward(const float *grad_out, const float *mask, float *grad_in,
                      std::size_t n) {
  if (n == 0) {
    return;
  }
  dropout_backward_kernel<<<grid_1d(n), kBlock>>>(grad_out, mask, grad_in, n);
  launch_sync();
}

void maxpool_forward(const float *in, float *out, int *indices_host, int N,
                     int C, int H, int W, int Kh, int Kw, int Sh, int Sw,
                     int Ho, int Wo) {
  const int total = N * C * Ho * Wo;
  if (total <= 0) {
    return;
  }
  int *indices = nullptr;
  cuda_check(cudaMalloc(&indices, static_cast<std::size_t>(total) * sizeof(int)),
             "cudaMalloc maxpool indices");
  maxpool_forward_kernel<<<grid_1d(static_cast<std::size_t>(total)), kBlock>>>(
      in, out, indices, N, C, H, W, Kh, Kw, Sh, Sw, Ho, Wo);
  launch_sync();
  cuda_check(cudaMemcpy(indices_host, indices,
                        static_cast<std::size_t>(total) * sizeof(int),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy maxpool indices");
  cudaFree(indices);
}

void maxpool_backward(const float *grad_out, float *grad_in,
                      const int *indices_host, int N, int C, int H, int W,
                      int Ho, int Wo) {
  const int total = N * C * Ho * Wo;
  if (total <= 0) {
    return;
  }
  int *indices = nullptr;
  cuda_check(cudaMalloc(&indices, static_cast<std::size_t>(total) * sizeof(int)),
             "cudaMalloc maxpool indices");
  cuda_check(cudaMemcpy(indices, indices_host,
                        static_cast<std::size_t>(total) * sizeof(int),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy maxpool indices H2D");
  maxpool_backward_kernel<<<grid_1d(static_cast<std::size_t>(total)), kBlock>>>(
      grad_out, grad_in, indices, N, C, H, W, Ho, Wo);
  launch_sync();
  cudaFree(indices);
}

void gap_forward(const float *in, float *out, int N, int C, int H, int W) {
  const int total = N * C;
  if (total <= 0) {
    return;
  }
  gap_forward_kernel<<<grid_1d(static_cast<std::size_t>(total)), kBlock>>>(
      in, out, N, C, H, W);
  launch_sync();
}

void gap_backward(const float *grad_out, float *grad_in, int N, int C, int H,
                  int W) {
  const int total = N * C * H * W;
  if (total <= 0) {
    return;
  }
  gap_backward_kernel<<<grid_1d(static_cast<std::size_t>(total)), kBlock>>>(
      grad_out, grad_in, N, C, H, W);
  launch_sync();
}

void temporal_mean_forward(const float *feat, float *out, int N, int T, int C) {
  const int total = N * C;
  if (total <= 0) {
    return;
  }
  temporal_mean_forward_kernel<<<grid_1d(static_cast<std::size_t>(total)),
                                 kBlock>>>(feat, out, N, T, C);
  launch_sync();
}

void temporal_mean_backward(const float *grad_out, float *grad_in, int N, int T,
                            int C) {
  const int total = N * T * C;
  if (total <= 0) {
    return;
  }
  temporal_mean_backward_kernel<<<grid_1d(static_cast<std::size_t>(total)),
                                  kBlock>>>(grad_out, grad_in, N, T, C);
  launch_sync();
}

__device__ int temporal_clamp(int t, int T) {
  return t < 0 ? 0 : (t >= T ? T - 1 : t);
}

__global__ void temporal_conv1d_forward_kernel(const float *x, const float *w,
                                               const float *b, float *y, int N,
                                               int T, int Cin, int Cout, int K,
                                               int pad, int use_bias) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = N * T * Cout;
  if (idx >= total) {
    return;
  }
  const int co = idx % Cout;
  const int t = (idx / Cout) % T;
  const int n = idx / (Cout * T);
  float sum = (use_bias && b != nullptr) ? b[co] : 0.0f;
  for (int k = 0; k < K; ++k) {
    const int ti = temporal_clamp(t + k - pad, T);
    const float *xrow = x + static_cast<long long>(n * T + ti) * Cin;
    const float *wk = w + (static_cast<long long>(co) * Cin * K + k);
    for (int ci = 0; ci < Cin; ++ci) {
      sum += wk[static_cast<long long>(ci) * K] * xrow[ci];
    }
  }
  y[static_cast<long long>(n * T + t) * Cout + co] = sum;
}

__global__ void temporal_conv1d_input_grad_kernel(const float *w,
                                                  const float *go, float *gx,
                                                  int N, int T, int Cin,
                                                  int Cout, int K, int pad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = N * T * Cin;
  if (idx >= total) {
    return;
  }
  const int ci = idx % Cin;
  const int ti = (idx / Cin) % T;
  const int n = idx / (Cin * T);
  float sum = 0.0f;
  for (int t = 0; t < T; ++t) {
    for (int k = 0; k < K; ++k) {
      if (temporal_clamp(t + k - pad, T) != ti) {
        continue;
      }
      const float *grow = go + static_cast<long long>(n * T + t) * Cout;
      const float *wk = w + (static_cast<long long>(ci) * K + k);
      for (int co = 0; co < Cout; ++co) {
        sum += grow[co] *
               wk[static_cast<long long>(co) * Cin * K];
      }
    }
  }
  gx[idx] = sum;
}

__global__ void temporal_conv1d_weight_grad_kernel(const float *x,
                                                   const float *go, float *gw,
                                                   int N, int T, int Cin,
                                                   int Cout, int K, int pad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = Cout * Cin * K;
  if (idx >= total) {
    return;
  }
  const int k = idx % K;
  const int ci = (idx / K) % Cin;
  const int co = idx / (K * Cin);
  float sum = 0.0f;
  for (int n = 0; n < N; ++n) {
    for (int t = 0; t < T; ++t) {
      const int ti = temporal_clamp(t + k - pad, T);
      sum += go[static_cast<long long>(n * T + t) * Cout + co] *
             x[static_cast<long long>(n * T + ti) * Cin + ci];
    }
  }
  gw[idx] += sum;
}

void temporal_conv1d_forward(const float *x, const float *weight,
                             const float *bias, float *y, int N, int T, int Cin,
                             int Cout, int K, int pad, int use_bias) {
  const int total = N * T * Cout;
  if (total <= 0) {
    return;
  }
  temporal_conv1d_forward_kernel<<<grid_1d(static_cast<std::size_t>(total)),
                                   kBlock>>>(x, weight, bias, y, N, T, Cin,
                                             Cout, K, pad, use_bias);
  launch_sync();
}

void temporal_conv1d_backward(const float *x, const float *weight,
                              const float *grad_out, float *grad_in,
                              float *grad_w, float *grad_b, int N, int T,
                              int Cin, int Cout, int K, int pad, int use_bias) {
  const int in_total = N * T * Cin;
  const int w_total = Cout * Cin * K;
  if (in_total > 0) {
    temporal_conv1d_input_grad_kernel<<<grid_1d(static_cast<std::size_t>(in_total)),
                                        kBlock>>>(weight, grad_out, grad_in, N,
                                                  T, Cin, Cout, K, pad);
    launch_sync();
  }
  if (w_total > 0) {
    temporal_conv1d_weight_grad_kernel<<<grid_1d(static_cast<std::size_t>(w_total)),
                                         kBlock>>>(x, grad_out, grad_w, N, T,
                                                   Cin, Cout, K, pad);
    launch_sync();
  }
  if (use_bias && grad_b != nullptr && N * T > 0) {
    bias_grad(grad_out, grad_b, N * T, Cout);
  }
}

void bn2d_forward(const float *x, float *y, float *xhat, const float *gamma,
                  const float *beta, float *batch_mean, float *inv_std,
                  float *running_mean, float *running_var, int N, int C, int H,
                  int W, int training, float momentum, float eps) {
  if (C <= 0 || N <= 0) {
    return;
  }
  const int threads = 128;
  const int cgrid = (C + threads - 1) / threads;
  float *var = bn_scratch(C);

  if (training) {
    bn2d_stats_kernel<<<cgrid, threads>>>(x, batch_mean, var, N, C, H, W);
    launch_sync();
    bn2d_inv_std_kernel<<<cgrid, threads>>>(var, inv_std, C, eps);
    launch_sync();
    bn2d_running_kernel<<<cgrid, threads>>>(running_mean, running_var,
                                            batch_mean, var, C, momentum);
    launch_sync();
  } else {
    bn2d_inv_std_kernel<<<cgrid, threads>>>(running_var, inv_std, C, eps);
    launch_sync();
    cuda_check(cudaMemcpy(batch_mean, running_mean,
                          static_cast<std::size_t>(C) * sizeof(float),
                          cudaMemcpyDeviceToDevice),
               "bn eval mean");
  }

  const int total = N * C * H * W;
  bn2d_apply_kernel<<<grid_1d(static_cast<std::size_t>(total)), kBlock>>>(
      x, y, xhat, training ? batch_mean : running_mean, inv_std, gamma, beta, N,
      C, H, W);
  launch_sync();
}

void bn2d_backward(const float *xhat, const float *gamma, const float *inv_std,
                   const float *grad_out, float *grad_in, float *grad_gamma,
                   float *grad_beta, int N, int C, int H, int W) {
  if (C <= 0 || N <= 0) {
    return;
  }
  float *dxhat_sum = bn_scratch(C * 2);
  float *dxhat_xhat_sum = dxhat_sum + C;
  const int threads = 128;
  const int cgrid = (C + threads - 1) / threads;
  bn2d_grad_reduce_kernel<<<cgrid, threads>>>(
      xhat, gamma, grad_out, grad_gamma, grad_beta, dxhat_sum, dxhat_xhat_sum,
      N, C, H, W);
  launch_sync();
  const int total = N * C * H * W;
  bn2d_grad_input_kernel<<<grid_1d(static_cast<std::size_t>(total)), kBlock>>>(
      xhat, inv_std, gamma, grad_out, dxhat_sum, dxhat_xhat_sum, grad_in, N, C,
      H, W);
  launch_sync();
}

} // namespace har::cuda_ops

#endif
