#pragma once

#include <cstddef>

#ifdef HAR_HAS_CUDA

namespace har::cuda_ops {

auto active() -> bool;
void synchronize();

void fill(float *p, std::size_t n, float value);
void copy(float *dst, const float *src, std::size_t n);

void add(float *dst, const float *a, const float *b, std::size_t n);
void sub(float *dst, const float *a, const float *b, std::size_t n);
void mul(float *dst, const float *a, const float *b, std::size_t n);
void div(float *dst, const float *a, const float *b, std::size_t n);

void add_scalar(float *dst, const float *a, float s, std::size_t n);
void sub_scalar(float *dst, const float *a, float s, std::size_t n);
void mul_scalar(float *dst, const float *a, float s, std::size_t n);
void div_scalar(float *dst, const float *a, float s, std::size_t n);
void rsub_scalar(float *dst, float s, const float *a, std::size_t n);
void negate(float *dst, const float *src, std::size_t n);

void matmul(const float *a, const float *b, float *c, int M, int K, int N);
void transpose(const float *a, float *b, int rows, int cols);
void add_bias_rows(float *out, const float *bias, int batch, int features);
void bias_grad(const float *grad_out, float *grad_bias, int batch, int features);

void conv2d_forward(const float *input, const float *weight, const float *bias,
                    float *output, int N, int Cin, int H, int W, int Cout,
                    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw, int Ho,
                    int Wo, int use_bias);

void conv2d_backward(const float *input, const float *weight,
                     const float *grad_out, float *grad_in, float *grad_w,
                     float *grad_b, int N, int Cin, int H, int W, int Cout,
                     int Kh, int Kw, int Sh, int Sw, int Ph, int Pw, int Ho,
                     int Wo, int use_bias);

void relu_forward(const float *in, float *out, std::size_t n);
void relu_backward(const float *in, const float *grad_out, float *grad_in,
                   std::size_t n);

void dropout_forward(const float *in, float *out, float *mask, std::size_t n,
                     float keep, unsigned long long seed);
void dropout_backward(const float *grad_out, const float *mask, float *grad_in,
                      std::size_t n);

void maxpool_forward(const float *in, float *out, int *indices_host, int N,
                     int C, int H, int W, int Kh, int Kw, int Sh, int Sw,
                     int Ho, int Wo);
void maxpool_backward(const float *grad_out, float *grad_in,
                      const int *indices_host, int N, int C, int H, int W,
                      int Ho, int Wo);

void gap_forward(const float *in, float *out, int N, int C, int H, int W);
void gap_backward(const float *grad_out, float *grad_in, int N, int C, int H,
                  int W);
void temporal_mean_forward(const float *feat, float *out, int N, int T, int C);
void temporal_mean_backward(const float *grad_out, float *grad_in, int N, int T,
                            int C);

void temporal_conv1d_forward(const float *x, const float *weight,
                             const float *bias, float *y, int N, int T, int Cin,
                             int Cout, int K, int pad, int use_bias);
void temporal_conv1d_backward(const float *x, const float *weight,
                              const float *grad_out, float *grad_in,
                              float *grad_w, float *grad_b, int N, int T,
                              int Cin, int Cout, int K, int pad, int use_bias);

void bn2d_forward(const float *x, float *y, float *xhat, const float *gamma,
                  const float *beta, float *batch_mean, float *inv_std,
                  float *running_mean, float *running_var, int N, int C, int H,
                  int W, int training, float momentum, float eps);
void bn2d_backward(const float *xhat, const float *gamma, const float *inv_std,
                   const float *grad_out, float *grad_in, float *grad_gamma,
                   float *grad_beta, int N, int C, int H, int W);

} // namespace har::cuda_ops

#endif
