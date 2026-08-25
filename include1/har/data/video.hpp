#pragma once

#include "har/tensor.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef HAR_HAS_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#endif

#ifndef HAR_STB_IMAGE_IMPLEMENTED
#define HAR_STB_IMAGE_IMPLEMENTED
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#endif

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace har::data {

struct FrameExtractOptions {
  int width{224};
  int height{224};
  int max_frames{16};
  int frame_stride{1};
  bool rgb{true};
  bool uniform{false};
};

namespace detail {

inline auto to_lower(std::string s) -> std::string {
  std::ranges::transform(s, s.begin(),
                         [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

inline auto quote_path(const std::filesystem::path &p) -> std::string {
  return std::format("\"{}\"", p.string());
}

#ifdef _WIN32
inline auto run_process(const std::string &cmd, bool capture)
    -> std::pair<int, std::string> {
  SECURITY_ATTRIBUTES sa{};
  sa.nLength = sizeof(sa);
  sa.bInheritHandle = TRUE;

  HANDLE read_pipe = nullptr;
  HANDLE write_pipe = nullptr;
  if (capture) {
    if (CreatePipe(&read_pipe, &write_pipe, &sa, 0) == 0) {
      return {static_cast<int>(GetLastError()), {}};
    }
    SetHandleInformation(read_pipe, HANDLE_FLAG_INHERIT, 0);
  } else {
    write_pipe = CreateFileA("NUL", GENERIC_WRITE,
                             FILE_SHARE_READ | FILE_SHARE_WRITE, &sa,
                             OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (write_pipe == INVALID_HANDLE_VALUE) {
      return {static_cast<int>(GetLastError()), {}};
    }
  }

  STARTUPINFOA si{};
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
  si.hStdOutput = write_pipe;
  si.hStdError = write_pipe;

  PROCESS_INFORMATION pi{};
  std::string mutable_cmd = cmd;
  const BOOL ok =
      CreateProcessA(nullptr, mutable_cmd.data(), nullptr, nullptr, TRUE,
                     CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
  CloseHandle(write_pipe);
  if (ok == 0) {
    if (read_pipe != nullptr) {
      CloseHandle(read_pipe);
    }
    return {static_cast<int>(GetLastError()), {}};
  }

  std::string out;
  if (capture && read_pipe != nullptr) {
    char tmp[512];
    DWORD n = 0;
    while (ReadFile(read_pipe, tmp, sizeof(tmp), &n, nullptr) != 0 && n > 0) {
      out.append(tmp, tmp + n);
    }
    CloseHandle(read_pipe);
  }

  WaitForSingleObject(pi.hProcess, INFINITE);
  DWORD code = 1;
  GetExitCodeProcess(pi.hProcess, &code);
  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);
  return {static_cast<int>(code), out};
}

inline auto run_capture(const std::string &cmd) -> std::pair<int, std::string> {
  return run_process(cmd, true);
}

inline auto run_ok(const std::string &cmd) -> bool {
  return run_process(cmd, false).first == 0;
}
#else
inline auto run_capture(const std::string &cmd) -> std::pair<int, std::string> {
  FILE *pipe = popen(cmd.c_str(), "r");
  if (pipe == nullptr) {
    return {1, {}};
  }
  std::string out;
  char buf[512];
  while (std::fgets(buf, sizeof(buf), pipe) != nullptr) {
    out += buf;
  }
  const int rc = pclose(pipe);
  return {rc, out};
}

inline auto run_ok(const std::string &cmd) -> bool {
  return std::system((cmd + " >/dev/null 2>&1").c_str()) == 0;
}
#endif

inline auto trim(std::string s) -> std::string {
  const auto not_space = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::ranges::find_if(s, not_space));
  s.erase(std::ranges::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s;
}

inline auto find_tool(const char *name) -> std::string {
  static std::string cache_name;
  static std::string cache_path;
  if (cache_name == name && !cache_path.empty()) {
    return cache_path;
  }

#ifdef _WIN32
  const auto [rc, out] = run_capture(std::format("where {}", name));
#else
  const auto [rc, out] = run_capture(std::format("command -v {}", name));
#endif
  if (rc == 0) {
    const auto line_end = out.find_first_of("\r\n");
    auto path = trim(out.substr(0, line_end));
    if (!path.empty()) {
      cache_name = name;
      cache_path = path;
      return path;
    }
  }

#ifdef _WIN32
  const char *exe = std::getenv("LOCALAPPDATA");
  if (exe != nullptr) {
    const auto packages =
        std::filesystem::path(exe) / "Microsoft" / "WinGet" / "Packages";
    std::error_code ec;
    if (std::filesystem::exists(packages, ec)) {
      for (const auto &pkg : std::filesystem::directory_iterator(packages, ec)) {
        const auto fname = pkg.path().filename().string();
        if (fname.find("FFmpeg") == std::string::npos &&
            fname.find("ffmpeg") == std::string::npos) {
          continue;
        }
        for (auto it = std::filesystem::recursive_directory_iterator(pkg.path(),
                                                                     ec);
             it != std::filesystem::recursive_directory_iterator();
             it.increment(ec)) {
          if (!it->is_regular_file()) {
            continue;
          }
          const auto stem = it->path().stem().string();
          const auto file = it->path().filename().string();
          if (file == name || file == std::string(name) + ".exe" ||
              stem == std::string(name)) {
            cache_name = name;
            cache_path = it->path().string();
            return cache_path;
          }
        }
      }
    }
  }
  const auto local = std::getenv("LOCALAPPDATA");
  std::vector<std::filesystem::path> guesses;
  if (local != nullptr) {
    guesses.emplace_back(std::filesystem::path(local) / "ffmpeg" / "bin" / name);
  }
  guesses.insert(guesses.end(), {
      std::filesystem::path("C:/ffmpeg/bin") / name,
      std::filesystem::path("C:/Program Files/ffmpeg/bin") / name,
      std::filesystem::path("C:/Program Files/Gyan/FFmpeg/bin") / name,
  });
  for (const auto &g : guesses) {
    std::filesystem::path candidate = g;
#ifdef _WIN32
    if (candidate.extension().empty()) {
      candidate += ".exe";
    }
#endif
    if (std::filesystem::exists(candidate)) {
      cache_name = name;
      cache_path = candidate.string();
      return cache_path;
    }
  }
#endif
  return {};
}

inline auto sample_indices(size_t total, size_t max_frames) -> std::vector<size_t> {
  std::vector<size_t> idx;
  if (total == 0 || max_frames == 0) {
    return idx;
  }
  const size_t n = std::min(total, max_frames);
  idx.resize(n);
  if (n == 1) {
    idx[0] = total / 2;
    return idx;
  }
  for (size_t i = 0; i < n; ++i) {
    idx[i] = i * (total - 1) / (n - 1);
  }
  return idx;
}

inline void store_hwc(Tensor<float> &out, size_t frame, size_t C, size_t H, size_t W,
                      const unsigned char *hwc) {
  for (size_t h = 0; h < H; ++h) {
    for (size_t w = 0; w < W; ++w) {
      for (size_t c = 0; c < C; ++c) {
        out.at(frame, c, h, w) =
            static_cast<float>(hwc[(h * W + w) * C + c]) / 255.0f;
      }
    }
  }
}

inline auto load_image_file(const std::filesystem::path &path, int width, int height,
                            bool rgb) -> std::vector<unsigned char> {
  int w = 0;
  int h = 0;
  int ch = 0;
  const int want = rgb ? 3 : 1;
  unsigned char *pixels = stbi_load(path.string().c_str(), &w, &h, &ch, want);
  if (pixels == nullptr) {
    throw std::runtime_error("Failed to load frame image: " + path.string());
  }

  std::vector<unsigned char> out(static_cast<size_t>(width) * static_cast<size_t>(height) *
                                 static_cast<size_t>(want));
  if (w == width && h == height) {
    std::copy(pixels, pixels + static_cast<ptrdiff_t>(out.size()), out.begin());
    stbi_image_free(pixels);
    return out;
  }

  // Nearest-neighbor resize fallback (ffmpeg/OpenCV should already resize).
  for (int y = 0; y < height; ++y) {
    const int sy = std::min(h - 1, y * h / height);
    for (int x = 0; x < width; ++x) {
      const int sx = std::min(w - 1, x * w / width);
      for (int c = 0; c < want; ++c) {
        out[(static_cast<size_t>(y) * static_cast<size_t>(width) +
             static_cast<size_t>(x)) *
                static_cast<size_t>(want) +
            static_cast<size_t>(c)] =
            pixels[(static_cast<size_t>(sy) * static_cast<size_t>(w) +
                    static_cast<size_t>(sx)) *
                       static_cast<size_t>(want) +
                   static_cast<size_t>(c)];
      }
    }
  }
  stbi_image_free(pixels);
  return out;
}

inline auto list_sorted_images(const std::filesystem::path &dir)
    -> std::vector<std::filesystem::path> {
  std::vector<std::filesystem::path> files;
  if (!std::filesystem::exists(dir)) {
    return files;
  }
  for (const auto &entry : std::filesystem::directory_iterator(dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    auto ext = to_lower(entry.path().extension().string());
    if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") {
      files.push_back(entry.path());
    }
  }
  std::ranges::sort(files);
  return files;
}

struct TempDir {
  std::filesystem::path path;
  explicit TempDir(std::filesystem::path p) : path(std::move(p)) {
    std::filesystem::create_directories(path);
  }
  TempDir(const TempDir &) = delete;
  auto operator=(const TempDir &) -> TempDir & = delete;
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
};

inline auto probe_duration_seconds(const std::filesystem::path &video) -> double {
  const auto ffprobe = find_tool("ffprobe.exe");
  const auto bin = ffprobe.empty() ? find_tool("ffprobe") : ffprobe;
  if (bin.empty()) {
    return 0.0;
  }
  const auto cmd = std::format(
      "{} -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {}",
      quote_path(bin), quote_path(video));
  const auto [rc, out] = run_capture(cmd);
  if (rc != 0) {
    return 0.0;
  }
  try {
    return std::stod(trim(out));
  } catch (...) {
    return 0.0;
  }
}

inline auto extract_with_ffmpeg(const std::filesystem::path &video,
                                const std::filesystem::path &out_dir,
                                const FrameExtractOptions &opt) -> size_t {
  auto ffmpeg = find_tool("ffmpeg.exe");
  if (ffmpeg.empty()) {
    ffmpeg = find_tool("ffmpeg");
  }
  if (ffmpeg.empty()) {
    throw std::runtime_error(
        "No video decoder available. Install OpenCV (rebuild) or FFmpeg in PATH.");
  }

  std::filesystem::create_directories(out_dir);
  const int max_frames = std::max(1, opt.max_frames);
  const int stride = std::max(1, opt.frame_stride);
  const double duration = probe_duration_seconds(video);

  std::string vf;
  if (opt.uniform && duration > 0.05) {
    const double fps = static_cast<double>(max_frames) / duration;
    vf = std::format("fps={:.6f},scale={}:{}", fps, opt.width, opt.height);
  } else if (opt.uniform) {
    vf = std::format("fps=1,scale={}:{}", opt.width, opt.height);
  } else {
    vf = std::format("framestep={},scale={}:{}", stride, opt.width, opt.height);
  }

  const auto pattern = (out_dir / "frame_%04d.png").string();
  const auto cmd = std::format(
      "{} -hide_banner -loglevel error -y -i {} -vf {} -frames:v {} -q:v 2 {}",
      quote_path(ffmpeg), quote_path(video), quote_path(vf), max_frames,
      quote_path(pattern));

  if (!run_ok(cmd)) {
    throw std::runtime_error("FFmpeg failed to decode: " + video.string());
  }
  return list_sorted_images(out_dir).size();
}

#ifdef HAR_HAS_OPENCV
inline auto mat_to_hwc(const cv::Mat &src, bool rgb) -> std::vector<unsigned char> {
  cv::Mat converted;
  if (rgb) {
    if (src.channels() == 3) {
      cv::cvtColor(src, converted, cv::COLOR_BGR2RGB);
    } else if (src.channels() == 4) {
      cv::cvtColor(src, converted, cv::COLOR_BGRA2RGB);
    } else {
      cv::cvtColor(src, converted, cv::COLOR_GRAY2RGB);
    }
  } else if (src.channels() == 1) {
    converted = src;
  } else if (src.channels() == 4) {
    cv::cvtColor(src, converted, cv::COLOR_BGRA2GRAY);
  } else {
    cv::cvtColor(src, converted, cv::COLOR_BGR2GRAY);
  }
  if (!converted.isContinuous()) {
    converted = converted.clone();
  }
  return {converted.datastart, converted.dataend};
}

inline auto open_video_capture(const std::filesystem::path &video) -> cv::VideoCapture {
  const auto path = video.string();
  cv::VideoCapture cap;
  const int backends[] = {cv::CAP_FFMPEG, cv::CAP_MSMF, cv::CAP_ANY};
  for (int backend : backends) {
    cap.open(path, backend);
    if (cap.isOpened()) {
      cap.set(cv::CAP_PROP_CONVERT_RGB, 1);
      return cap;
    }
  }
  throw std::runtime_error("OpenCV failed to open video: " + path);
}

inline auto extract_with_opencv(const std::filesystem::path &video,
                                const FrameExtractOptions &opt)
    -> std::vector<std::vector<unsigned char>> {
  auto cap = open_video_capture(video);

  const int max_frames = std::max(1, opt.max_frames);
  const int stride = std::max(1, opt.frame_stride);
  const size_t reported =
      static_cast<size_t>(std::max(0.0, cap.get(cv::CAP_PROP_FRAME_COUNT)));

  std::vector<std::vector<unsigned char>> frames;
  cv::Mat raw;
  cv::Mat resized;

  auto push_resized = [&](const cv::Mat &src) {
    if (src.empty()) {
      return;
    }
    cv::resize(src, resized, cv::Size(opt.width, opt.height));
    frames.push_back(mat_to_hwc(resized, opt.rgb));
  };

  if (opt.uniform && reported > 1) {
    const auto wanted = sample_indices(reported, static_cast<size_t>(max_frames));
    for (size_t idx : wanted) {
      cap.set(cv::CAP_PROP_POS_FRAMES, static_cast<double>(idx));
      if (cap.read(raw)) {
        push_resized(raw);
      }
    }
    if (!frames.empty()) {
      return frames;
    }
    cap.set(cv::CAP_PROP_POS_FRAMES, 0.0);
  }

  std::vector<size_t> wanted;
  if (opt.uniform && reported > 0) {
    wanted = sample_indices(reported, static_cast<size_t>(max_frames));
  }

  size_t n = 0;
  size_t want_i = 0;
  while (frames.size() < static_cast<size_t>(max_frames) && cap.read(raw)) {
    bool keep = false;
    if (!wanted.empty()) {
      if (want_i < wanted.size() && n == wanted[want_i]) {
        keep = true;
        ++want_i;
      }
    } else if (n % static_cast<size_t>(stride) == 0) {
      keep = true;
    }

    if (keep) {
      push_resized(raw);
    }
    ++n;
  }

  if (frames.empty()) {
    throw std::runtime_error("OpenCV extracted 0 frames: " + video.string());
  }
  return frames;
}
#endif

inline auto frames_to_tensor(const std::vector<std::vector<unsigned char>> &frames,
                             const FrameExtractOptions &opt) -> Tensor<float> {
  if (frames.empty()) {
    throw std::runtime_error("No frames extracted from video");
  }
  const size_t F = frames.size();
  const size_t C = opt.rgb ? 3 : 1;
  const size_t H = static_cast<size_t>(opt.height);
  const size_t W = static_cast<size_t>(opt.width);
  Tensor<float> out({F, C, H, W}, Device::CPU);
  for (size_t f = 0; f < F; ++f) {
    if (frames[f].size() != C * H * W) {
      throw std::runtime_error("Frame size mismatch");
    }
    store_hwc(out, f, C, H, W, frames[f].data());
  }
  return out;
}

inline auto images_to_tensor(const std::vector<std::filesystem::path> &files,
                             const FrameExtractOptions &opt) -> Tensor<float> {
  std::vector<std::vector<unsigned char>> frames;
  frames.reserve(files.size());
  for (const auto &f : files) {
    frames.push_back(load_image_file(f, opt.width, opt.height, opt.rgb));
  }
  return frames_to_tensor(frames, opt);
}

} // namespace detail

inline auto extract_video_frames(const std::string &video_path,
                                 const FrameExtractOptions &opt) -> Tensor<float> {
  const std::filesystem::path path(video_path);
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("Video not found: " + video_path);
  }
  if (opt.width <= 0 || opt.height <= 0 || opt.max_frames <= 0) {
    throw std::invalid_argument("FrameExtractOptions: width/height/max_frames must be > 0");
  }

#ifdef HAR_HAS_OPENCV
  return detail::frames_to_tensor(detail::extract_with_opencv(path, opt), opt);
#else
  const auto stamp = std::hash<std::string>{}(std::filesystem::absolute(path).string());
  detail::TempDir tmp(std::filesystem::temp_directory_path() /
                      std::format("har_frames_{}", stamp));
  detail::extract_with_ffmpeg(path, tmp.path, opt);
  auto files = detail::list_sorted_images(tmp.path);
  if (static_cast<int>(files.size()) > opt.max_frames) {
    files.resize(static_cast<size_t>(opt.max_frames));
  }
  return detail::images_to_tensor(files, opt);
#endif
}

inline auto extract_video_frames_to_dir(const std::string &video_path,
                                        const std::string &output_dir,
                                        const FrameExtractOptions &opt) -> size_t {
  const std::filesystem::path path(video_path);
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("Video not found: " + video_path);
  }
  std::filesystem::create_directories(output_dir);

#ifdef HAR_HAS_OPENCV
  auto frames = detail::extract_with_opencv(path, opt);
  const size_t C = opt.rgb ? 3 : 1;
  for (size_t i = 0; i < frames.size(); ++i) {
    const auto out = std::filesystem::path(output_dir) /
                     std::format("frame_{:04}.png", i + 1);
    if (stbi_write_png(out.string().c_str(), opt.width, opt.height,
                       static_cast<int>(C), frames[i].data(),
                       opt.width * static_cast<int>(C)) == 0) {
      throw std::runtime_error("Failed to write " + out.string());
    }
  }
  return frames.size();
#else
  return detail::extract_with_ffmpeg(path, output_dir, opt);
#endif
}

inline auto video_decoder_name() -> const char * {
#ifdef HAR_HAS_OPENCV
  return "OpenCV";
#else
  return "FFmpeg";
#endif
}

inline auto average_video_frames(const Tensor<float> &frames) -> Tensor<float> {
  if (frames.dims() != 4 || frames.shape()[0] == 0) {
    throw std::invalid_argument("average_video_frames: expected [F,C,H,W]");
  }
  const size_t F = frames.shape()[0];
  const size_t C = frames.shape()[1];
  const size_t H = frames.shape()[2];
  const size_t W = frames.shape()[3];
  Tensor<float> out({1, C, H, W}, 0.0f);
  const size_t plane = C * H * W;
  for (size_t f = 0; f < F; ++f) {
    for (size_t i = 0; i < plane; ++i) {
      out[i] += frames[f * plane + i];
    }
  }
  out /= static_cast<float>(F);
  return out;
}

} // namespace har::data
