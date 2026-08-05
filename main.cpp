#include "har/har.hpp"

#include <cstdlib>
#include <format>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace {

void print_usage() {
  std::cout <<
      R"(Usage:
  har_cnn smoke
  har_cnn extract <video> [outdir]
  har_cnn train --data <UCF11_root> [options]

Train options:
  --data PATH         Path to UCF11_updated_mpg / action_youtube_naudio
  --epochs N          Default: 5
  --batch N           Default: 4
  --lr FLOAT          Default: 0.01
  --size N            Input HxW (divisible by 4). Default: 64
  --frames N          Frames averaged per video. Default: 8
  --val-ratio FLOAT   Default: 0.2
  --max-per-class N   Limit videos/class (0=all). Useful for dry runs
  --no-cache          Disable frame cache under data/ucf11_cache
  --seed N            Default: 42

Download UCF11:
  https://www.crcv.ucf.edu/data/UCF_YouTube_Action.php
  or run: scripts/download_ucf11.ps1
)";
}

auto run_cnn_smoke_test() -> int {
  std::cout << "Smoke test: tiny CNN on synthetic NCHW images\n";

  constexpr size_t batch = 32;
  constexpr size_t epochs = 80;
  constexpr size_t H = 16;
  constexpr size_t W = 16;

  har::network::Sequential<float> model;
  model.add(std::make_unique<har::layers::Conv2D<float>>(1, 4, 3, 1, 1));
  model.add(std::make_unique<har::layers::ReLU<float>>());
  model.add(std::make_unique<har::layers::MaxPool2D<float>>(2));
  model.add(std::make_unique<har::layers::Flatten<float>>());
  model.add(std::make_unique<har::layers::Linear<float>>(4 * 8 * 8, 2));
  model.train();

  har::loss::CrossEntropyLoss<float> criterion;
  har::optim::SGD<float> optimizer(0.05f);

  std::mt19937 gen{7};
  std::uniform_real_distribution<float> noise(0.0f, 0.3f);

  har::Tensor<float> x({batch, 1, H, W});
  har::Tensor<float> y({batch});

  float first_loss = 0.0f;
  float last_loss = 0.0f;

  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    for (size_t n = 0; n < batch; ++n) {
      const bool label = (n % 2 == 0);
      y[n] = label ? 1.0f : 0.0f;
      for (size_t h = 0; h < H; ++h) {
        for (size_t w = 0; w < W; ++w) {
          float v = noise(gen);
          const bool center = (h >= 4 && h < 12 && w >= 4 && w < 12);
          if (label && center) {
            v += 0.7f;
          }
          if (!label && !center) {
            v += 0.7f;
          }
          x.at(n, 0, h, w) = v;
        }
      }
    }

    model.zero_grad();
    auto logits = model.forward(x);
    const float loss = criterion.forward(logits, y);
    auto grad = criterion.backward();
    model.backward(grad);
    optimizer.step(model.parameters());

    if (epoch == 0) {
      first_loss = loss;
    }
    last_loss = loss;

    if (epoch % 20 == 0 || epoch + 1 == epochs) {
      std::cout << std::format("epoch {:>3}  loss = {:.4f}\n", epoch, loss);
    }
  }

  model.eval();
  auto logits = model.forward(x);
  size_t correct = 0;
  for (size_t i = 0; i < batch; ++i) {
    const size_t pred = logits.at(i, 0) > logits.at(i, 1) ? 0 : 1;
    if (pred == static_cast<size_t>(y[i])) {
      ++correct;
    }
  }

  const float acc = static_cast<float>(correct) / static_cast<float>(batch);
  std::cout << std::format("train loss: {:.4f} -> {:.4f}\n", first_loss,
                           last_loss);
  std::cout << std::format("eval accuracy: {:.1f}% ({}/{})\n", acc * 100.0f,
                           correct, batch);

  if (!(last_loss < first_loss) || acc < 0.8f) {
    std::cout << "WARNING: CNN smoke test failed thresholds\n";
    return 1;
  }

  std::cout << "CNN smoke test passed.\n";
  return 0;
}

#ifdef HAR_HAS_OPENCV
auto run_extract_frames(const std::string &video_path,
                        const std::string &output_dir) -> int {
  har::data::FrameExtractOptions opt;
  opt.width = 224;
  opt.height = 224;
  opt.max_frames = 16;
  opt.frame_stride = 2;

  const size_t saved =
      har::data::extract_video_frames_to_dir(video_path, output_dir, opt);
  auto tensor = har::data::extract_video_frames(video_path, opt);

  std::cout << std::format(
      "Extracted {} frames to '{}'\nTensor shape: [{}, {}, {}, {}]\n", saved,
      output_dir, tensor.shape()[0], tensor.shape()[1], tensor.shape()[2],
      tensor.shape()[3]);
  return 0;
}

auto parse_train_args(int argc, char **argv, har::data::UCF11Options &data_opt,
                      har::train::TrainConfig &train_opt, std::string &data_root)
    -> bool {
  data_root.clear();
  for (int i = 2; i < argc; ++i) {
    const std::string_view arg = argv[i];
    auto need = [&](const char *name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("Missing value for ") + name);
      }
      return argv[++i];
    };

    if (arg == "--data") {
      data_root = need("--data");
    } else if (arg == "--epochs") {
      train_opt.epochs = static_cast<size_t>(std::stoul(need("--epochs")));
    } else if (arg == "--batch") {
      train_opt.batch_size = static_cast<size_t>(std::stoul(need("--batch")));
    } else if (arg == "--lr") {
      train_opt.learning_rate = std::stof(need("--lr"));
    } else if (arg == "--size") {
      const int s = std::stoi(need("--size"));
      data_opt.frames.width = s;
      data_opt.frames.height = s;
    } else if (arg == "--frames") {
      data_opt.frames.max_frames = std::stoi(need("--frames"));
    } else if (arg == "--val-ratio") {
      data_opt.val_ratio = std::stof(need("--val-ratio"));
    } else if (arg == "--max-per-class") {
      data_opt.max_per_class =
          static_cast<size_t>(std::stoul(need("--max-per-class")));
    } else if (arg == "--no-cache") {
      data_opt.use_cache = false;
    } else if (arg == "--seed") {
      const auto seed = static_cast<unsigned>(std::stoul(need("--seed")));
      data_opt.seed = seed;
      train_opt.seed = seed;
    } else if (arg == "--help" || arg == "-h") {
      return false;
    } else {
      throw std::runtime_error("Unknown argument: " + std::string(arg));
    }
  }
  return !data_root.empty();
}

auto run_train(int argc, char **argv) -> int {
  har::data::UCF11Options data_opt;
  har::train::TrainConfig train_opt;
  // CPU-friendly defaults for the naive Conv2D implementation.
  data_opt.frames.width = 64;
  data_opt.frames.height = 64;
  data_opt.frames.max_frames = 8;
  train_opt.epochs = 5;
  train_opt.batch_size = 4;
  train_opt.learning_rate = 0.01f;

  std::string data_root;
  if (!parse_train_args(argc, argv, data_opt, train_opt, data_root)) {
    print_usage();
    return 1;
  }

  har::data::UCF11Dataset dataset(data_root, data_opt);
  return har::train::train_ucf11(dataset, train_opt);
}
#endif

} // namespace

int main(int argc, char **argv) {
  std::cout << "Human Action Recognition using CNN\n";

  if (argc < 2) {
    print_usage();
    return run_cnn_smoke_test();
  }

  const std::string cmd = argv[1];
  try {
    if (cmd == "smoke" || cmd == "--smoke") {
      return run_cnn_smoke_test();
    }
    if (cmd == "help" || cmd == "--help" || cmd == "-h") {
      print_usage();
      return 0;
    }
#ifdef HAR_HAS_OPENCV
    if (cmd == "extract") {
      if (argc < 3) {
        print_usage();
        return 1;
      }
      const std::string out_dir = (argc >= 4) ? argv[3] : "data/frames";
      return run_extract_frames(argv[2], out_dir);
    }
    if (cmd == "train") {
      return run_train(argc, argv);
    }
#else
    if (cmd == "extract" || cmd == "train") {
      std::cerr << "Built without OpenCV. Rebuild with OpenCV enabled.\n";
      return 1;
    }
#endif
    // Backward compatible: har_cnn <video> [outdir]
#ifdef HAR_HAS_OPENCV
    if (argc >= 2) {
      const std::string out_dir = (argc >= 3) ? argv[2] : "data/frames";
      return run_extract_frames(argv[1], out_dir);
    }
#endif
    print_usage();
    return 1;
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << '\n';
    return 1;
  }
}
