#include "har/har.hpp"

#include <cctype>
#include <cstdlib>
#include <format>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void print_usage() {
  std::cout <<
      R"(Usage:
  har_cnn smoke
  har_cnn extract <video> [outdir]
  har_cnn train [--dataset ucf101|ucf11] --data <root> [options]
  har_cnn predict [--dir checkAcc] [--weights PATH] [--top K]
  har_cnn eval [--dataset ucf11] [--data PATH] [--weights PATH] [--split test]

Train options:
  --dataset NAME      ucf101 (default) or ucf11
  --data PATH         Video root. Default: data/UCF-101 or data/UCF11_updated_mpg
  --split N           UCF101 official split 1/2/3. Default: 1
  --epochs N          Default: 10
  --batch N           Default: 32 (auto-capped for large size/frames)
  --workers N         Prefetch next batch on a background thread (0=off). Default: 8
  --lr FLOAT          Default: 0.003
  --lr-step N         Decay every N epochs. Default: 4 (0=off)
  --lr-gamma FLOAT    Multiply lr at each step. Default: 0.5
  --momentum FLOAT    Default: 0.9
  --wd FLOAT          Weight decay. Default: 1e-4
  --dropout FLOAT     Default: 0.3
  --size N            Input HxW (divisible by 4). Default: 64
  --frames N          Frames kept per video (temporal dim). Default: 8
  --val-ratio FLOAT   Default: 0.1 (from official train for UCF101)
  --test-ratio FLOAT  UCF11 only. Default: 0.1
  --max-per-class N   Limit videos/class (0=all). Useful for dry runs
  --no-cache          Disable decoded-clip cache
  --seed N            Default: 42
  --weights PATH      Save checkpoint. Default: models/<dataset>_videocnn.harw
  --resume PATH       Load checkpoint and continue training
  --target-acc FLOAT  Stop when test accuracy reaches this (0.95 or 95). Off by default

Predict options:
  --dir PATH          Folder of videos to classify. Default: checkAcc
  --weights PATH      Checkpoint to load. Default: models/ucf11_videocnn.harw
  --top K             Print top-K classes. Default: 3

Eval options:
  --dataset NAME      ucf11 (default) or ucf101
  --data PATH         Video root. Default: data/UCF11_updated_mpg
  --weights PATH      Checkpoint to load. Default: models/ucf11_videocnnBest.harw
  --split NAME        test (default), val, or train
  --batch N           Default: 16
  --seed N            UCF11 hold-out seed. Default: 42

Download UCF101 (recommended HAR dataset):
  powershell -ExecutionPolicy Bypass -File scripts/download_ucf101.ps1

Download UCF11:
  powershell -ExecutionPolicy Bypass -File scripts/download_ucf11.ps1
)";
}

auto run_cnn_smoke_test() -> int {
  std::cout << "Smoke test: tiny CNN on synthetic NCHW images\n";
  std::cout << std::format("Device: {}\n", har::device_name());

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

    x.sync();
    y.sync();

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
  logits.sync();
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

struct TrainCli {
  std::string dataset{"ucf101"};
  std::string data_root;
  har::data::UCF11Options ucf11;
  har::data::UCF101Options ucf101;
  har::train::TrainConfig train;
  bool weights_set{false};
  bool help{false};
};

auto parse_train_args(int argc, char **argv, TrainCli &cli) -> void {
  for (int i = 2; i < argc; ++i) {
    const std::string_view arg = argv[i];
    auto need = [&](const char *name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("Missing value for ") + name);
      }
      return argv[++i];
    };

    if (arg == "--dataset") {
      cli.dataset = need("--dataset");
    } else if (arg == "--data") {
      cli.data_root = need("--data");
    } else if (arg == "--split") {
      cli.ucf101.split = std::stoi(need("--split"));
    } else if (arg == "--epochs") {
      cli.train.epochs = static_cast<size_t>(std::stoul(need("--epochs")));
    } else if (arg == "--batch") {
      cli.train.batch_size = static_cast<size_t>(std::stoul(need("--batch")));
    } else if (arg == "--workers") {
      cli.train.num_workers = static_cast<size_t>(std::stoul(need("--workers")));
    } else if (arg == "--lr") {
      cli.train.learning_rate = std::stof(need("--lr"));
    } else if (arg == "--lr-step") {
      cli.train.lr_step = static_cast<size_t>(std::stoul(need("--lr-step")));
    } else if (arg == "--lr-gamma") {
      cli.train.lr_gamma = std::stof(need("--lr-gamma"));
    } else if (arg == "--momentum") {
      cli.train.momentum = std::stof(need("--momentum"));
    } else if (arg == "--wd") {
      cli.train.weight_decay = std::stof(need("--wd"));
    } else if (arg == "--dropout") {
      cli.train.dropout = std::stof(need("--dropout"));
    } else if (arg == "--size") {
      const int s = std::stoi(need("--size"));
      cli.ucf11.frames.width = s;
      cli.ucf11.frames.height = s;
      cli.ucf101.frames.width = s;
      cli.ucf101.frames.height = s;
    } else if (arg == "--frames") {
      const int n = std::stoi(need("--frames"));
      cli.ucf11.frames.max_frames = n;
      cli.ucf101.frames.max_frames = n;
    } else if (arg == "--val-ratio") {
      const float v = std::stof(need("--val-ratio"));
      cli.ucf11.val_ratio = v;
      cli.ucf101.val_ratio = v;
    } else if (arg == "--test-ratio") {
      cli.ucf11.test_ratio = std::stof(need("--test-ratio"));
    } else if (arg == "--max-per-class") {
      const auto n = static_cast<size_t>(std::stoul(need("--max-per-class")));
      cli.ucf11.max_per_class = n;
      cli.ucf101.max_per_class = n;
    } else if (arg == "--no-cache") {
      cli.ucf11.use_cache = false;
      cli.ucf101.use_cache = false;
    } else if (arg == "--seed") {
      const auto seed = static_cast<unsigned>(std::stoul(need("--seed")));
      cli.ucf11.seed = seed;
      cli.ucf101.seed = seed;
      cli.train.seed = seed;
    } else if (arg == "--weights") {
      cli.train.weights_path = need("--weights");
      cli.weights_set = true;
    } else if (arg == "--resume") {
      cli.train.resume_path = need("--resume");
    } else if (arg == "--target-acc") {
      float v = std::stof(need("--target-acc"));
      if (v > 1.0f) {
        v *= 0.01f;
      }
      cli.train.target_test_acc = v;
    } else if (arg == "--help" || arg == "-h") {
      cli.help = true;
    } else {
      throw std::runtime_error("Unknown argument: " + std::string(arg));
    }
  }
}

auto run_train(int argc, char **argv) -> int {
  TrainCli cli;
  cli.ucf11.frames.width = 64;
  cli.ucf11.frames.height = 64;
  cli.ucf11.frames.max_frames = 8;
  cli.ucf101.frames.width = 64;
  cli.ucf101.frames.height = 64;
  cli.ucf101.frames.max_frames = 8;
  cli.train.epochs = 10;
  cli.train.batch_size = 32;
  cli.train.num_workers = 8;
  cli.train.learning_rate = 0.003f;

  parse_train_args(argc, argv, cli);
  if (cli.help) {
    print_usage();
    return 1;
  }

  std::string dataset = cli.dataset;
  for (char &c : dataset) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }

  if (cli.data_root.empty()) {
    cli.data_root = (dataset == "ucf11") ? "data/UCF11_updated_mpg" : "data";
  }
  if (!cli.weights_set) {
    cli.train.weights_path = (dataset == "ucf11")
                                 ? "models/ucf11_videocnn.harw"
                                 : "models/ucf101_videocnn.harw";
  }

  if (dataset == "ucf11") {
    har::data::UCF11Dataset ds(cli.data_root, cli.ucf11);
    return har::train::train_video_cnn(ds, cli.train, "UCF11");
  }
  if (dataset == "ucf101") {
    har::data::UCF101Dataset ds(cli.data_root, cli.ucf101);
    return har::train::train_video_cnn(ds, cli.train, "UCF101");
  }
  throw std::runtime_error("Unknown dataset: " + cli.dataset +
                           " (expected ucf101 or ucf11)");
}

auto run_predict(int argc, char **argv) -> int {
  har::train::PredictConfig cfg;
  for (int i = 2; i < argc; ++i) {
    const std::string_view arg = argv[i];
    auto need = [&](const char *name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("Missing value for ") + name);
      }
      return argv[++i];
    };
    if (arg == "--dir" || arg == "--data") {
      cfg.dir = need("--dir");
    } else if (arg == "--weights") {
      cfg.weights_path = need("--weights");
    } else if (arg == "--top") {
      cfg.top_k = static_cast<size_t>(std::stoul(need("--top")));
      if (cfg.top_k == 0) {
        throw std::runtime_error("--top must be >= 1");
      }
    } else if (arg == "--help" || arg == "-h") {
      print_usage();
      return 0;
    } else {
      throw std::runtime_error("Unknown argument: " + std::string(arg));
    }
  }
  return har::train::run_predict(cfg);
}

auto run_eval(int argc, char **argv) -> int {
  har::train::EvalConfig cfg;
  for (int i = 2; i < argc; ++i) {
    const std::string_view arg = argv[i];
    auto need = [&](const char *name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("Missing value for ") + name);
      }
      return argv[++i];
    };
    if (arg == "--dataset") {
      cfg.dataset = need("--dataset");
    } else if (arg == "--dir" || arg == "--data") {
      cfg.data_root = need("--data");
    } else if (arg == "--weights") {
      cfg.weights_path = need("--weights");
    } else if (arg == "--split") {
      cfg.split = need("--split");
    } else if (arg == "--batch") {
      cfg.batch_size = static_cast<size_t>(std::stoul(need("--batch")));
    } else if (arg == "--workers") {
      cfg.num_workers = static_cast<size_t>(std::stoul(need("--workers")));
    } else if (arg == "--seed") {
      cfg.seed = static_cast<unsigned>(std::stoul(need("--seed")));
    } else if (arg == "--help" || arg == "-h") {
      print_usage();
      return 0;
    } else {
      throw std::runtime_error("Unknown argument: " + std::string(arg));
    }
  }
  return har::train::run_eval(cfg);
}

} // namespace

int main(int argc, char **argv) {
  std::cout << std::unitbuf;
  std::cerr << std::unitbuf;
  std::cout << "Human Action Recognition using CNN\n";
  std::cout << std::format("Compute device: {}\n", har::device_name());
  std::cout << std::format("Video decoder: {}\n", har::data::video_decoder_name());

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
    if (cmd == "predict" || cmd == "check") {
      return run_predict(argc, argv);
    }
    if (cmd == "eval" || cmd == "test") {
      return run_eval(argc, argv);
    }
    // Backward compatible: har_cnn <video> [outdir]
    if (argc >= 2) {
      const std::string out_dir = (argc >= 3) ? argv[2] : "data/frames";
      return run_extract_frames(argv[1], out_dir);
    }
    print_usage();
    return 1;
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << '\n';
    return 1;
  }
}
