/**
 * Benchmarks the inversion of 1 random square matrix of increasing size
 **/

#include <benchmark/benchmark.h>

#include <arrayfire.h>
#include <Eigen/Eigen>
#include <torch/torch.h>

auto BM_PYTORCH_INV = [](benchmark::State& state, int size, torch::Device device) {
  torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kFloat64).layout(torch::kStrided).device(device).requires_grad(true);

  torch::manual_seed(0);
  torch::Tensor tensor0 = torch::rand({ size, size }, options);

  torch::Tensor result = torch::rand({ size, size }, options);
  for (auto _ : state)
  {
    benchmark::DoNotOptimize(result = tensor0.inverse());
  }
};

auto BM_EIGEN_INV = [](benchmark::State& state, int size) {
  Eigen::MatrixXd matrix0 = Eigen::MatrixXd::Random(size, size);
  Eigen::MatrixXd result = Eigen::MatrixXd::Random(size, size);
  for (auto _ : state)
  {
    benchmark::DoNotOptimize(result = matrix0.inverse());
  }
};

// TODO: Make sure the ArrayFire benchmarks are actually executing since they normally use lazy execution
auto BM_ARRAYFIRE_INV = [](benchmark::State& state, int size, auto device) {
  af::setBackend(device);

  af::setSeed(0);
  af::array array0 = af::randu(size, size);

  af::array result = af::randu(size, size);
  for (auto _ : state)
  {
    benchmark::DoNotOptimize(result = af::inverse(array0));
  }
};

int main(int argc, char** argv)
{
  for (auto& test_input : { 2, 3, 4, 8, 16, 32, 64, 128, 256, 512 })
  {
    std::string name = "BM_PYTORCH_CPU_INV_Size_" + std::to_string(test_input);
    benchmark::RegisterBenchmark(name.c_str(), BM_PYTORCH_INV, test_input, torch::kCPU)->UseRealTime()->Unit(benchmark::TimeUnit::kMicrosecond);
  }
  for (auto& test_input : { 2, 3, 4, 8, 16, 32, 64, 128, 256, 512 })
  {
    std::string name = "BM_PYTORCH_GPU_INV_Size_" + std::to_string(test_input);
    benchmark::RegisterBenchmark(name.c_str(), BM_PYTORCH_INV, test_input, torch::kCUDA)->UseRealTime()->Unit(benchmark::TimeUnit::kMicrosecond);
  }
  for (auto& test_input : { 2, 3, 4, 8, 16, 32, 64, 128, 256, 512 })
  {
    std::string name = "BM_EIGEN_INV_Size_" + std::to_string(test_input);
    benchmark::RegisterBenchmark(name.c_str(), BM_EIGEN_INV, test_input)->UseRealTime()->Unit(benchmark::TimeUnit::kMicrosecond);
  }
  for (auto& test_input : { 2, 3, 4, 8, 16, 32, 64, 128, 256, 512 })
  {
    std::string name = "BM_ARRAYFIRE_CPU_INV_Size_" + std::to_string(test_input);
    benchmark::RegisterBenchmark(name.c_str(), BM_ARRAYFIRE_INV, test_input, AF_BACKEND_CPU)->UseRealTime()->Unit(benchmark::TimeUnit::kMicrosecond);
  }
  for (auto& test_input : { 2, 3, 4, 8, 16, 32, 64, 128, 256, 512 })
  {
    std::string name = "BM_ARRAYFIRE_CUDA_INV_Size_" + std::to_string(test_input);
    benchmark::RegisterBenchmark(name.c_str(), BM_ARRAYFIRE_INV, test_input, AF_BACKEND_CUDA)->UseRealTime()->Unit(benchmark::TimeUnit::kMicrosecond);
  }
  for (auto& test_input : { 2, 3, 4, 8, 16, 32, 64, 128, 256, 512 })
  {
    std::string name = "BM_ARRAYFIRE_OPENCL_INV_Size_" + std::to_string(test_input);
    benchmark::RegisterBenchmark(name.c_str(), BM_ARRAYFIRE_INV, test_input, AF_BACKEND_OPENCL)->UseRealTime()->Unit(benchmark::TimeUnit::kMicrosecond);
  }
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
}
