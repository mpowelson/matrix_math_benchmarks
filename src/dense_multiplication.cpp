/**
 * @file dense_multiplication.cpp
 * @brief Benchmarks the multiplication of 2 random square matrices of increasing size
 *
 * @author Matthew Powelson
 * @date April 1, 2020
 * @version TODO
 * @bug No known bugs
 *
 * @copyright Copyright (c) 2020, Southwest Research Institute
 *
 * @par License
 * Software License Agreement (Apache License)
 * @par
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * @par
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <benchmark/benchmark.h>

#include <arrayfire.h>
#include <Eigen/Eigen>
#include <torch/torch.h>

auto BM_PYTORCH_MM = [](benchmark::State& state, int size, torch::Device device) {
  torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kFloat64).layout(torch::kStrided).device(device).requires_grad(true);

  torch::manual_seed(0);
  torch::Tensor tensor0 = torch::rand({ size, size }, options);
  torch::manual_seed(1);
  torch::Tensor tensor1 = torch::rand({ size, size }, options);
  torch::Tensor result = torch::rand({ size, size }, options);
  for (auto _ : state)
  {
    benchmark::DoNotOptimize(result = torch::mm(tensor0, tensor1));
  }
};

auto BM_EIGEN_MM = [](benchmark::State& state, int size) {
  Eigen::MatrixXd matrix0 = Eigen::MatrixXd::Random(size, size);
  Eigen::MatrixXd matrix1 = Eigen::MatrixXd::Random(size, size);
  Eigen::MatrixXd result = Eigen::MatrixXd::Random(size, size);
  for (auto _ : state)
  {
    benchmark::DoNotOptimize(result = matrix0 * matrix1);
  }
};

auto BM_ARRAYFIRE_MM = [](benchmark::State& state, int size, auto device) {
  af::setBackend(device);

  af::setSeed(0);
  af::array array0 = af::randu(size, size);
  af::setSeed(1);
  af::array array1 = af::randu(size, size);

  af::array result = af::randu(size, size);
  for (auto _ : state)
  {
    benchmark::DoNotOptimize(result = af::matmul(array0, array1));
  }
};

int main(int argc, char** argv)
{
  for (auto& test_input : { 2, 3, 4, 8, 16, 32, 64, 128, 256, 512 })
  {
    std::string name = "BM_PYTORCH_CPU_MM_Size_" + std::to_string(test_input);
    benchmark::RegisterBenchmark(name.c_str(), BM_PYTORCH_MM, test_input, torch::kCPU)
        ->UseRealTime()
        ->Unit(benchmark::TimeUnit::kMicrosecond);
  }
  for (auto& test_input : { 2, 3, 4, 8, 16, 32, 64, 128, 256, 512 })
  {
    std::string name = "BM_PYTORCH_GPU_MM_Size_" + std::to_string(test_input);
    benchmark::RegisterBenchmark(name.c_str(), BM_PYTORCH_MM, test_input, torch::kCUDA)
        ->UseRealTime()
        ->Unit(benchmark::TimeUnit::kMicrosecond);
  }
  for (auto& test_input : { 2, 3, 4, 8, 16, 32, 64, 128, 256, 512 })
  {
    std::string name = "BM_EIGEN_MM_Size_" + std::to_string(test_input);
    benchmark::RegisterBenchmark(name.c_str(), BM_EIGEN_MM, test_input)
        ->UseRealTime()
        ->Unit(benchmark::TimeUnit::kMicrosecond);
  }
  for (auto& test_input : { 2, 3, 4, 8, 16, 32, 64, 128, 256, 512 })
  {
    std::string name = "BM_ARRAYFIRE_CPU_MM_Size_" + std::to_string(test_input);
    benchmark::RegisterBenchmark(name.c_str(), BM_ARRAYFIRE_MM, test_input, AF_BACKEND_CPU)
        ->UseRealTime()
        ->Unit(benchmark::TimeUnit::kMicrosecond);
  }
  for (auto& test_input : { 2, 3, 4, 8, 16, 32, 64, 128, 256, 512 })
  {
    std::string name = "BM_ARRAYFIRE_CUDA_MM_Size_" + std::to_string(test_input);
    benchmark::RegisterBenchmark(name.c_str(), BM_ARRAYFIRE_MM, test_input, AF_BACKEND_CUDA)
        ->UseRealTime()
        ->Unit(benchmark::TimeUnit::kMicrosecond);
  }
  for (auto& test_input : { 2, 3, 4, 8, 16, 32, 64, 128, 256, 512 })
  {
    std::string name = "BM_ARRAYFIRE_OPENCL_MM_Size_" + std::to_string(test_input);
    benchmark::RegisterBenchmark(name.c_str(), BM_ARRAYFIRE_MM, test_input, AF_BACKEND_OPENCL)
        ->UseRealTime()
        ->Unit(benchmark::TimeUnit::kMicrosecond);
  }
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
}
