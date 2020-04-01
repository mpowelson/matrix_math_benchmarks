/**
 * @file dense_chained_multiplication.cpp
 * @brief Benchmarks the chained multiplication of 6 random square matrices of increasing size
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

auto BM_PYTORCH_CM = [](benchmark::State& state, int size, torch::Device device) {
  torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kFloat64).layout(torch::kStrided).device(device).requires_grad(true);

  torch::manual_seed(0);
  torch::Tensor tensor0 = torch::rand({ size, size }, options);
  torch::manual_seed(1);
  torch::Tensor tensor1 = torch::rand({ size, size }, options);
  torch::manual_seed(2);
  torch::Tensor tensor2 = torch::rand({ size, size }, options);
  torch::manual_seed(3);
  torch::Tensor tensor3 = torch::rand({ size, size }, options);
  torch::manual_seed(4);
  torch::Tensor tensor4 = torch::rand({ size, size }, options);
  torch::manual_seed(5);
  torch::Tensor tensor5 = torch::rand({ size, size }, options);

  torch::Tensor result = torch::rand({ size, size }, options);

  for (auto _ : state)
  {
    benchmark::DoNotOptimize(
        result = torch::mm(torch::mm(torch::mm(torch::mm(torch::mm(tensor0, tensor1), tensor2), tensor3), tensor4),
                           tensor5));
  }
};

auto BM_EIGEN_CM = [](benchmark::State& state, int size) {
  srand(0);
  Eigen::MatrixXd matrix1 = Eigen::MatrixXd::Random(size, size);
  srand(1);
  Eigen::MatrixXd matrix2 = Eigen::MatrixXd::Random(size, size);
  srand(2);
  Eigen::MatrixXd matrix3 = Eigen::MatrixXd::Random(size, size);
  srand(3);
  Eigen::MatrixXd matrix4 = Eigen::MatrixXd::Random(size, size);
  srand(4);
  Eigen::MatrixXd matrix5 = Eigen::MatrixXd::Random(size, size);
  srand(5);
  Eigen::MatrixXd matrix6 = Eigen::MatrixXd::Random(size, size);

  Eigen::MatrixXd result = Eigen::MatrixXd::Random(size, size);
  for (auto _ : state)
  {
    benchmark::DoNotOptimize(result = matrix1 * matrix2 * matrix3 * matrix4 * matrix5 * matrix6);
  }
};

auto BM_ARRAYFIRE_CM = [](benchmark::State& state, int size, auto device) {
  af::setBackend(device);

  af::setSeed(0);
  af::array array0 = af::randu(size, size);
  af::setSeed(1);
  af::array array1 = af::randu(size, size);
  af::setSeed(2);
  af::array array2 = af::randu(size, size);
  af::setSeed(3);
  af::array array3 = af::randu(size, size);
  af::setSeed(4);
  af::array array4 = af::randu(size, size);
  af::setSeed(5);
  af::array array5 = af::randu(size, size);

  af::array result = af::randu(size, size);
  for (auto _ : state)
  {
    benchmark::DoNotOptimize(result = af::matmul(af::matmul(array0, array1, array2, array3), array4, array5));
  }
};

int main(int argc, char** argv)
{
  for (auto& test_input : { 2, 3, 4, 8, 16, 32, 64, 128, 256, 512 })
  {
    std::string name = "BM_PYTORCH_CPU_CM_Size_" + std::to_string(test_input);
    benchmark::RegisterBenchmark(name.c_str(), BM_PYTORCH_CM, test_input, torch::kCPU)
        ->UseRealTime()
        ->Unit(benchmark::TimeUnit::kMicrosecond);
  }
  for (auto& test_input : { 2, 3, 4, 8, 16, 32, 64, 128, 256, 512 })
  {
    std::string name = "BM_PYTORCH_GPU_CM_Size_" + std::to_string(test_input);
    benchmark::RegisterBenchmark(name.c_str(), BM_PYTORCH_CM, test_input, torch::kCUDA)
        ->UseRealTime()
        ->Unit(benchmark::TimeUnit::kMicrosecond);
  }
  for (auto& test_input : { 2, 3, 4, 8, 16, 32, 64, 128, 256, 512 })
  {
    std::string name = "BM_EIGEN_CM_Size_" + std::to_string(test_input);
    benchmark::RegisterBenchmark(name.c_str(), BM_EIGEN_CM, test_input)
        ->UseRealTime()
        ->Unit(benchmark::TimeUnit::kMicrosecond);
  }
  for (auto& test_input : { 2, 3, 4, 8, 16, 32, 64, 128, 256, 512 })
  {
    std::string name = "BM_ARRAYFIRE_CPU_CM_Size_" + std::to_string(test_input);
    benchmark::RegisterBenchmark(name.c_str(), BM_ARRAYFIRE_CM, test_input, AF_BACKEND_CPU)
        ->UseRealTime()
        ->Unit(benchmark::TimeUnit::kMicrosecond);
  }
  for (auto& test_input : { 2, 3, 4, 8, 16, 32, 64, 128, 256, 512 })
  {
    std::string name = "BM_ARRAYFIRE_CUDA_CM_Size_" + std::to_string(test_input);
    benchmark::RegisterBenchmark(name.c_str(), BM_ARRAYFIRE_CM, test_input, AF_BACKEND_CUDA)
        ->UseRealTime()
        ->Unit(benchmark::TimeUnit::kMicrosecond);
  }
  for (auto& test_input : { 2, 3, 4, 8, 16, 32, 64, 128, 256, 512 })
  {
    std::string name = "BM_ARRAYFIRE_OPENCL_CM_Size_" + std::to_string(test_input);
    benchmark::RegisterBenchmark(name.c_str(), BM_ARRAYFIRE_CM, test_input, AF_BACKEND_OPENCL)
        ->UseRealTime()
        ->Unit(benchmark::TimeUnit::kMicrosecond);
  }
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
}
