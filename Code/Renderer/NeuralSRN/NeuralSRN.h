#pragma once
#include <torch/script.h> 

namespace NeuralSRN{
  class TracedNeuralModel{
  private:
    torch::jit::script::Module module;
  public:
    TracedNeuralModel (std::string modelLocation);
    at::Tensor at_phys_pos (torch::Tensor queryPoints);
    at::Tensor grad_at_phys_pos (torch::Tensor queryPoints);
    void to(torch::Device device);
  };
}