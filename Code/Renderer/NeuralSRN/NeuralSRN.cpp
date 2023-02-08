#include "NeuralSRN.h"
#include <torch/script.h>

namespace NeuralSRN{
  TracedNeuralModel::TracedNeuralModel (std::string modelLocation) {
    try {
      // Deserialize the ScriptModule from a file using torch::jit::load().
      module = torch::jit::load(modelLocation);
    }
    catch (const c10::Error& e) {
      std::cerr << "error loading the model\n";
    }
  }
  at::Tensor TracedNeuralModel::at_phys_pos (torch::Tensor queryPoints) {
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(queryPoints);
    at::Tensor output = module.forward(inputs).toTensor();
    return output;
  }
  at::Tensor TracedNeuralModel::grad_at_phys_pos (torch::Tensor queryPoints) {
    queryPoints.requires_grad_(true);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(queryPoints);

    at::Tensor output = module.get_method("grad_at")(inputs).toTensor();
    return output;
  }
  void TracedNeuralModel::to(torch::Device device){
    module.to(device);
  }
}