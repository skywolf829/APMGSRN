#include <torch/script.h> // One-stop header.
#include <iostream>

class TracedNeuralModel {
  torch::jit::script::Module module;
  public:
    TracedNeuralModel(std::string);
    at::Tensor at_phys_pos (torch::Tensor);
    at::Tensor grad_at_phys_pos (torch::Tensor);
};
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

/*
int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: model <path-to-exported-script-module>\n";
    return -1;
  }
  

  TracedNeuralModel model = TracedNeuralModel(argv[1]);
  std::cout << "Loaded model.\n";

  // Create a vector of inputs.
  torch::Tensor inputs = torch::tensor({{-0.3,0.1,0.2}});
  torch::Tensor batch_inputs = torch::tensor({{-0.3,0.1,0.2},{0.0,0.0,0.0}});

  // Execute the model and turn its output into a tensor.
  torch::Tensor output = model.at_phys_pos(inputs);
  torch::Tensor grad_output = model.grad_at_phys_pos(inputs);

  torch::Tensor batch_output = model.at_phys_pos(batch_inputs);
  torch::Tensor batch_grad_output = model.grad_at_phys_pos(batch_inputs);
  
  std::cout << "Output: " << output << '\n';
  std::cout << "Grad output: " << grad_output << '\n';
  std::cout << "Batch output: " << batch_output << '\n';
  std::cout << "Batch grad output: " << batch_grad_output << '\n';
}
*/