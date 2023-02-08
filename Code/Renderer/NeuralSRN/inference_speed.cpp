#include "NeuralSRN.cpp"
#include <torch/script.h>
#include <chrono>

using namespace std;
using namespace std::chrono;
using namespace NeuralSRN;

typedef high_resolution_clock Clock;
typedef Clock::time_point ClockTime;

void printExecutionTime(ClockTime start_time, ClockTime end_time);

torch::Device device(torch::kCUDA);

int main(int argc, const char* argv[]) {

    string location = "D:\\GitHub\\DistributedINR\\SavedModels\\Nyx_AMGSRN_small_convert_to_pytorch\\traced_model.pt";
    TracedNeuralModel model = TracedNeuralModel(location);
    model.to(device);

    torch::Tensor inputs;
    torch::Tensor outputs;
    
    int batchSize = 1000000;
    int numRuns = 1000;

    inputs = torch::rand({batchSize, 3}, device);

    ClockTime start_time = Clock::now();
    for(int i = 0; i < numRuns; i++){
        outputs = model.at_phys_pos(inputs);
    }
    ClockTime end_time = Clock::now();

    printExecutionTime(start_time, end_time);
    return 0;
}

void printExecutionTime(ClockTime start_time, ClockTime end_time)
{
    auto execution_time_ns = duration_cast<nanoseconds>(end_time - start_time).count();
    auto execution_time_ms = duration_cast<microseconds>(end_time - start_time).count();
    auto execution_time_sec = duration_cast<seconds>(end_time - start_time).count();
    auto execution_time_min = duration_cast<minutes>(end_time - start_time).count();
    auto execution_time_hour = duration_cast<hours>(end_time - start_time).count();

    cout << "\nExecution Time: ";
    if(execution_time_hour > 0)
    cout << "" << execution_time_hour << " Hours, ";
    if(execution_time_min > 0)
    cout << "" << execution_time_min % 60 << " Minutes, ";
    if(execution_time_sec > 0)
    cout << "" << execution_time_sec % 60 << " Seconds, ";
    if(execution_time_ms > 0)
    cout << "" << execution_time_ms % long(1E+3) << " MicroSeconds, ";
    //if(execution_time_ns > 0)
    //cout << "" << execution_time_ns % long(1E+6) << " NanoSeconds, ";
}