#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <iostream>

vector<string> split(string& input, char delimiter)
{
   using namespace std;
    istringstream stream(input);
    string field;
    vector<string> result;
    while (getline(stream, field, delimiter)) {
        result.push_back(field);
    }
    return result;
}


int wf_denoising_onnxruntime()
{
   // This macro use the trained model to apply to data using ONNX runtime.
   // To use ONNX runtime
   // % export LD_LIBRARY_PATH=../onnxruntime-linux-x64-1.4.0/lib:$LD_LIBRARY_PATH
   // % export ROOT_INCLUDE_PATH=$ROOT_INCLUDE_PATH:/home/uchiyama/MegAnalysis/ml/onnxruntime-linux-x64-1.4.0/include
 


   // ONNX model file
   string modelfile = "CDCHWfDenoising_356990_20201123_0.onnx";
   // string modelfile = "wf_denoising.onnx";

   // Layer names
   // input and output layer names must be identical to those defined when the model was built
   const char* input_names[] = {"input_1"};
   const char* output_names[] = {"conv1d_6"};

   // Model parameters
   static constexpr const int npoints_ = 1024;
   const Double_t scale = 5;
   const Double_t offset = 0.05;


   auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

   // Input and output arrays
   // std::array<float, npoints_> inputs{};
   // std::array<float, npoints_> outputs{};
   std::vector<float> inputs(npoints_);
   std::vector<float> outputs(npoints_);

   // Input and output tensors
   std::array<int64_t, 3> input_shape_{1, npoints_, 1};
   std::array<int64_t, 3> output_shape_{1, npoints_, 1};
   auto input_tensor_ = Ort::Value::CreateTensor<float>
         (memory_info, inputs.data(),
          inputs.size(), input_shape_.data(), input_shape_.size());
   auto output_tensor_ = Ort::Value::CreateTensor<float>
         (memory_info, outputs.data(), outputs.size(), 
          output_shape_.data(), output_shape_.size());


   // ONNX runtime session
   Ort::Env env;
   std::unique_ptr<Ort::Session> session_;
   try {
      session_ = std::make_unique<Ort::Session>(env, modelfile.c_str(), Ort::SessionOptions{nullptr});
   } catch (const Ort::Exception& exception) {
      std::cerr << exception.what() << std::endl;
      return 1;
   }
   
   std::vector<std::string> input_names2 = session_->GetInputNames();
   for (auto &&name: input_names2) {
      cout<<"NNNNN "<<name<<endl;
   }
   

   // load data
   ifstream ifs("./wf328469.csv");
   string line;
   std::vector<float> t(1024);
   while (getline(ifs, line)) {
      vector<string> strvec = split(line, ',');
      for (std::size_t i=0; i<strvec.size();i++){
         // Preprocess (normalization)
         inputs[i] = stof(strvec[i]) * scale;
         inputs[i] += offset * scale;
      }

      for (size_t i = 0; i < inputs.size(); i++) {
         t[i] = i;
      }

      // Apply the model (infer) to the data
      session_->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_, 1,
                   output_names, &output_tensor_, 1);


      for (size_t i = 0; i < outputs.size(); i++) {
         // Postprocess (revert the normalization)
         outputs[i] -= offset * scale;
         outputs[i] /= scale;
         inputs[i] -= offset * scale;
         inputs[i] /= scale;
      }
      


      TGraph *graphIn  = new TGraph(inputs.size(), t.data(), inputs.data());
      graphIn->Draw("AL");
      TGraph *graphOut  = new TGraph(inputs.size(), t.data(), outputs.data());
      graphOut->SetLineColor(kRed);
      graphOut->Draw("L");
      gPad->Update();
      gPad->WaitPrimitive();

   }
   return 0;
}
