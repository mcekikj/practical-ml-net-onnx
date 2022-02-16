using System.IO;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.Http;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using System.Collections.Generic;
using System.Linq;

namespace LeadScoringONNXFunc
{
    public static class LeadScoringFunc
    {
        // https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime/
        // Microsoft.ML.OnnxRuntime -> Install-Package Microsoft.ML.OnnxRuntime -Version 1.10.0

        [FunctionName("LeadScoringFunc")]
        public static async Task<IActionResult> Run(
            [HttpTrigger(AuthorizationLevel.Function, "get", "post", Route = null)] HttpRequest req,
            ILogger log)
        {
            string requestBody = await new StreamReader(req.Body).ReadToEndAsync();
            dynamic data = JsonConvert.DeserializeObject(requestBody);

            // modelParameters is consist of all input features or body parameters sent via the POST body
            float parameter = data?.modelParameters.parameterName;

            // ONNX Model local path
            var onnxModelPath = @"C:\Users\TestUser\Model.onnx";

            // Converting the LeadScoring input object into an object of type Tensor<float> so that we can pass it directly into the ONNX model
            // new int[] { 1, number of inputFeatures }
            var inputTensor = new DenseTensor<float>(new float[] {
                                                         parameter,
                                                         },
                                                         new int[] { 1, 1 });

            // 'feature_input' named in time of creating the ONNX format 
            var features_input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<float>("feature_input", inputTensor) };

            // Initializing and running the inference session where the actual scoring is happening
            var inferenceSession = new InferenceSession(onnxModelPath);
            var sessionOutput = inferenceSession.Run(features_input);

            // Unwrapping and parsing the output result`s probabilities
            var rawResult = (DisposableNamedOnnxValue)sessionOutput.ToArray()[1];
            var onnxValue = (IDisposableReadOnlyCollection<DisposableNamedOnnxValue>)rawResult.Value;
            var probalities = (Dictionary<long, float>)onnxValue.ToArray()[0].Value;

            // Probabilities (ranks) indicating the current lead`s conversion
            var leadScoreNotConverted = probalities.Values.ToArray()[0].ToString();
            var leadScoreConverted = probalities.Values.ToArray()[1].ToString();

            var responseMessage = $"The probability lead score of converting this lead is {leadScoreConverted}, " +
                $"while not being converted is {leadScoreNotConverted}.";

            sessionOutput.Dispose();

            return new JsonResult(responseMessage);
        }
    }
}