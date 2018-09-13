using System;

using ReSharpNN;
using ReSharpNN.DataSet;

using static System.IO.Path;

using static ReSharpNN.ErrorFunctions;
using static ReSharpNN.UpdateFunctions;
using static ReSharpNN.WeightsInitializeFunctions;

namespace ReSharpNNTest {

  class Program {
    static void Main(string[] args) {
      //var separator = DirectorySeparatorChar.ToString();
      //var mnistPath = $".{separator}mnist{separator}";
      //var mnist = new Mnist(mnistPath);
      var xor = new XorDataSet();
      //var imagePath = $"{mnistPath}png{separator}";
      //mnist.OutputImages(imagePath);
      
      Network.Factory.New();
      Network.Factory.AddLayer(2, Identity);
      Network.Factory.AddLayer(200, ReLU);
      Network.Factory.AddConnection(He);
      Network.Factory.AddLayer(1, ReLU, false);
      Network.Factory.AddConnection(He);
      Network.Factory.SetErrorFunction(MeanSquared);
      Network.Factory.Create(out var network);

      Trainer.Training(network, xor, 1e-5f, true);
      Trainer.RegressionTest(network, xor);
    }
  }

}