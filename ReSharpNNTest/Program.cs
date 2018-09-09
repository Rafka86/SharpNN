using System;

using ReSharpNN.DataSet;

using static System.IO.Path;

using static ReSharpNN.Network.Factory;
using static ReSharpNN.UpdateFunctions;
using static ReSharpNN.WeightsInitializeFunctions;

namespace ReSharpNNTest {

  class Program {
    static void Main(string[] args) {
      var separator = DirectorySeparatorChar.ToString();
      var mnistPath = $".{separator}mnist{separator}";
      var mnist = new Mnist(mnistPath);
      var xor = new XorDataSet();
      //var imagePath = $"{mnistPath}png{separator}";
      //mnist.OutputImages(imagePath);
      
      New();
      AddLayer(2, Identity, true);
      AddLayer(20, ReLU, true);
      AddConnection(SparseUniform);
      AddLayer(1, Identity);
      AddConnection(He);
      var network = Create();

      foreach (var data in xor.TrainingData()) {
        network.ForwardPropagation(data.Input);
        //Console.WriteLine(network.Dump(2));
        Console.WriteLine(string.Join(' ', network.Output));
      }
    }
  }

}