using SharpNN;
using SharpNN.DataSet;

using static System.IO.Path;

using static SharpNN.ErrorFunctions;
using static SharpNN.ActivateFunctions;

namespace SharpNNTest {

  class Program {
    static void Main(string[] args) {
      var separator = DirectorySeparatorChar.ToString();
      var mnistPath = $".{separator}mnist-fashion{separator}";
      var mnist = new Mnist(mnistPath);
      var xor = new XorDataSet();
      //var imagePath = $"{mnistPath}png{separator}";
      //mnist.OutputImages(imagePath);

      //Network.Factory.DefalutOptimizer = new Adam();
      
      Network.Factory.New();
      Network.Factory.AddLayer(xor.InputDataSize, Identity);
      Network.Factory.AddLayer(100, ReLU);
      Network.Factory.AddLayer(xor.OutputDataSize, Identity);
      Network.Factory.SetErrorFunction(MeanSquared);
      Network.Factory.Create(out var network);

      Trainer.Training(network, xor, limitError: 1e-10f, printLog: true);
      Trainer.RegressionTest(network, xor);

      Network.Factory.New();
      Network.Factory.AddLayer(mnist.InputDataSize, Identity);
      Network.Factory.AddLayer(400, ReLU);
      Network.Factory.AddLayer(100, ReLU);
      Network.Factory.AddLayer(50, ReLU);
      Network.Factory.AddLayer(mnist.OutputDataSize, Softmax);
      Network.Factory.SetErrorFunction(CrossEntropy);
      Network.Factory.Create(out network);

      //Trainer.PreTraining(network, mnist, epoch: 4, batchSize: 30, printLog: true);
      Trainer.Training(network, mnist, epoch: 1, batchSize: 50, printLog: true);
      Trainer.ClusteringTest(network, mnist);
    }
  }

}