using SharpNN;
using SharpNN.DataSet;
using SharpNN.Optimizer;

using static System.IO.Path;

using static SharpNN.ErrorFunctions;
using static SharpNN.ActivateFunctions;
using static SharpNN.WeightsInitializeFunctions;

namespace SharpNNTest {

  class Program {
    static void Main(string[] args) {
      var separator = DirectorySeparatorChar.ToString();
      var mnistPath = $".{separator}mnist-fashion{separator}";
      var mnist = new Mnist(mnistPath);
      var xor = new XorDataSet();
      //var imagePath = $"{mnistPath}png{separator}";
      //mnist.OutputImages(imagePath);

      //Trainer.DefaultOptimizer = new AdaDelta();
      //Trainer.DefaultOptimizer = new RMSProp();
      //Trainer.DefaultOptimizer = new NAG();
      Trainer.DefaultOptimizer = new Adam();
      
      Network.Factory.New();
      Network.Factory.AddLayer(xor.InputDataSize, Identity);
      Network.Factory.AddLayer(100, ReLU);
      Network.Factory.AddConnection(He);
      Network.Factory.AddLayer(xor.OutputDataSize, Identity, false);
      Network.Factory.AddConnection(He);
      Network.Factory.SetErrorFunction(MeanSquared);
      Network.Factory.Create(out var network);

      Trainer.Training(network, xor, limitError: 1e-10f, printLog: true);
      Trainer.RegressionTest(network, xor);

      Network.Factory.New();
      Network.Factory.AddLayer(mnist.InputDataSize, Identity);
      Network.Factory.AddLayer(400, ReLU);
      Network.Factory.AddConnection(He);
      Network.Factory.AddLayer(100, ReLU);
      Network.Factory.AddConnection(He);
      Network.Factory.AddLayer(50, ReLU);
      Network.Factory.AddConnection(He);
      Network.Factory.AddLayer(10, ReLU);
      Network.Factory.AddConnection(He);
      Network.Factory.AddLayer(mnist.OutputDataSize, Softmax, false);
      Network.Factory.AddConnection(He);
      Network.Factory.SetErrorFunction(CrossEntropy);
      Network.Factory.Create(out network);

      //Trainer.PreTraining(network, mnist, epoch: 4, batchSize: 30, printLog: true);
      Trainer.Training(network, mnist, epoch: 5, batchSize: 50, printLog: true);
      Trainer.ClusteringTest(network, mnist);
    }
  }

}