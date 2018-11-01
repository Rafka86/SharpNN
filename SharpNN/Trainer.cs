using SharpMKL;

using static System.Console;

namespace SharpNN {

  public static class Trainer {
    public static void Training(Network network, DataSet.DataSet data,
                                int epoch = 1, int batchSize = 1, bool printLog = false) {
      var trainingDataSize = data.TrainingDataSize;
      var iterationSize    = trainingDataSize / batchSize;
      for (var i = 0; i < epoch; i++) {
        WriteLine($"Epoch {i}");
        if (printLog) Write("Error : ");
        for (var j = 0; j < iterationSize; j++) {
          network.ClearDeltaW();
          var error = 0.0f;
          foreach (var datum in data.MiniBatch(batchSize)) {
            network.SetInputs(datum.Input);
            network.ForwardPropagation();
            if (printLog) error += network.Error(datum.Output);
            network.BackPropagation(datum.Output);
          }
          network.UpdateWeights();
          if (printLog) Write($"\rError : {error / batchSize: ##0.00000;-##0.00000}");
        }
        if (printLog) WriteLine();
      }
      network.FinishLearning();
    }

    public static void Training(Network network, DataSet.DataSet data,
                                float limitError = 1e-5f, bool printLog = false) {
      var error = float.MaxValue;
      var it    = 0;
      while (error > limitError) {
        if (printLog) WriteLine($"Epoch {it++}");
        error = 0.0f;
        network.ClearDeltaW();
        foreach (var datum in data.TrainingData()) {
            network.SetInputs(datum.Input);
            network.ForwardPropagation();
            error += network.Error(datum.Output);
            network.BackPropagation(datum.Output);
        }
        network.UpdateWeights();
        error /= data.TrainingDataSize;
        if (printLog) WriteLine($"Error : {error: ##0.00000;-##0.00000}");
      }
      network.FinishLearning();
    }

    public static void RegressionTest(Network network, DataSet.DataSet data) {
      foreach (var datum in data.TestData()) {
        WriteLine($"Case : {string.Join(' ', datum.Input)}");
        network.SetInputs(datum.Input);
        network.ForwardPropagation();
        WriteLine($"Output : {string.Join(' ', network.Output)}");
      }
    }

    public static void ClusteringTest(Network network, DataSet.DataSet data) {
      var correct = 0.0f;
      var count = 0.0f;
      WriteLine("Testing.");
      Write($"Success Rate : {0.0f:##0.00%}");
      foreach (var datum in data.TestData()) {
        count += 1.0f;
        network.SetInputs(datum.Input);
        network.ForwardPropagation();
        var maxIdx = Blas1.iamax(network.Output.Length, network.Output, 1);
        if (maxIdx == Blas1.iamax(datum.Output.Length, datum.Output, 1)) correct += 1.0f;
        Write($"\rSuccess Rate : {correct / count:##0.00%}");
      }
      WriteLine();
    }
  }

}