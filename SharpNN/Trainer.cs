using System;
using System.Collections.Generic;

namespace SharpNN {

  public static class Trainer {
    public static float LearningRate { get; set; } = 0.01f;

    public static void Training(Network network, DataSet.DataSet data, int epoch = 1, int batchSize = 1,
                                bool    printLog = false) {
      var trainingDataSize = data.TrainingDataSize;
      var iterationSize    = trainingDataSize / batchSize;
      for (var i = 0; i < epoch; i++) {
        Console.WriteLine($"Epoch {i}");
        if (printLog) Console.Write("Error : ");
        for (var j = 0; j < iterationSize; j++) {
          network.ClearDeltaW();
          var error = 0.0f;
          foreach (var datum in data.MiniBatch(batchSize)) {
            network.SetInputs(datum.Input);
            network.ForwardPropagation();
            error += network.Error(datum.Output);
            network.BackPropagation(datum.Output);
          }

          network.UpdateWeights(LearningRate);
          if (printLog) Console.Write($"\rError : {error / batchSize}");
        }

        if (printLog) Console.WriteLine();
      }
    }

    public static void Training(Network network, DataSet.DataSet data, float limitError = 1e-5f,
                                bool    printLog = false) {
      var error = float.MaxValue;
      var it    = 0;
      while (error > limitError) {
        Console.WriteLine($"Epoch {it++}");
        error = 0.0f;
        foreach (var datum in data.TrainingData()) {
            network.SetInputs(datum.Input);
            network.ForwardPropagation();
            error += network.Error(datum.Output);
            network.BackPropagation(datum.Output);
        }
        network.UpdateWeights(LearningRate);
        error /= data.TrainingDataSize;
        if (printLog) Console.WriteLine($"Error : {error}");
      }
    }

    public static void RegressionTest(Network network, DataSet.DataSet data) {
      foreach (var datum in data.TestData()) {
        Console.WriteLine($"Case : {string.Join(' ', datum.Input)}");
        network.SetInputs(datum.Input);
        network.ForwardPropagation();
        Console.WriteLine($"Output : {string.Join(' ', network.Output)}");
      }
    }

    public static void ClusteringTest(Network network, DataSet.DataSet data) {
      var correct = 0.0f;
      var count = 0.0f;
      Console.WriteLine("Testing.");
      Console.Write($"Success Rate : {0.0f:  0.00%}");
      foreach (var datum in data.TestData()) {
        count += 1.0f;
        network.SetInputs(datum.Input);
        network.ForwardPropagation();
        var maxIdx = GetMaxIndex(network.Output);
        if (maxIdx == GetMaxIndex(datum.Output)) correct += 1.0f;
        Console.Write($"\rSuccess Rate : {correct / count:  0.00%}");
      }

      int GetMaxIndex(IReadOnlyList<float> values) {
        var res = 0;
        var max = float.MinValue;
        for (var i = 0; i < values.Count; i++) {
          if (max >= values[i]) continue;
          res = i;
          max = values[i];
        }
        return res;
      }
    }
  }

}