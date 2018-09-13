using static System.Console;

namespace ReSharpNN {

  public static class Trainer {
    public static float LearningRate { get; set; } = 0.5f;

    public static void Training(Network network, DataSet.DataSet data, int epoch = 1, int batchSize = 1,
                                bool    printLog = false) {
      var trainingDataSize = data.TrainingDataSize;
      var iterationSize    = trainingDataSize / batchSize;
      for (var i = 0; i < epoch; i++) {
        WriteLine($"Epoch {i}");
        if (printLog) Write("Error : ");
        var error = 0.0f;
        for (var j = 0; j < iterationSize; j++) {
          network.ClearDeltaW();
          foreach (var datum in data.MiniBatch(batchSize)) {
            network.SetInputs(datum.Input);
            network.ForwardPropagation();
            error += network.Error(datum.Output);
            network.BackPropagation(datum.Output);
          }

          network.UpdateWeights(LearningRate);
          if (printLog) Write($"\rError : {error / batchSize}");
        }

        if (printLog) WriteLine();
      }
    }

    public static void Training(Network network, DataSet.DataSet data, float limitError = 1e-5f,
                                bool    printLog = false) {
      var error = float.MaxValue;
      var it    = 0;
      while (error > limitError) {
        WriteLine($"Epoch {it++}");
        error = 0.0f;
        foreach (var datum in data.TrainingData()) {
            network.SetInputs(datum.Input);
            network.ForwardPropagation();
            error += network.Error(datum.Output);
            network.BackPropagation(datum.Output);
        }
        network.UpdateWeights(LearningRate);
        error /= data.TrainingDataSize;
        if (printLog) WriteLine($"Error : {error}");
      }
    }

    public static void RegressionTest(Network network, DataSet.DataSet data) {
      foreach (var datum in data.TestData()) {
        WriteLine($"Case : {string.Join(' ', datum.Input)}");
        network.SetInputs(datum.Input);
        network.ForwardPropagation();
        WriteLine($"Output : {string.Join(' ', network.Output)}");
      }
    }
  }

}