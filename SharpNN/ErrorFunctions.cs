using System;

namespace SharpNN {

  public delegate float ErrorFunction(float[] output, float[] teacher);

  public static class ErrorFunctions {
    public static float MeanSquared(float[] output, float[] teacher) {
      var sum = 0.0f;
      for (var i = 0; i < output.Length; i++) sum += (output[i] - teacher[i]) * (output[i] - teacher[i]);
      return sum * 0.5f;
    }
    public static float CrossEntropy(float[] output, float[] teacher) {
      var sum = 0.0f;
      for (var i = 0; i < output.Length; i++) sum += teacher[i] * MathF.Log(output[i] > 1e-8f ? output[i] : 1e-8f);
      return -sum;
    }
  }

}