using System;
using System.Linq;

namespace SharpNN {

  public delegate float ErrorFunction(float[] output, float[] teacher);

  public static class ErrorFunctions {
    public static float MeanSquared(float[] output, float[] teacher) {
      var sum = output.Zip(teacher, (o, t) => (o - t) * (o - t)).Sum();
      return sum * 0.5f;
    }
    public static float CrossEntropy(float[] output, float[] teacher) {
      var sum = output.Zip(teacher, (o, t) => t * MathF.Log(o > 1e-8f ? o : 1e-8f)).Sum();
      return -sum;
    }
  }

}