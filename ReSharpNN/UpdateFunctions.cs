using System;

namespace ReSharpNN {

  public delegate float[] UpdateFunction(float[] value);

  public delegate float[] DiffFunction(float[] value);
  
  public static class UpdateFunctions {

    public static float[] Tanh(float[] value) {
      var res = new float[value.Length];
      for (var i = 0; i < res.Length; i++) res[i] = MathF.Tanh(value[i]);
      return res;
    }

    public static float[] Sigmoid(float[] value) {
      var res = new float[value.Length];
      for (var i = 0; i < res.Length; i++) res[i] = 1.0f / (1.0f + MathF.Exp(-value[i]));
      return res;
    }

    public static float[] ReLU(float[] value) {
      var res = new float[value.Length];
      for (var i = 0; i < res.Length; i++) res[i] = value[i] > 0.0f ? value[i] : 0.0f;
      return res;
    }

  }

}