using System;
using System.Linq;

namespace ReSharpNN {

  public delegate float[] UpdateFunction(float[] values);

  public delegate float[] DiffFunction(float[] values);
  
  public static class UpdateFunctions {
    public static float[] Identity(float[] values) {
      var res = new float[values.Length];
      for (var i = 0; i < res.Length; i++) res[i] = values[i];
      return res;
    }
    
    public static float[] Tanh(float[] values) {
      var res = new float[values.Length];
      for (var i = 0; i < res.Length; i++) res[i] = MathF.Tanh(values[i]);
      return res;
    }

    public static float[] Sigmoid(float[] values) {
      var res = new float[values.Length];
      for (var i = 0; i < res.Length; i++) res[i] = 1.0f / (1.0f + MathF.Exp(-values[i]));
      return res;
    }

    public static float[] ReLU(float[] values) {
      var res = new float[values.Length];
      for (var i = 0; i < res.Length; i++) res[i] = values[i] > 0.0f ? values[i] : 0.0f;
      return res;
    }

    public static float[] Softmax(float[] values) {
      var res = new float[values.Length];
      var exp = values.Select(MathF.Exp).ToArray();
      var sum = exp.Sum();
      for (var i = 0; i < res.Length; i++)
        res[i] = exp[i] / sum;
      return res;
    }
  }

}