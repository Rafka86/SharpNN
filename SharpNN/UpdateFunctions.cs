using System;
using System.Linq;
using System.Threading.Tasks;

namespace SharpNN {

  public delegate float[] ActivateFunction(float[] values);

  public delegate float[] DiffFunction(float[] values);
  
  public static class ActivateFunctions {
    internal static float Alpha = 0.05f;

    public static float LeakyReLUGrad {
      set {
        Alpha = value;
        ActivateDiffFunctions.Alpha = value;
      }
    }
    
    public static float[] Identity(float[] values) {
      var res = new float[values.Length];
      Parallel.For(0, res.Length, i => { res[i] = values[i]; });
      return res;
    }
    
    public static float[] Tanh(float[] values) {
      var res = new float[values.Length];
      Parallel.For(0, res.Length, i => { res[i] = MathF.Tanh(values[i]); });
      return res;
    }

    public static float[] Sigmoid(float[] values) {
      var res = new float[values.Length];
      Parallel.For(0, res.Length, i => { res[i] = 1.0f / (1.0f + MathF.Exp(-values[i])); });
      return res;
    }

    public static float[] ReLU(float[] values) {
      var res = new float[values.Length];
      Parallel.For(0, res.Length, i => { res[i] = values[i] > 0.0f ? values[i] : 0.0f; });
      return res;
    }

    public static float[] LeakyReLU(float[] values) {
      var res = new float[values.Length];
      Parallel.For(0, res.Length, i => { res[i] = values[i] > 0.0f ? values[i] : Alpha * values[i]; });
      return res;
    }
    
    public static float[] Softmax(float[] values) {
      var res = new float[values.Length];
      var max = values.Max();
      var exp = values.AsParallel().Select(val => MathF.Exp(val - max)).ToArray();
      var sum = exp.Sum();
      Parallel.For(0, res.Length, i => { res[i] = exp[i] / sum; });
      return res;
    }
  }

  public static class ActivateDiffFunctions {
    internal static float Alpha = 0.05f;

    public static float LeakyReLUGrad {
      set {
        Alpha = value;
        ActivateFunctions.Alpha = value;
      }
    }
    
    public static float[] DiffIdentity(float[] values) {
      var res = new float[values.Length];
      Parallel.For(0, res.Length, i => { res[i] = 1.0f; });
      return res;
    }

    public static float[] DiffTanh(float[] values) {
      var res = new float[values.Length];
      Parallel.For(0, res.Length, i => { res[i] = 1.0f - values[i] * values[i]; });
      return res;
    }

    public static float[] DiffSigmoid(float[] values) {
      var res = new float[values.Length];
      Parallel.For(0, res.Length, i => { res[i] = values[i] * (1.0f - values[i]); });
      return res;
    }

    public static float[] DiffReLU(float[] values) {
      var res = new float[values.Length];
      Parallel.For(0, res.Length, i => { res[i] = values[i] > 0.0f ? 1.0f : 0.0f; });
      return res;
    }

    public static float[] DiffLeakyReLU(float[] values) {
      var res = new float[values.Length];
      Parallel.For(0, res.Length, i => { res[i] = values[i] > 0.0f ? 1.0f : Alpha; });
      return res;
    }
  }

}