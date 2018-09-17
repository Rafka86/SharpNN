using System;
using System.Linq;
using System.Threading.Tasks;

using SharpNN.Utils;

namespace SharpNN {

  public delegate float[] WeightsInitializeFunction(int sizeRow, int sizeColumn);
  
  public static class WeightsInitializeFunctions {
    private static readonly XorShift Random = new XorShift();
    public static float ConnectiveRate { get; set; } = 0.5f;

    public static float[] AlwaysOne(int sizeRow, int sizeColumn) {
      var res = new float[sizeRow * sizeColumn];
      Parallel.For(0, res.Length, i => { res[i] = 1.0f; });
      return res;
    }

    public static float[] Uniform(int sizeRow, int sizeColumn) {
      var res = new float[sizeRow * sizeColumn];
      Parallel.For(0, res.Length, i => { res[i] = Random.RandFloat(-1.0f); });
      return res;
    }

    public static float[] SparseUniform(int sizeRow, int sizeColumn) {
      var res = new float[sizeRow * sizeColumn];
      Parallel.For(0, res.Length,
                   i => { res[i] = Random.RandFloat() < ConnectiveRate ? Random.RandFloat(-1.0f) : 0.0f; });
      return res;
    }

    public static float[] Normal(int sizeRow, int sizeColumn) {
      var res = new float[sizeRow * sizeColumn];
      Parallel.For(0, res.Length,
                   i => {
                     res[i] = MathF.Sqrt(-2.0f          * MathF.Log(Random.RandFloat(1e-5f))) *
                              MathF.Sin(2.0f * MathF.PI * Random.RandFloat(1e-5f));
                   });
      return res;
    }

    public static float[] Xavier(int sizeRow, int sizeColumn) {
      var tmp = Normal(sizeRow, sizeColumn);
      var std = 1.0f / MathF.Sqrt(sizeColumn);
      return tmp.AsParallel().Select(n => n * std).ToArray();
    }

    public static float[] He(int sizeRow, int sizeColumn) {
      var tmp = Normal(sizeRow, sizeColumn);
      var std = MathF.Sqrt(2.0f / sizeColumn);
      return tmp.AsParallel().Select(n => n * std).ToArray();
    }
  }

}