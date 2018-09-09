using ReSharpNN.Utils;

using static System.MathF;

namespace ReSharpNN {

  public delegate float[] WeightsInitializeFunction(int sizeRow, int sizeColumn);
  
  public static class WeightsInitializeFunctions {
    private static readonly XorShift random = new XorShift();
    public static float ConnectiveRate { get; set; } = 0.5f;

    public static float[] AlwaysOne(int sizeRow, int sizeColumn) {
      var res = new float[sizeRow * sizeColumn];
      for (var i = 0; i < sizeRow; i++)
        for (var j = 0; j < sizeColumn; j++)
          res[i * sizeColumn + j] = 1.0f;
      return res;
    }

    public static float[] Uniform(int sizeRow, int sizeColumn) {
      var res = new float[sizeRow * sizeColumn];
      for (var i = 0; i < sizeRow; i++)
        for (var j = 0; j < sizeColumn; j++)
          res[i * sizeColumn + j] = random.RandFloat(-1.0f);
      return res;
    }

    public static float[] SparseUniform(int sizeRow, int sizeColumn) {
      var res = Uniform(sizeRow, sizeColumn);
      for (var i = 0; i < res.Length; i++)
        res[i] = random.RandFloat() < ConnectiveRate ? res[i] : 0.0f;
      return res;
    }

    public static float[] Normal(int sizeRow, int sizeColumn) {
      var res = new float[sizeRow * sizeColumn];
      for (var i = 0; i < sizeRow; i++)
        for (var j = 0; j < sizeColumn; j++)
          res[i * sizeColumn + j] =
            Sqrt(-2.0f * Log(random.RandFloat(1e-8f)) * Sin(2.0f * PI * random.RandFloat(1e-8f)));
      return res;
    }

    public static float[] Xavier(int sizeRow, int sizeColumn) {
      var res = Normal(sizeRow, sizeColumn);
      var std = 1.0f / Sqrt(sizeColumn);
      for (var i = 0; i < res.Length; i++)
        res[i] *= std;
      return res;
    }

    public static float[] He(int sizeRow, int sizeColumn) {
      var res = Normal(sizeRow, sizeColumn);
      var std = Sqrt(2.0f / sizeColumn);
      for (var i = 0; i < res.Length; i++)
        res[i] *= std;
      return res;
    }
  }

}