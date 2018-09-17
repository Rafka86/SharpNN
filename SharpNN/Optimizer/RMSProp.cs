using System;
using System.Threading.Tasks;

using SharpMKL;

namespace SharpNN.Optimizer {

  public class RMSProp : Optimizer {
    public float RememberRate { get; }
    private float[] _v;
    
    public RMSProp(float learningRate = 0.01f, float rememberRate = 0.99f) {
      LearningRate = learningRate;
      RememberRate = rememberRate;
    }
    
    internal override void Update(float[] weight, float[] delta) {
      var v = _v ?? new float[delta.Length];
      var sq = new float[delta.Length];
      Parallel.For(0, sq.Length, i => { sq[i] = delta[i] * delta[i]; });
      Blas1.scal(v.Length, RememberRate, v, 1);
      Blas1.axpy(sq.Length, 1.0f - RememberRate, sq, 1, v, 1);
      Parallel.For(0, weight.Length, i => { weight[i] -= LearningRate / (MathF.Sqrt(v[i]) + 1e-8f) * delta[i]; });
      _v = v;
    }

    internal override Optimizer Clone() => new RMSProp(LearningRate, RememberRate);

    internal override float[] PostProcess() {
      throw new NotImplementedException();
    }
  }

}