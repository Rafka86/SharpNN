using System;
using System.Threading.Tasks;

using SharpMKL;

namespace SharpNN.Optimizer {

  public class AdaDelta : Optimizer {
    public float RememberRate { get; }

    private float[] _v;
    private float[] _s;
    private float[] _dw;

    private const float Eps = 1e-8f;

    public AdaDelta(float rememberRate = 0.99f) => RememberRate = rememberRate;
    
    internal override void Update(float[] weight, float[] delta) {
      var v = _v ?? new float[delta.Length];
      var s = _s ?? new float[delta.Length];
      var dw = _dw ?? new float[delta.Length];

      var sq = new float[delta.Length];
      Parallel.For(0, delta.Length, i => {
                                      sq[i] = delta[i] * delta[i];
                                      dw[i] *= dw[i];
                                    });
      Blas1.scal(v.Length, RememberRate, v, 1);
      Blas1.scal(s.Length, RememberRate, s, 1);
      Blas1.axpy(v.Length, 1.0f - RememberRate, sq, 1, v, 1);
      Blas1.axpy(v.Length, 1.0f - RememberRate, dw, 1, s, 1);
      Parallel.For(0, delta.Length, i => {
                                      dw[i] = -MathF.Sqrt((s[i] + Eps) / (v[i] + Eps)) * delta[i];
                                      weight[i] += dw[i];
                                    });
      _v = v;
      _s = s;
      _dw = dw;
    }

    internal override Optimizer Clone() => new AdaDelta(RememberRate);

    internal override float[] PostProcess() {
      throw new System.NotImplementedException();
    }
  }

}