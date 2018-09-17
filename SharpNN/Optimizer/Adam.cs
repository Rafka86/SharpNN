using System;
using System.Threading.Tasks;

using SharpMKL;

namespace SharpNN.Optimizer {

  public class Adam : Optimizer {
    public float Beta1 { get; }
    public float Beta2 { get; }

    private float[] _m;
    private float[] _v;
    private float _beta1 = 1.0f;
    private float _beta2 = 1.0f;
    
    private const float Eps = 1e-8f;
    
    public Adam(float learningRate = 0.01f, float beta1 = 0.9f, float beta2 = 0.999f) {
      LearningRate = learningRate;
      Beta1 = beta1;
      Beta2 = beta2;
    }
    
    internal override void Update(float[] weight, float[] delta) {
      var m = _m ?? new float[delta.Length];
      var v = _v ?? new float[delta.Length];
      var sq = new float[delta.Length];
      Parallel.For(0, sq.Length, i => { sq[i] = delta[i] * delta[i]; });
      _beta1 *= Beta1;
      _beta2 *= Beta2;
      
      Blas1.scal(m.Length, Beta1, m, 1);
      Blas1.scal(v.Length, Beta2, v, 1);
      Blas1.axpy(m.Length, 1.0f - Beta1, delta, 1, m, 1);
      Blas1.axpy(v.Length, 1.0f - Beta2,    sq, 1, v, 1);

      Parallel.For(0, weight.Length, i => {
                                     var mHat = m[i] / (1.0f - _beta1);
                                     var vHat = v[i] / (1.0f - _beta2);
                                     weight[i] -= LearningRate / (MathF.Sqrt(vHat) + Eps) * mHat;
                                   });
      
      _m = m;
      _v = v;
    }

    internal override Optimizer Clone() => new Adam(LearningRate, Beta1, Beta2);

    internal override float[] PostProcess() {
      throw new NotImplementedException();
    }
  }

}