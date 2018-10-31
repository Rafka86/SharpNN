using System;
using System.Threading.Tasks;

using SharpMKL;

namespace SharpNN {

  public class AdaGrad : Optimizer {
    private float[] _v;
    
    public AdaGrad(float learningRate = 0.01f) => LearningRate = learningRate;
    
    internal override void Update(float[] weight, float[] delta) {
      var velocity = _v ?? new float[delta.Length];
      var sq = new float[delta.Length];
      Parallel.For(0, delta.Length, i => { sq[i] = delta[i] * delta[i]; });
      Blas1.axpy(velocity.Length, 1.0f, sq, 1, velocity, 1);
      Parallel.For(0, weight.Length,
                   i => { weight[i] -= LearningRate / (MathF.Sqrt(velocity[i]) + 1e-8f) * delta[i]; });
      _v = velocity;
    }

    internal override Optimizer Clone() => new AdaGrad(LearningRate);

    internal override float[] PostProcess() {
      throw new NotImplementedException();
    }
  }

}