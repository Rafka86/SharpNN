using SharpMKL;

namespace SharpNN.Optimizer {

  public class Momentum : Optimizer {
    public float MomentumValue { get; }
    private float[] _velocity;
    
    public Momentum(float learningRate = 0.01f, float momentum = 0.9f) {
      LearningRate = learningRate;
      MomentumValue = momentum;
    }

    internal override void Update(float[] weight, float[] delta) {
      var velocity = _velocity ?? new float[delta.Length];
      Blas1.scal(velocity.Length, MomentumValue, velocity, 1);
      Blas1.axpy(velocity.Length, -LearningRate, delta, 1, velocity, 1);
      _velocity = velocity;
      Blas1.axpy(weight.Length, 1.0f, velocity, 1, weight, 1);
    }

    internal override Optimizer Clone() => new Momentum(LearningRate, MomentumValue);
    
    internal override float[] PostProcess() {
      throw new System.NotImplementedException();
    }
  }

}