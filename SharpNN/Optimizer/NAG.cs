using SharpMKL;

namespace SharpNN.Optimizer {

  public class NAG : Optimizer {
    public float MomentumValue { get; }
    private float[] _velocity;
    private float[] _weightMemory;
    
    public NAG(float learningRate = 0.01f, float momentum = 0.9f) {
      LearningRate = learningRate;
      MomentumValue = momentum;

      NeedPostProcess = true;
    }

    internal override void Update(float[] weight, float[] delta) {
      if (_weightMemory == null) Blas1.copy(weight.Length, weight, 1, out _weightMemory, 1);
      var velocity = _velocity ?? new float[weight.Length];
      Blas1.scal(velocity.Length, MomentumValue, velocity, 1);
      Blas1.axpy(velocity.Length, -LearningRate, delta, 1, velocity, 1);
      _velocity = velocity;
      Blas1.axpy(_weightMemory.Length, 1.0f, velocity, 1, _weightMemory, 1);
      Blas1.copy(_velocity.Length, _velocity, 1, out var ahead, 1);
      Blas1.scal(ahead.Length, MomentumValue, ahead, 1);
      Blas1.copy(_weightMemory.Length, _weightMemory, 1, weight, 1);
      Blas1.axpy(weight.Length, 1.0f, ahead, 1, weight, 1);
    }

    internal override Optimizer Clone() => new NAG(LearningRate, MomentumValue);

    internal override float[] PostProcess() => _weightMemory;
  }

}