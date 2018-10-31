using SharpMKL;

namespace SharpNN {

  public class SGD : Optimizer {
    public SGD(float learningRate = 0.01f) => LearningRate = learningRate;

    internal override void Update(float[] weight, float[] delta) {
      Blas1.axpy(weight.Length, -LearningRate, delta, 1, weight, 1);
    }

    internal override Optimizer Clone() => new SGD(LearningRate);
    
    internal override float[] PostProcess() {
      throw new System.NotImplementedException();
    }
  }

}