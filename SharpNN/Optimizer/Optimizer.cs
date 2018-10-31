namespace SharpNN {

  public abstract class Optimizer {
    public float LearningRate { get; set; } = 0.01f;
    internal bool NeedPostProcess { get; private protected set; }

    internal abstract void Update(float[] weight, float[] delta);
    internal abstract Optimizer Clone();
    internal abstract float[] PostProcess();
  }

}