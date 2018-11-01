using System.Threading.Tasks;

using static SharpNN.ActivateDiffFunctions;

namespace SharpNN {

  public abstract class Layer {
    internal float[] Input { get; }
    internal float[] Output { get; }
    internal float[] Delta { get; }

    /// <summary>
    /// The number of units including bias.
    /// </summary>
    internal int OutputSize => Output.Length;
    /// <summary>
    /// The number of units excluding bias.
    /// </summary>
    internal int InputSize => Input.Length;
    
    internal bool IsInputLayer { get; }
    internal bool IsOutputLayer { get; set; } = true;
    
    internal ActivateFunction Function { get; }
    private protected readonly DiffFunction _diff;

    private protected readonly Layer _preLayer;
    internal float[] Weight { get; private protected set; }
    private protected float[] dW { get; set; }
    private readonly Optimizer _optimizer;

    internal Layer(int size, ActivateFunction func, Layer pre, Optimizer opt) {
      Input = new float[size];
      Output = new float[size + 1];
      Output[size] = 1.0f;
      Delta = new float[size];

      Function = func;
      _diff    = ChooseDiffFunction();

      IsInputLayer = pre == null;

      _preLayer = pre;
      _optimizer = opt;

      DiffFunction ChooseDiffFunction() {
        switch (func.Method.Name) {
          case "Identity":  return DiffIdentity;
          case "Tanh":      return DiffTanh;
          case "Sigmoid":   return DiffSigmoid;
          case "ReLU":      return DiffReLU;
          case "LeakyReLU": return DiffLeakyReLU;
          default:          return null;
        }
      }
    }
    
    internal virtual void SetOutputs(float[] values) {
      Parallel.For(0, values.Length, i => { Output[i] = values[i];});
    }
    
    internal virtual void ClearDeltaW() {
      Parallel.For(0, dW.Length, i => { dW[i] = 0.0f; });
    }
    
    internal virtual void ApplyDeltaW() {
      _optimizer.Update(Weight, dW);
    }
    
    internal virtual void PostProcess() {
      if (!_optimizer.NeedPostProcess) return;
      Weight = _optimizer.PostProcess();
    }

    internal abstract void Update();
    internal abstract void CalculationOutputLayerDelta(float[] teacher);
    internal abstract void CalculationDelta(Layer postLayer);
    internal abstract string Dump(int dumpLevel);
  }

}