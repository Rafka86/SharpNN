using System.Diagnostics;
using System.Text;
using System.Threading.Tasks;

using SharpMKL;

using static SharpNN.ActivateDiffFunctions;

namespace SharpNN {

  internal class Layer {
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
    private readonly DiffFunction _diff;

    private readonly Layer _preLayer;
    internal float[] Weight { get; private set; }
    private readonly float[] _dw;
    private readonly Optimizer _optimizer;
    
    internal Layer(int size,
                   ActivateFunction updateFunction,
                   Layer preLayer,
                   WeightsInitializeFunction wInitFunction,
                   Optimizer optimizer) {
      Input = new float[size];
      Output = new float[size + 1];
      Output[size] = 1.0f;
      Delta = new float[size];

      Function = updateFunction;
      _diff = ChooseDiffFunction();

      IsInputLayer = preLayer == null;
      
      _preLayer = preLayer;
      if (preLayer == null) return;
      _optimizer = optimizer;
      Weight = wInitFunction(InputSize, _preLayer.OutputSize);
      _dw = new float[_preLayer.OutputSize * InputSize];
      
      DiffFunction ChooseDiffFunction() {
        switch (updateFunction.Method.Name) {
          case "Identity": return DiffIdentity;
          case "Tanh": return DiffTanh;
          case "Sigmoid": return DiffSigmoid;
          case "ReLU": return DiffReLU;
          case "LeakyReLU": return DiffLeakyReLU;
          default: return null;
        }
      }
    }

    internal void PrepareForTraining(int batchSize) => _optimizer.LearningRate /= batchSize;

    internal void SetOutputs(float[] values) {
      Parallel.For(0, values.Length, i => { Output[i] = values[i]; });
    }
    
    internal void Update() {
      Blas2.gemv(BlasLayout.RowMajor, BlasTranspose.NoTrans,
                 InputSize, _preLayer.OutputSize, 1.0f, Weight,
                 _preLayer.OutputSize, _preLayer.Output, 1, 0.0f, Input, 1);
      Blas1.copy(InputSize, Function(Input), 1, Output, 1);
    }

    internal void CalculationOutputLayerDelta(float[] teacher) {
      Parallel.For(0, Delta.Length, i => { Delta[i] = Output[i] - teacher[i]; });
      CalculationDeltaW();
    }

    internal void CalculationDelta(Layer postLayer) {
      var diff = _diff(Output);
      Parallel.For(0, Delta.Length, i => {
                                      Delta[i] = 0.0f;
                                      for (var k = 0; k < postLayer.Delta.Length; k++)
                                        Delta[i] += postLayer.Delta[k] * postLayer.Weight[OutputSize * k + i];
                                      Delta[i] *= diff[i];
                                    });
      if (!IsInputLayer)
        CalculationDeltaW();
    }
    
    internal void ClearDeltaW() {
      Parallel.For(0, _dw.Length, i => { _dw[i] = 0.0f; });
    }
    
    internal void CalculationDeltaW() {
      Parallel.For(0, InputSize, i => {
                                   for (var j = 0; j < _preLayer.OutputSize; j++)
                                     _dw[_preLayer.OutputSize* i + j] += Delta[i] * _preLayer.Output[j];
                                 });
    }

    internal void CalculationBiasDeltaW() {
      Parallel.For(0, InputSize,
                   i => {
                     _dw[_preLayer.OutputSize * i + _preLayer.InputSize]
                       += Delta[i] * _preLayer.Output[_preLayer.InputSize];
                   });
    }

    internal void ApplyDeltaW() {
      _optimizer.Update(Weight, _dw);
    }

    internal void PostProcess() {
      if (!_optimizer.NeedPostProcess) return;
      Weight = _optimizer.PostProcess();
    }

    internal string Dump(int dumpLevel) {
      var sb = new StringBuilder();
      sb.Clear();
      
      DumpLayer();
      DumpConnection();
      
      return sb.ToString();

      void DumpLayer() {
        sb.AppendLine("==================== Layer Infos ====================");
        sb.AppendLine($"Number of neurons     : {OutputSize.ToString()}");
        sb.AppendLine($"This layer's position : {(IsInputLayer ? "Input" : IsOutputLayer ? "Output" : "Middle")} Layer");
        sb.AppendLine($"This layer's function : {Function.Method.Name}");
        sb.AppendLine($"This layer's delta    : {_diff?.Method.Name ?? "None"}");

        if (dumpLevel > 1) {
          sb.AppendLine("-------------------- Activities ---------------------");
          for (var i = 0; i < Input.Length; i++)
            sb.AppendLine($"  Input[{i,4}] = {Input[i],-8: ##0.###;-##0.###}"
                        + $"\tUnit[{i,4}] = {Output[i],-8: ##0.###;-##0.###}"
                        + $"\tDelta[{i,4}] = {Delta[i],-8: ##0.###;-##0.###}");
          sb.AppendLine("-----------------------------------------------------");
        }
        
        sb.AppendLine("=====================================================");
      }

      void DumpConnection() {
        sb.AppendLine("================== Connection Infos =================");
        sb.AppendLine($"Number of pre layer's units  : {_preLayer.OutputSize.ToString()}");
        sb.AppendLine($"Number of post layer's units : {InputSize.ToString()}");
        sb.AppendLine($"Number of weights            : {Weight.Length.ToString()}");

        if (dumpLevel > 1) {
          sb.AppendLine("---------------------- Weights ----------------------");
          for (var i = 0; i < InputSize; i++)
            for (var j = 0; j < _preLayer.OutputSize; j++) {
              sb.Append($"  Weight[{i,4}, {j,4}] = {Weight[_preLayer.OutputSize* i + j]: ##0.0##;-##0.0##}");
              sb.AppendLine($"\tdW[{i,4}, {j,4}] = {_dw[_preLayer.OutputSize* i + j]: ##0.0##;-##0.0##}");
            }
          sb.AppendLine("-----------------------------------------------------");
        }
        
        sb.AppendLine("=====================================================");
      }
    }
  }

}