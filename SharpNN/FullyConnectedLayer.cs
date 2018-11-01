using System.Text;
using System.Threading.Tasks;

using SharpMKL;

namespace SharpNN {

  internal class FullyConnectedLayer : Layer {
    internal FullyConnectedLayer(int                       size,
                                 ActivateFunction          updateFunction,
                                 Layer                     preLayer,
                                 WeightsInitializeFunction wInitFunction,
                                 Optimizer                 optimizer)
      : base(size, updateFunction, preLayer, optimizer) {
      if (preLayer == null) return;
      Weight = wInitFunction(InputSize, _preLayer.OutputSize);
      dW     = new float[_preLayer.OutputSize * InputSize];
    }

    internal override void Update() {
      Blas2.gemv(BlasLayout.RowMajor, BlasTranspose.NoTrans,
                 InputSize, _preLayer.OutputSize, 1.0f, Weight,
                 _preLayer.OutputSize, _preLayer.Output, 1, 0.0f, Input, 1);
      Blas1.copy(InputSize, Function(Input), 1, Output, 1);
    }

    internal override void CalculationOutputLayerDelta(float[] teacher) {
      Parallel.For(0, Delta.Length, i => { Delta[i] = Output[i] - teacher[i]; });
      CalculationDeltaW();
    }

    internal override void CalculationDelta(Layer postLayer) {
      var diff = _diff(Output);
      Parallel.For(0, Delta.Length, i => {
                                      Delta[i] = 0.0f;
                                      for (var k = 0; k < postLayer.Delta.Length; k++)
                                        Delta[i] += postLayer.Delta[k] * postLayer.Weight[OutputSize * k + i];
                                      Delta[i] *= diff[i];
                                    });
      CalculationDeltaW();
    }
    
    internal void CalculationDeltaW() {
      Parallel.For(0, InputSize, i => {
                                   for (var j = 0; j < _preLayer.OutputSize; j++)
                                     dW[_preLayer.OutputSize* i + j] += Delta[i] * _preLayer.Output[j];
                                 });
    }

    internal override string Dump(int dumpLevel) {
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
              sb.AppendLine($"\tdW[{i,4}, {j,4}] = {dW[_preLayer.OutputSize* i + j]: ##0.0##;-##0.0##}");
            }
          sb.AppendLine("-----------------------------------------------------");
        }
        
        sb.AppendLine("=====================================================");
      }
    }
  }

}