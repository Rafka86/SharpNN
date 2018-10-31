using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace SharpNN {

  public partial class Network {
    private readonly List<Layer> _layers = new List<Layer>();
    private Layer _inputLayer;
    private Layer _outputLayer;

    private ErrorFunction _error;
    
    public int LayersCount => _layers.Count;

    public void SetInputs(float[] input) {
      _inputLayer.SetOutputs(input);
    }
    
    public void ForwardPropagation() {
      for (var i = 1; i < _layers.Count; i++)
        _layers[i].Update();
    }

    public void BackPropagation(float[] teacher) {
      _outputLayer.CalculationOutputLayerDelta(teacher);

      for (var i = LayersCount - 2; i > 0; i--)
        _layers[i].CalculationDelta(_layers[i + 1]);
    }

    internal void PrepareForTraining(int batchSize) {
      for (var i = 1; i < _layers.Count; i++)
        _layers[i].PrepareForTraining(batchSize);
    }

    internal void FinishLearning() {
      for (var i = 1; i < _layers.Count; i++)
        _layers[i].PostProcess();
    }
    
    public void ClearDeltaW() {
      for (var i = 1; i < _layers.Count; i++)
        _layers[i].ClearDeltaW();
    }

    public void UpdateWeights() {
      for (var i = 1; i < _layers.Count; i++)
        _layers[i].ApplyDeltaW();
    }
    
    public float[] Output => _outputLayer.Output.AsSpan(0, _outputLayer.InputSize).ToArray();

    public float[] Input => _inputLayer.Output.AsSpan(0, _inputLayer.InputSize).ToArray();

    public float Error(float[] teacher) => _error(_outputLayer.Output, teacher);

    internal float Error() => _error(_outputLayer.Output,
                                   _layers[LayersCount - 3].Output.AsSpan(0, _layers[LayersCount - 3].InputSize).ToArray());

    public bool CheckStatus() {
      var checkRelErrAct = true;
      if (_error.Method.Name == "CrossEntropy") checkRelErrAct = _outputLayer.Function.Method.Name == "Softmax";
      Console.WriteLine($"Check the relation between error function and output layer activate function. : {checkRelErrAct}");
      Console.WriteLine($"Error function : {_error.Method.Name}");
      Console.WriteLine($"Activate function of the output layer : {_outputLayer.Function.Method.Name}");

      return checkRelErrAct;
    }

    public void Print(string filePath = null, int dumpLevel = 1) {
      if (filePath == null) Console.WriteLine(Dump(dumpLevel));
      else using (var sw = new StreamWriter(filePath)) sw.WriteLine(Dump(dumpLevel));
    }
    
    public string Dump(int dumpLevel = 1) {
      var sb = new StringBuilder();
      sb.Clear();
      
      sb.AppendLine("=================== Network Infos ===================");
      sb.AppendLine($"Number of layers : {_layers.Count.ToString()}");
      sb.AppendLine($"       Layers[{0,3}] : {_layers[0].Output.Length.ToString()}(units) {_layers[0].IsOutputLayer.ToString()}(has bias)");
      for (var i = 1; i < _layers.Count; i++) {
        sb.AppendLine($"       Layers[{i,3}] : {_layers[i].Output.Length.ToString()}(neurons) {_layers[i].IsOutputLayer.ToString()}(has bias)");
      }
      sb.AppendLine("=====================================================");
      if (dumpLevel <= 0) return sb.ToString();

      sb.AppendLine();
      sb.AppendLine(">> Layer 0");
      sb.AppendLine(_layers[0].Dump(dumpLevel));
      for (var i = 1; i < _layers.Count; i++) {
        sb.AppendLine($">> Connection Layer {i - 1} -> Layer {i}");
        sb.AppendLine($">> Layer {i}");
        sb.AppendLine(_layers[i].Dump(dumpLevel));
      }

      return sb.ToString();
    }

    public override string ToString() => Dump();
  }

}