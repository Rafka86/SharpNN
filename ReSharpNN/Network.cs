using System;
using System.Collections.Generic;
using System.Text;

using static ReSharpNN.UpdateFunctions;
using static ReSharpNN.WeightsInitializeFunctions;

namespace ReSharpNN {

  public class Network {
    private readonly List<Layer> _layers = new List<Layer>();
    private readonly List<Connection> _connections = new List<Connection>();

    public static class Factory {
      private static Network _network;

      public static void New() => _network = new Network();

      public static void AddLayer(int size, UpdateFunction function = null, bool hasBias = false)
        => _network._layers.Add(new Layer(size, function ?? ReLU, hasBias));

      public static void AddConnection(WeightsInitializeFunction initFunction = null) {
        if (_network._layers.Count < 2) throw new ApplicationException("Not enough layers.");
        if (_network._layers.Count - 1 == _network._connections.Count)
          throw new ApplicationException("There are enough connections.");
        
        var lastIndex = _network._layers.Count - 1;
        var preLayerSize = _network._layers[lastIndex - 1].Size;
        var postLayerSize = _network._layers[lastIndex].PureSize;
        _network._connections.Add(new Connection(preLayerSize, postLayerSize, initFunction ?? He));
      }

      public static Network Create() => _network;
    }

    public void ForwardPropagation(float[] input) {
      var inputLayer = _layers[0];
      for (var i = 0; i < input.Length; i++)
        inputLayer.Unit[i] = input[i];

      for (var i = 1; i < _layers.Count; i++) {
        var preLayer = _layers[i - 1];
        var connection = _connections[i - 1];
        var layer = _layers[i];
        layer.Update(preLayer, connection);
      }
    }

    public float[] Output => _layers[_layers.Count - 1].Unit;
    
    public string Dump(int dumpLevel = 1) {
      var sb = new StringBuilder();
      sb.Clear();
      
      sb.AppendLine("=================== Network Infos ===================");
      sb.AppendLine($"Number of layers : {_layers.Count.ToString()}");
      sb.AppendLine($"       Layers[{0,3}] : {_layers[0].Unit.Length.ToString()}(units) {_layers[0].HasBias.ToString()}(has bias)");
      for (var i = 1; i < _layers.Count; i++) {
        sb.AppendLine($"  Connections[{i - 1,3}] : {_connections[i - 1].PreLayerSize.ToString()}(pre neurons) {_connections[i - 1].PostLayerSize.ToString()}(post neurons)");
        sb.AppendLine($"       Layers[{i,3}] : {_layers[i].Unit.Length.ToString()}(neurons) {_layers[i].HasBias.ToString()}(has bias)");
      }
      sb.AppendLine("=====================================================");
      if (dumpLevel <= 0) return sb.ToString();

      sb.AppendLine();
      sb.AppendLine(">> Layer 0");
      sb.AppendLine(_layers[0].Dump(dumpLevel));
      for (var i = 1; i < _layers.Count; i++) {
        sb.AppendLine($">> Connection Layer {i - 1} -> Layer {i}");
        sb.AppendLine(_connections[i - 1].Dump(dumpLevel));
        sb.AppendLine($">> Layer {i}");
        sb.AppendLine(_layers[i].Dump(dumpLevel));
      }

      return sb.ToString();
    }

    public override string ToString() => Dump();
  }

}