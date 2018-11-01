using System;

using static SharpNN.ActivateFunctions;
using static SharpNN.WeightsInitializeFunctions;

namespace SharpNN {

  public partial class Network {
    public static class Factory {
      private static Network _network;
      
      public static Optimizer DefalutOptimizer { get; set; } = new NAG();

      public static void New() => _network = new Network();

      public static void SetErrorFunction(ErrorFunction errFunc) => _network._error = errFunc;

      public static void AddLayer(int                       size,
                                  ActivateFunction          function      = null,
                                  WeightsInitializeFunction wInitFunction = null,
                                  Optimizer                 optimizer     = null) {
        var preLayer = _network.LayersCount != 0 ? _network._layers[_network.LayersCount - 1] : null;
        if (preLayer != null) preLayer.IsOutputLayer = false;
        _network._layers.Add(new FullyConnectedLayer(size,
                                                     function ?? ReLU,
                                                     preLayer,
                                                     wInitFunction ?? He,
                                                     optimizer     ?? DefalutOptimizer.Clone()));
      }
      
      public static Network Create() {
        if (_network._error == null) throw new ArgumentException("Error function is not defined.");
        _network._inputLayer = _network._layers[0];
        _network._outputLayer = _network._layers[_network.LayersCount - 1];
        var res = _network;
        _network = null;
        return res;
      }

      public static void Create(out Network network) => network = Create();

      public static void DockIn(Network network) => _network = network;

      public static void OutOfDock() {
        _network = null;
      }
    }
  }

}