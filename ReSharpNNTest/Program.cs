using ReSharpNN.DataSet;

using static System.IO.Path;

namespace ReSharpNNTest {

  class Program {
    static void Main(string[] args) {
      var separator = DirectorySeparatorChar.ToString();
      var mnistPath = $".{separator}mnist{separator}";
      var mnist = new Mnist(mnistPath);
      //var imagePath = $"{mnistPath}png{separator}";
      //mnist.OutputImages(imagePath);
    }
  }

}