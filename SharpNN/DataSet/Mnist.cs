using System;
using System.Collections.Generic;
using System.IO;

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace SharpNN.DataSet {

  public class Mnist : DataSet {
    private readonly int _imageSizeX;
    private readonly int _imageSizeY;
    
    public Mnist(string path,
                 string trainingLabelFile = "train-labels-idx1-ubyte",
                 string trainingImageFile = "train-images-idx3-ubyte",
                 string testLabelFile     = "t10k-labels-idx1-ubyte",
                 string testImageFile     = "t10k-images-idx3-ubyte") {
      //Setup data file paths.
      if (!Directory.Exists(path)) throw new ArgumentException($"Not found mnist directory at {path} .");
      var trainingLabelFilePath = $"{path}{Path.DirectorySeparatorChar.ToString()}{trainingLabelFile}";
      if (!File.Exists(trainingLabelFilePath))
        throw new ArgumentException($"Not found training label file at {trainingLabelFilePath} .");
      var trainingImageFilePath = $"{path}{Path.DirectorySeparatorChar.ToString()}{trainingImageFile}";
      if (!File.Exists(trainingImageFilePath))
        throw new ArgumentException($"Not found training image file at {trainingImageFilePath} .");
      var testLabelFilePath = $"{path}{Path.DirectorySeparatorChar.ToString()}{testLabelFile}";
      if (!File.Exists(trainingLabelFilePath))
        throw new ArgumentException($"Not found training label file at {testLabelFilePath} .");
      var testImageFilePath = $"{path}{Path.DirectorySeparatorChar.ToString()}{testImageFile}";
      if (!File.Exists(trainingImageFilePath))
        throw new ArgumentException($"Not found training image file at {testImageFilePath} .");

      //Loading training data.
      using (var laFile = new BinaryReader(File.OpenRead(trainingLabelFilePath)))
      using (var imFile = new BinaryReader(File.OpenRead(trainingImageFilePath)))
        (_imageSizeX, _imageSizeY) = MakeData(laFile, imFile, TrainingDataList);

      //Loading test data.
      using (var laFile = new BinaryReader(File.OpenRead(testLabelFilePath)))
      using (var imFile = new BinaryReader(File.OpenRead(testImageFilePath)))
        MakeData(laFile, imFile, TestDataList);

      InputDataSize = _imageSizeX * _imageSizeY;
      OutputDataSize = 10;

      (int, int) MakeData(BinaryReader labelFile, BinaryReader imageFile, ICollection<Datum> dstList) {
        //Read and discard magic numbers.
        labelFile.ReadInt32();
        imageFile.ReadInt32();

        var dataSize = labelFile.ReadInt32BigEndian();
        if (dataSize != imageFile.ReadInt32BigEndian()) throw new ArgumentException("Invalid mnist files.");
        var imgSizeX = imageFile.ReadInt32BigEndian();
        var imgSizeY = imageFile.ReadInt32BigEndian();
        var imageSize = imgSizeX * imgSizeY;

        for (var i = 0; i < dataSize; i++) {
          var input = new float[imageSize];
          var output = new float[10]; // 10 is the size of output layer's units.
          for (var j = 0; j < input.Length; j++) input[j] = imageFile.ReadByte() / 255.0f;
          output[labelFile.ReadByte()] = 1.0f;
          dstList.Add(new Datum(input, output));
        }
        
        // Check the number of loaded data.
        if (dataSize != dstList.Count) throw new ArgumentException("Mismatching the number of loaded data.");
        return (imgSizeX, imgSizeY);
      }
    }

    public void OutputImages(string path) {
      MakeImages(TrainingDataList, $"training{Path.DirectorySeparatorChar.ToString()}");
      MakeImages(TestDataList, $"test{Path.DirectorySeparatorChar.ToString()}");
      
      void MakeImages(IReadOnlyList<Datum> targetList, string extraPath) {
        for (var i = 0; i < targetList.Count; i++) {
          using (var image = new Image<Rgba32>(_imageSizeX, _imageSizeY)) {
            var imageData = targetList[i].Input;
            for (var y = 0; y < image.Height; y++)
              for (var x = 0; x < image.Width; x++)
                image[x, y] = new Rgba32(imageData[y * image.Width + x],
                                         imageData[y * image.Width + x],
                                         imageData[y * image.Width + x]);
            image.Save($"{path}{extraPath}{i:D5}.png");
          }
        }
      }
    }
  }

  internal static class BinaryReaderExtendedMethods {
    private static byte[] ReadBytesBigEndian(this BinaryReader br, int size) {
      var bytes = new byte[size];
      for (var i = size - 1; i >= 0; i--) bytes[i] = br.ReadByte();
      return bytes;
    }
    internal static int ReadInt32BigEndian(this BinaryReader br) => BitConverter.ToInt32(br.ReadBytesBigEndian(4));
  }

}