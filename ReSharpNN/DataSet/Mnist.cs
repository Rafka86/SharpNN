using System;
using System.Collections.Generic;
using System.IO;

namespace ReSharpNN.DataSet {

  public class Mnist : DataSet {
    public Mnist(string path = @"./mnist", string trainingLabelFile = "train-labels-idx1-ubyte",
                                           string trainingImageFile = "train-images-idx3-ubyte",
                                           string testLabelFile     = "t10k-labels-idx1-ubyte",
                                           string testImageFile     = "t10k-images-idx3-ubyte") {
      //Setup data file paths.
      if (!Directory.Exists(path)) throw new ArgumentException($"Not found mnist directory at {path}.");
      var trainingLabelFilePath = path + "/" + trainingLabelFile;
      if (!File.Exists(trainingLabelFilePath))
        throw new ArgumentException($"Not found training label file at {trainingLabelFilePath}");
      var trainingImageFilePath = path + "/" + trainingImageFile;
      if (!File.Exists(trainingImageFilePath))
        throw new ArgumentException($"Not found training image file at {trainingImageFilePath}");
      var testLabelFilePath = path + "/" + testLabelFile;
      if (!File.Exists(trainingLabelFilePath))
        throw new ArgumentException($"Not found training label file at {testLabelFilePath}");
      var testImageFilePath = path + "/" + testImageFile;
      if (!File.Exists(trainingImageFilePath))
        throw new ArgumentException($"Not found training image file at {testImageFilePath}");

      //Loading training data.
      using (var laFile = new BinaryReader(File.OpenRead(trainingLabelFilePath)))
      using (var imFile = new BinaryReader(File.OpenRead(trainingImageFilePath)))
        MakeData(laFile, imFile, TrainingDataList);

      //Loading test data.
      using (var laFile = new BinaryReader(File.OpenRead(testLabelFilePath)))
      using (var imFile = new BinaryReader(File.OpenRead(testImageFilePath)))
        MakeData(laFile, imFile, TestDataList);

      void MakeData(BinaryReader labelFile, BinaryReader imageFile, List<Datum> dstList) {
        //Read and discard magic numbers.
        labelFile.ReadInt32();
        imageFile.ReadInt32();

        var dataSize = labelFile.ReadInt32BigEndian();
        if (dataSize != imageFile.ReadInt32BigEndian()) throw new ArgumentException("Invalid mnist files.");
        var imageSizeX = imageFile.ReadInt32BigEndian();
        var imageSizeY = imageFile.ReadInt32BigEndian();
        var imageSize = imageSizeX * imageSizeY;

        for (var i = 0; i < dataSize; i++) {
          var input = new float[imageSize];
          var output = new float[10]; // 10 is the size of output layer's units.
          for (var j = 0; j < input.Length; j++) input[j] = imageFile.ReadByte() / 255.0f;
          output[labelFile.ReadByte()] = 1.0f;
          dstList.Add(new Datum(input, output));
        }
        
        // Check the number of loaded data.
        if (dataSize != dstList.Count) throw new ArgumentException("Mismatching the number of loaded data.");
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