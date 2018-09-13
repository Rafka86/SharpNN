using System;
using System.Collections.Generic;

namespace SharpNN.DataSet {

  public abstract class DataSet {
    protected List<Datum> TrainingDataList { get; } = new List<Datum>();
    protected List<Datum> TestDataList { get; } = new List<Datum>();
    public int TrainingDataSize => TrainingDataList.Count;
    public int TestDataSize => TestDataList.Count;
    public int InputDataSize { get; protected set; } = 1;
    public int OutputDataSize { get; protected set; } = 1;

    public virtual IEnumerable<Datum> TrainingData() {
      foreach (var datum in TrainingDataList) yield return datum;
    }
    public virtual IEnumerable<Datum> TestData() {
      foreach (var datum in TestDataList) yield return datum;
    }
    public virtual IEnumerable<Datum> MiniBatch(int size) {
      var rand = new Random();
      for (var i = 0; i < size; i++) yield return TrainingDataList[rand.Next(TrainingDataList.Count)];
    }
  }

}