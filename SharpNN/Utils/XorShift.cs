using System;
using System.Collections.Generic;

namespace SharpNN.Utils {

  internal class XorShift {
    private readonly IEnumerator<uint> _r;

    internal XorShift() {
      var w = (uint) Environment.TickCount;
      var x = w << 13;
      var y = (w >> 9) ^ (w << 6);
      var z = y >> 7;
      _r = RandGen();

      IEnumerator<uint> RandGen() {
        uint t;
        while (true) {
          t = x ^ (x << 11);
          x = y;
          y = z;
          z = w;
          yield return w = w ^ (w >> 19) ^ t ^ (t >> 8);
        }
      }
    }

    internal uint Rand {
      get {
        _r.MoveNext();
        return _r.Current;
      }
    }

    internal int RandInt(int min = 0, int max = 0x7FFFFFFF) => (int) (Rand % (max - min + 1)) + min;

    internal float RandFloat(float min = 0.0f, float max = 1.0f)
      => (float) (Rand % 0xFFFF) / 0xFFFF * (max - min) + min;
  }

}