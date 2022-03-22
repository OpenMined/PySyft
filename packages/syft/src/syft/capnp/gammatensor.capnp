@0xe1bb8d6f94aba00f

struct GammaTensor {
  magicHeader @0 :Data;
  value @1 :List(Data);
  dataSubjects @2 :List(Data);
  minVal @3: Float32,
  maxVal @4 :Float32;
  is_linear @5 :Bool;
  state @6 :;
}
