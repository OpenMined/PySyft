@0xd6061f86774b0046;

struct GammaTensor {
  magicHeader @0 :Data;
  value @1 :List(Data);
  inputs @2 :List(Data);
  entitiesIndexed @3 :List(Data);
  oneHotLookup @4 :List(Data);
  valueMetadata @5 :TensorMetadata;
  inputsMetadata @6 :TensorMetadata;
  entitiesIndexedMetadata @7 :TensorMetadata;
  oneHotLookupMetadata @8 :TensorMetadata;
  minVal @9 :Float64;
  maxVal @10 :Float64;
  isLinear @11 :Bool;
  id @12 :Text;

  struct TensorMetadata {
    dtype @0 :Text;
    decompressedSize @1 :UInt64;
  }
}
