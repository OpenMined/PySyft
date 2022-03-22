@0xd6061f86774b0046;

struct GammaTensor {
  magicHeader @0 :Data;
  value @1 :List(Data);
  entitiesIndexed @2 :List(Data);
  oneHotLookup @3 :List(Data);
  minVal @4 :Float64;
  maxVal @5 :Float64;
  isLinear @6 :Bool;
  inputs @7 :List(Data);
  valueMetadata @8 :TensorMetadata;
  entitiesIndexedMetadata @9 :TensorMetadata;
  oneHotLookupMetadata @10 :TensorMetadata;
  inputsMetadata @11 :TensorMetadata;


  struct TensorMetadata {
    dtype @0 :Text;
    decompressedSize @1 :UInt64;
  }
}
