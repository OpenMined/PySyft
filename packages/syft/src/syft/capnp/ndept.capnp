@0x8f9c60bd7a9842fc;

struct NDEPT {
  magicHeader @0 :Data;
  child @1 :List(Data);
  minVals @2 :List(Data);
  maxVals @3 :List(Data);
  entitiesIndexed @4 :List(Data);
  childMetadata @5 :TensorMetadata;
  minValsMetadata @6 :TensorMetadata;
  maxValsMetadata @7 :TensorMetadata;
  entitiesIndexedMetadata @8 :TensorMetadata;
  oneHotLookup @9 :List(Text);

  struct TensorMetadata {
    dtype @0 :Text;
    decompressedSize @1 :UInt64;
  }
}
