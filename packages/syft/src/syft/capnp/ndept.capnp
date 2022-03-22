@0x8f9c60bd7a9842fc;

struct NDEPT {
  magicHeader @0 :Data;
  child @1 :List(Data);
  minVals @2 :List(Data);
  maxVals @3 :List(Data);
  entitiesIndexed @4 :List(Data);
  oneHotLookup @5 :List(Data);
  childMetadata @6 :TensorMetadata;
  minValsMetadata @7 :TensorMetadata;
  maxValsMetadata @8 :TensorMetadata;
  entitiesIndexedMetadata @9 :TensorMetadata;
  oneHotLookupMetadata @10 :TensorMetadata;

  struct TensorMetadata {
    dtype @0 :Text;
    decompressedSize @1 :UInt64;
  }
}
