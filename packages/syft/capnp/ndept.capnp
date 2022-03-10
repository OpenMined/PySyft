@0x8f9c60bd7a9842fc;

struct NDEPT {
  child @0 :List(Data);
  minVals @1 :List(Data);
  maxVals @2 :List(Data);
  entitiesIndexed @3 :List(Data);
  childMetadata @4 :TensorMetadata;
  minValsMetadata @5 :TensorMetadata;
  maxValsMetadata @6 :TensorMetadata;
  entitiesIndexedMetadata @7 :TensorMetadata;
  oneHotLookup @8 :List(Text);

  struct TensorMetadata {
    dtype @0 :Text;
    decompressedSize @1 :UInt64;
  }
}
