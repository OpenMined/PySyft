@0xea6d71e3d9f61e0e;

struct DataSubjectLedger {
  magicHeader @0 :Data;
  constants @1 :List(Data);
  entitiesIndexed @2 :List(Data);
  constantsMetadata @3 :TensorMetadata;
  entitiesIndexedMetadata @4 :TensorMetadata;
  updateNumber @5 :UInt64;
  timestamp @6 :Float64;

  struct TensorMetadata {
    dtype @0 :Text;
    decompressedSize @1 :UInt64;
  }
}
