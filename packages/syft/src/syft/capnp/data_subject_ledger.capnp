@0xea6d71e3d9f61e0e;

struct DataSubjectLedger {
  magicHeader @0 :Data;
  constants @1 :List(Data);
  constantsMetadata @2 :TensorMetadata;
  updateNumber @3 :UInt64;
  timestamp @4 :Float64;

  struct TensorMetadata {
    dtype @0 :Text;
    decompressedSize @1 :UInt64;
  }
}
