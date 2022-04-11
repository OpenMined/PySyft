@0xea6d71e3d9f61e0e;
using Array = import "array.capnp".Array;

struct DataSubjectLedger {
  magicHeader @0 :Data;
  constants @1 :Array;
  updateNumber @2 :UInt64;
  timestamp @3 :Float64;
}
