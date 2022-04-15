@0x8f9c60bd7a9842fc;
using Array = import "array.capnp".Array;

struct NDEPT {
  magicHeader @0 :Data;
  child @1 :Array;
  minVals @2 :Array;
  maxVals @3 :Array;
  dataSubjectsIndexed @4 :Array;
  oneHotLookup @5 :Array;
}
