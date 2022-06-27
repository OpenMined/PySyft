@0xd6061f86774b0046;
using Array = import "array.capnp".Array;

struct GammaTensor {
  magicHeader @0 :Data;
  child @1 :Data;
  state @2 :Data;
  dataSubjectsIndexed @3 :Array;
  oneHotLookup @4 :Array;
  minVal @5 :Data;
  maxVal @6 :Data;
  isLinear @7 :Bool;
  id @8 :Text;
}
