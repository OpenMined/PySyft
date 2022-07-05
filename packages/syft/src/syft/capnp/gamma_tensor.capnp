@0xd6061f86774b0046;
using Array = import "array.capnp".Array;

struct GammaTensor {
  magicHeader @0 :Data;
  child @1 :Data;
  state @2 :Data;
  dataSubjects @3 :Array;
  minVal @4 :Data;
  maxVal @5 :Data;
  isLinear @6 :Bool;
  id @7 :Text;
}
