@0xd6061f86774b0046;
using Array = import "array.capnp".Array;

struct GammaTensor {
  magicHeader @0 :Data;
  value @1 :Array;
  inputs @2 :Array;
  dataSubjectsIndexed @3 :Array;
  oneHotLookup @4 :Array;
  minVal @5 :Float64;
  maxVal @6 :Float64;
  isLinear @7 :Bool;
  id @8 :Text;
}
