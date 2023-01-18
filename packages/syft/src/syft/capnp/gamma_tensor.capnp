@0xd6061f86774b0046;
using Array = import "array.capnp".Array;

struct GammaTensor {
  magicHeader @0 :Data;
  child @1 :List(Data);
  sources @2 :Data;
  isLinear @3 :Bool;
  id @4 :Text;
  isNumpy @5 :Bool;
  jaxOp @6 :Data;
}
