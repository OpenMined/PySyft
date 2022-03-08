@0x8f9c60bd7a9842fc;

struct REPT {
  child @0 :List(Data);
  minVals @1 :List(Data);
  maxVals @2 :List(Data);
  oneHotLookup @3 :List(Text);
  entitiesIndexed @4 : List(Data);
  childDtype @4 :Text;
  minValsDtype @5 :Text;
  maxValsDtype @6 :Text;
}
