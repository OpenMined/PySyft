@0xd0fa899d748562f4;

struct ShareTensor {
  magicHeader @0 :Data;
  child @1 :List(Data);
  rank @2 :UInt32;
  partiesInfo @3 :Data;
  seedPrzs @4 :UInt64;
  ringSize @5 :Text;
  isNumpy @6 :Bool;
}
