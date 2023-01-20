@0xcd0709e35fffa8d8;

struct Array{
    array @0 :List(Data);
    arrayMetadata @1 :TensorMetadata;

    struct TensorMetadata {
    dtype @0 :Text;
    decompressedSize @1 :UInt64;
  }
}

