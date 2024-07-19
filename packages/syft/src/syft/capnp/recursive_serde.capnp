@0xd7dd27f3820d22ee;

struct RecursiveSerde {
    fieldsName @0 :List(Text);
    fieldsData @1 :List(List(Data));
    nonrecursiveBlob @2 :List(Data);
    canonicalName @3 :Text;
    version @4 :Int32;
}
