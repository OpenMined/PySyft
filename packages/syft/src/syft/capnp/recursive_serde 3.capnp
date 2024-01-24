@0xd7dd27f3820d22ee;

struct RecursiveSerde {
    fieldsName @0 :List(Text);
    fieldsData @1 :List(List(Data));
    fullyQualifiedName @2 :Text;
    nonrecursiveBlob @3 :List(Data);
}
