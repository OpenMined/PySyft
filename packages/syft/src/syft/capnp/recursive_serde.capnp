@0xd7dd27f3820d22ee;

struct RecursiveSerde {
    magicHeader @0 :Text;
    fieldsName @1 :List(Text);
    fieldsData @2 :List(Data);
    fullyQualifiedName @3 :Text;
    nonrecursiveBlob @4 :Data;
}
