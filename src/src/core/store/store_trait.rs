trait Store<K, V> {
    fn sizeof(&self) -> u64;
    fn keys(&self) -> Vec<K>;
    fn values(&self) -> Vec<V>;
    fn to_string(&self) -> str;
    fn len(&self) -> u64;
    fn contains(&self, key: K) -> bool;
    fn get(&self, key: K) -> Option<V>;
    fn set(&self, key: K, value: V);
    fn clear(&self);

    // python interface
    fn __str__(&self) -> str;
    fn __len__(&self) -> u64;
    fn __contains__(&self, key: K) -> bool;
    fn __getitem__(&self, key: K) -> V;
    fn __setitem__(&self, key: K) -> V;
    fn __delitem__(&self, key: K);
    fn delete(&self, key: K);
    fn get_object(self, key: K) -> Option<V>;
}
