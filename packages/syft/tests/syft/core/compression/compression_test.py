from syft.core.compression.bytes_compressor import BytesCompressor


def test_bytes_compress_decompress() -> None:
    test_str = 'was compression successful?'
    test_bytes = test_str.encode('utf-8')

    compressed = BytesCompressor.compress(test_bytes)
    decompressed = BytesCompressor.decompress(compressed)

    after_comde = decompressed.decode('utf-8')
    assert test_str == after_comde

def test_bytes_uncompressed_decompress() -> None:
    test_str = 'was compression successful?'
    test_bytes = test_str.encode('utf-8')

    decompressed = BytesCompressor.decompress(test_bytes)

    after_comde = decompressed.decode('utf-8')
    assert test_str == after_comde