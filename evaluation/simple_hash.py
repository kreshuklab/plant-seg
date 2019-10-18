import hashlib

def simple_hash(numpy_array1, numpy_array2):
    hash1, hash2 = hashlib.md5(), hashlib.md5()
    hash1.update(numpy_array1.data.tobytes())
    hash2.update(numpy_array2.data.tobytes())
    return (hash1.hexdigest(), hash2.hexdigest())
