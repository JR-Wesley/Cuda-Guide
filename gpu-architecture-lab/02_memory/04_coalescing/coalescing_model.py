"""单条 warp load 的地址覆盖模型；不模拟缓存、wavefront 或物理传输。"""


def covered_segments(addresses, width, segment_bytes=32):
    """返回活跃 lane 覆盖的地址段编号集合，所有输入单位均为字节。

    addresses 每项对应一个活跃 lane，重复地址保留线程含义，但覆盖只算一次。
    32B 用于 sector 模型；128B 仅用于 L1 行边界的地址对照。
    """
    addresses = tuple(addresses)
    if width not in (4, 8, 16):
        raise ValueError("width must be 4, 8 or 16 bytes")
    if segment_bytes not in (32, 128):
        raise ValueError("segment_bytes must be 32 or 128 bytes")
    if len(addresses) > 32:
        raise ValueError("a warp has at most 32 active lanes")

    segments = set()
    for address in addresses:
        if not isinstance(address, int) or address < 0 or address % width:
            raise ValueError("each address must be a nonnegative, naturally aligned integer")
        # 末字节是 address + width - 1，不能把恰好相邻的下一个段算进去。
        first = address // segment_bytes
        last = (address + width - 1) // segment_bytes
        segments.update(range(first, last + 1))
    return segments
