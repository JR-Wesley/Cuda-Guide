"""验证 README 中的地址推导，不调用 CUDA 或读取历史 GPU 产物。"""
import unittest

from coalescing_model import covered_segments


class CoalescingModelTests(unittest.TestCase):
    def test_contiguous_widths(self):
        for width, count in ((4, 4), (8, 8), (16, 16)):
            with self.subTest(width=width):
                self.assertEqual(
                    covered_segments([width * lane for lane in range(32)], width),
                    set(range(count)),
                )

    def test_same_ownership_scalar_repeats_sectors(self):
        vector = covered_segments([16 * lane for lane in range(32)], 16)
        scalar = [covered_segments([16 * lane + 4 * k for lane in range(32)], 4)
                  for k in range(4)]
        self.assertEqual([len(s) for s in scalar], [16, 16, 16, 16])
        self.assertEqual(sum(map(len, scalar)), 64)
        self.assertEqual(set.union(*scalar), vector)
        self.assertEqual(len(vector), 16)

    def test_lane_contiguous_scalar_uses_disjoint_sectors(self):
        scalar = [covered_segments([128 * k + 4 * lane for lane in range(32)], 4)
                  for k in range(4)]
        self.assertEqual(scalar, [set(range(4 * k, 4 * k + 4)) for k in range(4)])
        self.assertEqual(sum(map(len, scalar)), 16)
        self.assertEqual(set.union(*scalar), set(range(16)))

    def test_layouts_cover_same_bytes_but_change_ownership(self):
        # 用逐字节集合独立核对三种布局，避免只比较 sector 数而漏掉数据差异。
        vector = [set(range(16 * lane, 16 * lane + 16)) for lane in range(32)]
        same = [{16 * lane + 4 * k + b for k in range(4) for b in range(4)}
                for lane in range(32)]
        striped = [{128 * k + 4 * lane + b for k in range(4) for b in range(4)}
                   for lane in range(32)]
        self.assertEqual(vector, same)
        self.assertNotEqual(vector[0], striped[0])
        self.assertEqual(set.union(*vector), set(range(512)))
        self.assertEqual(set.union(*striped), set(range(512)))

    def test_stride_32(self):
        self.assertEqual(covered_segments([32 * lane for lane in range(32)], 4),
                         set(range(32)))

    def test_duplicate_addresses(self):
        self.assertEqual(covered_segments([0] * 32, 4), {0})

    def test_partial_warp(self):
        self.assertEqual(covered_segments([0, 16, 32], 16), {0, 1})
        self.assertEqual(covered_segments([16 * lane for lane in range(0, 32, 2)], 16),
                         set(range(16)))

    def test_empty_active_lanes(self):
        self.assertEqual(covered_segments([], 16), set())

    def test_vector_offsets(self):
        for offset, sectors, lines in ((0, 16, 4), (16, 17, 5), (32, 16, 5)):
            with self.subTest(offset=offset):
                addresses = [offset + 16 * lane for lane in range(32)]
                self.assertEqual(len(covered_segments(addresses, 16)), sectors)
                self.assertEqual(len(covered_segments(addresses, 16, 128)), lines)
        self.assertAlmostEqual(512 / (17 * 32), 16 / 17)

    def test_scalar_offset(self):
        self.assertEqual(covered_segments([4 + 4 * lane for lane in range(32)], 4),
                         set(range(5)))

    def test_segment_end_is_inclusive(self):
        self.assertEqual(covered_segments([16], 16), {0})
        self.assertEqual(covered_segments([32], 16), {1})
        self.assertEqual(covered_segments([112], 16, 128), {0})
        self.assertEqual(covered_segments([128], 16, 128), {1})

    def test_bytewise_reference(self):
        # 对自然对齐偏移和 lane 子集做小规模枚举，参照逐字节算法。
        for width in (4, 8, 16):
            for segment in (32, 128):
                for base in range(0, 128, width):
                    for stride in (width, 32, 64):
                        addresses = [base + stride * lane for lane in (0, 1, 7, 31)]
                        expected = {byte // segment for address in addresses
                                    for byte in range(address, address + width)}
                        with self.subTest(width=width, segment=segment,
                                          base=base, stride=stride):
                            self.assertEqual(covered_segments(addresses, width, segment),
                                             expected)

    def test_invalid_alignment_and_addresses(self):
        for addresses, width in (([4], 16), ([8], 16), ([4], 8), ([-16], 16),
                                 ([0, 20], 16), ([0.0], 4)):
            with self.subTest(addresses=addresses, width=width):
                with self.assertRaisesRegex(ValueError, "naturally aligned"):
                    covered_segments(addresses, width)

    def test_unsupported_width_segment_and_lane_count(self):
        with self.assertRaisesRegex(ValueError, "width"):
            covered_segments([0], 12)
        with self.assertRaisesRegex(ValueError, "segment_bytes"):
            covered_segments([0], 4, 64)
        with self.assertRaisesRegex(ValueError, "32 active lanes"):
            covered_segments([0] * 33, 4)


if __name__ == "__main__":
    unittest.main()
