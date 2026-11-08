"""Model ONE collective rendezvous with all lanes still alive.

This is an explicit educational model, not a CUDA source/CFG verifier.
Invalid masks are diagnosed here rather than launched as undefined GPU code.
"""


def mask_errors(calls):
    errors = set()
    for lane, mask in calls.items():
        if not mask & (1 << lane):
            errors.add("caller_not_in_mask")
        for peer in range(32):
            if mask & (1 << peer):
                if peer not in calls:
                    errors.add("named_lane_does_not_call")
                elif calls[peer] != mask:
                    errors.add("overlapping_inconsistent_masks")
    return errors


def main():
    cases = [
        ("valid four lanes", {i: 0xf for i in range(4)}, set()),
        ("valid disjoint halves", {i: 0xffff if i < 16 else 0xffff0000 for i in range(32)}, set()),
        ("caller omitted", {i: 3 for i in range(4)}, {"caller_not_in_mask"}),
        ("noncalling lanes named", {i: 0x7f for i in range(4)}, {"named_lane_does_not_call"}),
        ("overlapping masks", {i: 1 if i == 0 else 0xffffffff for i in range(32)},
         {"overlapping_inconsistent_masks"}),
        ("hex literal 0xb1111 is not binary 0b1111", {i: 0xb1111 for i in range(4)},
         {"caller_not_in_mask", "named_lane_does_not_call"}),
    ]
    for name, calls, wanted in cases:
        errors = mask_errors(calls)
        if errors != wanted:
            raise AssertionError((name, errors, wanted))
        print(f"PASS {name}: {','.join(sorted(errors)) or 'valid'}")

    # Five participants and down(2): lanes 3/4 request live, nonparticipating lanes 5/6.
    invalid_sources = [(lane, lane + 2) for lane in range(5) if not (0x1f & (1 << (lane + 2)))]
    assert invalid_sources == [(3, 5), (4, 6)]
    print(f"PASS shuffle source diagnostic: {invalid_sources}; mask does not compact lanes")
    assert not (31 > 0 and 31 <= 32 and (31 & (31 - 1)) == 0)
    print("PASS width=31 rejected: only 1,2,4,8,16,32 are valid")

    # The pasted second branch reads shared[31-lane], outside its synchronized half.
    crossing = [(lane, 31 - lane) for lane in range(16, 32)
                if not (0xffff0000 & (1 << (31 - lane)))]
    assert len(crossing) == 16
    assert all(16 <= 47 - lane < 32 for lane in range(16, 32))
    print("PASS shared-memory diagnostic: 16 cross-half reads lack a covering barrier; "
          "use 47-lane for half reversal, or full-warp sync for cross-half exchange")
    print("PASS 9 contract diagnostics (host model; no invalid CUDA kernel launched)")


if __name__ == "__main__":
    main()
