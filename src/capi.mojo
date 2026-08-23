from std.sys.info import simd_width_of


comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime U8Ptr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()


def gcd_positive(a_in: Int, b_in: Int) -> Int:
    var a = a_in
    var b = b_in
    while b != 0:
        var remainder = a % b
        a = b
        b = remainder
    return a


def rescale_positive(value: Int, numerator: Int, denominator: Int, rounding: Int) -> Int:
    var whole = value // denominator
    var remainder = value % denominator
    var result = whole * numerator
    var scaled_remainder = remainder * numerator
    var extra = scaled_remainder // denominator
    if rounding == 0:
        pass
    elif rounding == 1:
        if scaled_remainder % denominator >= (denominator + 1) // 2:
            extra += 1
    elif rounding == 2:
        pass
    else:
        if scaled_remainder % denominator != 0:
            extra += 1
    return result + extra


@export("mav_rescale_many")
def mav_rescale_many(
    src_addr: Int,
    dst_addr: Int,
    count: Int,
    src_num: Int,
    src_den: Int,
    dst_num: Int,
    dst_den: Int,
    rounding: Int,
) abi("C") -> Int:
    if count < 0 or src_num <= 0 or src_den <= 0 or dst_num <= 0 or dst_den <= 0:
        return -1
    if count == 0:
        return 0
    if src_addr == 0 or dst_addr == 0:
        return -1
    var n1 = src_num
    var n2 = dst_den
    var d1 = src_den
    var d2 = dst_num
    var common = gcd_positive(n1, d1)
    n1 //= common
    d1 //= common
    common = gcd_positive(n1, d2)
    n1 //= common
    d2 //= common
    common = gcd_positive(n2, d1)
    n2 //= common
    d1 //= common
    common = gcd_positive(n2, d2)
    n2 //= common
    d2 //= common
    var max_i64 = 9223372036854775807
    if n1 > max_i64 // n2 or d1 > max_i64 // d2:
        return -2
    var numerator = n1 * n2
    var denominator = d1 * d2
    var src = I64Ptr(unsafe_from_address=src_addr)
    var dst = I64Ptr(unsafe_from_address=dst_addr)
    for i in range(count):
        var value = Int(src[i])
        if value == -9223372036854775807 - 1:
            dst[i] = Int64(value)
        else:
            var magnitude = value if value >= 0 else -value
            var effective_rounding = rounding
            if value < 0 and rounding == 2:
                effective_rounding = 3
            elif value < 0 and rounding == 3:
                effective_rounding = 2
            var whole = magnitude // denominator
            var remainder = magnitude % denominator
            if whole > max_i64 // numerator or remainder > max_i64 // numerator:
                return -2
            var base = whole * numerator
            var scaled_remainder = remainder * numerator
            var extra = scaled_remainder // denominator
            if effective_rounding == 1:
                if scaled_remainder % denominator >= (denominator + 1) // 2:
                    extra += 1
            elif effective_rounding == 3 and scaled_remainder % denominator != 0:
                extra += 1
            if base > max_i64 - extra:
                return -2
            var scaled = base + extra
            dst[i] = Int64(scaled if value >= 0 else -scaled)
    return 0


@export("mav_durations_from_pts")
def mav_durations_from_pts(
    pts_addr: Int, dst_addr: Int, count: Int, final_duration: Int
) abi("C") -> Int:
    if count < 0:
        return -1
    if count == 0:
        return 0
    if pts_addr == 0 or dst_addr == 0:
        return -1
    var pts = I64Ptr(unsafe_from_address=pts_addr)
    var dst = I64Ptr(unsafe_from_address=dst_addr)
    durations_range(pts, dst, 0, count - 1)
    dst[count - 1] = Int64(final_duration)
    return 0


def durations_range(pts: I64Ptr, dst: I64Ptr, start: Int, end: Int):
    var nopts = Int64(-9223372036854775807 - 1)
    var vector_end = start + ((end - start) // W) * W
    for i in range(start, vector_end, W):
        var current = pts.load[width=W](i)
        var following = pts.load[width=W](i + 1)
        var invalid = current.eq(nopts) | following.eq(nopts)
        dst.store(i, invalid.select(SIMD[DType.int64, W](nopts), following - current))
    for i in range(vector_end, end):
        var current = pts[i]
        var following = pts[i + 1]
        if current == nopts or following == nopts:
            dst[i] = nopts
        else:
            dst[i] = following - current


def read_u32_le(data: U8Ptr, offset: Int) -> Int:
    return (
        Int(data[offset])
        | (Int(data[offset + 1]) << 8)
        | (Int(data[offset + 2]) << 16)
        | (Int(data[offset + 3]) << 24)
    )


def read_i64_le(data: U8Ptr, offset: Int) -> Int64:
    var value = UInt64(data[offset])
    for j in range(1, 8):
        value |= UInt64(data[offset + j]) << UInt64(8 * j)
    return Int64(value)


@export("mav_scan_ivf")
def mav_scan_ivf(
    data_addr: Int,
    length: Int,
    offsets_addr: Int,
    sizes_addr: Int,
    timestamps_addr: Int,
    keyframes_addr: Int,
    capacity: Int,
) abi("C") -> Int:
    if length < 32 or capacity < 0 or data_addr == 0:
        return -1
    if capacity > 0 and (
        offsets_addr == 0 or sizes_addr == 0 or timestamps_addr == 0
        or keyframes_addr == 0
    ):
        return -1
    var data = U8Ptr(unsafe_from_address=data_addr)
    var offsets = I64Ptr(unsafe_from_address=offsets_addr)
    var sizes = I64Ptr(unsafe_from_address=sizes_addr)
    var timestamps = I64Ptr(unsafe_from_address=timestamps_addr)
    var keyframes = I64Ptr(unsafe_from_address=keyframes_addr)
    var position = 32
    var count = 0
    while position < length:
        if position + 12 > length:
            return -2
        if count >= capacity:
            return -3
        var size = read_u32_le(data, position)
        if size < 0 or position + 12 + size > length:
            return -2
        offsets[count] = Int64(position)
        sizes[count] = Int64(size)
        timestamps[count] = read_i64_le(data, position + 4)
        if size > 0 and (Int(data[position + 12]) & 1) == 0:
            keyframes[count] = 1
        else:
            keyframes[count] = 0
        position += 12 + size
        count += 1
    return count


@export("mav_fixed_packets")
def mav_fixed_packets(
    data_size: Int,
    block_align: Int,
    samples_per_packet: Int,
    offsets_addr: Int,
    sizes_addr: Int,
    durations_addr: Int,
    capacity: Int,
) abi("C") -> Int:
    if data_size < 0 or block_align <= 0 or samples_per_packet <= 0 or capacity < 0:
        return -1
    var max_i64 = 9223372036854775807
    if block_align > max_i64 // samples_per_packet:
        return -1
    var packet_bytes = block_align * samples_per_packet
    var needed = (data_size + packet_bytes - 1) // packet_bytes
    if needed > capacity:
        return -2
    if needed == 0:
        return 0
    if offsets_addr == 0 or sizes_addr == 0 or durations_addr == 0:
        return -1
    var offsets = I64Ptr(unsafe_from_address=offsets_addr)
    var sizes = I64Ptr(unsafe_from_address=sizes_addr)
    var durations = I64Ptr(unsafe_from_address=durations_addr)
    for i in range(needed):
        var offset = i * packet_bytes
        var size = min(packet_bytes, data_size - offset)
        offsets[i] = Int64(offset)
        sizes[i] = Int64(size)
        durations[i] = Int64(size // block_align)
    return needed
