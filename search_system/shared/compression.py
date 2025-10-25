"""
Compression utilities for integer sequences.
Implements Variable-Byte (VarByte) encoding and decoding.
"""

from typing import List

def varbyte_encode(numbers: List[int]) -> bytes:
    """
    Encode a list of non-negative integers using VarByte encoding.
    - Break each integer into groups of 7 bits.
    - Write those groups least significant first.
    - Set the MSB (bit 7) to 1 on all bytes except the last, which signals the end of that integer.
    """
    encoded_bytes = bytearray()

    for num in numbers:
        # Repeatedly take lowest 7 bits and shift right
        while True:
            byte = num & 0x7F   # get lowest 7 bits (mask 0x7F = 01111111)
            num >>= 7           # drop those 7 bits (logical right shift)
            if num:
                # If bits remain, set MSB = 1 to mark continuation (add 0x80)
                encoded_bytes.append(byte | 0x80)
            else:
                # If none remain, append as final byte
                encoded_bytes.append(byte)
                break
    
    # Convert accumulated bytearray to immutable bytes
    return bytes(encoded_bytes)

def varbyte_decode(encoded_bytes: bytes) -> List[int]:
    """
    Decode a VarByte-encoded byte stream back into integers.
    - Read one byte at a time.
    - Extract lowest 7 bits and accumulate until a byte with MSB = 0 is found.
    - Combine all groups to reconstruct the original integer.
    """
    decoded_numbers: List[int] = []
    num = shift = 0
    
    for byte in encoded_bytes:
        # Get lowest 7 bits and add into current number
        num |= (byte & 0x7F) << shift
        if byte & 0x80:
            # Increase by 7 to make room for next 7-bit group
            shift += 7
        else:
            # Append completed integer and reset
            decoded_numbers.append(num)
            num = shift = 0
    
    return decoded_numbers