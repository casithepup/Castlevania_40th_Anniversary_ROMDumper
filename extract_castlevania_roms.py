#!/usr/bin/env python3
"""
Extract the embedded Castlevania NES images from libnes.so.

Usage:
    python extract_castlevania_roms.py libnes.so
    python extract_castlevania_roms.py libnes.so --out-dir recovered

Works with the ELF32 and ELF64 little-endian libnes.so files from the
Castlevania_1.0.1 package. Uses only the Python standard library.
"""

import argparse
import struct
import sys
from pathlib import Path

ROM_SIZE = 0x20010  # 16-byte iNES header + 128 KiB PRG data
XOR_KEYS = {
    "ROM_EN": bytes.fromhex("c299426c6e19726a15eb98f5e47461ae"),
    "ROM_JP": bytes.fromhex("d36897773bb816ea5549916b2b379e74"),
}
OUTPUT_NAMES = {
    "ROM_EN": "Castlevania_English_embedded.nes",
    "ROM_JP": "Castlevania_Japanese_embedded.nes",
}


def parse_elf(data: bytes):
    """Return (sections, symbol_values) for the named ROM symbols."""
    if len(data) < 16 or data[:4] != b"\x7fELF":
        raise ValueError("Input is not an ELF shared library (libnes.so).")

    elf_class = data[4]  # 1 = ELF32, 2 = ELF64
    byte_order = data[5]  # 1 = little-endian, 2 = big-endian
    if byte_order != 1:
        raise ValueError("Only little-endian ELF files are supported.")
    if elf_class not in (1, 2):
        raise ValueError(f"Unsupported ELF class: {elf_class}")

    if elf_class == 1:
        # ELF32 header: e_shoff at 32, e_shentsize at 46, e_shnum at 48
        shoff = struct.unpack_from("<I", data, 32)[0]
        shentsize, shnum = struct.unpack_from("<HH", data, 46)
        sh_fmt = "<IIIIIIIIII"
        sym_fmt = "<IIIBBH"
    else:
        # ELF64 header: e_shoff at 40, e_shentsize at 58, e_shnum at 60
        shoff = struct.unpack_from("<Q", data, 40)[0]
        shentsize, shnum = struct.unpack_from("<HH", data, 58)
        sh_fmt = "<IIQQQQIIQQ"
        sym_fmt = "<IBBHQQ"

    expected_sh_size = struct.calcsize(sh_fmt)
    if shentsize < expected_sh_size or shoff + shentsize * shnum > len(data):
        raise ValueError("ELF section-header table is invalid or truncated.")

    sections = []
    for i in range(shnum):
        fields = struct.unpack_from(sh_fmt, data, shoff + i * shentsize)
        if elf_class == 1:
            name, typ, flags, addr, off, size, link, info, align, entsize = fields
        else:
            name, typ, flags, addr, off, size, link, info, align, entsize = fields
        sections.append({
            "name_offset": name, "type": typ, "addr": addr,
            "offset": off, "size": size, "link": link, "entsize": entsize,
        })

    # SHT_SYMTAB = 2; SHT_DYNSYM = 11
    symbol_values = {}
    for sec in sections:
        if sec["type"] not in (2, 11) or sec["link"] >= len(sections):
            continue
        strsec = sections[sec["link"]]
        str_start = strsec["offset"]
        str_end = str_start + strsec["size"]
        strings = data[str_start:str_end]
        ent_size = sec["entsize"] or struct.calcsize(sym_fmt)
        if ent_size < struct.calcsize(sym_fmt):
            continue
        start, end = sec["offset"], sec["offset"] + sec["size"]
        for pos in range(start, min(end, len(data)), ent_size):
            if pos + struct.calcsize(sym_fmt) > len(data):
                break
            fields = struct.unpack_from(sym_fmt, data, pos)
            if elf_class == 1:
                name_off, value, size, info, other, shndx = fields
            else:
                name_off, info, other, shndx, value, size = fields
            if name_off >= len(strings):
                continue
            name = strings[name_off:].split(b"\0", 1)[0].decode("ascii", "replace")
            if name in XOR_KEYS:
                symbol_values[name] = (value, size, shndx)

    return sections, symbol_values


def recover(data: bytes, sections, symbol_values, symbol: str) -> bytes:
    if symbol not in symbol_values:
        raise ValueError(
            f"Symbol {symbol} was not found. Use the matching unstripped "
            "libnes.so from the original package."
        )
    value, size, section_index = symbol_values[symbol]
    if size != ROM_SIZE:
        raise ValueError(
            f"{symbol} has unexpected size 0x{size:x} (expected 0x{ROM_SIZE:x})."
        )
    if section_index == 0 or section_index >= len(sections):
        raise ValueError(f"{symbol} does not point to a normal ELF section.")

    section = sections[section_index]
    offset = section["offset"] + (value - section["addr"])
    encrypted = data[offset:offset + size]
    if len(encrypted) != size:
        raise ValueError(f"{symbol} data extends beyond the end of the file.")

    key = XOR_KEYS[symbol]
    rom = bytes(byte ^ key[i % len(key)] for i, byte in enumerate(encrypted))
    if rom[:4] != b"NES\x1a":
        raise ValueError(
            f"{symbol} was decoded, but the iNES signature is invalid. "
            "The library may use a different version or layout."
        )

    # Confirm the standard iNES PRG size agrees with the known image size.
    prg_banks = rom[4]
    chr_banks = rom[5]
    declared_size = 16 + prg_banks * 16_384 + chr_banks * 8_192
    if declared_size > len(rom):
        raise ValueError(
            f"{symbol} has an iNES header declaring {declared_size} bytes, "
            f"larger than the extracted {len(rom)} bytes."
        )
    return rom


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("libnes", type=Path, help="path to libnes.so")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("."),
        help="directory for extracted .nes files (default: current directory)",
    )
    args = parser.parse_args()

    try:
        data = args.libnes.read_bytes()
        sections, symbols = parse_elf(data)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for symbol, filename in OUTPUT_NAMES.items():
            rom = recover(data, sections, symbols, symbol)
            output = args.out_dir / filename
            output.write_bytes(rom)
            print(f"{symbol}: wrote {output} ({len(rom):,} bytes; iNES header OK)")
    except (OSError, ValueError, struct.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
