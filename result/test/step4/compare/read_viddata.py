"""Read-only extractor for this small BIFF8 .xls, using standard library.

Supports regular OLE Workbook streams, SST without continuation, LABELSST,
NUMBER/RK/MULRK, sheet names and merged cells. Reject unsupported SST
continuations or mini streams rather than silently misread source values.
"""
import struct
from pathlib import Path


def extract(path):
    data = Path(path).read_bytes()
    if data[:8] != bytes.fromhex('d0cf11e0a1b11ae1'):
        raise ValueError('Expected OLE compound file')
    size = 1 << struct.unpack_from('<H', data, 30)[0]
    def sector(i):
        return data[(i+1)*size:(i+2)*size]
    fat_sectors = [v for v in struct.unpack_from('<109I', data, 76) if v < 0xfffffffa]
    fat = []
    for index in fat_sectors:
        fat.extend(struct.unpack('<'+str(size//4)+'I', sector(index)))
    def chain(start):
        parts, seen = [], set()
        while start < 0xfffffffa:
            if start in seen:
                raise ValueError('Cyclic OLE chain')
            seen.add(start)
            parts.append(sector(start))
            start = fat[start]
        return b''.join(parts)
    directory = chain(struct.unpack_from('<I', data, 48)[0])
    workbook = None
    for offset in range(0, len(directory), 128):
        entry = directory[offset:offset+128]
        length = struct.unpack_from('<H', entry, 64)[0]
        name = entry[:max(0, length-2)].decode('utf-16le')
        if name in ('Workbook', 'Book'):
            stream_size = struct.unpack_from('<Q', entry, 120)[0]
            if stream_size < 4096:
                raise ValueError('Mini stream not supported')
            workbook = chain(struct.unpack_from('<I', entry, 116)[0])[:stream_size]
    if workbook is None:
        raise ValueError('Workbook stream missing')
    records, offset = [], 0
    while offset+4 <= len(workbook):
        kind, length = struct.unpack_from('<HH', workbook, offset)
        records.append((offset, kind, workbook[offset+4:offset+4+length]))
        offset += length+4
    strings, sheets = [], []
    for ri, (offset, kind, payload) in enumerate(records):
        if kind == 0x85:
            start = struct.unpack_from('<I', payload)[0]
            length, flags = payload[6:8]
            name = payload[8:8+length*(2 if flags & 1 else 1)].decode('utf-16le' if flags & 1 else 'latin1')
            sheets.append((start, name))
        if kind == 0xfc:
            if records[ri+1][1] == 0x3c:
                raise ValueError('SST continuation unsupported')
            unique = struct.unpack_from('<I', payload, 4)[0]
            pos = 8
            for _ in range(unique):
                length, flags = struct.unpack_from('<HB', payload, pos)
                pos += 3
                rich = struct.unpack_from('<H', payload, pos)[0] if flags & 8 else 0
                pos += 2 if flags & 8 else 0
                extension = struct.unpack_from('<I', payload, pos)[0] if flags & 4 else 0
                pos += 4 if flags & 4 else 0
                width = 2 if flags & 1 else 1
                strings.append(payload[pos:pos+length*width].decode('utf-16le' if width == 2 else 'latin1'))
                pos += length*width+rich*4+extension
    def rk(value):
        if value & 2:
            number = (value if value < 0x80000000 else value-0x100000000) >> 2
        else:
            number = struct.unpack('<d', struct.pack('<II', 0, value & 0xfffffffc))[0]
        return number/100 if value & 1 else number
    result = []
    for si, (start, name) in enumerate(sheets):
        end = sheets[si+1][0] if si+1 < len(sheets) else len(workbook)
        cells, merges = [], []
        for offset, kind, payload in records:
            if not start <= offset < end:
                continue
            if kind in (0xfd, 0x203, 0x27e):
                row, col = struct.unpack_from('<HH', payload)
                value = (strings[struct.unpack_from('<I', payload, 6)[0]] if kind == 0xfd
                         else struct.unpack_from('<d', payload, 6)[0] if kind == 0x203
                         else rk(struct.unpack_from('<I', payload, 6)[0]))
                cells.append(dict(row=row+1, column=col+1, value=value))
            elif kind == 0xbd:
                row, col = struct.unpack_from('<HH', payload)
                for pos in range(4, len(payload)-2, 6):
                    cells.append(dict(row=row+1, column=col+1,
                                      value=rk(struct.unpack_from('<I', payload, pos+2)[0])))
                    col += 1
            elif kind == 0xe5:
                count = struct.unpack_from('<H', payload)[0]
                merges.extend([list(struct.unpack_from('<4H', payload, 2+8*i)) for i in range(count)])
        result.append(dict(sheet=name, cells=sorted(cells, key=lambda c:(c['row'], c['column'])),
                           merged_ranges_zero_based=merges))
    return result


if __name__ == '__main__':
    import json
    root = Path(__file__).resolve().parents[4]
    result = extract(root/'DATA/labeled/VidData.xls')
    output = Path(__file__).resolve().parent.parent/'raw_mfdfa/viddata_extracted.json'
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    for sheet in result:
        print(sheet['sheet'])
        for cell in sheet['cells']:
            print(cell)
