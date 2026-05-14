# Createc `.dat` Reader Notes

ProbeFlow's Createc image reader is designed to preserve raw-file provenance
while returning display-safe scan arrays for analysis.

## Payload Boundary

Createc image `.dat` files contain a text header followed by a `DATA` marker and
a zlib-compressed little-endian `float32` payload. In a hex or text viewer this
often appears as `DATAx`, because the first zlib byte is commonly `0x78`
(`x`).

The reader now accepts only a `DATA` candidate that can be decompressed as zlib
from the byte immediately after the four marker bytes. This prevents header
comments or metadata values containing the word `DATA` from being mistaken for
the binary payload boundary.

## Header Parsing

Createc headers use simple `key=value` lines. Some lines can contain both an
internal and display name:

```text
InternalName / DisplayName=value
```

ProbeFlow stores the display key and also keeps the internal key as an alias.
This keeps exact lookups such as `Channels` distinct from `ScanChannels`, while
still allowing old and new Createc header spellings to be read.

Some Createc files spell Angstrom with the Latin-1 `Å` byte. Known Dacto aliases
such as `Dacto[Å]z` are normalized to `Dacto[A]z` for calibration lookup.

## Decoded Arrays Versus Original File Contents

`CreatecDatDecodeReport.decoded_channels_dac` is the canonical array field. It
contains decoded DAC values after ProbeFlow's safety cleanup:

1. the zlib payload is decoded as little-endian `float32`;
2. only the complete declared channel stack is reshaped;
3. trailing non-image payload floats are ignored and recorded;
4. incomplete rows are trimmed when explicit or heuristic evidence indicates a
   partial scan;
5. the first stored column is removed by default.

The historical `raw_channels_dac` property remains as a compatibility alias, but
it points to the same cleaned decoded arrays. Use `original_header`,
`original_Nx`, and `original_Ny` when the acquisition dimensions or raw header
matter.

## First-Column Artifact

Real Createc image fixtures show a strong scan-line-start artifact in the first
stored column. ProbeFlow removes that column from every decoded plane by
default, then updates the decoded `Num.X` header value to match the returned
arrays.

This behavior is intentional. It prevents the artificial vertical column from
reappearing in the browser, processing tools, exported images, and conversions.
The original dimensions remain available on the decode report for diagnostics.

## Partial Scans

When `ImageYPosMax` is present, complete real fixtures record it as `Num.Y + 1`.
ProbeFlow interprets it as the one-based next Y position, so completed rows are
`ImageYPosMax - 1`. If that value is inside the declared image height, the scan
is marked partial and decoded arrays are trimmed to the completed rows.

If `ImageYPosMax` is absent or outside the declared height, ProbeFlow falls back
to the existing channel-0 row heuristic. `is_partial_scan`, `image_y_pos_max`,
`original_Ny`, and `trimmed_Ny` are recorded in the decode report so callers can
distinguish a completed smaller image from an interrupted acquisition.
