
# mimg
[![mimg](https://github.com/brian-sinquin/mimg/actions/workflows/dev.yml/badge.svg)](https://github.com/brian-sinquin/mimg/actions/workflows/dev.yml)
---

A small CLI tool written in Zig for experimenting with image manipulation pipelines. It loads a source image, applies a chain of modifiers, and writes the processed result to disk.

## Features

- Load PNG images via [`zigimg`](https://github.com/Snektron/zigimg)
- Apply modifiers such as color inversion, nearest-neighbour resize, and cropping
- Chain multiple modifiers in a single invocation
- Configure the output filename

## Project layout

```text
src/
  app.zig      # Top-level orchestration and lifecycle management
  cli.zig      # Argument registry, parsing, and help text
  basic.zig    # Image modifier implementations
  types.zig    # Shared data structures and option metadata
  utils.zig    # IO helpers for loading/saving images and parsing args
```

## Building

```cmd
zig build
```

## Running

```cmd
zig build run -- examples\lena.png invert resize 200 200 -o out.png
```

Global options such as `--help` or `--output` can appear either before or after the image path:

```cmd
zig build run -- --output result.png examples\lena.png invert
```

### Available options

- `invert` &mdash; invert the image colours
- `resize <width> <height>` &mdash; resize the image using nearest neighbour sampling
- `crop <x> <y> <width> <height>` &mdash; crop the image from a top-left coordinate
- `rotate <degrees>` &mdash; rotate the image clockwise by any angle (canvas automatically resizes to fit)
- `--output <filename>` (`-o`) &mdash; set the output filename
- `--output-dir <directory>` (`-d`) &mdash; write the output image inside the given directory (created if missing)
- `--list-modifiers` (`-L`) &mdash; show available modifiers and exit
- `--verbose` (`-v`) &mdash; enable additional logging during processing
- `--help` (`-h`) &mdash; print usage information

The modifiers can be chained; they are executed in the order they appear on the command line.

## Example

```cmd
zig build run -- examples\lena.png invert crop 10 10 100 150 --output lena_crop.png
```

Rotate before cropping:

```cmd
zig build run -- examples\lena.png rotate 37.5 crop 10 10 100 150 --output lena_rot_crop.png
```

List modifiers without processing an image:

```cmd
zig build run -- --list-modifiers
```

## Examples Gallery

See [examples/gallery/gallery.md](examples/gallery/gallery.md) for a comprehensive gallery of all available modifiers with example images and usage details.

## Next steps

- Expand the modifier library (rotation, filters, colour adjustments)
- Improve argument validation and error messaging
- Add wildcard support to batch-process multiple images
- Introduce automated tests once more helpers are available
