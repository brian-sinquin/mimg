# Instructions for AI LLM Working on mimg Project

## Project Overview
mimg is a fast, SIMD-optimized command-line image processing tool written in Zig. It provides high-performance image manipulation with support for multiple formats (PNG, TGA, QOI, PAM, PBM, PGM, PPM, PCX), 40+ modifiers (color adjustments, filters, transforms), presets for reusable processing chains, and multithreaded batch processing.

## Requirements
- Zig 0.15.1 or later
- No external dependencies (uses zigimg library)

## Project Structure
```
build_exe.zig          # Executable build configuration
build_gallery.zig       # Gallery generation build config
build.zig               # Main build script
build.zig.zon           # Dependencies (zigimg)
src/
  main.zig              # Entry point
  app/app.zig           # Main application logic and CLI
  core/                 # Core functionality
    argument_parser.zig # Argument parsing
    cli.zig             # Command-line interface
    main.zig            # Core main logic
    types.zig           # Type definitions
    utils.zig           # Utility functions
  gallery/              # Gallery generation
    gallery_data.zig
    gallery.zig
  processing/           # Image processing modules
    basic.zig           # Basic operations
    color.zig           # Color adjustments
    filters.zig         # Image filters
    modifiers.zig       # Modifier system
    transforms.zig      # Geometric transforms
  testing/              # Testing and benchmarks
    benchmarks.zig
    tests.zig
  utils/                # Utilities
    simd_utils.zig      # SIMD operations
    worker.zig          # Multithreaded processing
docs/                   # Documentation
examples/               # Example files
batch_test/             # Batch testing
processed/              # Processed outputs
```

## Build Commands
- **Build**: `zig build`
- **Run**: `zig build run -- <args>`
- **Test**: `zig build -Dtests test`
- **Benchmarks**: `zig build -Dbenchmarks bench`
- **Quick build**: `zig build quick` (Debug, no LTO)
- **Release build**: `zig build release` (ReleaseFast, LTO, AVX2)
- **Clean**: `zig build clean`
- **Gallery**: `zig build -Dgallery` (enable gallery generation)

### Build Options
- `-Ddocs`: Generate documentation (not yet supported)
- `-Dgallery`: Enable gallery generation
- `-Dtests`: Enable test building
- `-Dbenchmarks`: Enable benchmark building
- `-Dtarget-name=<name>`: Target name for binary
- `-Dlto`: Enable Link Time Optimization (slower builds, faster runtime)
- `-Dstrip`: Strip debug symbols (smaller binary)
- `-Dstatic`: Force static linking
- `-Dcpu-features=<features>`: CPU features to enable (avx2, sse4_2)

## Key Features to Understand
1. **Modifiers**: Chainable image processing operations (brightness, saturation, sharpen, etc.)
2. **Presets**: Saved processing chains for reuse
3. **Batch Processing**: Multithreaded processing of multiple images
4. **SIMD Optimization**: High-performance operations using SIMD instructions
5. **Multiple Formats**: Support for various image formats via zigimg

## Development Guidelines
- Use SIMD operations where possible for performance
- Maintain multithreaded batch processing capabilities
- Follow Zig coding conventions and safety practices
- Add comprehensive tests for new features
- Update documentation for user-facing changes

## Useful MCPs for Zig 0.15.1 Development

### Library Documentation (mcp_context7_get-library-docs)
Use this to fetch up-to-date documentation for:
- **Zig Standard Library**: `/ziglang/zig` - Core Zig language features, data structures, I/O, etc.
- **zigimg**: `/zigimg/zigimg` - Image loading, saving, and format support
- **Other Zig libraries**: Search for specific packages as needed

### Memory/Knowledge Graph (mcp_memory_*)
Use for tracking project knowledge:
- `mcp_memory_create_entities`: Track key components, functions, or concepts
- `mcp_memory_create_relations`: Link related code elements
- `mcp_memory_search_nodes`: Find relevant code or concepts
- `mcp_memory_read_graph`: Review accumulated project knowledge

### Sequential Thinking (mcp_sequentialthi_sequentialthinking)
Use for complex problem-solving:
- Breaking down implementation tasks
- Debugging complex issues
- Planning refactoring or new features
- Analyzing performance problems

## Common Development Tasks
1. **Adding New Modifiers**: Implement in `processing/modifiers.zig`, register in the `modifiers` array, add function in appropriate module (`color.zig`, `filters.zig`, `transforms.zig`)
2. **Format Support**: Extend via zigimg integration in `core/utils.zig`
3. **Performance Optimization**: Use SIMD in `utils/simd_utils.zig`
4. **Testing**: Add unit tests in `testing/tests.zig`, benchmarks in `testing/benchmarks.zig`
5. **Documentation**: Update relevant files in `docs/`, especially `modifiers.md` which may be outdated
6. **Build System**: Modify `build.zig` for new options, update `build/build_exe.zig` for executable configuration

## TODO Items (Version 0.1.3)
- Enhance build system (optimize, etc.) - Partially completed with new build options
- Loading bar for long operations (dispatch, etc.)
- Better gallery generation (currently not elegant)
- Refactor image processing pipeline
- Optimize memory usage
- Implement caching for loaded images
- Update documentation to reflect all implemented modifiers
- Add sanitizers support (TODO in build_exe.zig)

## Code Quality Standards
- Use `zig fmt` for consistent formatting
- Run tests before commits: `zig build -Dtests test`
- Add benchmarks for performance-critical code
- Document public APIs and complex algorithms
- Follow Zig's error handling patterns with `!` and `catch`

## Debugging Tips
- Use `std.log` for debug output (controlled by verbose flag `-v`)
- Check SIMD alignment and memory boundaries
- Profile with benchmarks: `zig build -Dbenchmarks bench`
- Test with various image sizes and formats
- Use gallery generation to visualize processing results: `zig build -Dgallery`
- Run tests: `zig build -Dtests test`
- Check build options for optimization: `zig build release` for production builds

## Imperative behaviors
- Keep track of advancements in memory or markdown file
- Dont oversize your context
- Use sequential thinking when needed
- Update documentation after implementing new features
- Run tests and benchmarks before commits
- Maintain SIMD optimization for performance-critical code