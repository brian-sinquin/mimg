# mimg Enhancement Ideas and Roadmap - Version 0.2.0

Based on comprehensive analysis of the mimg project, this document outlines strategic enhancements for code size reduction, readability improvement, optimization, and better modularity planned for **Version 0.2.0**.

## Version 0.2.0 Goals

### Breaking Changes Allowed
- **Refactor core architecture** for better modularity and extensibility
- **Redesign parameter system** to support mixed parameter types (filepaths, colors, coordinates)
- **Restructure processing pipeline** for improved performance and maintainability
- **Update CLI interface** to be more intuitive and powerful

### Backward Compatibility
- **Maintain all 42+ existing modifiers** with same functionality
- **Preserve existing command-line syntax** while adding new capabilities
- **Keep gallery generation** and batch processing working seamlessly

## 1. Core Architecture Improvements (v0.2.0)

### Modular Processing Pipeline
- **Extract common modifier patterns** into reusable traits/interfaces
- **Consolidate similar operations** across color.zig, filters.zig, and transforms.zig
- **Create a unified modifier framework** to reduce code duplication

### Memory Management Optimization
- **Implement image buffer pooling** to reduce allocations during batch processing
- **Add SIMD-aligned memory allocators** for better performance
- **Create lazy loading system** for large images

## 2. Code Organization & Modularity (v0.2.0)

### Processing Module Restructure
```zig
// New base module for common modifier functionality
// src/processing/core/modifier_base.zig
pub const ModifierResult = union(enum) {
    success: void,
    error: ProcessingError,
};

pub const ModifierInterface = struct {
    name: []const u8,
    apply: *const fn(image: *Image, params: ModifierParams) ModifierResult,
    validate_params: *const fn(params: ModifierParams) bool,
    param_types: []const ParamType,
};
```

### Unified Parameter System (v0.2.0 - Mixed Types Support)
```zig
// src/processing/core/parameters.zig
pub const ParamType = enum { 
    int, 
    float, 
    bool, 
    color,
    filepath,     // NEW in v0.2.0: For file-based operations
    string,       // NEW in v0.2.0: For text/enum parameters
    coordinate,   // NEW in v0.2.0: For x,y pairs
    dimension,    // NEW in v0.2.0: For width,height pairs
};

pub const ParamValue = union(ParamType) {
    int: i32,
    float: f32,
    bool: bool,
    color: struct { r: u8, g: u8, b: u8 },
    filepath: []const u8,
    string: []const u8,
    coordinate: struct { x: i32, y: i32 },
    dimension: struct { width: u32, height: u32 },
};

pub const ModifierParams = struct {
    values: []const ParamValue,
    
    pub fn getFloat(self: *const ModifierParams, index: usize) !f32 {
        if (index >= self.values.len) return error.IndexOutOfBounds;
        return switch (self.values[index]) {
            .float => |f| f,
            .int => |i| @floatFromInt(i),
            else => error.InvalidParameterType,
        };
    }
    
    pub fn getFilepath(self: *const ModifierParams, index: usize) ![]const u8 {
        if (index >= self.values.len) return error.IndexOutOfBounds;
        return switch (self.values[index]) {
            .filepath => |path| path,
            .string => |str| str,
            else => error.InvalidParameterType,
        };
    }
    
    pub fn getInt(self: *const ModifierParams, index: usize) !i32 {
        if (index >= self.values.len) return error.IndexOutOfBounds;
        return switch (self.values[index]) {
            .int => |i| i,
            .float => |f| @intFromFloat(f),
            else => error.InvalidParameterType,
        };
    }
    
    pub fn getColor(self: *const ModifierParams, index: usize) !struct { r: u8, g: u8, b: u8 } {
        if (index >= self.values.len) return error.IndexOutOfBounds;
        return switch (self.values[index]) {
            .color => |c| c,
            else => error.InvalidParameterType,
        };
    }
};

pub const Parameter = struct {
    name: []const u8,
    type: ParamType,
    min: ?f32 = null,
    max: ?f32 = null,
    default: ?ParamValue = null,
    description: []const u8,
};
```

## 3. Performance Optimizations (v0.2.0)

### SIMD Vectorization Enhancement
- **Batch pixel operations** using wider SIMD registers (AVX-512 where available)
- **Implement template-based SIMD dispatching** based on CPU features
- **Add compile-time SIMD optimization selection**

### Multithreading Improvements
- **Tile-based processing** for better cache locality
- **Work-stealing queue** for better load balancing
- **Adaptive thread count** based on image size and operation complexity

## 4. Code Size Reduction Strategies (v0.2.0)

### Macro-Based Code Generation (Updated for Mixed Parameters)
```zig
// src/processing/macros/modifier_gen.zig
// Generate repetitive modifier boilerplate
fn generateModifier(comptime name: []const u8, comptime func: anytype, comptime params: []const Parameter) type {
    return struct {
        pub const modifier_name = name;
        pub const parameters = params;
        pub fn apply(image: *Image, args: ModifierParams) !void {
            // Validate parameter types match expected
            if (args.values.len != params.len) return error.InvalidParameterCount;
            for (args.values, params) |value, param| {
                if (@as(ParamType, value) != param.type) return error.InvalidParameterType;
            }
            // Generated implementation with type-safe parameter access
            return func(image, args);
        }
    };
}
```

### Consolidate Similar Operations
- **Merge similar color adjustments** into parameterized functions
- **Create generic convolution kernel system** for filters
- **Unify geometric transform operations**

## 5. Error Handling & Validation (v0.2.0)

### Centralized Error Management
```zig
// src/core/errors.zig
pub const ProcessingError = error{
    InvalidParameters,
    InvalidParameterType,      // NEW in v0.2.0
    InvalidParameterCount,     // NEW in v0.2.0
    FileNotFound,             // NEW in v0.2.0: For filepath parameters
    InvalidFilePath,          // NEW in v0.2.0: For filepath validation
    UnsupportedFormat,
    OutOfMemory,
    InvalidImageDimensions,
    IOError,
};

pub fn validateAndExecute(modifier: ModifierInterface, image: *Image, params: ModifierParams) !void {
    if (!modifier.validate_params(params)) return ProcessingError.InvalidParameters;
    return modifier.apply(image, params);
}
```

## 6. Build System & Configuration (v0.2.0)

### Conditional Compilation
```zig
// build/feature_flags.zig
pub const Features = struct {
    simd_level: enum { none, sse4_2, avx2, avx512 } = .avx2,
    threading: bool = true,
    debug_output: bool = false,
    gallery_generation: bool = true,
    file_operations: bool = true,    // NEW in v0.2.0: Enable file-based modifiers
};
```

## 7. Testing & Benchmarking Framework (v0.2.0)

### Automated Performance Testing
- **Regression testing** for performance metrics
- **Memory usage profiling** during development
- **Cross-platform benchmark comparisons**
- **File I/O testing** for filepath-based modifiers

## 8. Documentation & Code Generation (v0.2.0)

### Self-Documenting Modifiers
```zig
// src/processing/core/documentation.zig
pub fn generateModifierDocs(comptime modifiers: []const ModifierInterface) []const u8 {
    // Generate markdown documentation from modifier definitions
    // Include parameter types and example usage for mixed parameter types
}
```

## 9. Implementation Roadmap (v0.2.0)

### Phase 1: Core Refactoring (v0.2.0)
1. Create `ModifierInterface` and `ModifierParams` with mixed type support
2. Extract common SIMD operations to shared utilities
3. Implement unified parameter validation system supporting multiple types
4. Add filepath parameter validation and file existence checking

### Phase 2: Performance Optimization (v0.2.0)
1. Add memory pooling for image buffers
2. Implement tile-based processing for large images
3. Optimize hot paths with profile-guided optimization
4. Cache loaded images for file-based operations

### Phase 3: Code Size Reduction (v0.2.0)
1. Generate modifier boilerplate with macros supporting mixed parameters
2. Consolidate similar operations
3. Remove duplicate validation logic
4. Create parameter parsing utilities for different types

### Phase 4: Advanced Features (v0.2.0)
1. Add preset system with JSON configuration supporting mixed parameter types
2. Implement plugin architecture for custom modifiers
3. Add real-time progress reporting
4. File operation modifiers (overlay, composite, texture mapping)

## 10. Immediate Actionable Steps (v0.2.0)

### High Priority
1. **Design ModifierParams system** to handle mixed parameter types
2. **Audit current modifier implementations** for parameter type patterns
3. **Implement parameter parsing** for filepaths, colors, coordinates
4. **Create type-safe parameter accessors** in ModifierParams
5. **Add file validation utilities** for filepath parameters

### Medium Priority
6. Extract common validation logic into shared utilities
7. Implement generic convolution kernel system for filters
8. Add memory pooling for frequent allocations
9. Create tile-based processing for large images
10. Implement adaptive threading based on workload

### Low Priority
11. Add plugin architecture for custom modifiers
12. Implement real-time progress reporting with loading bars
13. Create cross-platform benchmark suite
14. Add preset system with JSON configuration
15. Generate documentation from code annotations

## 11. Example File-Based Modifiers (v0.2.0)

### New Modifiers Using Filepaths
```zig
// Examples of modifiers planned for v0.2.0 that use filepath parameters:
// overlay <filepath> <x> <y> <opacity>          - Overlay another image
// texture <filepath> <blend_mode> <intensity>   - Apply texture from file
// mask <filepath> <invert>                      - Apply mask from file
// watermark <filepath> <position> <scale>       - Add watermark from file
// composite <filepath> <blend_mode>             - Composite with another image
// displacement <filepath> <strength>            - Displacement mapping
// normal-map <filepath> <strength>              - Apply normal map lighting
```

### Parameter Parsing Examples (v0.2.0)
```zig
// Command line examples for v0.2.0:
// mimg input.jpg overlay watermark.png 10 10 0.5 -o output.jpg
// mimg photo.jpg texture fabric.jpg multiply 0.3 -o textured.jpg
// mimg base.png composite overlay.png screen -o result.png
```

## 12. Code Quality Improvements (v0.2.0)

### Memory Safety
- **Add bounds checking** for all array accesses
- **Implement RAII patterns** for resource management
- **Use defer statements** for cleanup operations

### Error Handling
- **Standardize error types** across all modules
- **Add context to error messages** for better debugging
- **Implement error recovery** where appropriate

### Testing Strategy
- **Unit tests** for each modifier
- **Integration tests** for modifier chains
- **Performance regression tests**
- **Memory leak detection**
- **Cross-platform compatibility tests**

## 13. Performance Monitoring (v0.2.0)

### Metrics to Track
- **Processing time per modifier**
- **Memory usage patterns**
- **SIMD instruction utilization**
- **Thread utilization efficiency**
- **Cache hit rates**

### Benchmarking Infrastructure
- **Automated performance testing** in CI/CD
- **Historical performance tracking**
- **Comparison against reference implementations**
- **Real-world workload simulation**

## 14. Future Considerations (v0.2.0+)

### Extensibility
- **Plugin system** for third-party modifiers
- **Scripting interface** for complex processing chains
- **GPU acceleration** for compute-intensive operations
- **Distributed processing** for very large images

### User Experience
- **Interactive mode** for real-time parameter adjustment
- **GUI wrapper** for visual parameter tuning
- **Batch processing templates**
- **Undo/redo functionality**

## 15. Technical Debt Reduction (v0.2.0)

### Current Issues to Address
- **Duplicate validation logic** across modifiers
- **Inconsistent error handling** patterns
- **Manual parameter parsing** instead of automated
- **Hard-coded SIMD implementations** instead of generic
- **Limited test coverage** for edge cases

### Refactoring Priorities
1. Extract common validation into shared utilities
2. Standardize error handling across all modules
3. Implement generic parameter parsing system
4. Create template-based SIMD operations
5. Add comprehensive test coverage

## 16. Migration Plan from v0.1.3 to v0.2.0

### Breaking Changes
- **ModifierParams system** replaces simple float arrays
- **Enhanced parameter validation** with type checking
- **New file-based modifiers** require filepath handling
- **Restructured processing pipeline** for better modularity

### Compatibility Layer
- **Legacy parameter parsing** for existing command-line syntax
- **Automatic type conversion** where possible
- **Deprecation warnings** for old patterns
- **Migration guide** for custom modifications

## Notes (v0.2.0)
- **Version 0.2.0 allows breaking changes** for architectural improvements
- **Performance remains the top priority** - optimizations must not be sacrificed
- **SIMD optimizations** remain a core focus with enhanced capabilities
- **Mixed parameter system** must be type-safe and validated at runtime
- **File-based operations** should handle missing files gracefully with clear error messages
- **All 42+ existing modifiers** must remain functional with same behavior
- **Gallery generation** should continue to work as visual verification tool
- **Batch processing capabilities** must be preserved and enhanced
- **Parameter validation** should catch type mismatches early with helpful error messages
- **File I/O operations** should be cached when processing multiple images
- **Backward compatibility** maintained where possible, with clear migration path for breaking changes
- **Documentation updates** required for all new parameter types and file-based modifiers
- **Testing coverage** must include all new parameter types and edge cases