# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.5] - 2025-12-31

### Added

- New comprehensive website structure with dedicated sections:
  - Getting Started guide with quick examples and tips
  - Features & Modifiers reference with detailed documentation
  - Download page with platform-specific instructions
  - API Reference for library usage
- Styled download buttons for Windows, Linux, and macOS platforms
- Artist palette emoji (🎨) as website favicon
- Gallery preview section on homepage

### Changed

- Restructured website content for better navigation and user experience
- Updated navigation menu with cleaner, more intuitive organization
- Migrated all documentation from `/docs` folder to website
- Improved gallery grid layout with smaller cards (220px min width for more items per row)
- Simplified download links to always point to latest release (version-less artifact names)
- Updated CI/CD workflow to create stable artifact names without version numbers
- Fixed browser title display (removed `<span>` tags issue)

### Removed

- `/docs` folder (content migrated to website)
- Old website pages: modifiers.smd, installation.smd, examples.smd, formats.smd, presets.smd

## [0.1.4] - 2025-10-15

### Changed

- Major performance improvements across image resizing, gradients, and color operations
- Optimized core math and filter routines for faster processing
- Expanded filter capabilities and improved code organization
- All existing commands and APIs remain fully compatible

### Fixed

- All tests and builds pass
- No breaking changes

## [0.1.3] and earlier

See git history for previous releases.
