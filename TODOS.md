# Future Features

## Options

- Custom output path
- ~~Custom output directory~~ (done)
- ~~List modifiers option~~ (done)
- ~~Verbose flag~~ (done)

## Modifiers

Functions that take arguments (or not) to modify the loaded image or derive a new one.
Each modifier should support being chained in sequence.

## Better argument parsing

Expect CLI options such as `--foo value` (with a shorter alias `-f`).
After options, the remaining arguments should describe a list of modifiers with their parameters.

Example: `mimg file.png invert resize 200 200 rotate90`

A good strategy would be to treat the argument list as a stack and pop values as needed depending on the current context.
For maintainability, base this on a registry of modifiers that records the parameter count, a description, and a callback to the real function.

## Wildcard

Add the possibility to apply the chain of modifiers to a set of files using a wildcard.
Example: `mimg file_*.png invert resize 200 200 rotate90`

