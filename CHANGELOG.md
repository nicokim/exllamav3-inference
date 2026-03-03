# Changelog

## [0.4.0](https://github.com/nicokim/exllamav3-inference/compare/v0.3.5...v0.4.0) (2026-03-03)


### ⚠ BREAKING CHANGES

* ExLlamaV3 is now vendorized — no external install needed. A single `pip install` compiles both CUDA extensions (exllamav3_ext + exllamav3_opt_ext).

### Features

* vendorize exllamav3 runtime and fix stale RoPE positions ([#19](https://github.com/nicokim/exllamav3-inference/issues/19)) ([fe50bb7](https://github.com/nicokim/exllamav3-inference/commit/fe50bb7bb69d22d2baa6f5f8d9f4717951c4f3a6))

## [0.3.5](https://github.com/nicokim/exllamav3-inference/compare/v0.3.4...v0.3.5) (2026-03-02)


### Bug Fixes

* stale RoPE positions in decode loop and PrefixCache recurrent state ([#17](https://github.com/nicokim/exllamav3-inference/issues/17)) ([b8884a8](https://github.com/nicokim/exllamav3-inference/commit/b8884a82087c8afbb74bbfc5820c07ad73b878ea))

## [0.3.4](https://github.com/nicokim/exllamav3-inference/compare/v0.3.3...v0.3.4) (2026-03-02)


### Bug Fixes

* remove last_tokens_only from prefill to fix GatedDeltaNet state ([#15](https://github.com/nicokim/exllamav3-inference/issues/15)) ([b9204c0](https://github.com/nicokim/exllamav3-inference/commit/b9204c0867ef30274083a8c3897b9d7bfae922f8))

## [0.3.3](https://github.com/nicokim/exllamav3-inference/compare/v0.3.2...v0.3.3) (2026-03-02)


### Bug Fixes

* prefix cache copy_ crash inside inference_mode ([#13](https://github.com/nicokim/exllamav3-inference/issues/13)) ([6a6a5b1](https://github.com/nicokim/exllamav3-inference/commit/6a6a5b108249c39ce27a5190a24191ea39e139cd))

## [0.3.2](https://github.com/nicokim/exllamav3-inference/compare/v0.3.1...v0.3.2) (2026-03-02)


### Bug Fixes

* add refactor/perf/docs to release-please changelog sections ([#11](https://github.com/nicokim/exllamav3-inference/issues/11)) ([4de9df1](https://github.com/nicokim/exllamav3-inference/commit/4de9df18e66c2ba02865f8a6affbd78c26a46ee4))


### Refactors

* remove strategy abstraction, use exllamav3 native APIs ([#10](https://github.com/nicokim/exllamav3-inference/issues/10)) ([ec5d260](https://github.com/nicokim/exllamav3-inference/commit/ec5d260af0da19a683796e38336349ca28dd4f47))

## [0.3.1](https://github.com/nicokim/exllamav3-inference/compare/v0.3.0...v0.3.1) (2026-03-02)


### Bug Fixes

* correct exllamav3 upstream URL in README ([#6](https://github.com/nicokim/exllamav3-inference/issues/6)) ([b98d3a4](https://github.com/nicokim/exllamav3-inference/commit/b98d3a494123d313b210216b64114b5a4bd9b71a))

## [0.3.0](https://github.com/nicokim/exllamav3-inference/compare/v0.2.0...v0.3.0) (2026-03-02)


### Features

* fused CUDA kernels, FP8 cache ABC, cleanup dead modules ([#1](https://github.com/nicokim/exllamav3-inference/issues/1)) ([89c5746](https://github.com/nicokim/exllamav3-inference/commit/89c5746dded5c3c5e375d5f4bc4ceb544faab027))


### Bug Fixes

* use plain v* tags without package prefix in release-please ([#4](https://github.com/nicokim/exllamav3-inference/issues/4)) ([1fbd15d](https://github.com/nicokim/exllamav3-inference/commit/1fbd15d8302c0128dff6c216eb5a5c9da20d4bc2))

## [0.2.0](https://github.com/nicokim/exllamav3-inference/compare/exllamav3-inference-v0.1.0...exllamav3-inference-v0.2.0) (2026-03-02)


### Features

* fused CUDA kernels, FP8 cache ABC, cleanup dead modules ([#1](https://github.com/nicokim/exllamav3-inference/issues/1)) ([89c5746](https://github.com/nicokim/exllamav3-inference/commit/89c5746dded5c3c5e375d5f4bc4ceb544faab027))
