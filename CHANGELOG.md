# Changelog

## [0.0.8](https://github.com/FBumann/fluxopt/compare/v0.0.7-alpha.0...v0.0.8) (2026-05-31)


### Features

* **api:** allow period-varying flow profiles via DataFrame; rename TimeSeries → Variate ([#165](https://github.com/FBumann/fluxopt/issues/165)) ([a8b535c](https://github.com/FBumann/fluxopt/commit/a8b535cd5ed0c4f4b3cd065136c594e5ad7c2d9f))


### Bug Fixes

* exclude all highspy 1.14.x versions ([#181](https://github.com/FBumann/fluxopt/issues/181)) ([5046e46](https://github.com/FBumann/fluxopt/commit/5046e466260b7c85070a355d8d504a13303d8736))


### Miscellaneous Chores

* release 0.0.8 ([d375ec5](https://github.com/FBumann/fluxopt/commit/d375ec59c02f50a5c9183ca5b91ec4d5e5c93f02))

## [0.0.7-alpha.0](https://github.com/FBumann/fluxopt/compare/v0.0.6-alpha.0...v0.0.7-alpha.0) (2026-04-29)


### Features

* add component-level status on Storage ([#145](https://github.com/FBumann/fluxopt/issues/145)) ([a536e4f](https://github.com/FBumann/fluxopt/commit/a536e4f3e40801392271e9881f57276bec2f260a))
* cache per-contributor effect contributions in Result ([#140](https://github.com/FBumann/fluxopt/issues/140)) ([2f9b8d0](https://github.com/FBumann/fluxopt/commit/2f9b8d048541f01686269b6418b6b81f4bc7f82a))
* piecewise conversion via linopy add_piecewise_formulation ([#147](https://github.com/FBumann/fluxopt/issues/147)) ([bf25e39](https://github.com/FBumann/fluxopt/commit/bf25e3999add3ff9a3bff056bf9af558499404fb))


### Bug Fixes

* **ci:** let config control prerelease default, dispatch overrides ([#111](https://github.com/FBumann/fluxopt/issues/111)) ([aadf16c](https://github.com/FBumann/fluxopt/commit/aadf16c6602d763dfd51da8856d240219248f38b))
* exclude highspy 1.14.0 due to MIP presolve bug and temporarilly the nevest xarray version ([#129](https://github.com/FBumann/fluxopt/issues/129)) ([2f753e4](https://github.com/FBumann/fluxopt/commit/2f753e4c8883cfdc99dfd12a0d1446638dda266a))
* include Investment costs in compute_effect_contributions ([#135](https://github.com/FBumann/fluxopt/issues/135)) ([6bc2d4b](https://github.com/FBumann/fluxopt/commit/6bc2d4b76352066794193a66d1373018a3aa98ce))
* scale effect per-hour bounds by timestep duration ([#127](https://github.com/FBumann/fluxopt/issues/127)) ([3473d9c](https://github.com/FBumann/fluxopt/commit/3473d9c1d3776d8f64bb24038ccebea4404b3368))
* use xarray-native comparison in effect_contributions validation ([#139](https://github.com/FBumann/fluxopt/issues/139)) ([e34d7f7](https://github.com/FBumann/fluxopt/commit/e34d7f7264a5f0c5643884459f52c0184831a394))

## [0.0.6-alpha.0](https://github.com/FBumann/fluxopt/compare/v0.0.5-alpha.0...v0.0.6-alpha.0) (2026-03-14)


### Features

* make all effect parameters period-aware ([#94](https://github.com/FBumann/fluxopt/issues/94)) ([e0f8e66](https://github.com/FBumann/fluxopt/commit/e0f8e66e6237751f14322aeb88677a1c2a82d469))

## [0.0.5-alpha.0](https://github.com/FBumann/fluxopt/compare/v0.0.4-alpha.0...v0.0.5-alpha.0) (2026-03-14)


### Features

* add Investment for multi-period build-timing optimization ([#85](https://github.com/FBumann/fluxopt/issues/85)) ([fe70bf8](https://github.com/FBumann/fluxopt/commit/fe70bf87133def20288484e9ea658bea43a9e770))

## [0.0.4-alpha.0](https://github.com/FBumann/fluxopt/compare/v0.0.3-alpha.0...v0.0.4-alpha.0) (2026-03-13)


### Features

* add multi-period optimization ([#81](https://github.com/FBumann/fluxopt/issues/81)) ([2df9100](https://github.com/FBumann/fluxopt/commit/2df9100f226db05e4a5530262dce184ce7de55bf))

## [0.0.3-alpha.0](https://github.com/FBumann/fluxopt/compare/v0.0.2-alpha.0...v0.0.3-alpha.0) (2026-03-11)


### Features

* prepare core for package ecosystem ([#59](https://github.com/FBumann/fluxopt/issues/59)) ([0c77bd0](https://github.com/FBumann/fluxopt/commit/0c77bd06882d6b154c87649a4a75b52e0edb8a79))
* replace Bus with Carrier ([#60](https://github.com/FBumann/fluxopt/issues/60)) ([c3035e1](https://github.com/FBumann/fluxopt/commit/c3035e18e565d75f8c0c2de1220e3a0936826236))
* require explicit carriers in optimize() ([#65](https://github.com/FBumann/fluxopt/issues/65)) ([e6b27d6](https://github.com/FBumann/fluxopt/commit/e6b27d61e5e3029f661b9eeb8eacbb1fb54eec28))
* require explicit carriers in optimize() and ModelData.build() ([#64](https://github.com/FBumann/fluxopt/issues/64)) ([f07df79](https://github.com/FBumann/fluxopt/commit/f07df79bf0c016d256ba14d85c4f3978f09eda5f))

## Changelog
