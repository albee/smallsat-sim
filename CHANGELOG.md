# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0]

Internal release.

## [0.2.0]

Initial release for stable version of MuJuCo CPU simulation and in-work planning/control algorithms. Additional development and cleanup is planned for upcoming 1.0 release.

### Added

- ADA* adaptive waypoint planning
- MPCC and GP-MPCC mid-horizon planners
- In-progress mjx integration and RL simulation infrastructure
- Stable "lunar gateway" sample environment
- Visualization and collision detection utiltities, including gateway mesh

## [0.3.0]

Added support for RL training via mjx on GPU, including a simple 3DOF example.

- Complete RL training pipeline via mjx, including Docker environment
- A 3DOF RL training example, using PPO