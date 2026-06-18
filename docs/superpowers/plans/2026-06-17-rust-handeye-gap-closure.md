# Rust Handeye Gap Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the Rust hand-eye backend so it fully supports the remaining product gaps while keeping the frontend depth path disabled by default.

**Architecture:** Extend the existing Rust Tauri backend in `src-tauri/src/lib.rs`. Preserve the current frontend request shape, wire the dormant depth fields through the Rust calibration and ChArUco helpers, and lock the behavior down with focused Rust unit tests.

**Tech Stack:** Rust, Tauri, OpenCV, nalgebra, existing `cargo test` suite

---

### Task 1: Add regression tests for missing depth plumbing

**Files:**
- Modify: `src-tauri/src/lib.rs`
- Test: `src-tauri/src/lib.rs`

- [ ] Add a test that proves `detect_charuco` forwards `depth_path` into the Rust draw helper path and can return populated `camera_point` rows when depth is available.
- [ ] Add a test that proves the calibration path can report `depth_used: true` when `use_depth` is enabled and valid depth measurements are available.
- [ ] Run the focused failing Rust tests and confirm they fail for the current implementation.

### Task 2: Implement depth-capable Rust calibration plumbing

**Files:**
- Modify: `src-tauri/src/lib.rs`

- [ ] Parse `use_depth` into explicit off/optional/required behavior in the Rust backend.
- [ ] Match RGB images to optional depth files and derive `measured_object_to_camera` from depth when enabled, falling back to PnP when allowed.
- [ ] Preserve current frontend-default behavior by leaving `useDepth: "off"` in the React request path.
- [ ] Set `depth_used` from actual measurements instead of hardcoding `false`.

### Task 3: Implement depth-capable Rust ChArUco tool behavior

**Files:**
- Modify: `src-tauri/src/lib.rs`

- [ ] Forward `CharucoRequest.depth_path` into `charuco_detect_and_draw`.
- [ ] Populate `camera_point` rows from depth samples when depth is provided and valid.
- [ ] Keep non-depth detection behavior unchanged.

### Task 4: Verify end-to-end implementation evidence

**Files:**
- Modify: none

- [ ] Run `cargo test --manifest-path src-tauri/Cargo.toml`.
- [ ] Summarize which legacy implementation paths were removed and which runtime paths remain in the shipped app.
