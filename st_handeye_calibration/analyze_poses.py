#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation

with open('data/poses.csv') as f:
    lines = [line.strip() for line in f if line.strip()]

data = []
for line in lines:
    vals = [float(x) for x in line.split(',')]
    data.append(vals)

data = np.array(data)
tx, ty, tz = data[:, 0], data[:, 1], data[:, 2]
rx, ry, rz = data[:, 3], data[:, 4], data[:, 5]
n_frames = len(data)

print("=" * 80)
print("POSE MEASUREMENT QUALITY ANALYSIS")
print("=" * 80)
print(f"tx: mean={np.mean(tx):.2f}, std={np.std(tx):.2f}, range=[{np.min(tx):.2f}, {np.max(tx):.2f}]")
print(f"ty: mean={np.mean(ty):.2f}, std={np.std(ty):.2f}, range=[{np.min(ty):.2f}, {np.max(ty):.2f}]")
print(f"tz: mean={np.mean(tz):.2f}, std={np.std(tz):.2f}, range=[{np.min(tz):.2f}, {np.max(tz):.2f}]")
print(f"Translation magnitude: mean={np.mean(np.sqrt(tx**2+ty**2+tz**2)):.2f} mm")

# 2. Rotation angle analysis
print("\n2. ROTATION ANGLE ANALYSIS (degrees)")
print("-" * 40)
rx, ry, rz = data['rx'].values, data['ry'].values, data['rz'].values
print(f"rx: mean={np.mean(rx):.2f}, std={np.std(rx):.2f}")
print(f"ry: mean={np.mean(ry):.2f}, std={np.std(ry):.2f}")
print(f"rz: mean={np.mean(rz):.2f}, std={np.std(rz):.2f}")

# 3. 180° Flip Ambiguity Analysis
print("\n3. 180° FLIP AMBIGUITY ANALYSIS")
print("-" * 40)

# Check if rx is close to ±180°
rx_normalized = np.where(rx > 90, rx - 360, rx)  # Normalize to [-180, 180]
print("\nFrame-by-frame rx values (normalized to [-180, 180]):")
for i, (rx_raw, rx_norm) in enumerate(zip(rx, rx_normalized)):
    flip_sign = "+" if rx_raw > 0 else "-"
    near_180 = abs(abs(rx_norm) - 180) < 20
    status = "⚠️ NEAR ±180°" if near_180 else "✓ OK"
    print(f"  Frame {i+1:2d}: rx={rx_raw:8.2f}° (norm: {rx_norm:8.2f}°) {status}")

# Count frames near ±180°
near_pos_180 = np.sum(np.abs(rx - 180) < 20)
near_neg_180 = np.sum(np.abs(rx + 180) < 20)
print(f"\nFrames near +180°: {near_pos_180}/{n_frames}")
print(f"Frames near -180°: {near_neg_180}/{n_frames}")

# 4. Relative pose changes (consistency check)
print("\n4. CONSECUTIVE FRAME POSE CHANGES")
print("-" * 40)

# Convert to rotation matrices for proper comparison
def euler_to_rotmat(rx_deg, ry_deg, rz_deg):
    return Rotation.from_euler('xyz', [rx_deg, ry_deg, rz_deg], degrees=True).as_matrix()

def rotmat_to_euler(R):
    return Rotation.from_matrix(R).as_euler('xyz', degrees=True)

print("\nFrame | Δtx  | Δty  | Δtz  | Δrot (deg) | Δtrans (mm)")
print("-" * 60)
for i in range(1, n_frames):
    # Translation change
    dtx = tx[i] - tx[i-1]
    dty = ty[i] - ty[i-1]
    dtz = tz[i] - tz[i-1]
    dtrans = np.sqrt(dtx**2 + dty**2 + dtz**2)
    
    # Rotation change (using rotation matrices)
    R_prev = euler_to_rotmat(rx[i-1], ry[i-1], rz[i-1])
    R_curr = euler_to_rotmat(rx[i], ry[i], rz[i])
    R_rel = R_curr @ R_prev.T  # Relative rotation
    angle_axis = Rotation.from_matrix(R_rel).as_rotvec()
    drot_deg = np.linalg.norm(angle_axis) * 180 / np.pi
    
    print(f"  {i:2d}-{i+1:2d} | {dtx:+6.1f} | {dty:+6.1f} | {dtz:+6.1f} | {drot_deg:9.2f}  | {dtrans:8.2f}")

# 5. Identify problematic frames
print("\n5. SUSPICIOUS FRAMES")
print("-" * 40)

# Large rx deviation from ±180°
rx_deviation_from_180 = np.minimum(np.abs(rx - 180), np.abs(rx + 180))
suspicious_rx = np.where(rx_deviation_from_180 > 25)[0]
if len(suspicious_rx) > 0:
    print(f"Frames with rx far from ±180° (deviation > 25°):")
    for idx in suspicious_rx:
        print(f"  Frame {idx+1}: rx={rx[idx]:.2f}° (deviation: {rx_deviation_from_180[idx]:.2f}°)")

# Large consecutive rotation changes
large_rot_changes = []
for i in range(1, n_frames):
    R_prev = euler_to_rotmat(rx[i-1], ry[i-1], rz[i-1])
    R_curr = euler_to_rotmat(rx[i], ry[i], rz[i])
    R_rel = R_curr @ R_prev.T
    angle_axis = Rotation.from_matrix(R_rel).as_rotvec()
    drot_deg = np.linalg.norm(angle_axis) * 180 / np.pi
    if drot_deg > 30:
        large_rot_changes.append((i, drot_deg))

if len(large_rot_changes) > 0:
    print(f"\nLarge rotation changes between consecutive frames (>30°):")
    for idx, drot in large_rot_changes:
        print(f"  Frame {idx}→{idx+1}: {drot:.2f}°")

# 6. Quaternion representation (to verify no flip in actual rotation)
print("\n6. QUATERNION REPRESENTATION (w, x, y, z)")
print("-" * 40)
print("Frame |   w    |   x    |   y    |   z    |  rx(deg)")
print("-" * 60)
for i in range(n_frames):
    R = euler_to_rotmat(rx[i], ry[i], rz[i])
    q = Rotation.from_matrix(R).as_quat()  # (x, y, z, w) format
    print(f"  {i+1:2d}  | {q[3]:+.4f} | {q[0]:+.4f} | {q[1]:+.4f} | {q[2]:+.4f} | {rx[i]:+8.2f}")

# 7. Flip ambiguity detection using quaternion sign
print("\n7. FLIP AMBIGUITY ANALYSIS (Quaternion sign)")
print("-" * 40)
quats = []
for i in range(n_frames):
    R = euler_to_rotmat(rx[i], ry[i], rz[i])
    q = Rotation.from_matrix(R).as_quat()  # (x, y, z, w)
    quats.append(q)
quats = np.array(quats)

# Check quaternion signs consistency (w component)
w_values = quats[:, 3]
print(f"Quaternion w-component: min={np.min(w_values):.4f}, max={np.max(w_values):.4f}")
print(f"w > 0: {np.sum(w_values > 0)} frames, w < 0: {np.sum(w_values < 0)} frames")

# Identify flip frames
flip_frames = []
for i in range(1, n_frames):
    q_prev = quats[i-1]
    q_curr = quats[i]
    # Check if quaternions represent same rotation (q == q' or q == -q')
    dot = np.dot(q_prev, q_curr)
    if dot < 0:
        flip_frames.append(i)
        print(f"  Frame {i}: quaternion sign flip detected (dot={dot:.4f})")

print(f"\nTotal flip frames: {len(flip_frames)}/{n_frames-1}")

# 8. Summary and recommendations
print("\n" + "=" * 80)
print("SUMMARY AND RECOMMENDATIONS")
print("=" * 80)

print("""
ROOT CAUSES IDENTIFIED:

1. **180° FLIP AMBIGUITY**: {}/{} frames have rx near ±180°
   - This is a classic Euler angle singularity at gimbal lock
   - The rx ≈ 180° means the flange is nearly upside-down
   - Small noise causes flip between +180° and -180°
   
2. **EULER ANGLE INSTABILITY**: When rx ≈ 180°, the rotation
   representation becomes unstable - ry and rz can swing wildly
   while representing nearly the same rotation.

3. **POSE MEASUREMENT QUALITY**: The flip ambiguity affects
   hand-eye calibration constraint equations.

RECOMMENDATIONS:

1. **Use quaternion representation** internally instead of Euler angles
   - Quaternions avoid gimbal lock singularities
   - Signed flip (q vs -q) is still an issue but easier to handle

2. **Normalize quaternion signs** before optimization:
   - Ensure all quaternions have consistent sign (e.g., w > 0)
   - Or use the `flip_frames` identified above to filter

3. **Consider re-collecting data** with different robot orientations
   to avoid the rx ≈ ±180° region if possible

4. **Use graph optimization** which is more robust to local noise
   than closed-form methods like Tsai's algorithm

5. **Filter inconsistent frames** using AX=XB constraint checking
""".format(near_pos_180 + near_neg_180, n_frames))