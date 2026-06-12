import { useEffect, useMemo, useRef, useState } from "react";
import { RefreshCcw } from "lucide-react";

const DEFAULT_ORIGIN_RADIUS_M = 0.005;
const SELECTED_ORIGIN_RADIUS_M = 0.007;

export type PreviewFrame = {
  index: number;
  imagePath?: string;
  used: boolean;
  cameraInBase: number[][];
  boardInBase: number[][];
  boardInFocus?: number[][];
};

export type PreviewLayerVisibility = {
  showBaseAxes: boolean;
  showCameraFrames: boolean;
  showBoardFrames: boolean;
  showUnifiedBoardFrame: boolean;
  showLabels: boolean;
};

type PreviewLabel = {
  id: string;
  text: string;
  position: { x: number; y: number };
  kind: "camera" | "board" | "unified" | "dimension" | "robot";
};

type SceneState = {
  THREE: any;
  scene: any;
  camera: any;
  renderer: any;
  controls: {
    update: () => void;
    dispose: () => void;
    target: { set: (x: number, y: number, z: number) => void };
    minDistance: number;
  };
  baseAxes: any;
  frameRoot: any;
  referenceRoot: any;
  boundingBoxRoot: any;
  anchorMap: Map<number, { camera: any; board: any }>;
  referenceAnchor: any | null;
  baseAnchor: any;
  animationFrame: number | null;
  resizeObserver: ResizeObserver | null;
};

type BoardMatrixMode = "base" | "focus";

function fileNameFromPath(path: string) {
  return path.split(/[\\/]/).pop() || path;
}

function frameName(frame: PreviewFrame) {
  return `Frame ${String(frame.index).padStart(3, "0")}`;
}

function frameTitle(frame: PreviewFrame) {
  const imageName = frame.imagePath ? fileNameFromPath(frame.imagePath) : "未命名帧";
  return `${frameName(frame)} ${imageName}`;
}

function matrixTranslation(matrix: number[][]) {
  return [
    matrix[0]?.[3] ?? 0,
    matrix[1]?.[3] ?? 0,
    matrix[2]?.[3] ?? 0,
  ];
}

function boardMatrixForFrame(frame: PreviewFrame, mode: BoardMatrixMode) {
  return mode === "focus" ? (frame.boardInFocus ?? frame.boardInBase) : frame.boardInBase;
}

function storedBoardMatrix(object3D: any, mode: BoardMatrixMode) {
  return mode === "focus"
    ? (object3D.userData.focusMatrix ?? object3D.userData.baseMatrix)
    : (object3D.userData.baseMatrix ?? object3D.userData.focusMatrix);
}

function applyStoredBoardMatrix(THREE: any, object3D: any, mode: BoardMatrixMode) {
  applyMatrixToObject3D(THREE, object3D, storedBoardMatrix(object3D, mode));
}

export function CoordinatePreview3D({
  frames,
  selectedFrameIndex,
  hoveredFrameIndex,
  layers,
  axisScale,
  referenceBoardInBase,
  referenceBoardInFocus,
}: {
  frames: PreviewFrame[];
  selectedFrameIndex: number | null;
  hoveredFrameIndex: number | null;
  layers: PreviewLayerVisibility;
  axisScale: number;
  referenceBoardInBase?: number[][] | null;
  referenceBoardInFocus?: number[][] | null;
}) {
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const labelLayerRef = useRef<HTMLDivElement | null>(null);
  const sceneRef = useRef<SceneState | null>(null);
  const latestPreviewStateRef = useRef({
    frameMap: new Map<number, PreviewFrame>(),
    hoveredFrameIndex: null as number | null,
    layers,
    selectedFrameIndex: null as number | null,
  });
  const labelsSignatureRef = useRef("");
  const boardFocusRef = useRef(false);
  const savedRefOriginColorRef = useRef<number | null>(null);
  const boardFocusAxisLenRef = useRef(0);
  const [renderError, setRenderError] = useState("");
  const [labels, setLabels] = useState<PreviewLabel[]>([]);
  const [sceneReadyVersion, setSceneReadyVersion] = useState(0);

  const frameMap = useMemo(() => new Map(frames.map((frame) => [frame.index, frame])), [frames]);
  latestPreviewStateRef.current = {
    frameMap,
    hoveredFrameIndex,
    layers,
    selectedFrameIndex,
  };

  const syncLabels = () => {
    const state = sceneRef.current;
    const viewport = viewportRef.current;
    const latest = latestPreviewStateRef.current;
    if (!state || !viewport) return;
    if (!latest.layers.showLabels) {
      if (labelsSignatureRef.current !== "") {
        labelsSignatureRef.current = "";
        setLabels([]);
      }
      return;
    }
    state.scene.updateMatrixWorld(true);
    const activeIndices = [latest.selectedFrameIndex, latest.hoveredFrameIndex].filter((value): value is number => value !== null);
    const uniqueCameraIndices = Array.from(new Set(activeIndices));
    const nextLabels: PreviewLabel[] = [];
    if (!boardFocusRef.current) {
      for (const index of uniqueCameraIndices) {
        const frame = latest.frameMap.get(index);
        const anchors = state.anchorMap.get(index);
        if (!frame || !anchors) continue;
        nextLabels.push(projectLabel(state, viewport, anchors.camera, `camera-${index}`, `C${String(index).padStart(3, "0")}`, "camera", index));
      }
    }
    if (latest.selectedFrameIndex !== null) {
      const index = latest.selectedFrameIndex;
      const frame = latest.frameMap.get(index);
      const anchors = state.anchorMap.get(index);
      if (frame && anchors) {
      nextLabels.push(projectLabel(state, viewport, anchors.board, `board-${index}`, `B${String(index).padStart(3, "0")}`, "board", index));
      }
    }
    if (state.referenceAnchor && isCoordinateGroupVisible(state.referenceAnchor)) {
      nextLabels.push(projectLabel(state, viewport, state.referenceAnchor, "unified-board", "B统一", "unified", 0));
    }
    if (state.baseAnchor && !boardFocusRef.current) {
      nextLabels.push(projectLabel(state, viewport, state.baseAnchor, "robot-base", "robot", "robot", 0));
    }
    if (boardFocusRef.current && state.boundingBoxRoot?.userData.dimensions) {
      const { endpoints, sizes } = state.boundingBoxRoot.userData.dimensions;
      const axisNames = ["X", "Y", "Z"];
      for (let i = 0; i < endpoints.length; i++) {
        const sizeMm = sizes[i] * 1000;
        nextLabels.push(projectLabel(state, viewport, endpoints[i], `bbox-${i}`, `${axisNames[i]} ${sizeMm.toFixed(1)}mm`, "dimension", i));
      }
    }
    const signature = nextLabels
      .map((label) => `${label.id}:${Math.round(label.position.x)}:${Math.round(label.position.y)}`)
      .join("|");
    if (signature !== labelsSignatureRef.current) {
      labelsSignatureRef.current = signature;
      setLabels(nextLabels);
    }
  };

  const updateOriginSpheres = () => {
    const state = sceneRef.current;
    const viewport = viewportRef.current;
    if (!state || !viewport) return;

    const camera = state.camera;
    const fovRad = camera.fov * Math.PI / 180;
    const tanHalfFov = Math.tan(fovRad / 2);
    const vh = viewport.clientHeight || 600;

    const targetPx = boardFocusRef.current ? 5 : 7;
    const worldPos = new state.THREE.Vector3();

    const processOrigin = (origin: any, parentObj: any) => {
      if (!origin || !origin.visible) return;
      parentObj.getWorldPosition(worldPos);
      const dist = Math.max(camera.position.distanceTo(worldPos), 0.001);
      const desiredRadius = targetPx * dist * tanHalfFov / vh;
      const baseRadius = origin.geometry?.parameters?.radius;
      if (baseRadius) origin.scale.setScalar(desiredRadius / baseRadius);
    };

    for (const [, anchors] of state.anchorMap) {
      processOrigin(anchors.camera.userData.origin, anchors.camera);
      processOrigin(anchors.board.userData.origin, anchors.board);
    }
    processOrigin(state.referenceAnchor?.userData.origin, state.referenceAnchor);
  };

  const restoreBoardFocus = () => {
    const state = sceneRef.current;
    if (!state || !boardFocusRef.current) return;
    boardFocusRef.current = false;
    boardFocusAxisLenRef.current = 0;

    if (state.referenceAnchor?.userData.dimMarkers) {
      for (const m of state.referenceAnchor.userData.dimMarkers) {
        state.referenceAnchor.remove(m);
      }
      state.referenceAnchor.userData.dimMarkers = undefined;
    }
    if (state.referenceAnchor?.userData.axisEndpoints) {
      for (const ep of state.referenceAnchor.userData.axisEndpoints) {
        state.referenceAnchor.remove(ep);
      }
      state.referenceAnchor.userData.axisEndpoints = undefined;
    }
    for (const line of state.referenceAnchor?.userData.axisLines ?? []) {
      line.scale.setScalar(1);
      const originalLength = line.userData.originalLength;
      const axis = line.userData.axisIndex;
      if (originalLength && axis !== undefined) {
        line.position.setComponent(axis, originalLength / 2);
      }
    }
    if (state.referenceAnchor?.userData.origin) {
      const origin = state.referenceAnchor.userData.origin;
      if (savedRefOriginColorRef.current !== null) {
        origin.material.color.setHex(savedRefOriginColorRef.current);
        savedRefOriginColorRef.current = null;
      }
    }

    state.camera.near = 0.01;
    state.camera.far = 100;
    state.camera.updateProjectionMatrix();

    state.boundingBoxRoot.clear();
    state.controls.minDistance = 0;
    state.baseAxes.visible = layers.showBaseAxes;
    for (const [, anchors] of state.anchorMap) {
      applyStoredBoardMatrix(state.THREE, anchors.board, "base");
      setCoordinateGroupDisplay(anchors.camera, layers.showCameraFrames, true, true);
      setCoordinateGroupDisplay(anchors.board, layers.showBoardFrames, true, true);
    }
    if (state.referenceAnchor) {
      applyStoredBoardMatrix(state.THREE, state.referenceAnchor, "base");
      const visible = Boolean(state.referenceAnchor.userData.hasBaseMatrix) && layers.showUnifiedBoardFrame;
      setCoordinateGroupDisplay(state.referenceAnchor, layers.showBoardFrames, true, visible);
    }
  };

  const applyBoardFocusVisibility = () => {
    const state = sceneRef.current;
    if (!state) return;
    state.baseAxes.visible = false;
    boardFocusAxisLenRef.current = 0;

    for (const [, anchors] of state.anchorMap) {
      applyStoredBoardMatrix(state.THREE, anchors.board, "focus");
      setCoordinateGroupDisplay(anchors.camera, false, false, false);
      setCoordinateGroupDisplay(anchors.board, false, true, true);
    }
    if (state.referenceAnchor) {
      applyStoredBoardMatrix(state.THREE, state.referenceAnchor, "focus");
      setCoordinateGroupDisplay(state.referenceAnchor, true, true, true);
      const origin = state.referenceAnchor.userData.origin;
      if (origin) {
        savedRefOriginColorRef.current = origin.material.color.getHex();
        origin.material.color.setHex(0x00e5ff);
      }
      if (state.referenceAnchor.userData.axisEndpoints) {
        for (const ep of state.referenceAnchor.userData.axisEndpoints) {
          state.referenceAnchor.remove(ep);
        }
        state.referenceAnchor.userData.axisEndpoints = undefined;
      }
      if (state.referenceAnchor.userData.dimMarkers) {
        for (const m of state.referenceAnchor.userData.dimMarkers) {
          state.referenceAnchor.remove(m);
        }
        state.referenceAnchor.userData.dimMarkers = undefined;
      }
      for (const line of state.referenceAnchor.userData.axisLines ?? []) {
        line.scale.setScalar(1);
      }
    }

    state.boundingBoxRoot.clear();
    const positions: any[] = [];
    for (const [, anchors] of state.anchorMap) {
      const pos = new state.THREE.Vector3();
      anchors.board.getWorldPosition(pos);
      positions.push(pos);
    }
    if (state.referenceAnchor) {
      const pos = new state.THREE.Vector3();
      state.referenceAnchor.getWorldPosition(pos);
      positions.push(pos);
    }
    if (positions.length < 2) return;

    const box = new state.THREE.Box3().setFromPoints(positions);
    const size = box.getSize(new state.THREE.Vector3());
    const center = box.getCenter(new state.THREE.Vector3());

    const hw = size.x / 2, hh = size.y / 2, hd = size.z / 2;
    const axColors = [0xd94a36, 0x2b9b5f, 0x2d6fd1];
    const axisEdgeSets = [
      { pts: [[-hw,-hh,-hd],[ hw,-hh,-hd],[-hw, hh,-hd],[ hw, hh,-hd],[-hw,-hh, hd],[ hw,-hh, hd],[-hw, hh, hd],[ hw, hh, hd]], color: 0 },
      { pts: [[-hw,-hh,-hd],[-hw, hh,-hd],[ hw,-hh,-hd],[ hw, hh,-hd],[-hw,-hh, hd],[-hw, hh, hd],[ hw,-hh, hd],[ hw, hh, hd]], color: 1 },
      { pts: [[-hw,-hh,-hd],[-hw,-hh, hd],[ hw,-hh,-hd],[ hw,-hh, hd],[-hw, hh,-hd],[-hw, hh, hd],[ hw, hh,-hd],[ hw, hh, hd]], color: 2 },
    ];
    for (const { pts, color } of axisEdgeSets) {
      const geom = new state.THREE.BufferGeometry();
      geom.setAttribute("position", new state.THREE.BufferAttribute(new Float32Array(pts.flat()), 3));
      const line = new state.THREE.LineSegments(
        geom,
        new state.THREE.LineBasicMaterial({ color: axColors[color], transparent: true, opacity: 0.5 }),
      );
      line.position.copy(center);
      state.boundingBoxRoot.add(line);
    }

    /* Scale unified board reference axes to be proportional to board spread */
    if (state.referenceAnchor) {
      const maxDim = Math.max(size.x, size.y, size.z);
      const adaptiveLength = maxDim * 0.35;
      boardFocusAxisLenRef.current = adaptiveLength;
      for (const line of state.referenceAnchor.userData.axisLines ?? []) {
        const originalLength = line.userData.originalLength;
        const axis = line.userData.axisIndex;
        if (originalLength && originalLength > 0 && axis !== undefined) {
          const f = adaptiveLength / originalLength;
          line.scale.y = f;
          line.position.setComponent(axis, (originalLength / 2) * f);
          /* Scale shaft radius proportionally to length so it stays thinner than origin sphere */
          const radiusRatio = 0.02;
          const geomRadius = line.geometry.parameters.radiusTop;
          if (geomRadius && geomRadius > 0) {
            const rScale = (adaptiveLength * radiusRatio) / geomRadius;
            line.scale.x = rScale;
            line.scale.z = rScale;
          }
        }
      }

      /* Dimension labels attached to unified board coordinate axes */
      if (maxDim > 0) {
        const dimEndpointMarkers = ["X", "Y", "Z"].map((_, axis) => {
          const marker = new state.THREE.Object3D();
          const pos = new state.THREE.Vector3(0, 0, 0);
          pos.setComponent(axis, adaptiveLength);
          marker.position.copy(pos);
          state.referenceAnchor.add(marker);
          return marker;
        });
        state.referenceAnchor.userData.dimMarkers = dimEndpointMarkers;
        state.boundingBoxRoot.userData.dimensions = {
          endpoints: dimEndpointMarkers,
          sizes: [size.x, size.y, size.z],
        };
      }
    }
  };

  const focusFrames = (target: "all" | "selected" | "boards") => {
    const state = sceneRef.current;
    if (!state || (state.anchorMap.size === 0 && !state.referenceAnchor)) return;

    if (target === "boards") {
      const center = new state.THREE.Vector3();
      const referenceBoard = referenceBoardInFocus ?? referenceBoardInBase;
      if (referenceBoard) {
        const t = matrixTranslation(referenceBoard);
        center.set(t[0], t[1], t[2]);
      } else {
        const boardPositions = frames.map((frame) => {
          const t = matrixTranslation(boardMatrixForFrame(frame, "focus"));
          return new state.THREE.Vector3(t[0], t[1], t[2]);
        });
        if (boardPositions.length === 0) return;
        const box = new state.THREE.Box3();
        for (const pos of boardPositions) box.expandByPoint(pos);
        box.getCenter(center);
      }
      let maxDist = 0;
      for (const frame of frames) {
        const t = matrixTranslation(boardMatrixForFrame(frame, "focus"));
        const pos = new state.THREE.Vector3(t[0], t[1], t[2]);
        const dist = pos.distanceTo(center);
        if (dist > maxDist) maxDist = dist;
      }
      boardFocusRef.current = true;
      applyBoardFocusVisibility();
      const radius = Math.max(maxDist * 3, 0.0005);
      state.controls.minDistance = Math.max(radius * 0.05, 0.002);
      state.controls.target.set(center.x, center.y, center.z);
      state.camera.position.set(center.x + radius, center.y - radius, center.z + radius * 0.75);
      state.camera.lookAt(center);
      const camDist = state.camera.position.distanceTo(center);
      state.camera.near = Math.max(camDist * 0.001, 0.0001);
      state.camera.far = Math.max(camDist * 50, 10);
      state.camera.updateProjectionMatrix();
      syncLabels();
      return;
    }

    restoreBoardFocus();

    const indices = target === "selected" && selectedFrameIndex !== null
      ? [selectedFrameIndex]
      : frames.map((frame) => frame.index);
    const objects = indices.flatMap((index) => {
      const anchors = state.anchorMap.get(index);
      return anchors ? [anchors.camera, anchors.board] : [];
    });
    if (target === "all" && state.referenceAnchor && isCoordinateGroupVisible(state.referenceAnchor)) {
      objects.push(state.referenceAnchor);
    }
    if (objects.length === 0) return;
    const box = new state.THREE.Box3();
    for (const object3D of objects) {
      box.expandByObject(object3D);
    }
    const center = box.getCenter(new state.THREE.Vector3());
    const size = box.getSize(new state.THREE.Vector3());
    const radius = Math.max(size.length() * 0.65, axisScale * 5, 0.6);
    state.controls.target.set(center.x, center.y, center.z);
    state.camera.position.set(center.x + radius, center.y - radius, center.z + radius * 0.75);
    state.camera.lookAt(center);
    syncLabels();
  };

  const resetView = () => {
    const state = sceneRef.current;
    if (!state) return;
    restoreBoardFocus();
    state.controls.target.set(0, 0, 0);
    state.camera.position.set(1.8, -2.2, 1.5);
    state.camera.lookAt(0, 0, 0);
    syncLabels();
  };

  useEffect(() => {
    let disposed = false;
    let cleanup: (() => void) | null = null;

    (async () => {
      if (!viewportRef.current) return;
      if (navigator.userAgent.includes("jsdom") || (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env?.VITEST) {
        return;
      }
      try {
        const THREE = await import("three");
        const controlsModule = await import("three/examples/jsm/controls/OrbitControls.js");
        if (disposed || !viewportRef.current) return;

        const { OrbitControls } = controlsModule;
        const scene = new THREE.Scene();
        scene.background = new THREE.Color("#eef4f7");

        const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 100);
        camera.up.set(0, 0, 1);
        camera.position.set(1.8, -2.2, 1.5);

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        viewportRef.current.replaceChildren(renderer.domElement);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.target.set(0, 0, 0);

        const ambientLight = new THREE.AmbientLight(0xffffff, 1.2);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1.4);
        directionalLight.position.set(2, -3, 3);
        scene.add(ambientLight, directionalLight);

        const baseAxes = new THREE.AxesHelper(0.35);
        scene.add(baseAxes);

        const baseAnchor = new THREE.Object3D();
        baseAnchor.position.set(0, 0, 0);
        scene.add(baseAnchor);

        const frameRoot = new THREE.Group();
        scene.add(frameRoot);
        const referenceRoot = new THREE.Group();
        scene.add(referenceRoot);
        const boundingBoxRoot = new THREE.Group();
        scene.add(boundingBoxRoot);

        const resize = () => {
          if (!viewportRef.current) return;
          const width = Math.max(viewportRef.current.clientWidth, 1);
          const height = Math.max(viewportRef.current.clientHeight, 1);
          camera.aspect = width / height;
          camera.updateProjectionMatrix();
          renderer.setSize(width, height, false);
          syncLabels();
        };
        resize();

        const resizeObserver = typeof ResizeObserver === "undefined"
          ? null
          : new ResizeObserver(() => resize());
        resizeObserver?.observe(viewportRef.current);
        window.addEventListener("resize", resize);

        const state: SceneState = {
          THREE,
          scene,
          camera,
          renderer,
          controls,
          baseAxes,
          frameRoot,
          referenceRoot,
          boundingBoxRoot,
          anchorMap: new Map(),
          referenceAnchor: null,
          baseAnchor,
          animationFrame: null,
          resizeObserver,
        };
        sceneRef.current = state;
        setSceneReadyVersion((version) => version + 1);

        const renderLoop = () => {
          if (!sceneRef.current) return;
          controls.update();
          updateOriginSpheres();
          renderer.render(scene, camera);
          syncLabels();
          state.animationFrame = window.requestAnimationFrame(renderLoop);
        };
        renderLoop();

        cleanup = () => {
          window.removeEventListener("resize", resize);
          resizeObserver?.disconnect();
          if (state.animationFrame !== null) {
            window.cancelAnimationFrame(state.animationFrame);
          }
          controls.dispose();
          renderer.dispose();
          frameRoot.clear();
          referenceRoot.clear();
          boundingBoxRoot.clear();
          viewportRef.current?.replaceChildren();
          if (sceneRef.current === state) {
            sceneRef.current = null;
          }
        };
      } catch (error) {
        setRenderError(`Three.js 预览初始化失败：${String(error)}`);
      }
    })();

    return () => {
      disposed = true;
      cleanup?.();
    };
  }, []);

  useEffect(() => {
    const state = sceneRef.current;
    if (!state) return;

    state.baseAxes.visible = layers.showBaseAxes;
    state.frameRoot.clear();
    state.referenceRoot.clear();
    state.boundingBoxRoot.clear();
    state.anchorMap.clear();
    state.referenceAnchor = null;

    for (const frame of frames) {
      const isSelected = frame.index === selectedFrameIndex;
      const isHovered = frame.index === hoveredFrameIndex;
      const frameGroup = new state.THREE.Group();

      const cameraAxes = createAxesGroup(state.THREE, axisScale, {
        opacity: frame.used ? (isSelected ? 1 : 0.18) : 0.08,
        axisLengthScale: isSelected ? 1.12 : 1,
        boardStyle: false,
        hovered: isHovered,
        selected: isSelected,
      });
      const boardAxes = createAxesGroup(state.THREE, axisScale * 0.92, {
        opacity: frame.used ? (isSelected ? 0.85 : 0.12) : 0.06,
        axisLengthScale: isSelected ? 1.08 : 1,
        boardStyle: true,
        hovered: isHovered,
        selected: isSelected,
      });

      applyMatrixToObject3D(state.THREE, cameraAxes, frame.cameraInBase);
      const boardBaseMatrix = boardMatrixForFrame(frame, "base");
      const boardFocusMatrix = boardMatrixForFrame(frame, "focus");
      applyMatrixToObject3D(state.THREE, boardAxes, boardBaseMatrix);
      boardAxes.userData.baseMatrix = boardBaseMatrix;
      boardAxes.userData.focusMatrix = boardFocusMatrix;

      setCoordinateGroupDisplay(cameraAxes, layers.showCameraFrames, true);
      setCoordinateGroupDisplay(boardAxes, layers.showBoardFrames, true);
      frameGroup.add(cameraAxes, boardAxes);
      state.frameRoot.add(frameGroup);
      state.anchorMap.set(frame.index, { camera: cameraAxes, board: boardAxes });
    }
    if (referenceBoardInBase || referenceBoardInFocus) {
      const referenceMatrix = referenceBoardInBase ?? referenceBoardInFocus!;
      const referenceAxes = createAxesGroup(state.THREE, axisScale * 1.18, {
        opacity: 0.96,
        axisLengthScale: 1.12,
        boardStyle: true,
        hovered: false,
        selected: false,
        unifiedStyle: true,
      });
      applyMatrixToObject3D(state.THREE, referenceAxes, referenceMatrix);
      referenceAxes.userData.baseMatrix = referenceMatrix;
      referenceAxes.userData.focusMatrix = referenceBoardInFocus ?? referenceMatrix;
      referenceAxes.userData.hasBaseMatrix = Boolean(referenceBoardInBase);
      setCoordinateGroupDisplay(
        referenceAxes,
        layers.showBoardFrames,
        true,
        Boolean(referenceBoardInBase) && layers.showUnifiedBoardFrame,
      );
      state.referenceRoot.add(referenceAxes);
      state.referenceAnchor = referenceAxes;
    }

    syncLabels();
    if (frames.length > 0) {
      if (boardFocusRef.current) {
        focusFrames("boards");
      } else {
        focusFrames(selectedFrameIndex === null ? "all" : "selected");
      }
    } else {
      resetView();
    }
  }, [axisScale, frames, hoveredFrameIndex, layers, referenceBoardInBase, referenceBoardInFocus, sceneReadyVersion, selectedFrameIndex]);

  useEffect(() => {
    syncLabels();
  }, [hoveredFrameIndex, layers.showLabels, selectedFrameIndex]);

  return (
    <div className="coordinate-preview-shell">
      <div className="coordinate-preview-toolbar">
        <button type="button" className="secondary-action" onClick={() => focusFrames("all")}>
          全部显示
        </button>
        <button
          type="button"
          className="secondary-action"
          onClick={() => focusFrames("selected")}
          disabled={selectedFrameIndex === null}
        >
          聚焦选中帧
        </button>
        <button type="button" className="secondary-action" onClick={() => focusFrames("boards")}>
          聚焦标定板
        </button>
        <button type="button" className="secondary-action" onClick={resetView}>
          <RefreshCcw size={15} />
          重置视角
        </button>
      </div>
      <div className="coordinate-preview-stage">
        <div ref={viewportRef} className="coordinate-preview-canvas" aria-label="点云坐标转换三维预览" />
        <div ref={labelLayerRef} className="coordinate-preview-label-layer" aria-hidden="true">
          {labels.map((label) => (
            <div
              key={label.id}
              className={`coordinate-preview-label ${label.kind}`}
              style={{ left: `${label.position.x}px`, top: `${label.position.y}px` }}
            >
              {label.text}
            </div>
          ))}
        </div>
        {frames.length === 0 && (
          <div className="coordinate-preview-empty">
            <strong>等待生成预览</strong>
            <span>加载数据文件夹、位姿文件和标定结果后即可查看底座 / 相机 / 标定板坐标系。</span>
          </div>
        )}
        {renderError && <div className="coordinate-preview-error">{renderError}</div>}
      </div>
    </div>
  );
}

function applyMatrixToObject3D(
  THREE: any,
  object3D: any,
  matrixRows: number[][],
) {
  const matrix = new THREE.Matrix4().set(...matrixRowsToThreeSetArgs(matrixRows));
  object3D.matrixAutoUpdate = false;
  object3D.matrix.copy(matrix);
  object3D.matrix.decompose(object3D.position, object3D.quaternion, object3D.scale);
  object3D.matrixWorldNeedsUpdate = true;
}

function projectLabel(
  state: SceneState,
  viewport: HTMLDivElement,
  object3D: any,
  id: string,
  text: string,
  kind: PreviewLabel["kind"],
  index: number,
): PreviewLabel {
  const worldPosition = new state.THREE.Vector3();
  object3D.updateMatrixWorld(true);
  object3D.getWorldPosition(worldPosition);
  worldPosition.project(state.camera);
  const x = ((worldPosition.x + 1) / 2) * viewport.clientWidth;
  const y = ((1 - worldPosition.y) / 2) * viewport.clientHeight;
  const offset = labelOffset(kind, index);
  return {
    id,
    text,
    kind,
    position: { x: x + offset.x, y: y + offset.y },
  };
}

function labelOffset(kind: PreviewLabel["kind"], index: number) {
  if (kind === "robot") return { x: 12, y: 14 };
  if (kind === "camera") return { x: 12, y: -18 };
  if (kind === "unified") return { x: 14, y: 14 };
  if (kind === "dimension") return { x: 4 + index * 4, y: -10 - index * 2 };
  const stagger = index % 4;
  return {
    x: 10 + (stagger % 2) * 10,
    y: -18 - Math.floor(stagger / 2) * 12,
  };
}

function matrixRowsToThreeSetArgs(matrixRows: number[][]) {
  return [
    matrixRows[0]?.[0] ?? 1,
    matrixRows[0]?.[1] ?? 0,
    matrixRows[0]?.[2] ?? 0,
    matrixRows[0]?.[3] ?? 0,
    matrixRows[1]?.[0] ?? 0,
    matrixRows[1]?.[1] ?? 1,
    matrixRows[1]?.[2] ?? 0,
    matrixRows[1]?.[3] ?? 0,
    matrixRows[2]?.[0] ?? 0,
    matrixRows[2]?.[1] ?? 0,
    matrixRows[2]?.[2] ?? 1,
    matrixRows[2]?.[3] ?? 0,
    matrixRows[3]?.[0] ?? 0,
    matrixRows[3]?.[1] ?? 0,
    matrixRows[3]?.[2] ?? 0,
    matrixRows[3]?.[3] ?? 1,
  ];
}

function createAxesGroup(
  THREE: any,
  scale: number,
  options: {
    opacity: number;
    axisLengthScale: number;
    boardStyle: boolean;
    hovered: boolean;
    selected: boolean;
    unifiedStyle?: boolean;
  },
) {
  const group = new THREE.Group();
  const axisLines: any[] = [];
  const axisLength = scale * options.axisLengthScale;
  const colors = axisColors();
  const shaftRadius = scale * 0.008;

  for (const [axis, color] of colors.entries()) {
    const geom = new THREE.CylinderGeometry(shaftRadius, shaftRadius, axisLength, 6);
    const mat = new THREE.MeshStandardMaterial({
      color,
      transparent: true,
      opacity: options.opacity,
    });
    const mesh = new THREE.Mesh(geom, mat);
    const midpoint = new THREE.Vector3(0, 0, 0);
    midpoint.setComponent(axis, axisLength / 2);
    mesh.position.copy(midpoint);
    if (axis === 0) mesh.rotation.z = -Math.PI / 2;
    else if (axis === 2) mesh.rotation.x = Math.PI / 2;
    mesh.userData.originalLength = axisLength;
    mesh.userData.axisIndex = axis;
    axisLines.push(mesh);
    group.add(mesh);
  }

  const origin = new THREE.Mesh(
    new THREE.SphereGeometry(options.selected ? SELECTED_ORIGIN_RADIUS_M : DEFAULT_ORIGIN_RADIUS_M, 12, 12),
    new THREE.MeshStandardMaterial({
      color: originColor(options.boardStyle),
      transparent: true,
      opacity: Math.min(1, options.opacity + 0.1),
    }),
  );
  group.add(origin);
  group.userData.axisLines = axisLines;
  group.userData.origin = origin;
  return group;
}

function setCoordinateGroupDisplay(group: any, showAxes: boolean, showOrigin: boolean, visible = true) {
  group.visible = true;
  for (const line of group.userData.axisLines ?? []) {
    line.visible = visible && showAxes;
  }
  if (group.userData.origin) {
    group.userData.origin.visible = visible && showOrigin;
  }
}

function isCoordinateGroupVisible(group: any) {
  return Boolean(group.userData.origin?.visible || (group.userData.axisLines ?? []).some((line: any) => line.visible));
}

function axisColors() {
  return ["#ef4444", "#16a34a", "#2563eb"];
}

function originColor(boardStyle?: boolean) {
  return boardStyle ? "#ffd700" : "#ff4444";
}

export { axisColors, frameTitle, fileNameFromPath, matrixRowsToThreeSetArgs, matrixTranslation, originColor };
