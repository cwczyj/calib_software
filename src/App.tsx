import { type CSSProperties, type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";
import {
  Activity,
  BarChart3,
  Calculator,
  FolderOpen,
  Grid3X3,
  Image as ImageIcon,
  RefreshCcw,
  Rotate3D,
} from "lucide-react";
import { TitleBar } from "./TitleBar";
import {
  CalibrationMode,
  MarkerType,
  errorRows,
  modeDescriptions,
  modeLabels,
  poseSamples,
} from "./mockData";
import {
  CoordinatePreview3D,
  fileNameFromPath as previewFileNameFromPath,
  frameTitle as previewFrameTitle,
  matrixTranslation,
  type PreviewFrame,
  type PreviewLayerVisibility,
} from "./CoordinatePreview3D";

type Page = "calibration" | "results" | "tools";
type PathFieldKind = "file" | "directory";
type CalibrationPathField = "dataFolder" | "robotPoseFile";
type ToolPathField = "circleImageFile" | "circleDepthFile" | "conversionDataFolder" | "conversionRobotPoseFile";

type ImageFileEntry = {
  name: string;
  path: string;
};

type CalibrationImageEntry = ImageFileEntry & {
  displayName: string;
  depthImage?: ImageFileEntry | null;
};

type PoseFileRow = {
  index: number;
  content: string;
};

type CalibrationRun = {
  outputPath: string;
  imageDir?: string;
  posesFile?: string;
  stdout?: string;
  stderr?: string;
  setup?: CalibrationMode;
  primaryTransformName?: string;
  primaryMatrixRows?: string[];
  secondaryTransformName?: string;
  secondaryMatrixRows?: string[];
  matrixRows?: string[];
  averageErrorMm?: number;
  rotationErrorDeg?: number;
  reprojectionErrorPx?: number;
  reprojectionRmsPx?: number;
  baseConsistencyMeanMm?: number | null;
  baseConsistencyRmsMm?: number | null;
  baseConsistencyMaxMm?: number | null;
  baseConsistencyCount?: number | null;
  numImages?: number;
  numImagesUsed?: number;
  filteredImages?: number[];
  frameErrors?: FrameError[];
  previewFrames?: PreviewFrame[];
  depthUsed?: boolean;
  message?: string;
};

type FrameError = {
  index: number;
  imagePath?: string;
  used: boolean;
  usedChessboardFallback?: boolean;
  cornerCount?: number | null;
  reprojectionMeanPx?: number | null;
  reprojectionRmsPx?: number | null;
  reprojectionMaxPx?: number | null;
  referenceReprojectionMeanPx?: number | null;
  referenceReprojectionRmsPx?: number | null;
  referenceReprojectionMaxPx?: number | null;
  reprojectionErrorPx?: number | null;
  optimizedReprojectionErrorPx?: number | null;
  baseConsistencyMeanMm?: number | null;
  baseConsistencyRmsMm?: number | null;
  baseConsistencyMaxMm?: number | null;
  baseConsistencyCount?: number | null;
  translationErrorMm?: number | null;
  rotationErrorDeg?: number | null;
};

type CharucoDetection = {
  imagePath: string;
  outputPath: string;
  success: boolean;
  numCorners: number;
  numMarkers: number;
  usedChessboardFallback?: boolean;
  message: string;
  cornerRows: CharucoCornerRow[];
};

type CharucoCornerRow = {
  id: number;
  imagePoint: [number, number];
  cameraPoint: [number, number, number] | null;
};

type CameraIntrinsics = {
  cx: number;
  cy: number;
  fx: number;
  fy: number;
  distortionCoefficients?: number[];
};

type CameraIntrinsicField = "cx" | "cy" | "fx" | "fy";
type CameraIntrinsicsSource = "manual" | "file";

type CharucoBoardParams = {
  squaresX: number;
  squaresY: number;
  squareLength: number;
  markerLength: number;
  arucoDict: string;
};

type CharucoBoardParamValues = Record<keyof CharucoBoardParams, string>;

type CalibrationRequestPayload = {
  imageDir: string;
  posesFile: string;
  marker: MarkerType;
  cameraIntrinsics: CameraIntrinsics;
  cameraParams: string | null;
  setup: CalibrationMode;
  poseFormat: string;
  useDepth: string;
  squaresX: number;
  squaresY: number;
  squareLength: number;
  markerLength: number;
  arucoDict: string;
  filterInconsistent?: boolean;
  excludedImageIndices?: number[];
};

type SaveTextFileRequest = {
  path: string;
  content: string;
};

type ConversionPreviewRequestPayload = {
  imageDir: string;
  posesFile: string;
  setup: CalibrationMode;
  poseFormat: string;
  primaryTransformName: string;
  primaryMatrixRows: string[];
  secondaryTransformName: string;
  secondaryMatrixRows: string[];
  cameraIntrinsics?: CameraIntrinsics;
  squaresX?: number;
  squaresY?: number;
  squareLength?: number;
  markerLength?: number;
  arucoDict?: string;
};

type ConversionPreviewResult = {
  previewFrames: PreviewFrame[];
};

const defaultCharucoParams: CharucoBoardParamValues = {
  squaresX: "14",
  squaresY: "9",
  squareLength: "0.020",
  markerLength: "0.015",
  arucoDict: "DICT_5X5_50",
};

const arucoDictOptions = [
  "DICT_4X4_50",
  "DICT_4X4_100",
  "DICT_4X4_250",
  "DICT_4X4_1000",
  "DICT_5X5_50",
  "DICT_5X5_100",
  "DICT_5X5_250",
  "DICT_5X5_1000",
  "DICT_6X6_50",
  "DICT_6X6_100",
  "DICT_6X6_250",
  "DICT_6X6_1000",
  "DICT_7X7_50",
  "DICT_7X7_100",
  "DICT_7X7_250",
  "DICT_7X7_1000",
];

const poseFormatOptions = [
  "sxyz", "sxyx", "sxzy", "sxzx", "syzx", "syzy", "syxz", "syxy",
  "szxy", "szxz", "szyx", "szyz", "rzyx", "rxyx", "ryzx", "rxzx",
  "rxzy", "ryzy", "rzxy", "ryxy", "ryxz", "rzxz", "rxyz", "rzyz",
];

const pages: Array<{ id: Page; label: string; icon: React.ReactNode }> = [
  { id: "calibration", label: "标定", icon: <Grid3X3 size={14} /> },
  { id: "results", label: "结果 / 误差分析", icon: <BarChart3 size={14} /> },
  { id: "tools", label: "工具", icon: <RefreshCcw size={14} /> },
];

function formatPoint(values: readonly number[], digits: number) {
  return `(${values.map((value) => value.toFixed(digits)).join(", ")})`;
}

function previewFrameName(frame: PreviewFrame) {
  return `Frame ${String(frame.index).padStart(3, "0")}`;
}

function parseTransformRows(rows: string[]) {
  if (rows.length !== 4) return null;
  const matrix = rows.map((row) => row.split(",").map((value) => Number(value.trim())));
  if (matrix.some((row) => row.length !== 4 || row.some((value) => !Number.isFinite(value)))) {
    return null;
  }
  return matrix;
}

function fileNameFromPath(path: string) {
  return path.split(/[\\/]/).pop() || path;
}

function imageFramePrefix(name: string) {
  return name.match(/^(\d+)/)?.[1] ?? "";
}

function pairCalibrationImages(rgbImages: ImageFileEntry[], depthImages: ImageFileEntry[]): CalibrationImageEntry[] {
  const depthByPrefix = new Map(
    depthImages
      .map((depth) => [imageFramePrefix(depth.name), depth] as const)
      .filter(([prefix]) => prefix !== ""),
  );

  if (depthByPrefix.size === 0) {
    return rgbImages.map((rgb) => ({ ...rgb, displayName: rgb.name, depthImage: null }));
  }

  return rgbImages
    .flatMap<CalibrationImageEntry>((rgb) => {
      const depth = depthByPrefix.get(imageFramePrefix(rgb.name)) ?? null;
      if (!depth) return [];
      return [{
        ...rgb,
        displayName: `${rgb.name}|${depth.name}`,
        depthImage: depth,
      }];
    });
}

function formatOptionalNumber(value: number | null | undefined, digits = 6) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "";
}

function formatMetric(value: number | null | undefined, unit: string) {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(6)} ${unit}` : "--";
}

function formatDistortionCoefficients(values?: number[]) {
  if (!values?.length) return "";
  return values.map((value) => value.toFixed(6)).join(", ");
}

function parseMatrixRows(matrixRows: string[] | undefined) {
  return (matrixRows ?? []).map((row) => row.split(",").map((value) => Number(value.trim())));
}

function yamlScalar(value: boolean | number | string | null | undefined) {
  if (value === null || value === undefined) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "null";
  return value;
}

function yamlArray(values: Array<boolean | number | string | null | undefined>) {
  return `[${values.map((value) => yamlScalar(value)).join(", ")}]`;
}

export function buildFrontendCalibrationYaml(result: CalibrationRun) {
  const transformName = result.primaryTransformName || (result.setup === "eye-to-hand" ? "T_C2W" : "T_C2F");
  const matrix = parseMatrixRows(result.matrixRows);
  const lines = [
    `setup: ${result.setup || "eye-in-hand"}`,
    "transforms:",
    `  ${transformName}:`,
    ...matrix.map((row) => `    - ${yamlArray(row)}`),
    "metrics:",
    `  translation_mean_mm: ${yamlScalar(result.averageErrorMm)}`,
    `  rotation_mean_deg: ${yamlScalar(result.rotationErrorDeg)}`,
    `  reprojection_mean_px: ${yamlScalar(result.reprojectionErrorPx)}`,
    `  reprojection_rms_px: ${yamlScalar(result.reprojectionRmsPx)}`,
    `  base_consistency_mean_mm: ${yamlScalar(result.baseConsistencyMeanMm)}`,
    `  base_consistency_rms_mm: ${yamlScalar(result.baseConsistencyRmsMm)}`,
    `  base_consistency_max_mm: ${yamlScalar(result.baseConsistencyMaxMm)}`,
    `  base_consistency_count: ${yamlScalar(result.baseConsistencyCount)}`,
  ];

  return `${lines.join("\n")}\n`;
}

function firstNumber(...values: Array<number | null | undefined>) {
  return values.find((value) => typeof value === "number" && Number.isFinite(value));
}

function aggregateFrameRms(values: Array<number | null | undefined>) {
  const finite = values.filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  if (finite.length === 0) return null;
  return Math.sqrt(finite.reduce((sum, value) => sum + value * value, 0) / finite.length);
}

function parseCameraIntrinsics(
  values: Record<CameraIntrinsicField, string>,
  distortionCoefficients?: number[],
): CameraIntrinsics | null {
  if (Object.values(values).some((value) => value.trim() === "")) {
    return null;
  }
  const intrinsics = {
    cx: Number(values.cx),
    cy: Number(values.cy),
    fx: Number(values.fx),
    fy: Number(values.fy),
  };
  if (Object.values(intrinsics).some((value) => !Number.isFinite(value))) {
    return null;
  }
  return distortionCoefficients?.length ? { ...intrinsics, distortionCoefficients } : intrinsics;
}

function parseCharucoBoardParams(values: CharucoBoardParamValues): CharucoBoardParams | null {
  const squaresX = Number(values.squaresX);
  const squaresY = Number(values.squaresY);
  const squareLength = Number(values.squareLength);
  const markerLength = Number(values.markerLength);
  if (
    !Number.isInteger(squaresX)
    || !Number.isInteger(squaresY)
    || squaresX <= 1
    || squaresY <= 1
    || !Number.isFinite(squareLength)
    || !Number.isFinite(markerLength)
    || squareLength <= 0
    || markerLength <= 0
    || !values.arucoDict
  ) {
    return null;
  }
  return {
    squaresX,
    squaresY,
    squareLength,
    markerLength,
    arucoDict: values.arucoDict,
  };
}

export function App() {
  const [page, setPage] = useState<Page>("calibration");
  const [mode, setMode] = useState<CalibrationMode>("eye-in-hand");
  const [marker] = useState<MarkerType>("charuco");
  const [logs, setLogs] = useState<string[]>(["等待导入数据并计算"]);
  const [isCalculating, setIsCalculating] = useState(false);
  const [isRecalculating, setIsRecalculating] = useState(false);
  const [selectedRows, setSelectedRows] = useState(() => new Set(errorRows.map((row) => row.id)));
  const [selectedResultFrameIndices, setSelectedResultFrameIndices] = useState(() => new Set<number>());
  const [isResultSelectionDirty, setIsResultSelectionDirty] = useState(false);
  const [analysisMessage, setAnalysisMessage] = useState("已载入示例标定结果");
  const [resultExportMessage, setResultExportMessage] = useState("");
  const [calibrationResult, setCalibrationResult] = useState<CalibrationRun | null>(null);
  const [lastCalibrationRequest, setLastCalibrationRequest] = useState<CalibrationRequestPayload | null>(null);
  const [calibrationPaths, setCalibrationPaths] = useState<Partial<Record<CalibrationPathField, string>>>({});
  const [calibrationIntrinsics, setCalibrationIntrinsics] = useState<Record<CameraIntrinsicField, string>>({
    cx: "640",
    cy: "360",
    fx: "600",
    fy: "600",
  });
  const [calibrationDistortion, setCalibrationDistortion] = useState<number[] | undefined>(undefined);
  const [cameraIntrinsicsSource, setCameraIntrinsicsSource] = useState<CameraIntrinsicsSource>("manual");
  const [charucoParams, setCharucoParams] = useState<CharucoBoardParamValues>(defaultCharucoParams);
  const [poseFormat, setPoseFormat] = useState("sxyz");
  const [rgbImages, setRgbImages] = useState<ImageFileEntry[]>([]);
  const [depthImages, setDepthImages] = useState<ImageFileEntry[]>([]);
  const [selectedRgbImage, setSelectedRgbImage] = useState<CalibrationImageEntry | null>(null);
  const [selectedDepthPreview, setSelectedDepthPreview] = useState<ImageFileEntry | null>(null);
  const [poseRows, setPoseRows] = useState<PoseFileRow[]>(
    poseSamples.map((content, index) => ({ index: index + 1, content })),
  );

  const calibrationImages = useMemo(
    () => pairCalibrationImages(rgbImages, depthImages),
    [rgbImages, depthImages],
  );

  useEffect(() => {
    const depthImage = selectedRgbImage?.depthImage;
    if (!depthImage?.path) {
      setSelectedDepthPreview(null);
      return;
    }

    let canceled = false;
    setSelectedDepthPreview(null);
    Promise.resolve(invoke<ImageFileEntry>("create_depth_preview", { depthPath: depthImage.path }))
      .then((preview) => {
        if (!canceled) setSelectedDepthPreview(preview ?? null);
      })
      .catch(() => {
        if (!canceled) setSelectedDepthPreview(null);
      });

    return () => {
      canceled = true;
    };
  }, [selectedRgbImage?.depthImage?.path]);

  const averageError = useMemo(() => {
    const activeRows = errorRows.filter((row) => selectedRows.has(row.id));
    if (activeRows.length === 0) return 0;
    return activeRows.reduce((sum, row) => sum + row.error, 0) / activeRows.length;
  }, [selectedRows]);

  const addLog = (message: string) => {
    setLogs((current) => [...current, message]);
  };

  const applyCalibrationResult = (
    result: CalibrationRun,
    fallbackMessage: string,
    options: { preserveFrameSelection?: boolean } = {},
  ) => {
    if (options.preserveFrameSelection) {
      setCalibrationResult((current) => mergeCalibrationFrameErrors(result, current, selectedResultFrameIndices));
    } else {
      setCalibrationResult(result.matrixRows ? result : null);
      if (result.frameErrors?.length) {
        setSelectedResultFrameIndices(new Set(result.frameErrors.filter((row) => row.used).map((row) => row.index)));
      } else {
        setSelectedResultFrameIndices(new Set<number>());
      }
      setIsResultSelectionDirty(false);
    }
    const message = result.message || fallbackMessage;
    addLog(message);
    setAnalysisMessage(message);
    setResultExportMessage("");
  };

  return (
    <div className="app-shell">
      <TitleBar pages={pages} currentPage={page} onPageChange={(id) => setPage(id as Page)} />

      <main>
        {page === "calibration" && (
          <CalibrationPage
            mode={mode}
            marker={marker}
            logs={logs}
            isCalculating={isCalculating}
            paths={calibrationPaths}
            charucoParams={charucoParams}
            poseFormat={poseFormat}
            onModeChange={setMode}
            onCharucoParamChange={(field, value) => {
              setCharucoParams((current) => ({ ...current, [field]: value }));
            }}
            onPoseFormatChange={setPoseFormat}
            images={calibrationImages}
            selectedRgbImage={selectedRgbImage}
            selectedDepthPreview={selectedDepthPreview}
            onSelectRgbImage={setSelectedRgbImage}
            poseRows={poseRows}
            onCalculate={async () => {
              const imageDir = calibrationPaths.dataFolder;
              const posesFile = calibrationPaths.robotPoseFile;
              if (!imageDir || !posesFile) {
                addLog("请先选择 RGB 文件夹和机械臂位姿文件");
                return;
              }
              if (cameraIntrinsicsSource !== "file") {
                addLog("请先选择包含 camera_params.yaml 的 RGB 文件夹");
                return;
              }
              const intrinsics = parseCameraIntrinsics(calibrationIntrinsics, calibrationDistortion);
              if (!intrinsics) {
                addLog("读取到的相机内参无效");
                return;
              }
              const boardParams = parseCharucoBoardParams(charucoParams);
              if (!boardParams) {
                addLog("请填写有效的 ChArUco 标定板参数");
                return;
              }
              const request: CalibrationRequestPayload = {
                imageDir,
                posesFile,
                marker,
                cameraIntrinsics: intrinsics,
                cameraParams: null,
                setup: mode,
                poseFormat,
                useDepth: "off",
                ...boardParams,
              };
              setLastCalibrationRequest(request);
              setIsCalculating(true);
              addLog("标定计算中...");
              try {
                const result = await invoke<CalibrationRun>("run_handeye_calibration", {
                  request,
                });
                applyCalibrationResult({ ...result, imageDir, posesFile }, `RGB-only 标定完成：${result.outputPath}`);
              } catch (error) {
                addLog(`标定失败：${String(error)}`);
              } finally {
                setIsCalculating(false);
              }
            }}
            onPathChange={async (field, label, value) => {
              setCalibrationPaths((current) => ({ ...current, [field]: value }));
              addLog(`已选择${label}：${value}`);
              if (field === "dataFolder") {
                try {
                  const [files, depthFiles, folderIntrinsics] = await Promise.all([
                    invoke<ImageFileEntry[]>("list_rgb_images", { folder: value }),
                    invoke<ImageFileEntry[]>("list_depth_images", { folder: value }),
                    invoke<CameraIntrinsics | null>("read_camera_params", { folder: value }),
                  ]);
                  const pairedImages = pairCalibrationImages(files, depthFiles);
                  setRgbImages(files);
                  setDepthImages(depthFiles);
                  setSelectedRgbImage(pairedImages[0] ?? null);
                  if (folderIntrinsics) {
                    setCalibrationIntrinsics({
                      cx: String(folderIntrinsics.cx),
                      cy: String(folderIntrinsics.cy),
                      fx: String(folderIntrinsics.fx),
                      fy: String(folderIntrinsics.fy),
                    });
                    setCalibrationDistortion(folderIntrinsics.distortionCoefficients);
                    setCameraIntrinsicsSource("file");
                    addLog(`已载入 ${files.length} 张 RGB 图像，${depthFiles.length} 张深度图，并读取相机内参`);
                    addLog(
                      `已读取相机参数：cx=${folderIntrinsics.cx.toFixed(6)}, cy=${folderIntrinsics.cy.toFixed(6)}, fx=${folderIntrinsics.fx.toFixed(6)}, fy=${folderIntrinsics.fy.toFixed(6)}`,
                    );
                    if (folderIntrinsics.distortionCoefficients?.length) {
                      addLog(`已读取畸变参数：${formatDistortionCoefficients(folderIntrinsics.distortionCoefficients)}`);
                    }
                  } else {
                    setCalibrationDistortion(undefined);
                    setCameraIntrinsicsSource("manual");
                    addLog(`已载入 ${files.length} 张 RGB 图像，${depthFiles.length} 张深度图`);
                  }
                } catch (error) {
                  setRgbImages([]);
                  setDepthImages([]);
                  setSelectedRgbImage(null);
                  addLog(`载入图像失败：${String(error)}`);
                }
              }
              if (field === "robotPoseFile") {
                try {
                  const rows = await invoke<PoseFileRow[]>("read_pose_file", { file: value });
                  setPoseRows(rows);
                  addLog(`已载入 ${rows.length} 行机器人姿态数据`);
                } catch (error) {
                  setPoseRows([]);
                  addLog(`载入机器人姿态数据失败：${String(error)}`);
                }
              }
            }}
          />
        )}
        {page === "results" && (
          <ResultsPage
            mode={mode}
            result={calibrationResult}
            selectedRows={selectedRows}
            selectedResultFrameIndices={selectedResultFrameIndices}
            isResultSelectionDirty={isResultSelectionDirty}
            averageError={calibrationResult?.averageErrorMm ?? averageError}
            message={analysisMessage}
            exportMessage={resultExportMessage}
            isRecalculating={isRecalculating}
            onToggleRow={(id) => {
              setSelectedRows((current) => {
                const next = new Set(current);
                if (next.has(id)) next.delete(id);
                else next.add(id);
                return next;
              });
            }}
            onToggleResultFrame={(index) => {
              setIsResultSelectionDirty(true);
              setSelectedResultFrameIndices((current) => {
                const next = new Set(current);
                if (next.has(index)) next.delete(index);
                else next.add(index);
                return next;
              });
            }}
            onRecalculate={async () => {
              if (!calibrationResult?.frameErrors?.length || !lastCalibrationRequest) {
                setAnalysisMessage(`已按 ${selectedRows.size} 组有效数据重新计算 mock 误差`);
                return;
              }
              const excludedImageIndices = calibrationResult.frameErrors
                .filter((row) => !selectedResultFrameIndices.has(row.index))
                .map((row) => row.index);
              const request = { ...lastCalibrationRequest, excludedImageIndices, filterInconsistent: false };
              setIsRecalculating(true);
              setAnalysisMessage("按勾选点位重新计算中...");
              try {
                const result = await invoke<CalibrationRun>("run_handeye_calibration", { request });
                setLastCalibrationRequest(request);
                applyCalibrationResult({ ...result, imageDir: request.imageDir, posesFile: request.posesFile }, result.message || `重新计算完成：${result.outputPath}`, {
                  preserveFrameSelection: true,
                });
              } catch (error) {
                setAnalysisMessage(`重新计算失败：${String(error)}`);
              } finally {
                setIsRecalculating(false);
              }
            }}
            onSaveYaml={async () => {
              if (!calibrationResult?.matrixRows?.length) return;
              try {
                const defaultPath = calibrationResult.outputPath.replace(/[^/\\]+$/, "handeye_result.yaml");
                const targetPath = await save({
                  title: "保存手眼标定结果",
                  defaultPath,
                  filters: [{ name: "YAML", extensions: ["yaml", "yml"] }],
                });
                if (!targetPath) return;
                const content = buildFrontendCalibrationYaml(calibrationResult);
                await invoke<null>("save_text_file", {
                  path: targetPath,
                  content,
                } satisfies SaveTextFileRequest);
                setResultExportMessage(`结果已保存：${targetPath}`);
              } catch (error) {
                setResultExportMessage(`结果保存失败：${String(error)}`);
              }
            }}
          />
        )}
        {page === "tools" && (
          <ToolsPage
            mode={mode}
            poseFormat={poseFormat}
            calibrationResult={calibrationResult}
            cameraIntrinsics={calibrationIntrinsics}
            distortionCoefficients={calibrationDistortion}
            cameraIntrinsicsSource={cameraIntrinsicsSource}
            charucoParams={charucoParams}
            onCharucoParamChange={(field, value) => {
              setCharucoParams((current) => ({ ...current, [field]: value }));
            }}
          />
        )}
      </main>
    </div>
  );
}

function CalibrationPage({
  mode,
  marker,
  logs,
  isCalculating,
  paths,
  charucoParams,
  poseFormat,
  images,
  selectedRgbImage,
  selectedDepthPreview,
  onModeChange,
  onCharucoParamChange,
  onPoseFormatChange,
  onSelectRgbImage,
  poseRows,
  onCalculate,
  onPathChange,
}: {
  mode: CalibrationMode;
  marker: MarkerType;
  logs: string[];
  isCalculating: boolean;
  paths: Partial<Record<CalibrationPathField, string>>;
  charucoParams: CharucoBoardParamValues;
  poseFormat: string;
  images: CalibrationImageEntry[];
  selectedRgbImage: CalibrationImageEntry | null;
  selectedDepthPreview: ImageFileEntry | null;
  onModeChange: (mode: CalibrationMode) => void;
  onCharucoParamChange: (field: keyof CharucoBoardParams, value: string) => void;
  onPoseFormatChange: (poseFormat: string) => void;
  onSelectRgbImage: (image: CalibrationImageEntry) => void;
  poseRows: PoseFileRow[];
  onCalculate: () => void | Promise<void>;
  onPathChange: (field: CalibrationPathField, label: string, value: string) => void | Promise<void>;
}) {
  const defaultDataFolder = `D:/HandEyeCalibrationData/test_new/${mode}/`;
  const defaultPoseFile = `D:/HandEyeCalibrationData/test_new/${mode}/pose.txt`;

  const hasDepth = images.some((image) => image.depthImage);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <section className="calibration-workspace">
      <div className="panel main-panel">
        <PanelTitle icon={<Grid3X3 size={18} />} title="标记物标定" subtitle={modeDescriptions[mode]} />

        <fieldset className="group-box">
          <legend>深度图和 RGB 数据</legend>
          <Field
            kind="directory"
            label="深度图和 RGB 文件夹"
            value={paths.dataFolder ?? defaultDataFolder}
            onValueChange={(value) => onPathChange("dataFolder", "深度图和 RGB 文件夹", value)}
          />
          <div className="calibration-main-grid">
            <ImageFileList
              title={hasDepth ? "匹配图像（深度图 + RGB）" : "RGB 图像"}
              files={images}
              selectedIndex={images.findIndex((f) => f.path === selectedRgbImage?.path)}
              onSelect={onSelectRgbImage}
            />
            <MarkerPreview
              marker={marker}
              image={selectedRgbImage}
              depthImage={selectedRgbImage?.depthImage ?? null}
              depthPreviewImage={selectedDepthPreview}
            />
          </div>
          <div className="camera-controls-row">
            <label className="field marker-select">
              <span>标记物类型：</span>
              <select value={marker} onChange={() => undefined}>
                <option value="charuco">ChArUco</option>
              </select>
            </label>
            <button className="primary-action" onClick={onCalculate} disabled={isCalculating}>
              <Calculator size={16} />
              {isCalculating ? "计算中" : "计算"}
            </button>
            <div className="mode-radio-row">
              {Object.entries(modeLabels).map(([id, label]) => (
                <label key={id}>
                  <input
                    type="radio"
                    name="calibration-mode"
                    checked={mode === id}
                    onChange={() => onModeChange(id as CalibrationMode)}
                  />
                  {label}
                </label>
              ))}
            </div>
          </div>
          <CharucoBoardParamPanel
            className="calibration-board-card"
            values={charucoParams}
            onChange={onCharucoParamChange}
          />
          <div className="log-box" ref={logRef} role="log" aria-label="标定日志">
            {isCalculating && <div className="calculation-progress" aria-label="标定计算进度" />}
            {logs.map((text, index) => (
              <div className="log-line" key={`${index}-${text}`}>{text}</div>
            ))}
          </div>
          </fieldset>
      </div>

      <div className="panel side-panel">
        <PanelTitle icon={<Activity size={18} />} title="机器人姿态" subtitle="机械臂位姿数据" />

        <fieldset className="group-box robot-pose-box">
          <legend>机器人姿态数据</legend>
          <Field
            kind="file"
            label="机械臂位姿文件"
            value={paths.robotPoseFile ?? defaultPoseFile}
            onValueChange={(value) => onPathChange("robotPoseFile", "机械臂位姿文件", value)}
          />
          <label className="field">
            <span>机器人数据格式：</span>
            <select value={poseFormat} onChange={(event) => onPoseFormatChange(event.target.value)}>
              {poseFormatOptions.map((option) => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
          </label>
          <div className="pose-box">
            {poseRows.map((row) => (
              <div className="pose-row" key={`${row.index}-${row.content}`}>
                <span>{row.index}</span>
                <code>{row.content}</code>
              </div>
            ))}
          </div>
        </fieldset>
      </div>
    </section>
  );
}

function ResultsPage({
  mode,
  result,
  selectedRows,
  selectedResultFrameIndices,
  isResultSelectionDirty,
  averageError,
  message,
  exportMessage,
  isRecalculating,
  onToggleRow,
  onToggleResultFrame,
  onRecalculate,
  onSaveYaml,
}: {
  mode: CalibrationMode;
  result: CalibrationRun | null;
  selectedRows: Set<string>;
  selectedResultFrameIndices: Set<number>;
  isResultSelectionDirty: boolean;
  averageError: number;
  message: string;
  exportMessage: string;
  isRecalculating: boolean;
  onToggleRow: (id: string) => void;
  onToggleResultFrame: (index: number) => void;
  onRecalculate: () => void | Promise<void>;
  onSaveYaml: () => void | Promise<void>;
}) {
  const resultMatrixRows = result?.matrixRows?.length ? result.matrixRows : [];
  const usedCount = hasResultFrameSelection(result) && isResultSelectionDirty
    ? selectedResultFrameIndices.size
    : result?.numImagesUsed ?? selectedRows.size;
  const totalCount = result?.numImages ?? errorRows.length;
  const resultMessage = isRecalculating ? message : result?.message ?? message;
  const frameErrors = result?.frameErrors ?? [];
  const hasFrameErrors = frameErrors.length > 0;
  const twoDimensionalRms = result?.reprojectionRmsPx ?? result?.reprojectionErrorPx;
  const baseConsistencyRms = result?.baseConsistencyRmsMm ?? aggregateFrameRms(frameErrors.map((row) => row.baseConsistencyRmsMm));
  const consistencyLabel = mode === "eye-to-hand" ? "法兰3D RMS" : "底座3D RMS";
  const consistencyTitle = mode === "eye-to-hand"
    ? "角点转换到法兰末端坐标系后的跨帧三维一致性 RMS，作为整体三维偏差结果"
    : "角点转换到机械臂底座坐标系后的跨帧三维一致性 RMS，作为整体三维偏差结果";

  return (
    <section className="page-grid results-grid">
      <div className="panel">
        <PanelTitle icon={<BarChart3 size={18} />} title="误差分析" subtitle="显示理论误差，支持剔除异常数据后重新计算。" />
        <div className="metric-row">
          <div>
            <span>2D RMS</span>
            <strong>{formatMetric(twoDimensionalRms, "px")}</strong>
          </div>
          <div>
            <span>{consistencyLabel}</span>
            <strong>{formatMetric(baseConsistencyRms, "mm")}</strong>
          </div>
          <div>
            <span>有效数据</span>
            <strong>{usedCount} / {totalCount}</strong>
          </div>
          <button className="secondary-action" onClick={onRecalculate} disabled={isRecalculating}>
            <RefreshCcw size={15} />
            {isRecalculating ? "重新计算中" : "重新计算"}
          </button>
        </div>
        {isRecalculating && <div className="calculation-progress analysis-progress" aria-label="重新计算进度" />}
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>使用</th>
                <th>帧</th>
                <th>图像</th>
                <th>检测模式</th>
                <th>角点</th>
                <th title="最终全局手眼链路推导位姿的角点重投影 RMS">2D RMS(px)</th>
                <th title="参考位姿与全局手眼链路推导位姿之间的平移残差">平移残差(mm)</th>
                <th title="参考位姿与全局手眼链路推导位姿之间的最小旋转角残差">旋转残差(deg)</th>
                <th title={consistencyTitle}>{consistencyLabel}(mm)</th>
              </tr>
            </thead>
            <tbody>
              {hasFrameErrors ? (
                frameErrors.map((row) => (
                  <tr
                    key={`${row.index}-${row.imagePath ?? ""}`}
                    className={selectedResultFrameIndices.has(row.index) ? "selected" : ""}
                    onClick={() => onToggleResultFrame(row.index)}
                  >
                    <td>
                      <input
                        type="checkbox"
                        checked={selectedResultFrameIndices.has(row.index)}
                        onChange={() => onToggleResultFrame(row.index)}
                        onClick={(event) => event.stopPropagation()}
                      />
                    </td>
                    <td>{String(row.index).padStart(3, "0")}</td>
                    <td>{row.imagePath ? fileNameFromPath(row.imagePath) : ""}</td>
                    <td>{detectionModeLabel(row.usedChessboardFallback)}</td>
                    <td>{row.cornerCount ?? ""}</td>
                    <td>{formatOptionalNumber(row.reprojectionRmsPx)}</td>
                    <td>{formatOptionalNumber(row.translationErrorMm)}</td>
                    <td>{formatOptionalNumber(row.rotationErrorDeg)}</td>
                    <td>{formatOptionalNumber(row.baseConsistencyRmsMm)}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={9}>暂无逐帧误差数据</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div className="panel">
        <PanelTitle icon={<Activity size={18} />} title="标定结果" subtitle={`${modeLabels[mode]}：${mode === "eye-in-hand" ? "相机到末端执行器" : "相机到机器人基座"} 的变换。`} />
        <div className="result-actions">
          <button className="secondary-action" onClick={onSaveYaml} disabled={!result?.matrixRows?.length}>
            保存 YAML
          </button>
        </div>
        <div className="matrix-box">
          {resultMatrixRows.map((row) => (
            <code key={row}>{row}</code>
          ))}
        </div>
        <div className="result-note">{resultMessage}</div>
        {exportMessage ? <div className="result-note export-note">{exportMessage}</div> : null}
      </div>
    </section>
  );
}

function hasResultFrameSelection(result: CalibrationRun | null) {
  return Boolean(result?.frameErrors?.length);
}

function mergeCalibrationFrameErrors(
  next: CalibrationRun,
  current: CalibrationRun | null,
  selectedFrameIndices: Set<number>,
): CalibrationRun | null {
  if (!next.matrixRows) return null;
  if (!current?.frameErrors?.length) return next;

  const nextRows = new Map((next.frameErrors ?? []).map((row) => [row.index, row]));
  const currentRows = new Map(current.frameErrors.map((row) => [row.index, row]));
  for (const row of next.frameErrors ?? []) {
    currentRows.set(row.index, row);
  }

  const frameErrors = Array.from(currentRows.values())
    .map((row) => {
      const updated = nextRows.get(row.index);
      const selected = selectedFrameIndices.has(row.index);
      if (updated && selected) {
        return { ...updated, used: true };
      }
      return clearFrameErrorMetrics({ ...row, ...(updated ? { imagePath: updated.imagePath || row.imagePath } : {}) });
    })
    .sort((a, b) => a.index - b.index);

  return {
    ...next,
    frameErrors,
  };
}

function clearFrameErrorMetrics(row: FrameError): FrameError {
  return {
    ...row,
    used: false,
    cornerCount: null,
    reprojectionMeanPx: null,
    reprojectionRmsPx: null,
    reprojectionMaxPx: null,
    referenceReprojectionMeanPx: null,
    referenceReprojectionRmsPx: null,
    referenceReprojectionMaxPx: null,
    reprojectionErrorPx: null,
    optimizedReprojectionErrorPx: null,
    baseConsistencyMeanMm: null,
    baseConsistencyRmsMm: null,
    baseConsistencyMaxMm: null,
    translationErrorMm: null,
    rotationErrorDeg: null,
  };
}

function detectionModeLabel(usedChessboardFallback?: boolean) {
  return usedChessboardFallback ? "Chessboard fallback" : "ChArUco";
}

function CharucoBoardParamPanel({
  className,
  values,
  onChange,
}: {
  className?: string;
  values: CharucoBoardParamValues;
  onChange: (field: keyof CharucoBoardParams, value: string) => void;
}) {
  return (
    <div className={`parameter-card charuco-board-card ${className ?? ""}`} aria-label="ChArUco 标定板参数">
      <div className="section-label">ChArUco 标定板</div>
      <div className="charuco-param-grid">
        <label className="param-field">
          <span>横向格数</span>
          <input
            type="number"
            min="2"
            step="1"
            value={values.squaresX}
            onChange={(event) => onChange("squaresX", event.target.value)}
          />
        </label>
        <label className="param-field">
          <span>纵向格数</span>
          <input
            type="number"
            min="2"
            step="1"
            value={values.squaresY}
            onChange={(event) => onChange("squaresY", event.target.value)}
          />
        </label>
        <label className="param-field">
          <span>方格边长(m)</span>
          <input
            type="number"
            min="0"
            step="0.001"
            value={values.squareLength}
            onChange={(event) => onChange("squareLength", event.target.value)}
          />
        </label>
        <label className="param-field">
          <span>Marker边长(m)</span>
          <input
            type="number"
            min="0"
            step="0.001"
            value={values.markerLength}
            onChange={(event) => onChange("markerLength", event.target.value)}
          />
        </label>
        <label className="param-field dict-field">
          <span>字典</span>
          <select value={values.arucoDict} onChange={(event) => onChange("arucoDict", event.target.value)}>
            {arucoDictOptions.map((option) => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        </label>
      </div>
    </div>
  );
}

function ToolsPage({
  mode,
  poseFormat,
  calibrationResult,
  cameraIntrinsics,
  distortionCoefficients,
  cameraIntrinsicsSource,
  charucoParams,
  onCharucoParamChange,
}: {
  mode: CalibrationMode;
  poseFormat: string;
  calibrationResult: CalibrationRun | null;
  cameraIntrinsics: Record<CameraIntrinsicField, string>;
  distortionCoefficients?: number[];
  cameraIntrinsicsSource: CameraIntrinsicsSource;
  charucoParams: CharucoBoardParamValues;
  onCharucoParamChange: (field: keyof CharucoBoardParams, value: string) => void;
}) {
  const [toolMessage, setToolMessage] = useState("选择工具并输入示例数据");
  const [charucoDetection, setCharucoDetection] = useState<CharucoDetection | null>(null);
  const [paths, setPaths] = useState<Partial<Record<ToolPathField, string>>>({});
  const [conversionPreview, setConversionPreview] = useState<PreviewFrame[]>([]);
  const [isGeneratingPreview, setIsGeneratingPreview] = useState(false);
  const [selectedPreviewFrameIndex, setSelectedPreviewFrameIndex] = useState<number | null>(null);
  const [hoveredPreviewFrameIndex, setHoveredPreviewFrameIndex] = useState<number | null>(null);
  const [previewLayers, setPreviewLayers] = useState<PreviewLayerVisibility>({
    showBaseAxes: true,
    showCameraFrames: true,
    showBoardFrames: true,
    showUnifiedBoardFrame: true,
    showLabels: true,
  });
  const [axisScale, setAxisScale] = useState(0.12);
  const autoDetectCountRef = useRef(0);
  const primaryMatrixRows = calibrationResult?.primaryMatrixRows?.length
    ? calibrationResult.primaryMatrixRows
    : (calibrationResult?.matrixRows ?? []);
  const secondaryMatrixRows = calibrationResult?.secondaryMatrixRows ?? [];
  const previewFrames = conversionPreview.length ? conversionPreview : (calibrationResult?.previewFrames ?? []);
  const referenceBoardInBase = useMemo(
    () => calibrationResult?.setup === "eye-in-hand" ? parseTransformRows(secondaryMatrixRows) : null,
    [calibrationResult?.setup, secondaryMatrixRows],
  );
  const referenceBoardInFocus = useMemo(
    () => parseTransformRows(secondaryMatrixRows),
    [secondaryMatrixRows],
  );
  const frameErrorsByIndex = useMemo(
    () => new Map((calibrationResult?.frameErrors ?? []).map((row) => [row.index, row])),
    [calibrationResult?.frameErrors],
  );
  const selectedPreviewFrame = selectedPreviewFrameIndex === null
    ? null
    : previewFrames.find((frame) => frame.index === selectedPreviewFrameIndex) ?? null;

  useEffect(() => {
    if (!conversionPreview.length && calibrationResult?.previewFrames?.length) {
      setConversionPreview(calibrationResult.previewFrames);
    }
  }, [calibrationResult?.previewFrames, conversionPreview.length]);

  useEffect(() => {
    if (!previewFrames.length) {
      setSelectedPreviewFrameIndex(null);
      return;
    }
    if (selectedPreviewFrameIndex !== null && previewFrames.some((frame) => frame.index === selectedPreviewFrameIndex)) {
      return;
    }
    const defaultFrame = previewFrames.find((frame) => frame.used) ?? previewFrames[0];
    setSelectedPreviewFrameIndex(defaultFrame?.index ?? null);
  }, [previewFrames, selectedPreviewFrameIndex]);

  /* Auto-sync: when frame selected in right panel, show its charuco detection in left panel */
  useEffect(() => {
    if (!selectedPreviewFrame?.imagePath) return;
    setPaths((current) => ({ ...current, circleImageFile: selectedPreviewFrame.imagePath! }));
    setCharucoDetection(null);
    detectCharuco(selectedPreviewFrame.imagePath);
  }, [selectedPreviewFrameIndex]);

  const updatePath = (field: ToolPathField, label: string, value: string) => {
    setPaths((current) => ({ ...current, [field]: value }));
    if (field === "circleImageFile") {
      setCharucoDetection(null);
    }
    if (field === "conversionDataFolder" || field === "conversionRobotPoseFile") {
      setConversionPreview([]);
      setSelectedPreviewFrameIndex(null);
    }
    setToolMessage(`已选择${label}：${value}`);
  };
  const detectCharuco = async (imageOverride?: string) => {
    const imagePath = imageOverride ?? paths.circleImageFile;
    if (!imagePath) {
      setToolMessage("请先选择图片文件");
      return;
    }
    if (cameraIntrinsicsSource !== "file") {
      setToolMessage("请先在标定界面选择包含 camera_params.yaml 的数据文件夹");
      return;
    }
    const intrinsics = parseCameraIntrinsics(cameraIntrinsics, distortionCoefficients);
    if (!intrinsics) {
      setToolMessage("读取到的相机内参无效");
      return;
    }
    const boardParams = parseCharucoBoardParams(charucoParams);
    if (!boardParams) {
      setToolMessage("请填写有效的 ChArUco 标定板参数");
      return;
    }
    const currentDetect = ++autoDetectCountRef.current;
    setToolMessage("ChArUco 识别中...");
    try {
      const result = await invoke<CharucoDetection>("detect_charuco", {
        request: {
          imagePath,
          depthPath: paths.circleDepthFile ?? null,
          cameraParams: null,
          cameraIntrinsics: intrinsics,
          ...boardParams,
        },
      });
      if (currentDetect !== autoDetectCountRef.current) return;
      setCharucoDetection(result);
      setToolMessage(result.success ? "ChArUco 识别完成" : result.message);
    } catch (error) {
      if (currentDetect !== autoDetectCountRef.current) return;
      setCharucoDetection(null);
      setToolMessage(`ChArUco 识别失败：${String(error)}`);
    }
  };
  const previewImage = charucoDetection
    ? { name: "ChArUco 检测结果", path: charucoDetection.outputPath }
    : paths.circleImageFile
      ? { name: fileNameFromPath(paths.circleImageFile), path: paths.circleImageFile }
      : null;
  const buildConversionPreview = async () => {
    if (!paths.conversionDataFolder || !paths.conversionRobotPoseFile) {
      setToolMessage("请先选择数据文件夹和机器人姿态文件");
      return;
    }
    if (!calibrationResult?.setup || !primaryMatrixRows.length || !secondaryMatrixRows.length) {
      setToolMessage("请先完成一次手眼标定，确保主/辅变换矩阵可用");
      return;
    }
    const intrinsics = parseCameraIntrinsics(cameraIntrinsics, distortionCoefficients);
    if (!intrinsics) {
      setToolMessage("请先填写有效的相机内参（在标定页选择包含 camera_params.yaml 的数据文件夹）");
      return;
    }
    const boardParams = parseCharucoBoardParams(charucoParams);
    if (!boardParams) {
      setToolMessage("请填写有效的 ChArUco 标定板参数");
      return;
    }
    setIsGeneratingPreview(true);
    setToolMessage("输出预览生成中...");
    try {
      const canReuseCalibrationPreview = Boolean(
        calibrationResult.previewFrames?.length
        && calibrationResult.imageDir === paths.conversionDataFolder
        && calibrationResult.posesFile === paths.conversionRobotPoseFile,
      );
      if (canReuseCalibrationPreview) {
        const preview = calibrationResult.previewFrames ?? [];
        setConversionPreview(preview);
        const defaultFrame = preview.find((frame) => frame.used) ?? preview[0] ?? null;
        setSelectedPreviewFrameIndex(defaultFrame?.index ?? null);
        setToolMessage(`输出预览已生成：${preview.length} 帧`);
        return;
      }
      const result = await invoke<ConversionPreviewResult>("build_conversion_preview", {
        request: {
          imageDir: paths.conversionDataFolder,
          posesFile: paths.conversionRobotPoseFile,
          setup: calibrationResult.setup,
          poseFormat,
          primaryTransformName: calibrationResult.primaryTransformName || (calibrationResult.setup === "eye-to-hand" ? "T_C2W" : "T_C2F"),
          primaryMatrixRows,
          secondaryTransformName: calibrationResult.secondaryTransformName || (calibrationResult.setup === "eye-to-hand" ? "T_O2F" : "T_O2W"),
          secondaryMatrixRows,
          cameraIntrinsics: intrinsics,
          ...boardParams,
        } satisfies ConversionPreviewRequestPayload,
      });
      setConversionPreview(result.previewFrames ?? []);
      const defaultFrame = result.previewFrames?.find((frame) => frame.used) ?? result.previewFrames?.[0] ?? null;
      setSelectedPreviewFrameIndex(defaultFrame?.index ?? null);
      setToolMessage(`输出预览已生成：${result.previewFrames?.length ?? 0} 帧`);
    } catch (error) {
      setToolMessage(`输出预览生成失败：${String(error)}`);
      setConversionPreview([]);
      setSelectedPreviewFrameIndex(null);
    } finally {
      setIsGeneratingPreview(false);
    }
  };

  return (
    <section className="tool-workspace">
      <div className="panel tool-panel circle-tool">
        <PanelTitle icon={<ImageIcon size={18} />} title="计算标记物圆心坐标" subtitle="导入一组图片和深度信息，选择标记物类型后查看检测结果。" />
        <fieldset className="group-box tool-group">
          <legend>标记物检测</legend>
          <div className="tool-detection-grid">
            <MarkerPreview
              marker="charuco"
              compact
              image={previewImage}
            />
            <CharucoBoardParamPanel
              className="tool-board-card"
              values={charucoParams}
              onChange={onCharucoParamChange}
            />
            <div className="parameter-card tool-input-card">
              <div className="section-label">检测输入</div>
              <div className="tool-input-column">
                <Field
                  kind="file"
                  label="图片文件"
                  value={paths.circleImageFile ?? ""}
                  onValueChange={(value) => updatePath("circleImageFile", "图片文件", value)}
                />
                <Field
                  kind="file"
                  label="深度图"
                  value={paths.circleDepthFile ?? ""}
                  onValueChange={(value) => updatePath("circleDepthFile", "深度图", value)}
                />
                <div className="tool-input-actions">
                  <label className="field marker-type-field">
                    <span>标记物类型：</span>
                    <select value="charuco" aria-readonly="true" onChange={() => undefined}>
                      <option value="charuco">ChArUco</option>
                    </select>
                  </label>
                  <button className="secondary-action detect-action" onClick={() => detectCharuco()}>
                    <Calculator size={15} />
                    识别 ChArUco
                  </button>
                </div>
              </div>
            </div>
            {charucoDetection && (
              <div className="tool-detection-summary">
                <div className="detection-summary">
                  角点 {charucoDetection.numCorners} / 标记 {charucoDetection.numMarkers}
                </div>
                <div className="detection-summary">
                  {detectionModeLabel(charucoDetection.usedChessboardFallback)}
                </div>
              </div>
            )}
          </div>
        </fieldset>
      </div>

      <div className="panel tool-panel conversion-tool">
        <PanelTitle icon={<Rotate3D size={18} />} title="点云坐标转换" subtitle={`导入数据文件夹、机器人位姿和 ${modeLabels[mode]} 标定矩阵，预览转换后的点云重合效果。`} />
        <div className="point-cloud-tool-grid">
          <div className="conversion-sidebar">
            {selectedPreviewFrame && (
              <fieldset className="group-box conversion-frame-info-box">
                <legend>帧信息</legend>
                <div className="conversion-preview-metrics">
                  <div className="coordinate-preview-current">当前帧：{previewFrameName(selectedPreviewFrame)}</div>
                  <div>
                    <b>{previewFrameTitle(selectedPreviewFrame)}</b>
                    <span>{selectedPreviewFrame.used ? "参与标定" : "已过滤 / 未使用"}</span>
                  </div>
                  <code>Camera {formatPoint(matrixTranslation(selectedPreviewFrame.cameraInBase), 3)}</code>
                  <code>Board {formatPoint(matrixTranslation(selectedPreviewFrame.boardInBase), 3)}</code>
                  <code>{previewFileNameFromPath(selectedPreviewFrame.imagePath ?? `frame-${selectedPreviewFrame.index}`)}</code>
                </div>
              </fieldset>
            )}
            <fieldset className="conversion-frame-box">
              <legend>帧列表</legend>
              <div className="section-label">输出预览帧（{previewFrames.length}）</div>
              <div className="conversion-frame-list" aria-label="输出预览帧列表">
                {previewFrames.length === 0 ? (
                  <div className="conversion-frame-empty">完成标定后在这里筛选和高亮帧。</div>
                ) : (
                  previewFrames.map((frame) => {
                    const errorRow = frameErrorsByIndex.get(frame.index);
                    const metric = firstNumber(errorRow?.baseConsistencyRmsMm, errorRow?.translationErrorMm);
                    const metricLabel = errorRow?.baseConsistencyRmsMm != null ? "3D RMS" : "平移残差";
                    return (
                      <button
                        key={`${frame.index}-${frame.imagePath ?? "frame"}`}
                        type="button"
                        aria-label={previewFrameTitle(frame)}
                        className={`conversion-frame-item ${selectedPreviewFrameIndex === frame.index ? "selected" : ""} ${frame.used ? "" : "is-unused"}`}
                        aria-pressed={selectedPreviewFrameIndex === frame.index}
                        onMouseEnter={() => setHoveredPreviewFrameIndex(frame.index)}
                        onMouseLeave={() => setHoveredPreviewFrameIndex((current) => (current === frame.index ? null : current))}
                        onClick={() => setSelectedPreviewFrameIndex(frame.index)}
                      >
                        <strong>{previewFrameTitle(frame)}</strong>
                        <span>{frame.used ? "参与标定" : "已过滤 / 未使用"}</span>
                        <span>{metric == null ? "无附加误差指标" : `${metricLabel} ${metric.toFixed(3)}`}</span>
                      </button>
                    );
                  })
                )}
              </div>
            </fieldset>
            <fieldset className="group-box conversion-layer-box">
              <legend>图层控制</legend>
              <label className="toggle-row"><input type="checkbox" checked={previewLayers.showBaseAxes} onChange={() => setPreviewLayers((current) => ({ ...current, showBaseAxes: !current.showBaseAxes }))} /> 底座坐标系</label>
              <label className="toggle-row"><input type="checkbox" checked={previewLayers.showCameraFrames} onChange={() => setPreviewLayers((current) => ({ ...current, showCameraFrames: !current.showCameraFrames }))} /> 相机坐标轴（关闭时保留圆心）</label>
              <label className="toggle-row"><input type="checkbox" checked={previewLayers.showBoardFrames} onChange={() => setPreviewLayers((current) => ({ ...current, showBoardFrames: !current.showBoardFrames }))} /> 标定板坐标轴（关闭时保留圆心）</label>
              <label className="toggle-row"><input type="checkbox" checked={previewLayers.showUnifiedBoardFrame} onChange={() => setPreviewLayers((current) => ({ ...current, showUnifiedBoardFrame: !current.showUnifiedBoardFrame }))} /> 3D一致性统一标定板</label>
              <label className="toggle-row"><input type="checkbox" checked={previewLayers.showLabels} onChange={() => setPreviewLayers((current) => ({ ...current, showLabels: !current.showLabels }))} /> 编号标签</label>
              <label className="field axis-scale-field">
                <span>坐标轴长度：{axisScale.toFixed(2)} m</span>
                <input type="range" min="0.05" max="0.5" step="0.01" value={axisScale} onChange={(event) => setAxisScale(Number(event.target.value))} />
              </label>
            </fieldset>
          </div>
          <fieldset className="group-box cloud-output-box">
            <legend>2. 输出预览</legend>
            <div className="conversion-preview-panel">
              <CoordinatePreview3D
                frames={previewFrames}
                selectedFrameIndex={selectedPreviewFrameIndex}
                hoveredFrameIndex={hoveredPreviewFrameIndex}
                layers={previewLayers}
                axisScale={axisScale}
                referenceBoardInBase={referenceBoardInBase}
                referenceBoardInFocus={referenceBoardInFocus}
              />
            </div>
          </fieldset>
        </div>
      </div>
      <div className="tool-status">{toolMessage}</div>
    </section>
  );
}

function PanelTitle({ icon, title, subtitle }: { icon: ReactNode; title: string; subtitle: string }) {
  return (
    <div className="panel-title">
      <div className="title-icon">{icon}</div>
      <div>
        <h2>{title}</h2>
        <p>{subtitle}</p>
      </div>
    </div>
  );
}

function Field({
  kind,
  label,
  value,
  onValueChange,
}: {
  kind: PathFieldKind;
  label: string;
  value: string;
  onValueChange: (value: string) => void;
}) {
  const handleBrowse = async () => {
    const selected = await open({ directory: kind === "directory", multiple: false });
    if (typeof selected !== "string") return;
    onValueChange(selected);
  };

  return (
    <label className="field">
      <span>{label}</span>
      <div className="file-control">
        <input value={value} readOnly />
        <button type="button" aria-label={`${label}浏览`} onClick={handleBrowse}>
          <FolderOpen size={15} />
          浏览
        </button>
      </div>
    </label>
  );
}

function ImageFileList({
  title,
  files,
  selectedIndex,
  onSelect,
}: {
  title: string;
  files: CalibrationImageEntry[];
  selectedIndex: number;
  onSelect: (image: CalibrationImageEntry) => void;
}) {
  return (
    <div className="file-list image-file-list">
      <div className="section-label">{title}（{files.length}）</div>
      <ul>
        {files.map((file, index) => (
          <li key={`${file.name}-${file.path}`}>
            <button
              type="button"
              className={index === selectedIndex ? "selected-file" : ""}
              onClick={() => onSelect(file)}
            >
              {String(index + 1).padStart(3, "0")}. {file.displayName}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

function MarkerPreview({
  marker,
  compact = false,
  image,
  depthImage,
  depthPreviewImage,
}: {
  marker: MarkerType;
  compact?: boolean;
  image?: ImageFileEntry | CalibrationImageEntry | null;
  depthImage?: ImageFileEntry | null;
  depthPreviewImage?: ImageFileEntry | null;
}) {
  return (
    <div className={`marker-preview ${compact ? "compact-preview" : ""}`}>
      <div className="section-label">图像显示区域</div>
      {image?.path && depthImage?.path ? (
        <div className="paired-preview-grid">
          <figure className="preview-pane">
            <span className="preview-caption">RGB</span>
            <ZoomableImage src={convertFileSrc(image.path)} alt={`RGB ${image.name}`} />
          </figure>
          <figure className="preview-pane">
            <span className="preview-caption">Depth</span>
            {depthPreviewImage?.path ? (
              <ZoomableImage src={convertFileSrc(depthPreviewImage.path)} alt={`深度 ${depthImage.name}`} />
            ) : (
              <div className="preview-placeholder">生成中</div>
            )}
          </figure>
        </div>
      ) : image?.path ? (
        <ZoomableImage src={convertFileSrc(image.path)} alt={image.name} />
      ) : marker === "charuco" ? (
        <CalibrationBoard />
      ) : (
        <ConcentricMarker />
      )}
    </div>
  );
}

function ZoomableImage({ src, alt }: { src: string; alt: string }) {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef<{ x: number; y: number; panX: number; panY: number } | null>(null);
  const viewportRef = useRef<HTMLDivElement | null>(null);

  const updateZoom = (delta: number) => {
    setZoom((current) => {
      const next = Math.min(4, Math.max(1, Number((current + delta).toFixed(2))));
      const ratio = current === 0 ? 1 : next / current;
      const nextPan = next === 1
        ? { x: 0, y: 0 }
        : {
            x: Math.round(pan.x * ratio),
            y: Math.round(pan.y * ratio),
          };
      setPan(nextPan);
      if (viewportRef.current) {
        viewportRef.current.scrollLeft = nextPan.x;
        viewportRef.current.scrollTop = nextPan.y;
      }
      return next;
    });
  };

  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setIsDragging(false);
    dragStartRef.current = null;
    if (viewportRef.current) {
      viewportRef.current.scrollLeft = 0;
      viewportRef.current.scrollTop = 0;
    }
  };

  const isPannable = zoom > 1;
  const zoomPercent = `${Number((zoom * 100).toFixed(2))}%`;

  return (
    <div
      ref={viewportRef}
      className={`zoom-viewport ${isPannable ? "is-pannable" : ""} ${isDragging ? "is-dragging" : ""}`}
      aria-label={`图像预览 ${alt}`}
      data-zoom={zoom.toFixed(2)}
      data-pan-x={pan.x}
      data-pan-y={pan.y}
      onWheel={(event) => {
        event.preventDefault();
        updateZoom(event.deltaY < 0 ? 0.1 : -0.1);
      }}
      onDoubleClick={resetView}
      onMouseDown={(event) => {
        if (!isPannable) return;
        dragStartRef.current = {
          x: event.clientX,
          y: event.clientY,
          panX: pan.x,
          panY: pan.y,
        };
        setIsDragging(true);
      }}
      onMouseMove={(event) => {
        if (!dragStartRef.current) return;
        const nextPan = {
          x: dragStartRef.current.panX - (event.clientX - dragStartRef.current.x),
          y: dragStartRef.current.panY - (event.clientY - dragStartRef.current.y),
        };
        setPan(nextPan);
        if (viewportRef.current) {
          viewportRef.current.scrollLeft = nextPan.x;
          viewportRef.current.scrollTop = nextPan.y;
        }
      }}
      onMouseUp={() => {
        dragStartRef.current = null;
        setIsDragging(false);
      }}
      onMouseLeave={() => {
        dragStartRef.current = null;
        setIsDragging(false);
      }}
    >
      <div className="zoom-stage" style={{ width: zoomPercent, height: zoomPercent }}>
        <img
          src={src}
          alt={alt}
          draggable={false}
        />
      </div>
    </div>
  );
}

function CalibrationBoard() {
  return (
    <div className="board">
      {Array.from({ length: 44 }, (_, index) => (
        <span key={index} style={{ "--i": index } as CSSProperties} />
      ))}
    </div>
  );
}

function ConcentricMarker() {
  return (
    <div className="concentric">
      <span />
      <span />
      <span />
      <i />
    </div>
  );
}
