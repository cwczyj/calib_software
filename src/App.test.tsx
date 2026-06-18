import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";
import { axisColors, matrixRowsToThreeSetArgs, originColor } from "./CoordinatePreview3D";

const { mockOpen, mockSave } = vi.hoisted(() => ({
  mockOpen: vi.fn(),
  mockSave: vi.fn(),
}));
const { mockInvoke } = vi.hoisted(() => ({
  mockInvoke: vi.fn(),
}));

vi.mock("@tauri-apps/plugin-dialog", () => ({
  open: mockOpen,
  save: mockSave,
}));

vi.mock("@tauri-apps/api/core", () => ({
  invoke: mockInvoke,
  convertFileSrc: (path: string) => `asset://${path}`,
}));

vi.mock("@tauri-apps/api/path", () => ({
  convertFileSrc: (path: string) => `asset://${path}`,
}));

describe("HandEyeManager UI", () => {
  beforeEach(() => {
    mockOpen.mockReset();
    mockSave.mockReset();
    mockInvoke.mockReset();
  });

  it("keeps row-major transform translations in the three.js matrix set arguments", () => {
    expect(matrixRowsToThreeSetArgs([
      [1, 0, 0, 0.1],
      [0, 1, 0, 0.2],
      [0, 0, 1, 0.3],
      [0, 0, 0, 1],
    ])).toEqual([
      1, 0, 0, 0.1,
      0, 1, 0, 0.2,
      0, 0, 1, 0.3,
      0, 0, 0, 1,
    ]);
  });

  it("uses standard RGB colors for coordinate axes regardless of selection", () => {
    expect(axisColors()).toEqual(["#ef4444", "#16a34a", "#2563eb"]);
    expect(originColor(false)).toBe("#ff4444");
    expect(originColor(true)).toBe("#ffd700");
  });

  it("shows marker calibration without TCP touch calibration", () => {
    render(<App />);

    expect(screen.getByRole("heading", { name: "标记物标定" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "眼在手上" })).toBeChecked();
    expect(screen.queryByText("TCP触碰标定")).not.toBeInTheDocument();
    expect(screen.queryByText("流程检查")).not.toBeInTheDocument();
    expect(screen.queryByText("如本科技")).not.toBeInTheDocument();
    expect(screen.queryByText("HandEyeManager")).not.toBeInTheDocument();
    expect(screen.queryByText("算法：")).not.toBeInTheDocument();
  });

  it("switches calibration mode and requires selected data before calculation", async () => {
    const user = userEvent.setup();
    render(<App />);

    expect(screen.queryByText("0.png")).not.toBeInTheDocument();

    await user.click(screen.getByRole("radio", { name: "眼在手外" }));
    expect(screen.getByText("相机固定于外部基座，机器人末端携带标记物多姿态采集。")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "计算" }));
    expect(screen.getByText("请先选择 RGB 文件夹和机械臂位姿文件")).toBeInTheDocument();
  });

  it("provides result analysis and both tools", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("tab", { name: "结果 / 误差分析" }));
    expect(screen.getByRole("heading", { name: "误差分析" })).toBeInTheDocument();
    expect(screen.getByText("2D RMS")).toBeInTheDocument();
    expect(screen.getByText("底座3D RMS")).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "工具" }));
    expect(screen.getByRole("heading", { name: "计算标记物圆心坐标" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "点云坐标转换" })).toBeInTheDocument();
    expect(screen.getByText("2. 输出预览")).toBeInTheDocument();
    expect(screen.getByRole("group", { name: "帧列表" })).toBeInTheDocument();
    expect(screen.getByRole("group", { name: "图层控制" })).toBeInTheDocument();
    expect(screen.queryByText("1. 转换参数")).not.toBeInTheDocument();
    expect(screen.queryByText("圆心坐标结果查看")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("数据文件夹")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("点云文件夹")).not.toBeInTheDocument();
    expect(screen.queryByRole("combobox", { name: "姿态类型：" })).not.toBeInTheDocument();
    expect(screen.queryByRole("textbox", { name: "手眼标定结果矩阵（16个数值，逗号分隔）：" })).not.toBeInTheDocument();
    expect(screen.queryByLabelText("点云(mm)")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("位姿(mm)")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("角度")).not.toBeInTheDocument();
  });

  it("keeps the Tauri shell responsive below the desktop layout width", () => {
    const styles = readFileSync(resolve(process.cwd(), "src/styles.css"), "utf8");

    expect(styles).not.toMatch(/body\s*{[^}]*min-width:\s*1180px/s);
    expect(styles).not.toMatch(/body\s*{[^}]*min-height:\s*760px/s);
    expect(styles).not.toMatch(/main\s*{[^}]*overflow:\s*hidden/s);
    expect(styles).toContain("@media (max-width: 900px)");
  });

  it("enables the Tauri asset protocol for local image previews", () => {
    const config = JSON.parse(readFileSync(resolve(process.cwd(), "src-tauri/tauri.conf.json"), "utf8"));

    expect(config.app.security.assetProtocol).toEqual({
      enable: true,
      scope: {
        requireLiteralLeadingDot: false,
        allow: ["**/*"],
      },
    });
  });

  it("allows the Tauri save dialog required for YAML export", () => {
    const capability = JSON.parse(
      readFileSync(resolve(process.cwd(), "src-tauri/capabilities/default.json"), "utf8"),
    );

    expect(capability.permissions).toContain("dialog:allow-save");
  });

  it("keeps the calibration panels the same height with constrained overflow", () => {
    const styles = readFileSync(resolve(process.cwd(), "src/styles.css"), "utf8");

    expect(styles).toMatch(/\.calibration-workspace\s*{[^}]*height:\s*100%/s);
    expect(styles).toMatch(/\.calibration-workspace\s+\.main-panel\s*{[^}]*overflow:\s*auto/s);
  });

  it("does not add nested scrollbars inside calibration group boxes", () => {
    const styles = readFileSync(resolve(process.cwd(), "src/styles.css"), "utf8");

    expect(styles).not.toMatch(/\.calibration-workspace\s+\.group-box\s*{[^}]*overflow:\s*auto/s);
  });

  it("keeps the robot pose preview box fixed while its contents scroll", () => {
    const styles = readFileSync(resolve(process.cwd(), "src/styles.css"), "utf8");

    expect(styles).toMatch(/\.robot-pose-box \.pose-box\s*{[^}]*flex:\s*1/s);
    expect(styles).toMatch(/\.robot-pose-box \.pose-box\s*{[^}]*overflow:\s*auto/s);
  });

  it("keeps the results error list inside a fixed scrolling area", () => {
    const styles = readFileSync(resolve(process.cwd(), "src/styles.css"), "utf8");

    expect(styles).toMatch(/\.results-grid\s*{[^}]*height:\s*100%/s);
    expect(styles).toMatch(/\.results-grid \.panel:first-child\s*{[^}]*overflow:\s*hidden/s);
    expect(styles).toMatch(/\.results-grid \.panel:first-child \.table-wrap\s*{[^}]*max-height:\s*calc\(100vh - 220px\)/s);
    expect(styles).toMatch(/\.results-grid \.panel:first-child \.table-wrap\s*{[^}]*overflow:\s*auto/s);
  });

  it("opens folder and file pickers from path fields and stores the selected paths", async () => {
    const user = userEvent.setup();
    mockOpen
      .mockResolvedValueOnce("/tmp/handeye-data")
      .mockResolvedValueOnce("/tmp/handeye-data/pose.txt")
      .mockResolvedValueOnce(null);
    render(<App />);

    await user.click(screen.getByRole("button", { name: "深度图和 RGB 文件夹浏览" }));
    expect(mockOpen).toHaveBeenLastCalledWith({ directory: true, multiple: false });
    expect(screen.getByDisplayValue("/tmp/handeye-data")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "机械臂位姿文件浏览" }));
    expect(mockOpen).toHaveBeenLastCalledWith({ directory: false, multiple: false });
    expect(screen.getByDisplayValue("/tmp/handeye-data/pose.txt")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "机械臂位姿文件浏览" }));
    expect(screen.getByDisplayValue("/tmp/handeye-data/pose.txt")).toBeInTheDocument();
  });

  it("loads RGB images from a selected folder and previews the selected image", async () => {
    const user = userEvent.setup();
    mockOpen.mockResolvedValueOnce("/tmp/handeye-data");
    mockInvoke
      .mockResolvedValueOnce([
        { name: "001_Color.png", path: "/tmp/handeye-data/001_Color.png" },
        { name: "002_Color.png", path: "/tmp/handeye-data/002_Color.png" },
      ])
      .mockResolvedValueOnce([]);

    render(<App />);

    await user.click(screen.getByRole("button", { name: "深度图和 RGB 文件夹浏览" }));

    expect(mockInvoke).toHaveBeenCalledWith("list_rgb_images", { folder: "/tmp/handeye-data" });
    expect(screen.getByRole("button", { name: /001\. 001_Color\.png/ })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /002\. 002_Color\.png/ }));
    expect(screen.getByRole("img", { name: "002_Color.png" })).toHaveAttribute(
      "src",
      "asset:///tmp/handeye-data/002_Color.png",
    );
  });

  it("loads camera intrinsics from camera_params.yaml when selecting the RGB/depth folder", async () => {
    const user = userEvent.setup();
    mockOpen.mockResolvedValueOnce("/tmp/handeye-data");
    mockInvoke
      .mockResolvedValueOnce([{ name: "001_Color.png", path: "/tmp/handeye-data/001_Color.png" }])
      .mockResolvedValueOnce([{ name: "001_Depth.png", path: "/tmp/handeye-data/001_Depth.png" }])
      .mockResolvedValueOnce({
        cx: 321.5,
        cy: 242.25,
        fx: 610.75,
        fy: 611.5,
        distortionCoefficients: [0.1, -0.2, 0.01, 0.02, 0.03],
      });

    render(<App />);
    await user.click(screen.getByRole("button", { name: "深度图和 RGB 文件夹浏览" }));

    expect(mockInvoke).toHaveBeenCalledWith("read_camera_params", { folder: "/tmp/handeye-data" });
    expect(screen.queryByLabelText("cx")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("cy")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("fx")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("fy")).not.toBeInTheDocument();
    expect(screen.getByText("已载入 1 张 RGB 图像，1 张深度图，并读取相机内参")).toBeInTheDocument();
    expect(screen.getByText("已读取相机参数：cx=321.500000, cy=242.250000, fx=610.750000, fy=611.500000")).toBeInTheDocument();
    expect(screen.getByText("已读取畸变参数：0.100000, -0.200000, 0.010000, 0.020000, 0.030000")).toBeInTheDocument();
  });

  it("syncs loaded calibration camera parameters to tools and passes distortion to ChArUco detection", async () => {
    const user = userEvent.setup();
    mockOpen
      .mockResolvedValueOnce("/tmp/handeye-data")
      .mockResolvedValueOnce("/tmp/handeye-data/001_Color.png")
      .mockResolvedValueOnce("/tmp/handeye-data/001_Depth.png");
    mockInvoke.mockImplementation((command: string) => {
      if (command === "list_rgb_images") {
        return Promise.resolve([{ name: "001_Color.png", path: "/tmp/handeye-data/001_Color.png" }]);
      }
      if (command === "list_depth_images") {
        return Promise.resolve([{ name: "001_Depth.png", path: "/tmp/handeye-data/001_Depth.png" }]);
      }
      if (command === "read_camera_params") {
        return Promise.resolve({
          cx: 321.5,
          cy: 242.25,
          fx: 610.75,
          fy: 611.5,
          distortionCoefficients: [0.1, -0.2, 0.01, 0.02, 0.03],
        });
      }
      if (command === "detect_charuco") {
        return Promise.resolve({
          imagePath: "/tmp/handeye-data/001_Color.png",
          outputPath: "/tmp/handeye-data/detection/detection_001.png",
          success: true,
          numCorners: 1,
          numMarkers: 1,
          message: "ok",
          cornerRows: [],
        });
      }
      return Promise.resolve(null);
    });

    render(<App />);
    await user.click(screen.getByRole("button", { name: "深度图和 RGB 文件夹浏览" }));
    await user.click(screen.getByRole("tab", { name: "工具" }));

    expect(screen.queryByLabelText("cx")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("cy")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("fx")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("fy")).not.toBeInTheDocument();
    expect(screen.queryByText("相机内参")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "图片文件浏览" }));
    await user.click(screen.getByRole("button", { name: "深度图浏览" }));
    await user.click(screen.getByRole("button", { name: "识别 ChArUco" }));

    expect(mockInvoke).toHaveBeenLastCalledWith("detect_charuco", {
      request: expect.objectContaining({
        cameraIntrinsics: {
          cx: 321.5,
          cy: 242.25,
          fx: 610.75,
          fy: 611.5,
          distortionCoefficients: [0.1, -0.2, 0.01, 0.02, 0.03],
        },
      }),
    });
  });

  it("loads depth images separately and leaves the depth list empty when none are found", async () => {
    const user = userEvent.setup();
    mockOpen.mockResolvedValueOnce("/tmp/handeye-data");
    mockInvoke
      .mockResolvedValueOnce([
        { name: "001_Color.png", path: "/tmp/handeye-data/001_Color.png" },
        { name: "013_Color.png", path: "/tmp/handeye-data/013_Color.png" },
      ])
      .mockResolvedValueOnce([]);

    render(<App />);

    expect(screen.queryByText("0.ply")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "深度图和 RGB 文件夹浏览" }));

    expect(mockInvoke).toHaveBeenCalledWith("list_rgb_images", { folder: "/tmp/handeye-data" });
    expect(mockInvoke).toHaveBeenCalledWith("list_depth_images", { folder: "/tmp/handeye-data" });
    expect(screen.getByRole("button", { name: /001\. 001_Color\.png/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /002\. 013_Color\.png/ })).toBeInTheDocument();
    expect(screen.queryByText("001_Depth.png")).not.toBeInTheDocument();
  });

  it("pairs matching RGB and depth images and previews both when selected", async () => {
    const user = userEvent.setup();
    mockOpen.mockResolvedValueOnce("/tmp/handeye-data");
    mockInvoke.mockImplementation((command: string) => {
      if (command === "list_rgb_images") {
        return Promise.resolve([
          { name: "001_Color.png", path: "/tmp/handeye-data/001_Color.png" },
          { name: "002_Color.png", path: "/tmp/handeye-data/002_Color.png" },
        ]);
      }
      if (command === "list_depth_images") {
        return Promise.resolve([
          { name: "001_Depth.png", path: "/tmp/handeye-data/001_Depth.png" },
          { name: "002_Depth.png", path: "/tmp/handeye-data/002_Depth.png" },
        ]);
      }
      if (command === "read_camera_params") {
        return Promise.resolve({ cx: 640, cy: 360, fx: 600, fy: 600 });
      }
      if (command === "create_depth_preview") {
        return Promise.resolve({ name: "002_Depth_preview.png", path: "/tmp/handeye-data/preview/002_Depth_preview.png" });
      }
      return Promise.resolve(null);
    });

    render(<App />);
    await user.click(screen.getByRole("button", { name: "深度图和 RGB 文件夹浏览" }));

    expect(screen.getByRole("button", { name: /001\. 001_Color\.png\|001_Depth\.png/ })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /002\. 002_Color\.png\|002_Depth\.png/ }));

    expect(mockInvoke).toHaveBeenCalledWith("create_depth_preview", { depthPath: "/tmp/handeye-data/002_Depth.png" });
    expect(await screen.findByRole("img", { name: "RGB 002_Color.png" })).toHaveAttribute(
      "src",
      "asset:///tmp/handeye-data/002_Color.png",
    );
    expect(await screen.findByRole("img", { name: "深度 002_Depth.png" })).toHaveAttribute(
      "src",
      "asset:///tmp/handeye-data/preview/002_Depth_preview.png",
    );
  });

  it("keeps depth calibration off by default when paired depth images are available", async () => {
    const user = userEvent.setup();
    mockOpen
      .mockResolvedValueOnce("/tmp/handeye-data")
      .mockResolvedValueOnce("/tmp/handeye-data/poses.csv");
    mockInvoke.mockImplementation((command: string) => {
      if (command === "list_rgb_images") {
        return Promise.resolve([{ name: "001_Color.png", path: "/tmp/handeye-data/001_Color.png" }]);
      }
      if (command === "list_depth_images") {
        return Promise.resolve([{ name: "001_Depth.png", path: "/tmp/handeye-data/001_Depth.png" }]);
      }
      if (command === "read_camera_params") {
        return Promise.resolve({ cx: 640, cy: 360, fx: 600, fy: 600 });
      }
      if (command === "read_pose_file") {
        return Promise.resolve([{ index: 1, content: "0, 0, 0, 0, 0, 0" }]);
      }
      if (command === "create_depth_preview") {
        return Promise.resolve({ name: "001_Depth_preview.png", path: "/tmp/handeye-data/preview/001_Depth_preview.png" });
      }
      if (command === "run_handeye_calibration") {
        return Promise.resolve({
          outputPath: "/tmp/handeye-data/calibration_result.yaml",
          setup: "eye-in-hand",
          primaryTransformName: "T_C2F",
          matrixRows: ["1.0000000, 0.0000000, 0.0000000, 0.0000000"],
          averageErrorMm: 0,
          rotationErrorDeg: 0,
          reprojectionErrorPx: 0,
          numImages: 1,
          numImagesUsed: 1,
          filteredImages: [],
          depthUsed: false,
          frameErrors: [],
          message: "done",
        });
      }
      return Promise.resolve(null);
    });

    render(<App />);
    await user.click(screen.getByRole("button", { name: "深度图和 RGB 文件夹浏览" }));
    await user.click(screen.getByRole("button", { name: "机械臂位姿文件浏览" }));
    await user.click(screen.getByRole("button", { name: "计算" }));

    expect(mockInvoke).toHaveBeenLastCalledWith("run_handeye_calibration", {
      request: expect.objectContaining({
        useDepth: "off",
        cameraIntrinsics: { cx: 640, cy: 360, fx: 600, fy: 600 },
      }),
    });
  });

  it("does not truncate calibration image lists so scrolling can reveal every file", async () => {
    const user = userEvent.setup();
    const rgbFiles = Array.from({ length: 13 }, (_, index) => {
      const id = String(index + 1).padStart(3, "0");
      return { name: `${id}_Color.png`, path: `/tmp/handeye-data/${id}_Color.png` };
    });
    const depthFiles = Array.from({ length: 13 }, (_, index) => {
      const id = String(index + 1).padStart(3, "0");
      return { name: `${id}_Depth.png`, path: `/tmp/handeye-data/${id}_Depth.png` };
    });
    mockOpen.mockResolvedValueOnce("/tmp/handeye-data");
    mockInvoke.mockResolvedValueOnce(rgbFiles).mockResolvedValueOnce(depthFiles);

    render(<App />);
    await user.click(screen.getByRole("button", { name: "深度图和 RGB 文件夹浏览" }));

    expect(screen.getByRole("button", { name: /013\. 013_Color\.png/ })).toBeInTheDocument();
  });

  it("only offers ChArUco as the calibration marker type", () => {
    render(<App />);

    const markerSelect = screen.getByRole("combobox", { name: "标记物类型：" });
    expect(markerSelect).toHaveValue("charuco");
    expect(markerSelect).toHaveTextContent("ChArUco");
    expect(markerSelect).not.toHaveTextContent("同心圆");
    expect(markerSelect).not.toHaveTextContent("非对称黑底白圆标定板");
  });

  it("reads the selected robot pose file and displays each non-empty row with an index", async () => {
    const user = userEvent.setup();
    mockOpen.mockResolvedValueOnce("/tmp/handeye-data/poses.csv");
    mockInvoke.mockResolvedValueOnce([
      { index: 1, content: "762.4474, -128.1737, -28.2222, -171.6236, 13.3687, -78.3524" },
      { index: 2, content: "714.1447, -90.8957, -15.4468, -173.1541, 10.4620, -83.3007" },
    ]);

    render(<App />);
    await user.click(screen.getByRole("button", { name: "机械臂位姿文件浏览" }));

    expect(mockInvoke).toHaveBeenCalledWith("read_pose_file", { file: "/tmp/handeye-data/poses.csv" });
    expect(screen.getByText("1")).toBeInTheDocument();
    expect(screen.getByText("762.4474, -128.1737, -28.2222, -171.6236, 13.3687, -78.3524")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
    expect(screen.getByText("714.1447, -90.8957, -15.4468, -173.1541, 10.4620, -83.3007")).toBeInTheDocument();
  });

  it("runs RGB-only calibration with the selected mode and data paths", async () => {
    const user = userEvent.setup();
    mockOpen
      .mockResolvedValueOnce("/tmp/handeye-data")
      .mockResolvedValueOnce("/tmp/handeye-data/poses.csv");
    mockInvoke
      .mockResolvedValueOnce([{ name: "001_Color.png", path: "/tmp/handeye-data/001_Color.png" }])
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce({ cx: 640, cy: 360, fx: 600, fy: 600 })
      .mockResolvedValueOnce([{ index: 1, content: "762.4474, -128.1737, -28.2222, -171.6236, 13.3687, -78.3524" }])
      .mockResolvedValueOnce({
        outputPath: "/tmp/handeye-data/calibration_result.yaml",
        stdout: "Reprojection (derived): 0.42 px",
        stderr: "",
      });

    render(<App />);

    await user.click(screen.getByRole("button", { name: "深度图和 RGB 文件夹浏览" }));
    await user.click(screen.getByRole("button", { name: "机械臂位姿文件浏览" }));
    await user.click(screen.getByRole("radio", { name: "眼在手外" }));
    await user.click(screen.getByRole("button", { name: "计算" }));

    expect(mockInvoke).toHaveBeenLastCalledWith("run_handeye_calibration", {
      request: expect.objectContaining({
        imageDir: "/tmp/handeye-data",
        posesFile: "/tmp/handeye-data/poses.csv",
        setup: "eye-to-hand",
        useDepth: "off",
        squaresX: 14,
        squaresY: 9,
        squareLength: 0.02,
        markerLength: 0.015,
        arucoDict: "DICT_5X5_50",
        cameraIntrinsics: { cx: 640, cy: 360, fx: 600, fy: 600 },
      }),
    });
    expect(screen.getByText(/RGB-only 标定完成/)).toBeInTheDocument();
  });

  it("passes edited ChArUco board parameters to calibration", async () => {
    const user = userEvent.setup();
    mockOpen
      .mockResolvedValueOnce("/tmp/handeye-data")
      .mockResolvedValueOnce("/tmp/handeye-data/poses.csv");
    mockInvoke
      .mockResolvedValueOnce([{ name: "001_Color.png", path: "/tmp/handeye-data/001_Color.png" }])
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce({ cx: 640, cy: 360, fx: 600, fy: 600 })
      .mockResolvedValueOnce([{ index: 1, content: "pose" }])
      .mockResolvedValueOnce({
        outputPath: "/tmp/handeye-data/calibration_result.yaml",
        stdout: "",
        stderr: "",
      });

    render(<App />);
    await user.click(screen.getByRole("button", { name: "深度图和 RGB 文件夹浏览" }));
    await user.click(screen.getByRole("button", { name: "机械臂位姿文件浏览" }));
    await user.clear(screen.getByLabelText("横向格数"));
    await user.type(screen.getByLabelText("横向格数"), "9");
    await user.clear(screen.getByLabelText("纵向格数"));
    await user.type(screen.getByLabelText("纵向格数"), "6");
    await user.clear(screen.getByLabelText("方格边长(m)"));
    await user.type(screen.getByLabelText("方格边长(m)"), "0.025");
    await user.clear(screen.getByLabelText("Marker边长(m)"));
    await user.type(screen.getByLabelText("Marker边长(m)"), "0.018");
    await user.selectOptions(screen.getByLabelText("字典"), "DICT_4X4_50");
    await user.click(screen.getByRole("button", { name: "计算" }));

    expect(mockInvoke).toHaveBeenLastCalledWith("run_handeye_calibration", {
      request: expect.objectContaining({
        squaresX: 9,
        squaresY: 6,
        squareLength: 0.025,
        markerLength: 0.018,
        arucoDict: "DICT_4X4_50",
        cameraIntrinsics: { cx: 640, cy: 360, fx: 600, fy: 600 },
      }),
    });
  });

  it("passes calibration page selections, shows local progress, and renders returned results", async () => {
    const user = userEvent.setup();
    mockOpen
      .mockResolvedValueOnce("/tmp/handeye-data")
      .mockResolvedValueOnce("/tmp/handeye-data/poses.csv");
    mockInvoke
      .mockResolvedValueOnce([{ name: "001_Color.png", path: "/tmp/handeye-data/001_Color.png" }])
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce({ cx: 640, cy: 360, fx: 600, fy: 600 })
      .mockResolvedValueOnce([{ index: 1, content: "762.4474, -128.1737, -28.2222, -171.6236, 13.3687, -78.3524" }])
      .mockImplementationOnce(() => new Promise((resolve) => {
        setTimeout(() => resolve({
          outputPath: "/tmp/handeye-data/calibration_result.yaml",
          setup: "eye-in-hand",
          primaryTransformName: "T_C2F",
          matrixRows: [
            "1.0000000, 0.0000000, 0.0000000, 0.1000000",
            "0.0000000, 1.0000000, 0.0000000, 0.2000000",
            "0.0000000, 0.0000000, 1.0000000, 0.3000000",
            "0.0000000, 0.0000000, 0.0000000, 1.0000000",
          ],
          averageErrorMm: 2.5,
          rotationErrorDeg: 0.8,
          reprojectionErrorPx: 0.42,
          reprojectionRmsPx: 0.51,
          baseConsistencyRmsMm: 1.6,
          baseConsistencyMeanMm: 1.2,
          baseConsistencyMaxMm: 2.4,
          numImages: 5,
          numImagesUsed: 4,
          filteredImages: [3],
          depthUsed: false,
          frameErrors: [
            {
              index: 0,
              imagePath: "001_Color.png",
              used: true,
              usedChessboardFallback: false,
              cornerCount: 42,
              reprojectionMeanPx: 0.21,
              reprojectionRmsPx: 0.27,
              reprojectionMaxPx: 0.8,
              referenceReprojectionMeanPx: 0.18,
              reprojectionErrorPx: 0.21,
              optimizedReprojectionErrorPx: 0.18,
              baseConsistencyRmsMm: 1.3,
              translationErrorMm: 1.2,
              rotationErrorDeg: 0.3,
            },
            {
              index: 1,
              imagePath: "002_Color.png",
              used: true,
              usedChessboardFallback: true,
              cornerCount: 38,
              reprojectionMeanPx: 0.63,
              reprojectionRmsPx: 0.71,
              reprojectionMaxPx: 1.4,
              referenceReprojectionMeanPx: 0.44,
              reprojectionErrorPx: 0.63,
              optimizedReprojectionErrorPx: 0.44,
              baseConsistencyRmsMm: 1.9,
              translationErrorMm: 3.8,
              rotationErrorDeg: 1.1,
            },
          ],
          message: "T_C2F 标定完成；有效数据 4/5；平均平移误差 2.500 mm",
        }), 20);
      }));

    render(<App />);
    await user.click(screen.getByRole("button", { name: "深度图和 RGB 文件夹浏览" }));
    await user.click(screen.getByRole("button", { name: "机械臂位姿文件浏览" }));
    await user.selectOptions(screen.getByRole("combobox", { name: "机器人数据格式：" }), "rxyz");
    await user.click(screen.getByRole("button", { name: "计算" }));

    expect(screen.getByRole("button", { name: "计算中" })).toBeDisabled();
    expect(screen.getByText("标定计算中...")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "结果 / 误差分析" })).not.toBeDisabled();

    expect(await screen.findByText(/T_C2F 标定完成/)).toBeInTheDocument();
    expect(mockInvoke).toHaveBeenLastCalledWith("run_handeye_calibration", {
      request: expect.objectContaining({
        imageDir: "/tmp/handeye-data",
        posesFile: "/tmp/handeye-data/poses.csv",
        setup: "eye-in-hand",
        marker: "charuco",
        poseFormat: "rxyz",
        cameraIntrinsics: { cx: 640, cy: 360, fx: 600, fy: 600 },
      }),
    });

    await user.click(screen.getByRole("tab", { name: "结果 / 误差分析" }));
    expect(screen.getByText("0.510000 px")).toBeInTheDocument();
    expect(screen.getByText("1.600000 mm")).toBeInTheDocument();
    expect(screen.getByText("4 / 5")).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "角点" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "2D RMS(px)" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "底座3D RMS(mm)" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "平移残差(mm)" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "旋转残差(deg)" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "检测模式" })).toBeInTheDocument();
    expect(screen.getByText("001_Color.png")).toBeInTheDocument();
    expect(screen.getByText("002_Color.png")).toBeInTheDocument();
    expect(screen.getByText("ChArUco")).toBeInTheDocument();
    expect(screen.getByText("Chessboard fallback")).toBeInTheDocument();
    expect(screen.getByText("42")).toBeInTheDocument();
    expect(screen.getByText("38")).toBeInTheDocument();
    expect(screen.getByText("0.270000")).toBeInTheDocument();
    expect(screen.getByText("1.300000")).toBeInTheDocument();
    expect(screen.getByText("0.710000")).toBeInTheDocument();
    expect(screen.getByText("1.900000")).toBeInTheDocument();
    expect(screen.getByText("1.200000")).toBeInTheDocument();
    expect(screen.getByText("3.800000")).toBeInTheDocument();
    expect(screen.getByText("0.300000")).toBeInTheDocument();
    expect(screen.getByText("1.100000")).toBeInTheDocument();
    expect(screen.getByText("1.0000000, 0.0000000, 0.0000000, 0.1000000")).toBeInTheDocument();
    expect(screen.getByText("T_C2F 标定完成；有效数据 4/5；平均平移误差 2.500 mm")).toBeInTheDocument();
  });

  it("syncs the latest calibration matrix and pose format into the conversion tool", async () => {
    const user = userEvent.setup();
    mockOpen
      .mockResolvedValueOnce("/tmp/handeye-data")
      .mockResolvedValueOnce("/tmp/handeye-data/poses.csv");
    mockInvoke
      .mockResolvedValueOnce([{ name: "001_Color.png", path: "/tmp/handeye-data/001_Color.png" }])
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce({ cx: 640, cy: 360, fx: 600, fy: 600 })
      .mockResolvedValueOnce([{ index: 1, content: "pose" }])
      .mockResolvedValueOnce({
        outputPath: "/tmp/handeye-data/calibration_result.yaml",
        setup: "eye-in-hand",
        primaryTransformName: "T_C2F",
        matrixRows: [
          "0.9999000, 0.0000000, 0.0100000, 0.1234000",
          "0.0000000, 1.0000000, 0.0000000, 0.2345000",
          "-0.0100000, 0.0000000, 0.9999000, 0.3456000",
          "0.0000000, 0.0000000, 0.0000000, 1.0000000",
        ],
        averageErrorMm: 1.5,
        message: "done",
      });

    render(<App />);
    await user.click(screen.getByRole("button", { name: "深度图和 RGB 文件夹浏览" }));
    await user.click(screen.getByRole("button", { name: "机械臂位姿文件浏览" }));
    await user.selectOptions(screen.getByRole("combobox", { name: "机器人数据格式：" }), "rxyz");
    await user.click(screen.getByRole("button", { name: "计算" }));
    await screen.findByText("done");

    await user.click(screen.getByRole("tab", { name: "工具" }));
    expect(screen.queryByRole("combobox", { name: "姿态类型：" })).not.toBeInTheDocument();
    expect(screen.queryByRole("textbox", { name: "手眼标定结果矩阵（16个数值，逗号分隔）：" })).not.toBeInTheDocument();
    expect(screen.getByRole("group", { name: "帧列表" })).toBeInTheDocument();
  });

  it("shows calibration preview frames in the left frame list and highlights the selected frame", async () => {
    const user = userEvent.setup();
    mockOpen
      .mockResolvedValueOnce("/tmp/handeye-data")
      .mockResolvedValueOnce("/tmp/handeye-data/poses.csv");
    mockInvoke.mockImplementation((command: string) => {
      if (command === "list_rgb_images") {
        return Promise.resolve([
          { name: "001_Color.png", path: "/tmp/handeye-data/001_Color.png" },
          { name: "002_Color.png", path: "/tmp/handeye-data/002_Color.png" },
        ]);
      }
      if (command === "list_depth_images") {
        return Promise.resolve([]);
      }
      if (command === "read_camera_params") {
        return Promise.resolve({ cx: 640, cy: 360, fx: 600, fy: 600 });
      }
      if (command === "read_pose_file") {
        return Promise.resolve([{ index: 1, content: "pose" }, { index: 2, content: "pose" }]);
      }
      if (command === "run_handeye_calibration") {
        return Promise.resolve({
          outputPath: "/tmp/handeye-data/calibration_result.yaml",
          setup: "eye-in-hand",
          primaryTransformName: "T_C2F",
          primaryMatrixRows: [
            "1.0000000, 0.0000000, 0.0000000, 0.1000000",
            "0.0000000, 1.0000000, 0.0000000, 0.2000000",
            "0.0000000, 0.0000000, 1.0000000, 0.3000000",
            "0.0000000, 0.0000000, 0.0000000, 1.0000000",
          ],
          secondaryTransformName: "T_O2W",
          secondaryMatrixRows: [
            "1.0000000, 0.0000000, 0.0000000, 1.1000000",
            "0.0000000, 1.0000000, 0.0000000, 1.2000000",
            "0.0000000, 0.0000000, 1.0000000, 1.3000000",
            "0.0000000, 0.0000000, 0.0000000, 1.0000000",
          ],
          matrixRows: [
            "1.0000000, 0.0000000, 0.0000000, 0.1000000",
            "0.0000000, 1.0000000, 0.0000000, 0.2000000",
            "0.0000000, 0.0000000, 1.0000000, 0.3000000",
            "0.0000000, 0.0000000, 0.0000000, 1.0000000",
          ],
          frameErrors: [
            { index: 0, imagePath: "001_Color.png", used: true, translationErrorMm: 0.8 },
            { index: 1, imagePath: "002_Color.png", used: false, translationErrorMm: 3.2 },
          ],
          previewFrames: [
            {
              index: 0,
              imagePath: "/tmp/handeye-data/001_Color.png",
              used: true,
              cameraInBase: [
                [1, 0, 0, 0.1],
                [0, 1, 0, 0.2],
                [0, 0, 1, 0.3],
                [0, 0, 0, 1],
              ],
              boardInBase: [
                [1, 0, 0, 1.1],
                [0, 1, 0, 1.2],
                [0, 0, 1, 1.3],
                [0, 0, 0, 1],
              ],
            },
            {
              index: 1,
              imagePath: "/tmp/handeye-data/002_Color.png",
              used: false,
              cameraInBase: [
                [1, 0, 0, 0.4],
                [0, 1, 0, 0.5],
                [0, 0, 1, 0.6],
                [0, 0, 0, 1],
              ],
              boardInBase: [
                [1, 0, 0, 1.4],
                [0, 1, 0, 1.5],
                [0, 0, 1, 1.6],
                [0, 0, 0, 1],
              ],
            },
          ],
          message: "done",
        });
      }
      return Promise.resolve(null);
    });

    render(<App />);
    await user.click(screen.getByRole("button", { name: "深度图和 RGB 文件夹浏览" }));
    await user.click(screen.getByRole("button", { name: "机械臂位姿文件浏览" }));
    await user.click(screen.getByRole("button", { name: "计算" }));
    await screen.findByText("done");

    await user.click(screen.getByRole("tab", { name: "工具" }));
    expect(await screen.findByRole("button", { name: "Frame 000 001_Color.png" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "Frame 001 002_Color.png" })).toHaveClass("is-unused");
    expect(screen.getByText("当前帧：Frame 000")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Frame 001 002_Color.png" }));
    expect(screen.getByText("当前帧：Frame 001")).toBeInTheDocument();

    const outputPanel = screen.getByRole("group", { name: "2. 输出预览" });
    expect(outputPanel.querySelector(".conversion-frame-box")).toBeNull();
    expect(document.querySelector(".conversion-sidebar .conversion-frame-box")).not.toBeNull();
  });

  it("recalculates calibration after excluding selected result frames", async () => {
    const user = userEvent.setup();
    mockOpen
      .mockResolvedValueOnce("/tmp/handeye-data")
      .mockResolvedValueOnce("/tmp/handeye-data/poses.csv");
    mockInvoke
      .mockResolvedValueOnce([{ name: "001_Color.png", path: "/tmp/handeye-data/001_Color.png" }])
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce({ cx: 640, cy: 360, fx: 600, fy: 600 })
      .mockResolvedValueOnce([{ index: 1, content: "762.4474, -128.1737, -28.2222, -171.6236, 13.3687, -78.3524" }])
      .mockResolvedValueOnce({
        outputPath: "/tmp/handeye-data/calibration_result.yaml",
        setup: "eye-to-hand",
        primaryTransformName: "T_C2W",
        matrixRows: ["initial"],
        averageErrorMm: 2.5,
        numImages: 5,
        numImagesUsed: 4,
        frameErrors: [
          { index: 0, imagePath: "001_Color.png", used: true, cornerCount: 42, translationErrorMm: 1.2 },
          { index: 1, imagePath: "002_Color.png", used: true, cornerCount: 38, translationErrorMm: 3.8 },
          { index: 2, imagePath: "003_Color.png", used: false },
        ],
        message: "初次标定完成",
      })
      .mockImplementationOnce(() => new Promise((resolve) => {
        setTimeout(() => resolve({
          outputPath: "/tmp/handeye-data/calibration_result.yaml",
          setup: "eye-to-hand",
          primaryTransformName: "T_C2W",
          matrixRows: ["recalculated"],
          averageErrorMm: 1.5,
          reprojectionRmsPx: 0.33,
          numImages: 5,
          numImagesUsed: 3,
          filteredImages: [1],
          frameErrors: [
            { index: 0, imagePath: "001_Color.png", used: true, cornerCount: 42, translationErrorMm: 1.0 },
          ],
          message: "剔除后重新计算完成",
        }), 20);
      }));

    render(<App />);
    await user.click(screen.getByRole("button", { name: "深度图和 RGB 文件夹浏览" }));
    await user.click(screen.getByRole("button", { name: "机械臂位姿文件浏览" }));
    await user.click(screen.getByRole("radio", { name: "眼在手外" }));
    await user.click(screen.getByRole("button", { name: "计算" }));
    await screen.findByText("初次标定完成");

    await user.click(screen.getByRole("tab", { name: "结果 / 误差分析" }));
    const checkboxes = screen.getAllByRole("checkbox");
    await user.click(checkboxes[1]);
    await user.click(screen.getByRole("button", { name: "重新计算" }));

    expect(screen.getByText("按勾选点位重新计算中...")).toBeInTheDocument();
    expect(mockInvoke).toHaveBeenLastCalledWith("run_handeye_calibration", {
      request: expect.objectContaining({
        imageDir: "/tmp/handeye-data",
        posesFile: "/tmp/handeye-data/poses.csv",
        setup: "eye-to-hand",
        excludedImageIndices: [1, 2],
        filterInconsistent: false,
      }),
    });
    expect(await screen.findByText("剔除后重新计算完成")).toBeInTheDocument();
    expect(screen.getByText("0.330000 px")).toBeInTheDocument();
    expect(screen.getByText("recalculated")).toBeInTheDocument();
    expect(screen.getByText("002_Color.png")).toBeInTheDocument();
    expect(screen.getByText("003_Color.png")).toBeInTheDocument();
    const updatedCheckboxes = screen.getAllByRole("checkbox");
    expect(updatedCheckboxes[1]).not.toBeChecked();
    expect(updatedCheckboxes[2]).not.toBeChecked();
    expect(screen.queryByText("3.800000")).not.toBeInTheDocument();
  });

  it("saves the frontend calibration result to a YAML file chosen by the user", async () => {
    const user = userEvent.setup();
    mockOpen
      .mockResolvedValueOnce("/tmp/handeye-data")
      .mockResolvedValueOnce("/tmp/handeye-data/poses.csv");
    mockSave.mockResolvedValueOnce("/tmp/exported/front_result.yaml");
    mockInvoke
      .mockResolvedValueOnce([{ name: "001_Color.png", path: "/tmp/handeye-data/001_Color.png" }])
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce({ cx: 640, cy: 360, fx: 600, fy: 600 })
      .mockResolvedValueOnce([{ index: 1, content: "pose" }])
      .mockResolvedValueOnce({
        outputPath: "/tmp/handeye-data/calibration_result.yaml",
        setup: "eye-in-hand",
        primaryTransformName: "T_C2F",
        matrixRows: [
          "1.0000000, 0.0000000, 0.0000000, 0.1000000",
          "0.0000000, 1.0000000, 0.0000000, 0.2000000",
          "0.0000000, 0.0000000, 1.0000000, 0.3000000",
          "0.0000000, 0.0000000, 0.0000000, 1.0000000",
        ],
        averageErrorMm: 2.5,
        rotationErrorDeg: 0.8,
        reprojectionErrorPx: 0.42,
        reprojectionRmsPx: 0.51,
        baseConsistencyMeanMm: 1.1,
        baseConsistencyRmsMm: 1.5,
        baseConsistencyMaxMm: 2.3,
        baseConsistencyCount: 80,
        numImages: 5,
        numImagesUsed: 4,
        filteredImages: [3],
        depthUsed: false,
        frameErrors: [
          {
            index: 0,
            imagePath: "001_Color.png",
            used: true,
            cornerCount: 42,
            reprojectionMeanPx: 0.21,
            reprojectionRmsPx: 0.27,
            reprojectionMaxPx: 0.8,
            referenceReprojectionMeanPx: 0.18,
            translationErrorMm: 1.2,
            rotationErrorDeg: 0.3,
          },
        ],
        message: "done",
      })
      .mockResolvedValueOnce(null);

    render(<App />);
    await user.click(screen.getByRole("button", { name: "深度图和 RGB 文件夹浏览" }));
    await user.click(screen.getByRole("button", { name: "机械臂位姿文件浏览" }));
    await user.click(screen.getByRole("button", { name: "计算" }));
    await screen.findByText("done");
    await user.click(screen.getByRole("tab", { name: "结果 / 误差分析" }));
    await user.click(screen.getByRole("button", { name: "保存 YAML" }));

    expect(mockSave).toHaveBeenCalledWith({
      defaultPath: "/tmp/handeye-data/handeye_result.yaml",
      filters: [{ name: "YAML", extensions: ["yaml", "yml"] }],
      title: "保存手眼标定结果",
    });
    expect(mockInvoke).toHaveBeenLastCalledWith("save_text_file", {
      path: "/tmp/exported/front_result.yaml",
      content: expect.stringContaining("setup: eye-in-hand"),
    });
    const saveCall = mockInvoke.mock.calls.at(-1);
    expect(saveCall?.[1]?.content).toContain("T_C2F:");
    expect(saveCall?.[1]?.content).toContain("- [1, 0, 0, 0.1]");
    expect(saveCall?.[1]?.content).toContain("translation_mean_mm: 2.5");
    expect(saveCall?.[1]?.content).toContain("reprojection_rms_px: 0.51");
    expect(saveCall?.[1]?.content).not.toContain("frame_errors:");
    expect(saveCall?.[1]?.content).not.toContain("num_images:");
    expect(saveCall?.[1]?.content).not.toContain("num_images_used:");
    expect(saveCall?.[1]?.content).not.toContain("filtered_images:");
    expect(saveCall?.[1]?.content).not.toContain("depth_used:");
    expect(saveCall?.[1]?.content).not.toContain("001_Color.png");
    expect(screen.getByText("结果已保存：/tmp/exported/front_result.yaml")).toBeInTheDocument();
  });

  it("detects ChArUco markers in tools and displays the overlay image", async () => {
    const user = userEvent.setup();
    mockOpen
      .mockResolvedValueOnce("/tmp/handeye-data")
      .mockResolvedValueOnce("/tmp/handeye-data/001_Color.png");
    let detectCallCount = 0;
    mockInvoke.mockImplementation((command: string, args?: { request?: { depthPath?: string | null } }) => {
      if (command === "list_rgb_images") {
        return Promise.resolve([{ name: "001_Color.png", path: "/tmp/handeye-data/001_Color.png" }]);
      }
      if (command === "list_depth_images") {
        return Promise.resolve([{ name: "001_Depth.png", path: "/tmp/handeye-data/001_Depth.png" }]);
      }
      if (command === "read_camera_params") {
        return Promise.resolve({ cx: 640, cy: 360, fx: 600, fy: 600 });
      }
      if (command === "detect_charuco") {
        detectCallCount += 1;
        return Promise.resolve({
          imagePath: "/tmp/handeye-data/001_Color.png",
          outputPath: "/tmp/handeye-data/detection/detection_001.png",
          success: true,
          numCorners: 48,
          numMarkers: 24,
          usedChessboardFallback: true,
          message: "ok",
          cornerRows: [
            {
              id: 7,
              imagePoint: [766.78, 518.89],
              cameraPoint: detectCallCount === 1 || !args?.request?.depthPath ? null : [0.0029, 0.0011, 0.4378],
            },
          ],
        });
      }
      return Promise.resolve(null);
    });

    render(<App />);
    await user.click(screen.getByRole("button", { name: "深度图和 RGB 文件夹浏览" }));
    await user.click(screen.getByRole("tab", { name: "工具" }));

    expect(screen.getByLabelText("图片文件")).toHaveValue("");
    expect(screen.getByLabelText("深度图")).toHaveValue("");
    expect(screen.queryByLabelText("cx")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("cy")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("fx")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("fy")).not.toBeInTheDocument();
    expect(screen.queryByText("相机内参")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("数据文件夹")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("机器人姿态文件")).not.toBeInTheDocument();
    expect(screen.queryByText("0.png / 0.ply")).not.toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "标记物类型：" })).toHaveValue("charuco");
    expect(screen.getByRole("combobox", { name: "标记物类型：" })).toHaveTextContent("ChArUco");
    expect(screen.getByRole("combobox", { name: "标记物类型：" })).not.toHaveTextContent("同心圆");

    await user.click(screen.getByRole("button", { name: "图片文件浏览" }));
    expect(screen.getByRole("img", { name: "001_Color.png" })).toHaveAttribute(
      "src",
      "asset:///tmp/handeye-data/001_Color.png",
    );

    await user.click(screen.getByRole("button", { name: "识别 ChArUco" }));

    expect(mockInvoke).toHaveBeenLastCalledWith("detect_charuco", {
      request: expect.objectContaining({
        imagePath: "/tmp/handeye-data/001_Color.png",
        depthPath: null,
        cameraIntrinsics: { cx: 640, cy: 360, fx: 600, fy: 600 },
        squaresX: 14,
        squaresY: 9,
        squareLength: 0.02,
        markerLength: 0.015,
        arucoDict: "DICT_5X5_50",
      }),
    });

    mockOpen.mockResolvedValueOnce("/tmp/handeye-data/001_Depth.png");
    await user.click(screen.getByRole("button", { name: "深度图浏览" }));
    await user.click(screen.getByRole("button", { name: "识别 ChArUco" }));

    expect(mockInvoke).toHaveBeenLastCalledWith("detect_charuco", {
      request: expect.objectContaining({
        imagePath: "/tmp/handeye-data/001_Color.png",
        depthPath: "/tmp/handeye-data/001_Depth.png",
        cameraIntrinsics: { cx: 640, cy: 360, fx: 600, fy: 600 },
        squaresX: 14,
        squaresY: 9,
        squareLength: 0.02,
        markerLength: 0.015,
        arucoDict: "DICT_5X5_50",
      }),
    });
    expect(screen.getByText("角点 48 / 标记 24")).toBeInTheDocument();
    expect(screen.getByText("Chessboard fallback")).toBeInTheDocument();
    expect(document.querySelector(".tool-input-column .detection-summary")).not.toBeInTheDocument();
    expect(document.querySelector(".tool-detection-summary .detection-summary")).toHaveTextContent("角点 48 / 标记 24");
    expect(screen.getByRole("img", { name: "ChArUco 检测结果" })).toHaveAttribute(
      "src",
      "asset:///tmp/handeye-data/detection/detection_001.png",
    );
    expect(screen.queryByText("(766.78, 518.89)")).not.toBeInTheDocument();
    expect(screen.queryByText("(0.002900, 0.001100, 0.437800)")).not.toBeInTheDocument();
  });

  it("supports mouse-wheel zooming in the tool image preview", async () => {
    const user = userEvent.setup();
    mockOpen
      .mockResolvedValueOnce("/tmp/handeye-data")
      .mockResolvedValueOnce("/tmp/handeye-data/001_Color.png");
    mockInvoke.mockImplementation((command: string) => {
      if (command === "list_rgb_images") {
        return Promise.resolve([{ name: "001_Color.png", path: "/tmp/handeye-data/001_Color.png" }]);
      }
      if (command === "list_depth_images") {
        return Promise.resolve([]);
      }
      if (command === "read_camera_params") {
        return Promise.resolve({ cx: 640, cy: 360, fx: 600, fy: 600 });
      }
      return Promise.resolve(null);
    });

    render(<App />);
    await user.click(screen.getByRole("button", { name: "深度图和 RGB 文件夹浏览" }));
    await user.click(screen.getByRole("tab", { name: "工具" }));
    await user.click(screen.getByRole("button", { name: "图片文件浏览" }));

    const viewport = screen.getByLabelText("图像预览 001_Color.png");
    const zoomStage = viewport.firstElementChild as HTMLElement;

    expect(viewport).toHaveAttribute("data-zoom", "1.00");
    expect(viewport).toHaveAttribute("data-pan-x", "0");
    expect(viewport).toHaveAttribute("data-pan-y", "0");
    expect(zoomStage).toHaveStyle({ width: "100%", height: "100%" });

    fireEvent.wheel(viewport, { deltaY: -120 });
    expect(viewport).toHaveAttribute("data-zoom", "1.10");
    expect(zoomStage).toHaveStyle({ width: "110%", height: "110%" });

    fireEvent.wheel(viewport, { deltaY: 120 });
    expect(viewport).toHaveAttribute("data-zoom", "1.00");
    expect(viewport).toHaveAttribute("data-pan-x", "0");
    expect(viewport).toHaveAttribute("data-pan-y", "0");
    expect(zoomStage).toHaveStyle({ width: "100%", height: "100%" });
  });

  it("supports drag panning after zooming in the tool image preview", async () => {
    const user = userEvent.setup();
    mockOpen
      .mockResolvedValueOnce("/tmp/handeye-data")
      .mockResolvedValueOnce("/tmp/handeye-data/001_Color.png");
    mockInvoke.mockImplementation((command: string) => {
      if (command === "list_rgb_images") {
        return Promise.resolve([{ name: "001_Color.png", path: "/tmp/handeye-data/001_Color.png" }]);
      }
      if (command === "list_depth_images") {
        return Promise.resolve([]);
      }
      if (command === "read_camera_params") {
        return Promise.resolve({ cx: 640, cy: 360, fx: 600, fy: 600 });
      }
      return Promise.resolve(null);
    });

    render(<App />);
    await user.click(screen.getByRole("button", { name: "深度图和 RGB 文件夹浏览" }));
    await user.click(screen.getByRole("tab", { name: "工具" }));
    await user.click(screen.getByRole("button", { name: "图片文件浏览" }));

    const viewport = screen.getByLabelText("图像预览 001_Color.png");
    const zoomStage = viewport.firstElementChild as HTMLElement;

    fireEvent.wheel(viewport, { deltaY: -120 });
    expect(viewport).toHaveAttribute("data-zoom", "1.10");

    fireEvent.mouseDown(viewport, { clientX: 100, clientY: 80 });
    fireEvent.mouseMove(viewport, { clientX: 136, clientY: 108 });

    expect(viewport).toHaveAttribute("data-pan-x", "-36");
    expect(viewport).toHaveAttribute("data-pan-y", "-28");
    expect(zoomStage).toHaveStyle({ width: "110%", height: "110%" });

    fireEvent.mouseUp(viewport);
    fireEvent.wheel(viewport, { deltaY: 120 });

    expect(viewport).toHaveAttribute("data-zoom", "1.00");
    expect(viewport).toHaveAttribute("data-pan-x", "0");
    expect(viewport).toHaveAttribute("data-pan-y", "0");
    fireEvent.doubleClick(viewport);

    expect(viewport).toHaveAttribute("data-zoom", "1.00");
    expect(viewport).toHaveAttribute("data-pan-x", "0");
    expect(viewport).toHaveAttribute("data-pan-y", "0");
    expect(zoomStage).toHaveStyle({ width: "100%", height: "100%" });
  });

  it("passes edited ChArUco board parameters to tool detection and validates them", async () => {
    const user = userEvent.setup();
    mockOpen
      .mockResolvedValueOnce("/tmp/handeye-data")
      .mockResolvedValueOnce("/tmp/handeye-data/001_Color.png");
    mockInvoke
      .mockResolvedValueOnce([{ name: "001_Color.png", path: "/tmp/handeye-data/001_Color.png" }])
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce({ cx: 640, cy: 360, fx: 600, fy: 600 })
      .mockResolvedValueOnce({
        imagePath: "/tmp/handeye-data/001_Color.png",
        outputPath: "/tmp/handeye-data/detection/detection_001.png",
        success: true,
        numCorners: 48,
        numMarkers: 24,
        message: "ok",
        cornerRows: [],
      });

    render(<App />);
    await user.click(screen.getByRole("button", { name: "深度图和 RGB 文件夹浏览" }));
    await user.click(screen.getByRole("tab", { name: "工具" }));
    await user.click(screen.getByRole("button", { name: "图片文件浏览" }));
    await user.clear(screen.getByLabelText("横向格数"));
    await user.click(screen.getByRole("button", { name: "识别 ChArUco" }));
    expect(screen.getByText("请填写有效的 ChArUco 标定板参数")).toBeInTheDocument();

    await user.type(screen.getByLabelText("横向格数"), "10");
    await user.clear(screen.getByLabelText("纵向格数"));
    await user.type(screen.getByLabelText("纵向格数"), "7");
    await user.clear(screen.getByLabelText("方格边长(m)"));
    await user.type(screen.getByLabelText("方格边长(m)"), "0.03");
    await user.clear(screen.getByLabelText("Marker边长(m)"));
    await user.type(screen.getByLabelText("Marker边长(m)"), "0.02");
    await user.selectOptions(screen.getByLabelText("字典"), "DICT_6X6_250");
    await user.click(screen.getByRole("button", { name: "识别 ChArUco" }));

    expect(mockInvoke).toHaveBeenLastCalledWith("detect_charuco", {
      request: expect.objectContaining({
        squaresX: 10,
        squaresY: 7,
        squareLength: 0.03,
        markerLength: 0.02,
        arucoDict: "DICT_6X6_250",
      }),
    });
  });

  it("keeps the ChArUco coordinate table inside a fixed scrolling area", () => {
    const styles = readFileSync(resolve(process.cwd(), "src/styles.css"), "utf8");

    expect(styles).toMatch(/\.tool-workspace\s*{[^}]*grid-template-columns:\s*minmax\(0,\s*1fr\) minmax\(0,\s*2fr\)/s);
    expect(styles).toMatch(/\.tool-workspace\s*{[^}]*grid-template-rows:\s*minmax\(0,\s*1fr\) auto/s);
    expect(styles).toMatch(/\.tool-workspace\s*{[^}]*gap:\s*16px/s);
    expect(styles).toMatch(/\.tool-workspace\s*{[^}]*height:\s*100%/s);
    expect(styles).toMatch(/\.tool-workspace\s*{[^}]*overflow:\s*hidden/s);
    expect(styles).toMatch(/\.tool-panel\s*{[^}]*padding:\s*10px 12px 12px/s);
    expect(styles).toMatch(/\.tool-panel\s*{[^}]*overflow:\s*hidden/s);
    expect(styles).toMatch(/\.tool-detection-grid\s*{[^}]*grid-template-rows:\s*minmax\(0,\s*1fr\) auto auto/s);
    expect(styles).toMatch(/\.tool-detection-grid\s*{[^}]*min-height:\s*0/s);
    expect(styles).toMatch(/\.tool-detection-grid \.marker-preview\s*{[^}]*height:\s*100%/s);
    expect(styles).toMatch(/\.tool-detection-grid \.marker-preview\s*{[^}]*min-height:\s*0/s);
    expect(styles).toMatch(/\.zoom-viewport\s*{[^}]*overflow:\s*auto/s);
    expect(styles).toMatch(/\.zoom-stage\s*{[^}]*transition:\s*width 120ms ease-out,\s*height 120ms ease-out/s);
    expect(styles).toMatch(/\.zoom-viewport\.is-pannable\s*{[^}]*cursor:\s*grab/s);
    expect(styles).toMatch(/\.parameter-card\s*{[^}]*border:\s*1px solid #304652/s);
    expect(styles).toMatch(/\.tool-board-card \.charuco-param-grid\s*{[^}]*grid-template-columns:\s*repeat\(5,\s*minmax\(0,\s*1fr\)\)/s);
    expect(styles).toMatch(/\.tool-board-card \.param-field input,\s*\.tool-board-card \.param-field select\s*{[^}]*padding:\s*0 5px/s);
    expect(styles).toMatch(/\.tool-input-card\s*{[^}]*grid-column:\s*1 \/ -1/s);
    expect(styles).toMatch(/\.tool-input-card\s*{[^}]*display:\s*grid/s);
    expect(styles).toMatch(/\.tool-input-actions\s*{[^}]*grid-template-columns:\s*minmax\(0,\s*1fr\) 136px/s);
    expect(styles).toMatch(/\.point-cloud-tool-grid\s*{[^}]*grid-template-columns:\s*minmax\(250px,\s*0\.38fr\) minmax\(460px,\s*1\.62fr\)/s);
    expect(styles).toMatch(/\.point-cloud-tool-grid\s*{[^}]*grid-template-rows:\s*minmax\(0,\s*1fr\)/s);
    expect(styles).toMatch(/\.conversion-sidebar\s*{[^}]*grid-template-rows:\s*auto minmax\(0,\s*1fr\) auto/s);
    expect(styles).toMatch(/\.conversion-sidebar\s*{[^}]*height:\s*100%/s);
    expect(styles).toMatch(/\.conversion-sidebar\s*{[^}]*overflow:\s*hidden/s);
    expect(styles).toMatch(/\.conversion-frame-box\s*{[^}]*display:\s*flex/s);
    expect(styles).toMatch(/\.conversion-frame-box\s*{[^}]*min-height:\s*0/s);
    expect(styles).toMatch(/\.conversion-frame-list\s*{[^}]*grid-auto-flow:\s*row/s);
    expect(styles).toMatch(/\.conversion-frame-list\s*{[^}]*flex:\s*1 1 auto/s);
    expect(styles).toMatch(/\.conversion-frame-list\s*{[^}]*min-height:\s*0/s);
    expect(styles).toMatch(/\.conversion-frame-list\s*{[^}]*overflow-y:\s*auto/s);
    expect(styles).toMatch(/\.coordinate-preview-label\s*{[^}]*transform:\s*translate\(0,\s*-50%\)/s);
    expect(styles).toMatch(/\.coordinate-preview-stage\s*{[^}]*min-height:\s*0/s);
    expect(styles).toMatch(/\.coordinate-preview-canvas\s*{[^}]*min-height:\s*0/s);
    expect(styles).toMatch(/\.converted-cloud-view\s*{[^}]*min-height:\s*160px/s);
    expect(styles).toMatch(/\.converted-cloud-view\s*{[^}]*height:\s*100%/s);
  });
});
