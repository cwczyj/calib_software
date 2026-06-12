export type CalibrationMode = "eye-in-hand" | "eye-to-hand";
export type MarkerType = "charuco";

export const modeLabels: Record<CalibrationMode, string> = {
  "eye-in-hand": "眼在手上",
  "eye-to-hand": "眼在手外",
};

export const modeDescriptions: Record<CalibrationMode, string> = {
  "eye-in-hand": "相机安装于机器人末端，固定标记物，机器人多姿态采集图像、点云和位姿。",
  "eye-to-hand": "相机固定于外部基座，机器人末端携带标记物多姿态采集。",
};

export const imageFiles = Array.from({ length: 12 }, (_, index) => `${index}.png`);

export const poseSamples = [
  "762.4474, -128.1737, -28.2222, -171.6236, 13.3687, -78.3524",
  "714.1447, -90.8957, -15.4468, -173.1541, 10.4620, -83.3007",
  "741.5046, -18.0529, 43.8758, -169.7603, -4.7324, -78.1201",
];

export const errorRows = [
  { id: "000", group: 15, error: 0.463341 },
  { id: "001", group: 16, error: 0.42532 },
  { id: "002", group: 1, error: 1.540962 },
  { id: "003", group: 7, error: 1.03472 },
  { id: "004", group: 17, error: 0.39143 },
  { id: "005", group: 6, error: 1.110836 },
  { id: "006", group: 5, error: 1.113081 },
  { id: "007", group: 11, error: 0.782066 },
  { id: "008", group: 9, error: 0.876022 },
  { id: "009", group: 4, error: 1.387219 },
];

export const matrices: Record<CalibrationMode, string[]> = {
  "eye-in-hand": [
    "0.0250517, 0.9987762, 0.0126142, 41.7673600",
    "-0.9996355, 0.0219798, 0.0152387, -125.3131368",
    "0.0149610, -0.0130124, 0.9998032, 0.0011492",
    "0.0000000, 0.0000000, 0.0000000, 1.0000000",
  ],
  "eye-to-hand": [
    "0.9991024, -0.0124550, 0.0404981, 14.7674000",
    "0.0131840, 0.9997613, -0.0174250, 193.3180000",
    "-0.0402660, 0.0179530, 0.9990270, 174.7000000",
    "0.0000000, 0.0000000, 0.0000000, 1.0000000",
  ],
};

export const centerRows = [
  { name: "0.png / 0.ply", center: "(766.78, 518.89)", point: "(0.0029, 0.0011, 0.4378)" },
  { name: "1.png / 1.ply", center: "(641.03, 610.73)", point: "(17.2033, 154.8003, 446.6647)" },
  { name: "2.png / 2.ply", center: "(508.16, 521.80)", point: "(33.4144, 77.1584, 493.5819)" },
];
