# 数据清单 (探索用)

## 经典数据 (保留待用)

| 数据 | 位置 | 规模 | 说明 |
|------|------|------|------|
| 证件照 | test/testdata/face | 67 张 | 1 人 1 张, face1/face2 验证 (跨域) |
| ORL | test/testdata/orl/faces | 40人×10 | 5-shot 标准人脸库 |
| MNIST | E:\testdata\mnist | 60000/10000 | 手写数字 |
| 20newsgroups | E:\testdata\20newsgroups | 18846 | 文本聚类 |
| 验证码(旧) | E:\testdata\captcha_arial | train 500/test 100 | 字符模糊, 质量差 (弃用待用) |
| 验证码(硬) | E:\testdata\captcha_hard | - | 旋转+干扰线 |
| 视频 | test/testdata/video + video_dl | 7+3 个 MP4 | CPU 科普 + Blender 样片 |

## 高质量数据 (新下载)

| 数据 | 位置 | 规模 | 来源 |
|------|------|------|------|
| captcha_hq2 | test/testdata/captcha_hq2 | **4069 张** | orlov-ai/hcaptcha-dataset (8 类现代 hcaptcha) |
| face_crop | test/testdata/face_crop | **668 张** | ahmetozlu/face_recognition_crop (2人×250+) |
| face_attendance | test/testdata/face_attendance | 14 张 | 考勤人脸 |
| meGlass | test/testdata/meGlass | 10 张 | 高质量人脸样本 |
| captcha_hq | test/testdata/captcha_hq | 6 张 | 早期下载 (小) |

## 下载方式

GitHub API 搜索定位 → codeload zip (raw.githubusercontent 超时不可用)
→ 解压至 test/testdata/ (旧数据 E:\testdata 未动)

## 输出与日志

- 探索日志: test/explore_logs/ + 本 docs/
- 视频输出: test/video_output_v3/
- Agent 输出: test/agent_output/ surf_output/ behavior_output*/
- 人脸实验: test/face_*.py + face_*.log
