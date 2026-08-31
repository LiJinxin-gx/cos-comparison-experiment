# Dataset Inventory (Exploration Use)

## Classic Datasets (Retained for Future Use)

| Dataset | Location | Scale | Description |
|---------|----------|-------|-------------|
| ID photos | test/testdata/face | 67 images | 1 person 1 photo, face1/face2 validation (cross-domain) |
| Standard face benchmark | test/testdata/orl/faces | 40 persons x 10 | 5-shot standard face dataset |
| Handwritten digits | E:\testdata\mnist | 60000/10000 | Handwritten digit recognition |
| News groups | E:\testdata\20newsgroups | 18846 | Text clustering |
| Captcha (old) | E:\testdata\captcha_arial | train 500/test 100 | Blurred characters, low quality (deprecated) |
| Captcha (hard) | E:\testdata\captcha_hard | - | Rotation + interference lines |
| Video | test/testdata/video + video_dl | 7+3 MP4 | CPU educational + Blender samples |

## High-Quality Datasets (Newly Downloaded)

| Dataset | Location | Scale | Source |
|---------|----------|-------|--------|
| captcha_hq2 | test/testdata/captcha_hq2 | **4069 images** | Modern captcha dataset (8 classes) |
| face_crop | test/testdata/face_crop | **668 images** | Face recognition crop dataset (2 persons x 250+) |
| face_attendance | test/testdata/face_attendance | 14 images | Attendance face photos |
| meGlass | test/testdata/meGlass | 10 images | High-quality face samples |
| captcha_hq | test/testdata/captcha_hq | 6 images | Early download (small) |

## Download Method

Platform API search -> codeload zip (raw CDN timeout unavailable)
-> extract to test/testdata/ (old data E:\testdata untouched)

## Output and Logs

- Exploration logs: test/explore_logs/ + this docs/
- Video output: test/video_output_v3/
- Agent output: test/agent_output/ surf_output/ behavior_output*/
- Face experiments: test/face_*.py + face_*.log
