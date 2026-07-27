# Apple Surface Defect Detection Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Build a Python + OpenCV apple defect detection and quality grading program.

**Architecture:** Single-file CLI: preprocess -> segment_apple -> detect_defects -> classify_quality -> visualize_result.

**Tech Stack:** Python 3.13, OpenCV 4.13, NumPy 2.1

---

### Task 1: Project Scaffolding
**Files:** Create E:\apple\apple_defect_detector.py, images/, output/
- [ ] Step 1: Create directories
- [ ] Step 2: Write imports, constants, argparse CLI skeleton
- [ ] Step 3: Smoke test: python apple_defect_detector.py --help

### Task 2: Image Preprocessing
**Files:** Modify pple_defect_detector.py
- [ ] Step 1: Add preprocess() - resize to 500px width, GaussianBlur
- [ ] Step 2: Update process_single_image() to call preprocess

### Task 3: Apple Segmentation
**Files:** Modify pple_defect_detector.py
- [ ] Step 1: Add segment_apple() - HSV threshold (red+green), morph ops, find max contour
- [ ] Step 2: Update process_single_image() with segmentation

### Task 4: Defect Detection
**Files:** Modify pple_defect_detector.py
- [ ] Step 1: Add detect_defects() - HSV color threshold + Laplacian texture, combine, clean
- [ ] Step 2: Update process_single_image() with detection

### Task 5: Quality Classification
**Files:** Modify pple_defect_detector.py
- [ ] Step 1: Add classify_quality() - defect ratio <3% Premium, 3-15% Standard, >15% Reject
- [ ] Step 2: Update process_single_image() with classification

### Task 6: Visualization & Output
**Files:** Modify pple_defect_detector.py
- [ ] Step 1: Add isualize_result() - draw contours, overlay, info panel
- [ ] Step 2: Add save_result() - save annotated image to output/
- [ ] Step 3: Complete process_single_image() end-to-end

### Task 7: Test & Verify
**Files:** Create E:\apple\test_detector.py
- [ ] Step 1: Create test script with synthetic apple images (good + defective)
- [ ] Step 2: Run test: python test_detector.py
- [ ] Step 3: Run CLI: python apple_defect_detector.py output/test_bad.jpg
