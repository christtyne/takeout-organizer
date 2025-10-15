# 📸 Takeout Organizer
**Automatically organize and clean your Google Takeout photos and videos**

---

# ⚡ TL;DR - Quick Start

### 1️⃣ Install Python & dependencies
- Install **Python 3.8+**
- In a terminal, run:
```bash
pip install -r requirements.txt
```

### 2️⃣ Put your Google Takeout files in a folder
Example structure:
```
takeout-organizer/
├─ extract_archives.py
├─ takeout_setup.py
└─ your_google_takeout/     ← all the .zip or .tar files here
```

### 3️⃣ Extract the archives
```bash
python extract_archives.py --source your_google_takeout --target extracted --workers 8
```
This unpacks everything into `extracted/`.

### 4️⃣ Process and organize
From the project folder:
```bash
python takeout_setup.py --mode yyyy-mm
```
What this does (hands-free):
- reads metadata (JSON & EXIF)
- chooses the best timestamp
- computes hashes (for duplicates)
- removes dupes
- organizes into `organized/YYYY/MM/`

### 5️⃣ (Optional) Resume later
If you stop midway:
```bash
python takeout_setup.py --resume --mode yyyy-mm
```

### 6️⃣ (Optional) Clean up leftover empty folders
```bash
python clean_empty_folders.py
```
(Use `--dry-run` to preview without deleting.)


✅ Result: Everything organized under `organized/YYYY/MM/`, duplicates removed, and logs saved.


---

This tool helps you sort, deduplicate, and rename your Google Photos Takeout files — turning the huge, messy export into a clean, organized library.  
You don’t need to be a programmer to use it!

---

## 🌟 What It Does
When you download your Google Photos **Takeout**, Google gives you a lot of mixed files and JSONs.  
Takeout Organizer automates the cleanup:

| Step | What It Does |
|------|---------------|
| **1️⃣ Extract Archives** | Unpacks all `.zip` and `.tar` files from Google Takeout into one folder. |
| **2️⃣ Extract Metadata** | Reads each photo/video’s JSON file and saves metadata (timestamps, titles, etc.) into a database. |
| **3️⃣ Choose Best Timestamp** | Picks the most accurate date/time from EXIF, JSON, or filename. |
| **4️⃣ Compute Hashes** | Creates image/video fingerprints so duplicates can be found later. |
| **5️⃣ Deduplicate Files** | Detects exact and near-duplicates using advanced image/video comparison (SSIM and perceptual hash). |
| **6️⃣ Reorganize Files** | Moves everything into folders by year/month. |
| **7️⃣ Clean Empty Folders** | Removes leftover empty directories. |

After this process, your Takeout library becomes tidy, duplicate-free, and easy to browse.

---

## 🧰 Requirements

- **Python 3.8+**
- Works on **Windows, macOS, or Linux**
- Recommended: at least **4GB RAM** for large Takeouts

### Install dependencies
Open a terminal (or Command Prompt) in the project folder and run:
```bash
pip install -r requirements.txt
```
If you don’t have `pip`, see [Python’s installation guide](https://www.python.org/downloads/).

---

## 📂 Folder Setup

Prepare your folders like this:

```
takeout-organizer/
├── takeout_setup.py
├── extract_archives.py
├── extract_meta.py
├── choose_timestamp.py
├── dedupe.py
├── reorganize.py
├── clean_empty_folders.py
└── your_google_takeout/        ← put all Takeout .zip/.tar files here
```

---

## 🚀 How To Use

### Step 1 — Extract Google Takeout archives
```bash
python extract_archives.py --source your_google_takeout --target extracted --workers 8
```
This will unpack all `.zip` or `.tar` archives into `extracted/`.

---

### Step 2 — Run the Setup
Once everything is extracted, just run:
```bash
python takeout_setup.py
```
It will guide you through all processing steps automatically:
- Connects to the internal database  
- Extracts metadata from JSONs  
- Picks best timestamps  
- Computes image and video hashes  
- Detects duplicates  
- Reorganizes photos into folders by year/month  
- Cleans up leftovers  

You can stop and resume anytime — the progress is saved in a database file (`media_catalog.db`).

---

### Optional Flags

You can customize behavior with command-line options:

| Option | Example | Description |
|--------|----------|-------------|
| `--resume` | `python takeout_setup.py --resume` | Resume from where you left off |
| `--mode` | `python takeout_setup.py --mode yyyy-mm` | Organize into folders by year/month instead of only year |
| `--workers` | `python extract_archives.py --workers 8` | Control how many threads are used for extraction |
| `--dry-run` | `python clean_empty_folders.py --path organized --dry-run` | Show what would be removed, without deleting anything |

---

## 🧩 Outputs

After running, you’ll get:

```
organized/
├── 2023/
│   ├── 01/
│   ├── 02/
│   └── ...
├── 2024/
│   └── ...
└── logs/
```

All files are renamed to their **creation timestamp**, for example:
```
2020-08-12_14-35-21.jpg
2020-08-12_14-35-21_duplicate.jpg
```

---

## 🛠 Troubleshooting

- If you see errors like `ValueError: month must be in 1..12`, don’t worry — the program logs them and keeps going.
- Logs are saved in `logs/` next to each script.
- For a fresh start, you can safely delete the database file (`media_catalog.db`) and re-run the setup.

---

## ❤️ Credits

Developed for the **Google Photos Takeout** cleanup workflow.  
Uses:
- [Pillow](https://python-pillow.org/) for image handling  
- [OpenCV](https://opencv.org/) and [scikit-image](https://scikit-image.org/) for SSIM  
- [videohash](https://pypi.org/project/videohash/) for video fingerprints  
- [tqdm](https://tqdm.github.io/) for progress bars  

---

## ☕ Tip

If you have a huge Takeout (hundreds of GBs), start by testing on a small subset:
```bash
python takeout_setup.py --mode yyyy-mm
```
Once you’re happy with the results, run it on the full export.
