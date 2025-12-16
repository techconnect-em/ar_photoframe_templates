# AR Photoframe Templates

このリポジトリは、スマートフォンのブラウザ上で動作する **ARフォトフレーム Webアプリのテンプレ集**です。  
スマホカメラの映像に、静止PNGまたは透過フレーム（WebM / 連番PNGなど）を重ね、撮影・保存・共有までを行います。

受託制作の運用を想定しており、案件が来たら **要件に合うテンプレを選んでフォルダごとコピーし、素材を差し替えてデプロイ**する流れを基本にしています。

---

## テンプレ一覧（例）

- `face_basic/`：フレーム重ね＋撮影（ベーステンプレ）
- `face-crown/`：顔認識して王冠PNGを頭に追従（予定/開発中）
- `pose-medal/`：体（肩位置）認識してメダルPNGを首に追従（予定/開発中）
- `hand-trophy/`：手認識してトロフィーPNGを手元に追従（予定/開発中）

※ 実際に存在するテンプレ名は、リポジトリ内のフォルダ構成を参照してください。

---

## ディレクトリ構造（テンプレはフォルダ内で完結）

本リポジトリでは、**テンプレごとに自己完結**させる方針です。  
そのため、テンプレ間で素材やJSを共有しない前提で構成しています（詳細は `AGENTS.md` を参照してください）。

```txt
project-root/
├── README.md
├── AGENTS.md
├── index.html                 # （任意）テンプレ一覧ページ
└── <template-name>/           # 例: face_basic, face-crown, pose-medal, hand-trophy
    ├── index.html
    ├── main.js
    ├── styles.css
    └── assets/
        ├── frame.png
        ├── frame_alpha.webm
        └── ios_frame_*.png
