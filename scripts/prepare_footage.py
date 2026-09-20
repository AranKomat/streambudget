"""Acquire five small openly licensed real clips and retain source provenance.

No inference. Full originals remain local; previews are for evaluator annotation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

from streambudget.replay import prepare_video

SOURCES = {
    "c01": "File:Gigaset Cordless Telephone Production VII - Pneumatic Conveyor Belt.webm",
    "c02": "File:Gigaset Cordless Telephone Production II - Engel Injection Moulding Machine.webm",
    "c03": "File:JFC-UA Service Members Train NGOs, Liberians on Forklift 150110-A-YW926-001.webm",
    "c04": "File:Twin Cities METRO Blue Line Doors Closing Chime.webm",
    "c05": "File:Red green flashing pedestrian crossing.webm",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if not args.download:
        parser.error("Acquisition requires --download after reviewing source rights")
    args.out.mkdir(parents=True, exist_ok=False)
    provenance = []
    with httpx.Client(timeout=60, follow_redirects=True, headers={"User-Agent": "StreamBudgetResearch/0.1"}) as client:
        for cid, title in SOURCES.items():
            response = client.get("https://commons.wikimedia.org/w/api.php", params={
                "action": "query", "format": "json", "titles": title, "prop": "imageinfo",
                "iiprop": "url|size|sha1|extmetadata"})
            response.raise_for_status()
            page = next(iter(response.json()["query"]["pages"].values()))
            info = page["imageinfo"][0]
            meta = info["extmetadata"]
            license_name = meta["LicenseShortName"]["value"]
            if license_name not in {"CC BY 3.0", "CC0", "Public domain"} or meta.get("Restrictions", {}).get("value"):
                raise ValueError("Source rights changed; review before downloading")
            if info["size"] > 30_000_000:
                raise ValueError("Unexpectedly large source")
            folder = args.out / cid
            folder.mkdir()
            video = folder / "source.webm"
            content = client.get(info["url"])
            content.raise_for_status()
            if len(content.content) != info["size"]:
                raise ValueError("Source size mismatch")
            video.write_bytes(content.content)
            sha = hashlib.sha256(content.content).hexdigest()
            provenance.append({"id": cid, "title": title, "pageid": page["pageid"], "sha256": sha,
                               "info": info, "derivatives": "JPEG resize, contact sheets; audio not sent to models"})
            (args.out / "sources.json").write_text(json.dumps(provenance, indent=2) + "\n")
            prepare_video(video, folder / "prepared", source=cid, fps=2, max_seconds=60)
            rows = [json.loads(line) for line in (folder / "prepared/events.jsonl").read_text().splitlines()]
            # A whole-clip overview, followed by denser pages for transition inspection.
            groups = [[rows[round(i * (len(rows) - 1) / 15)] for i in range(16)]]
            groups.extend(rows[start:start + 16] for start in range(0, len(rows), 16))
            for n, group in enumerate(groups):
                sheet = Image.new("RGB", (1280, 4 * 205), "white")
                for i, row in enumerate(group):
                    image = Image.open(folder / "prepared" / row["media"])
                    image.thumbnail((320, 180))
                    x, y = i % 4 * 320, i // 4 * 205
                    sheet.paste(image, (x, y))
                    ImageDraw.Draw(sheet).text((x + 5, y + 183), f"{cid} t={row['ts']:.3f}", fill="black")
                sheet.save(folder / f"contact-{n:02}.jpg")
            print(json.dumps({"id": cid, "frames": len(rows), "last_ts": rows[-1]["ts"],
                              "license": license_name, "sha256": sha}), flush=True)


if __name__ == "__main__":
    main()
