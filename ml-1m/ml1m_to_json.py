#!/usr/bin/env python3
import json
import re
import argparse
from pathlib import Path

AGE_MAP = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44",
           45: "45-49", 50: "50-55", 56: "56+"}
OCC_MAP = {0:"other",1:"academic/educator",2:"artist",3:"clerical/admin",
           4:"college/grad student",5:"customer service",6:"doctor/health care",
           7:"executive/managerial",8:"farmer",9:"homemaker",10:"K-12 student",
           11:"lawyer",12:"programmer",13:"retired",14:"sales/marketing",
           15:"scientist",16:"self-employed",17:"technician/engineer",
           18:"tradesman/craftsman",19:"unemployed",20:"writer"}

YEAR_RE = re.compile(r"\((\d{4})\)\s*$")

def parse_users(path):
    users = {}
    with open(path, encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("::")
            if len(parts) != 5: continue
            uid, g, age, occ, zc = parts
            users[int(uid)] = {
                "gender": g,
                "age": AGE_MAP.get(int(age), int(age)),
                "occupation": OCC_MAP.get(int(occ), int(occ)),
                "zip_code": zc,
            }
    return users

def parse_movies(path):
    movies = {}
    with open(path, encoding="latin-1") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line: continue
            parts = line.split("::")
            if len(parts) != 3: continue
            mid, title, genres = parts
            m = YEAR_RE.search(title)
            year = int(m.group(1)) if m else None
            clean_title = YEAR_RE.sub("", title).strip()
            movies[int(mid)] = {
                "title": clean_title,
                "genre": [g.strip() for g in genres.split("|") if g.strip()],
                "year": year,
            }
    return movies

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--format", choices=["jsonl","array"], default="jsonl")
    args = ap.parse_args()

    d = Path(args.data_dir)
    users = parse_users(d/"users.dat")
    movies = parse_movies(d/"movies.dat")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    if args.format == "jsonl":
        fout = open(out_path, "w", encoding="utf-8")
    with open(d/"ratings.dat", encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("::")
            if len(parts) != 4: continue
            uid, mid, rating, ts = parts
            uid, mid = int(uid), int(mid)
            u = users.get(uid, {})
            m = movies.get(mid, {})
            rec = {
                "user_id": uid,
                "age": u.get("age"),
                "occupation": u.get("occupation"),
                "gender": u.get("gender"),
                "zip_code": u.get("zip_code"),
                "item_id": mid,
                "title": m.get("title"),
                "genre": m.get("genre"),
                "year": m.get("year"),
                "score": float(rating),
                "timestamp": int(ts),
            }
            if args.format == "jsonl":
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            else:
                records.append(rec)
    if args.format == "jsonl":
        fout.close()
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()