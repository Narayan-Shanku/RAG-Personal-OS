import argparse

from app.config import settings
from app.db import DB
from app.ingest.indexer import POSIndexer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="*", default=list(settings.modes))
    args = parser.parse_args()

    db = DB(settings.db_path)
    db.init()

    idx = POSIndexer(db=db)
    idx.ensure_dirs()

    modes = [m for m in args.modes if m in settings.modes]
    for m in modes:
        stats = idx.index_mode(m)
        print(stats)


if __name__ == "__main__":
    main()