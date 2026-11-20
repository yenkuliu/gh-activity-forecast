"""Download and verify complete hours from GH Archive."""
import argparse
from datetime import date, datetime, timedelta, timezone
import gzip
import http.client
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import build_opener, HTTPRedirectHandler, Request
import zlib


HOST = "data.gharchive.org"
HOUR = timedelta(hours=1)
BLOCK_SIZE = 1024 * 1024
GZIP_ERRORS = (gzip.BadGzipFile, EOFError, zlib.error)


class ArchiveRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urlsplit(newurl)
        if target.scheme != "https" or target.netloc != HOST:
            raise URLError("Redirect outside the GH Archive download host")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def parse_time(value):
    """Date-only inputs are UTC; timestamps must include an explicit UTC offset."""
    if len(value) == 10:
        result = datetime.combine(date.fromisoformat(value), datetime.min.time(), timezone.utc)
    else:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None:
            raise ValueError("Timestamps must include a UTC offset")
        result = result.astimezone(timezone.utc)
    if result.minute or result.second or result.microsecond:
        raise ValueError("The range must start and end on whole UTC hours")
    return result


def verify_gzip(path):
    with path.open("rb") as raw:
        if raw.read(2) != b"\x1f\x8b":
            raise gzip.BadGzipFile("Missing gzip header")
        raw.seek(0)
        with gzip.GzipFile(fileobj=raw) as stream:
            while stream.read(BLOCK_SIZE):
                pass  # Reading through EOF validates every member's checksum.


def fetch(hour, directory, opener):
    filename = f"{hour:%Y-%m-%d}-{hour.hour}.json.gz"
    destination = directory / filename
    if destination.exists():
        try:
            verify_gzip(destination)
            return "verified"
        except GZIP_ERRORS:
            pass  # Replace a damaged file only after a valid download succeeds.

    request = Request(f"https://{HOST}/{filename}", headers={
        "User-Agent": "GitHubActivityForecast/1.0", "Accept-Encoding": "identity",
    })
    for attempt in range(3):
        temp = None
        try:
            with opener.open(request, timeout=90) as response:
                with tempfile.NamedTemporaryFile(dir=directory, prefix=filename + ".", suffix=".part",
                                                 delete=False) as out:
                    temp = Path(out.name)
                    shutil.copyfileobj(response, out, BLOCK_SIZE)
                    length = response.headers.get("Content-Length")
                    if length is not None and out.tell() != int(length):
                        raise EOFError("Incomplete HTTP response")
            verify_gzip(temp)
            os.replace(temp, destination)
            return "downloaded"
        except HTTPError as err:
            err.close()
            if err.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
        except (URLError, TimeoutError, http.client.HTTPException, *GZIP_ERRORS):
            if attempt == 2:
                raise
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)
        time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="Inclusive UTC date or offset-aware timestamp")
    parser.add_argument("--end", required=True, help="Exclusive UTC date or offset-aware timestamp")
    parser.add_argument("--out", required=True, type=Path, help="Directory for the .json.gz files")
    args = parser.parse_args()
    try:
        start, end = parse_time(args.start), parse_time(args.end)
    except ValueError as err:
        parser.error(str(err))
    if start >= end:
        parser.error("--start must precede --end")
    latest = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    if end > latest:
        parser.error("--end would include an incomplete or future hour")
    args.out.mkdir(parents=True, exist_ok=True)
    opener = build_opener(ArchiveRedirect())
    hour, missing = start, 0
    while hour < end:
        try:
            status = fetch(hour, args.out, opener)
            print(f"{hour.isoformat()} {status}", flush=True)
        except HTTPError as err:
            if err.code != 404:
                parser.exit(1, f"{hour.isoformat()} failed: {err}\n")
            missing += 1
            print(f"{hour.isoformat()} missing from GH Archive (HTTP 404)", file=sys.stderr, flush=True)
        except (OSError, http.client.HTTPException, *GZIP_ERRORS) as err:
            parser.exit(1, f"{hour.isoformat()} failed: {err}\n")
        hour += HOUR
    if missing:
        parser.exit(1, f"Unavailable GH Archive hours: {missing}\n")


if __name__ == "__main__":
    main()
